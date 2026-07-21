"""
Unit tests for the Audio Correlation Agent.

llm_client.generate is monkeypatched so these tests never make a real LLM
call -- the deterministic scoring logic is what's under test here.
"""

from src.agents import audio_correlation_agent
from src.schemas import AgentFinding, Domain, Evidence, Severity


def _finding(agent_name: str, severity: Severity | None) -> AgentFinding:
    return AgentFinding(
        agent_name=agent_name,
        domain=Domain.AUDIO,
        category=None,
        severity=severity,
        confidence=0.8,
        evidence=Evidence(start_ms=0, end_ms=1000),
        summary=f"{agent_name} summary",
    )


def test_no_findings_yields_zero_score(monkeypatch):
    monkeypatch.setattr(audio_correlation_agent.llm_client, "generate", lambda *a, **k: "no notable signals")

    result = audio_correlation_agent.correlate_audio_findings([])

    assert result.domain == Domain.AUDIO
    assert result.score == 0.0
    assert result.contributing_findings == []


def test_single_high_severity_finding_sets_base_score(monkeypatch):
    monkeypatch.setattr(audio_correlation_agent.llm_client, "generate", lambda *a, **k: "elevated stress")

    findings = [_finding("emotion", Severity.HIGH)]
    result = audio_correlation_agent.correlate_audio_findings(findings)

    assert result.score == 75.0
    assert result.summary == "elevated stress"
    assert result.contributing_findings == findings


def test_corroborating_findings_add_a_bonus_capped_at_100(monkeypatch):
    monkeypatch.setattr(audio_correlation_agent.llm_client, "generate", lambda *a, **k: "summary")

    findings = [
        _finding("emotion", Severity.CRITICAL),
        _finding("prosody", Severity.HIGH),
        _finding("background_audio", Severity.MEDIUM),
    ]
    result = audio_correlation_agent.correlate_audio_findings(findings)

    # base=100 (critical) + bonus min(3-1,3)*5=10, capped at 100.
    assert result.score == 100.0


def test_findings_without_severity_are_ignored_in_scoring(monkeypatch):
    monkeypatch.setattr(audio_correlation_agent.llm_client, "generate", lambda *a, **k: "summary")

    findings = [_finding("prosody", None), _finding("background_audio", None)]
    result = audio_correlation_agent.correlate_audio_findings(findings)

    assert result.score == 0.0


def test_generate_is_called_with_findings_context(monkeypatch):
    captured = {}

    def fake_generate(prompt, *, system_prompt=None):
        captured["prompt"] = prompt
        captured["system_prompt"] = system_prompt
        return "summary"

    monkeypatch.setattr(audio_correlation_agent.llm_client, "generate", fake_generate)

    audio_correlation_agent.correlate_audio_findings([_finding("emotion", Severity.LOW)])

    assert "emotion" in captured["prompt"]
    assert captured["system_prompt"] is not None
