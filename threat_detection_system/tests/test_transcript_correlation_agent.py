"""
Unit tests for the Transcript Correlation Agent.

llm_client.generate is monkeypatched so these tests never make a real LLM
call -- the deterministic scoring logic is what's under test here.
"""

from src.agents import transcript_correlation_agent
from src.schemas import AgentFinding, Domain, Evidence, Severity, ThreatCategory


def _finding(agent_name: str, category: ThreatCategory, severity: Severity | None) -> AgentFinding:
    return AgentFinding(
        agent_name=agent_name,
        domain=Domain.TRANSCRIPT,
        category=category,
        severity=severity,
        confidence=0.8,
        evidence=Evidence(turn_index=0),
        summary=f"{agent_name} summary",
    )


def test_no_findings_yields_zero_score(monkeypatch):
    monkeypatch.setattr(transcript_correlation_agent.llm_client, "generate", lambda *a, **k: "no notable signals")

    result = transcript_correlation_agent.correlate_transcript_findings([])

    assert result.domain == Domain.TRANSCRIPT
    assert result.score == 0.0
    assert result.contributing_findings == []


def test_single_critical_finding_sets_base_score(monkeypatch):
    monkeypatch.setattr(transcript_correlation_agent.llm_client, "generate", lambda *a, **k: "explicit threat found")

    findings = [_finding("threat", ThreatCategory.THREAT_OF_VIOLENCE, Severity.CRITICAL)]
    result = transcript_correlation_agent.correlate_transcript_findings(findings)

    assert result.score == 100.0
    assert result.summary == "explicit threat found"
    assert result.contributing_findings == findings


def test_corroborating_findings_add_a_bonus(monkeypatch):
    monkeypatch.setattr(transcript_correlation_agent.llm_client, "generate", lambda *a, **k: "summary")

    findings = [
        _finding("verbal_abuse", ThreatCategory.VERBAL_ABUSE, Severity.HIGH),
        _finding("fraud_social_engineering", ThreatCategory.FRAUD_SOCIAL_ENGINEERING, Severity.MEDIUM),
    ]
    result = transcript_correlation_agent.correlate_transcript_findings(findings)

    # base=75 (high) + bonus min(2-1,3)*5=5.
    assert result.score == 80.0


def test_generate_is_called_with_category_context(monkeypatch):
    captured = {}

    def fake_generate(prompt, *, system_prompt=None):
        captured["prompt"] = prompt
        return "summary"

    monkeypatch.setattr(transcript_correlation_agent.llm_client, "generate", fake_generate)

    transcript_correlation_agent.correlate_transcript_findings(
        [_finding("compliance", ThreatCategory.COMPLIANCE_VIOLATION, Severity.LOW)]
    )

    assert "compliance_violation" in captured["prompt"]
