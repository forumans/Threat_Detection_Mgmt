"""
Unit tests for the Threat Correlation & Decision Agent.

llm_client.build_detection_agent is monkeypatched to a fake PydanticAI-shaped
agent, so these tests never make a real LLM call. The deterministic
risk-scoring and alert-threshold logic is what's primarily under test here.
"""

from types import SimpleNamespace

from src.agents import threat_correlation_decision_agent as agent_module
from src.schemas import Domain, DomainScore, Severity, ThreatCategory


class _FakeDecisionAgent:
    def __init__(self, final_category, final_severity, decision_summary):
        self._output = SimpleNamespace(
            final_category=final_category, final_severity=final_severity, decision_summary=decision_summary
        )

    def run_sync(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(output=self._output)


def _domain_score(domain: Domain, score: float, summary: str = "summary") -> DomainScore:
    return DomainScore(domain=domain, score=score, summary=summary, contributing_findings=[])


def test_low_scores_in_both_domains_do_not_alert(monkeypatch):
    fake_agent = _FakeDecisionAgent(None, None, "Nothing notable.")
    monkeypatch.setattr(agent_module.llm_client, "build_detection_agent", lambda output_type, prompt: fake_agent)

    audio = _domain_score(Domain.AUDIO, 10.0)
    transcript = _domain_score(Domain.TRANSCRIPT, 10.0)
    result = agent_module.make_threat_assessment(audio, transcript)

    assert result.risk_score == 10.0
    assert result.alert_decision is False


def test_single_domain_high_score_alerts_without_corroboration_bonus(monkeypatch):
    fake_agent = _FakeDecisionAgent(ThreatCategory.THREAT_OF_VIOLENCE, Severity.HIGH, "Explicit threat in audio.")
    monkeypatch.setattr(agent_module.llm_client, "build_detection_agent", lambda output_type, prompt: fake_agent)

    audio = _domain_score(Domain.AUDIO, 80.0)
    transcript = _domain_score(Domain.TRANSCRIPT, 0.0)
    result = agent_module.make_threat_assessment(audio, transcript)

    # No bonus: transcript score is below the corroboration minimum.
    assert result.risk_score == 80.0
    assert result.alert_decision is True
    assert result.final_category == ThreatCategory.THREAT_OF_VIOLENCE
    assert result.final_severity == Severity.HIGH


def test_corroborating_moderate_scores_add_bonus_and_reach_alert_threshold(monkeypatch):
    fake_agent = _FakeDecisionAgent(ThreatCategory.VERBAL_ABUSE, Severity.MEDIUM, "Corroborated across domains.")
    monkeypatch.setattr(agent_module.llm_client, "build_detection_agent", lambda output_type, prompt: fake_agent)

    audio = _domain_score(Domain.AUDIO, 50.0)
    transcript = _domain_score(Domain.TRANSCRIPT, 50.0)
    result = agent_module.make_threat_assessment(audio, transcript)

    # base=50 + bonus min(50,50)*0.2=10 -> 60, which meets the alert threshold.
    assert result.risk_score == 60.0
    assert result.alert_decision is True


def test_result_preserves_both_input_domain_scores(monkeypatch):
    fake_agent = _FakeDecisionAgent(None, None, "summary")
    monkeypatch.setattr(agent_module.llm_client, "build_detection_agent", lambda output_type, prompt: fake_agent)

    audio = _domain_score(Domain.AUDIO, 30.0, summary="audio summary")
    transcript = _domain_score(Domain.TRANSCRIPT, 20.0, summary="transcript summary")
    result = agent_module.make_threat_assessment(audio, transcript)

    assert result.audio_domain_score is audio
    assert result.transcript_domain_score is transcript
    assert "audio summary" in fake_agent.last_prompt
    assert "transcript summary" in fake_agent.last_prompt
