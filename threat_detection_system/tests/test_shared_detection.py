"""
Unit tests for the shared transcript-detection helper used by all four
transcript-domain detection agents.

llm_client.build_detection_agent is monkeypatched to a fake PydanticAI-shaped
agent, so these tests never make a real LLM call.
"""

from types import SimpleNamespace

from src.agents import _shared_detection
from src.schemas import Severity, ThreatCategory, TranscriptTurn


class _FakeSpan:
    def __init__(self, turn_index, severity, confidence, explanation):
        self.turn_index = turn_index
        self.severity = severity
        self.confidence = confidence
        self.explanation = explanation


class _FakeAgent:
    def __init__(self, spans):
        self._spans = spans

    def run_sync(self, prompt):
        return SimpleNamespace(output=SimpleNamespace(spans=self._spans))


def _turns() -> list[TranscriptTurn]:
    return [
        TranscriptTurn(turn_index=0, speaker_label="caller", start_ms=0, end_ms=1000, text="Hello"),
        TranscriptTurn(turn_index=1, speaker_label="agent", start_ms=1000, end_ms=2000, text="Hi there"),
    ]


def test_empty_transcript_short_circuits_without_calling_llm(monkeypatch):
    called = {"count": 0}

    def fake_build(output_type, system_prompt):
        called["count"] += 1
        return _FakeAgent([])

    monkeypatch.setattr(_shared_detection.llm_client, "build_detection_agent", fake_build)

    findings = _shared_detection.run_transcript_detection(
        [], agent_name="test_agent", category=ThreatCategory.VERBAL_ABUSE, system_prompt="prompt"
    )

    assert findings == []
    assert called["count"] == 0


def test_detected_spans_are_mapped_to_agent_findings(monkeypatch):
    spans = [_FakeSpan(turn_index=1, severity=Severity.HIGH, confidence=0.9, explanation="explicit threat")]
    monkeypatch.setattr(
        _shared_detection.llm_client, "build_detection_agent", lambda output_type, system_prompt: _FakeAgent(spans)
    )

    findings = _shared_detection.run_transcript_detection(
        _turns(), agent_name="threat", category=ThreatCategory.THREAT_OF_VIOLENCE, system_prompt="prompt"
    )

    assert len(findings) == 1
    f = findings[0]
    assert f.agent_name == "threat"
    assert f.category == ThreatCategory.THREAT_OF_VIOLENCE
    assert f.severity == Severity.HIGH
    assert f.confidence == 0.9
    assert f.evidence.turn_index == 1
    assert f.summary == "explicit threat"


def test_no_spans_detected_yields_empty_findings(monkeypatch):
    monkeypatch.setattr(
        _shared_detection.llm_client, "build_detection_agent", lambda output_type, system_prompt: _FakeAgent([])
    )

    findings = _shared_detection.run_transcript_detection(
        _turns(), agent_name="verbal_abuse", category=ThreatCategory.VERBAL_ABUSE, system_prompt="prompt"
    )

    assert findings == []
