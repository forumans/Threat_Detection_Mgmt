"""
Unit tests for the Compliance Detection Agent.

run_transcript_detection (the shared LLM-calling helper) is monkeypatched, so
these tests verify this agent wires up the *correct* category/agent_name/
prompt -- the LLM-calling plumbing itself is covered by test_shared_detection.py.
"""

from src.agents import compliance_detection_agent as agent_module
from src.schemas import ThreatCategory, TranscriptTurn


def test_detect_compliance_violations_uses_correct_category_and_agent_name(monkeypatch):
    captured = {}

    def fake_run(turns, *, agent_name, category, system_prompt):
        captured["turns"] = turns
        captured["agent_name"] = agent_name
        captured["category"] = category
        captured["system_prompt"] = system_prompt
        return ["sentinel-finding"]

    monkeypatch.setattr(agent_module, "run_transcript_detection", fake_run)

    turns = [
        TranscriptTurn(
            turn_index=0, speaker_label="agent", start_ms=0, end_ms=1000,
            text="Your card number is 4111 1111 1111 1111",
        )
    ]
    result = agent_module.detect_compliance_violations(turns)

    assert result == ["sentinel-finding"]
    assert captured["turns"] == turns
    assert captured["agent_name"] == "compliance"
    assert captured["category"] == ThreatCategory.COMPLIANCE_VIOLATION
    assert isinstance(captured["system_prompt"], str) and len(captured["system_prompt"]) > 0
