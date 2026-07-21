"""
Unit tests for the Fraud & Social Engineering Agent.

run_transcript_detection (the shared LLM-calling helper) is monkeypatched, so
these tests verify this agent wires up the *correct* category/agent_name/
prompt -- the LLM-calling plumbing itself is covered by test_shared_detection.py.
"""

from src.agents import fraud_social_engineering_agent as agent_module
from src.schemas import ThreatCategory, TranscriptTurn


def test_detect_fraud_uses_correct_category_and_agent_name(monkeypatch):
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
            turn_index=0, speaker_label="caller", start_ms=0, end_ms=1000,
            text="This is the IRS, pay now with gift cards",
        )
    ]
    result = agent_module.detect_fraud_and_social_engineering(turns)

    assert result == ["sentinel-finding"]
    assert captured["turns"] == turns
    assert captured["agent_name"] == "fraud_social_engineering"
    assert captured["category"] == ThreatCategory.FRAUD_SOCIAL_ENGINEERING
    assert isinstance(captured["system_prompt"], str) and len(captured["system_prompt"]) > 0
