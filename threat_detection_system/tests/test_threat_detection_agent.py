"""
Unit tests for the Threat Detection Agent.

run_transcript_detection (the shared LLM-calling helper) is monkeypatched, so
these tests verify this agent wires up the *correct* category/agent_name/
prompt -- the LLM-calling plumbing itself is covered by test_shared_detection.py.
"""

from src.agents import threat_detection_agent as agent_module
from src.schemas import ThreatCategory, TranscriptTurn


def test_detect_threats_uses_correct_category_and_agent_name(monkeypatch):
    captured = {}

    def fake_run(turns, *, agent_name, category, system_prompt):
        captured["turns"] = turns
        captured["agent_name"] = agent_name
        captured["category"] = category
        captured["system_prompt"] = system_prompt
        return ["sentinel-finding"]

    monkeypatch.setattr(agent_module, "run_transcript_detection", fake_run)

    turns = [TranscriptTurn(turn_index=0, speaker_label="caller", start_ms=0, end_ms=1000, text="I'll find you")]
    result = agent_module.detect_threats(turns)

    assert result == ["sentinel-finding"]
    assert captured["turns"] == turns
    assert captured["agent_name"] == "threat"
    assert captured["category"] == ThreatCategory.THREAT_OF_VIOLENCE
    assert isinstance(captured["system_prompt"], str) and len(captured["system_prompt"]) > 0
