"""
Unit tests for the Ground Truth Generator Agent.

Pure deterministic logic, no mocking needed -- these tests exercise the real
function directly.
"""

import pytest

from src.agents.ground_truth_generator_agent import generate_ground_truth
from src.schemas import Category, ChannelQuality, Conversation, ConversationTurnPlan, Scenario, Severity


def _conversation(injected_indices: list[int]) -> Conversation:
    return Conversation(
        conversation_id="conv-1",
        scenario_id="scenario-1",
        turns=[
            ConversationTurnPlan(turn_index=0, speaker_persona_id="p-caller", intended_content="a", intended_emotion="neutral"),
            ConversationTurnPlan(turn_index=1, speaker_persona_id="p-agent", intended_content="b", intended_emotion="professional"),
            ConversationTurnPlan(turn_index=2, speaker_persona_id="p-caller", intended_content="c", intended_emotion="angry"),
        ],
        injected_threat_turn_indices=injected_indices,
    )


def _scenario(category=Category.THREAT_OF_VIOLENCE, severity=Severity.HIGH) -> Scenario:
    return Scenario(
        scenario_id="scenario-1",
        category=category,
        severity=severity,
        setting="a call",
        locale="en-US",
        channel_quality=ChannelQuality.CLEAN,
        num_speakers=2,
    )


def test_benign_scenario_with_no_injected_turns_yields_no_labels():
    labels = generate_ground_truth(_conversation([]), _scenario(category=Category.BENIGN, severity=None))

    assert labels == []


def test_one_label_per_injected_turn():
    labels = generate_ground_truth(_conversation([2]), _scenario())

    assert len(labels) == 1
    assert labels[0].turn_index == 2
    assert labels[0].sample_id == "conv-1"
    assert labels[0].category == Category.THREAT_OF_VIOLENCE
    assert labels[0].severity == Severity.HIGH


def test_label_inherits_speaker_and_intended_emotion_from_the_turn_plan():
    labels = generate_ground_truth(_conversation([2]), _scenario())

    assert labels[0].speaker_persona_id == "p-caller"
    assert labels[0].expected_emotion == "angry"


def test_span_and_audio_timing_stay_unset_at_this_stage():
    labels = generate_ground_truth(_conversation([2]), _scenario())

    assert labels[0].span is None
    assert labels[0].audio_start_ms is None
    assert labels[0].audio_end_ms is None


def test_prosody_notes_scale_with_severity():
    low = generate_ground_truth(_conversation([2]), _scenario(severity=Severity.LOW))
    critical = generate_ground_truth(_conversation([2]), _scenario(severity=Severity.CRITICAL))

    assert "subtle" in low[0].expected_prosody_notes
    assert "sharply" in critical[0].expected_prosody_notes


def test_multiple_injected_turns_yield_multiple_labels_in_order():
    labels = generate_ground_truth(_conversation([0, 2]), _scenario())

    assert [label.turn_index for label in labels] == [0, 2]


def test_missing_turn_index_reference_raises():
    with pytest.raises(ValueError, match="missing turn_index"):
        generate_ground_truth(_conversation([99]), _scenario())
