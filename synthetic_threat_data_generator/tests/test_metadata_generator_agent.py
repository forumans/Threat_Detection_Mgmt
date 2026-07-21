"""
Unit tests for the Metadata Generator Agent.

No mocking needed -- Faker and the seeded RNG run for real (lightweight,
deterministic).
"""

from src.agents.metadata_generator_agent import generate_metadata
from src.schemas import ChannelQuality, Conversation, ConversationTurnPlan, Scenario, Category, Severity


def _conversation() -> Conversation:
    return Conversation(
        conversation_id="conv-1",
        scenario_id="scenario-1",
        turns=[
            ConversationTurnPlan(turn_index=0, speaker_persona_id="p-caller", intended_content="a", intended_emotion="neutral"),
            ConversationTurnPlan(turn_index=1, speaker_persona_id="p-agent", intended_content="b", intended_emotion="neutral"),
            ConversationTurnPlan(turn_index=2, speaker_persona_id="p-caller", intended_content="c", intended_emotion="angry"),
        ],
        injected_threat_turn_indices=[],
    )


def _scenario(channel_quality=ChannelQuality.CLEAN) -> Scenario:
    return Scenario(
        scenario_id="scenario-1",
        category=Category.BENIGN,
        severity=None,
        setting="a call",
        locale="en-US",
        channel_quality=channel_quality,
        num_speakers=2,
    )


def test_call_id_reuses_conversation_id():
    metadata = generate_metadata(_conversation(), _scenario())

    assert metadata.call_id == "conv-1"
    assert metadata.scenario_id == "scenario-1"
    assert metadata.locale == "en-US"


def test_participant_ids_count_matches_unique_speakers():
    metadata = generate_metadata(_conversation(), _scenario())

    # Conversation has 2 unique speakers: p-caller, p-agent.
    assert len(metadata.participant_ids) == 2


def test_channel_info_reflects_channel_quality():
    clean = generate_metadata(_conversation(), _scenario(channel_quality=ChannelQuality.CLEAN))
    degraded = generate_metadata(_conversation(), _scenario(channel_quality=ChannelQuality.DEGRADED))

    assert clean.channel_info["sample_rate_hz"] == 16000
    assert degraded.channel_info["sample_rate_hz"] == 8000


def test_duration_is_positive_and_scales_with_turn_count():
    short = generate_metadata(_conversation(), _scenario())

    long_conversation = _conversation()
    long_conversation.turns = long_conversation.turns * 5  # 15 turns instead of 3
    long_metadata = generate_metadata(long_conversation, _scenario())

    assert short.duration_ms > 0
    assert long_metadata.duration_ms > short.duration_ms


def test_same_conversation_id_reproduces_identical_metadata():
    first = generate_metadata(_conversation(), _scenario())
    second = generate_metadata(_conversation(), _scenario())

    assert first.start_timestamp == second.start_timestamp
    assert first.duration_ms == second.duration_ms
    assert first.participant_ids == second.participant_ids
