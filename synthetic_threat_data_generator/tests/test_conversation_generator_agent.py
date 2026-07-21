"""
Unit tests for the Conversation Generator Agent.

llm_client.build_generation_agent is monkeypatched to a fake PydanticAI-shaped
agent, so these tests never make a real LLM call.
"""

from types import SimpleNamespace

from src.agents import conversation_generator_agent as agent_module
from src.agents.conversation_generator_agent import _ConversationPlanOutput, _TurnPlanOutput
from src.schemas import Category, ChannelQuality, Persona, Scenario, Severity


class _FakeAgent:
    def __init__(self, plan: _ConversationPlanOutput):
        self._plan = plan

    def run_sync(self, prompt):
        return SimpleNamespace(output=self._plan)


def _scenario(category=Category.BENIGN, severity=None) -> Scenario:
    return Scenario(
        scenario_id="scenario-1",
        category=category,
        severity=severity,
        setting="a support call",
        locale="en-US",
        channel_quality=ChannelQuality.CLEAN,
        num_speakers=2,
    )


def _personas() -> list[Persona]:
    return [
        Persona(persona_id="p-caller", role="caller", voice_traits={"name": "Alex"}, emotional_baseline="calm"),
        Persona(persona_id="p-agent", role="agent", voice_traits={"name": "Sam"}, emotional_baseline="professional"),
    ]


def test_speaker_index_is_mapped_to_the_correct_persona_id(monkeypatch):
    plan = _ConversationPlanOutput(
        turns=[
            _TurnPlanOutput(turn_index=0, speaker_index=0, intended_content="greets", intended_emotion="neutral"),
            _TurnPlanOutput(turn_index=1, speaker_index=1, intended_content="responds", intended_emotion="neutral"),
        ]
    )
    monkeypatch.setattr(agent_module.llm_client, "build_generation_agent", lambda *a, **k: _FakeAgent(plan))

    conversation = agent_module.generate_conversation(_scenario(), _personas())

    assert conversation.turns[0].speaker_persona_id == "p-caller"
    assert conversation.turns[1].speaker_persona_id == "p-agent"


def test_out_of_range_speaker_index_is_clamped_not_raised(monkeypatch):
    plan = _ConversationPlanOutput(
        turns=[_TurnPlanOutput(turn_index=0, speaker_index=7, intended_content="x", intended_emotion="neutral")]
    )
    monkeypatch.setattr(agent_module.llm_client, "build_generation_agent", lambda *a, **k: _FakeAgent(plan))

    conversation = agent_module.generate_conversation(_scenario(), _personas())

    # 7 % 2 == 1 -> the second persona ("p-agent").
    assert conversation.turns[0].speaker_persona_id == "p-agent"


def test_benign_scenario_has_no_injected_turns(monkeypatch):
    plan = _ConversationPlanOutput(
        turns=[
            _TurnPlanOutput(turn_index=0, speaker_index=0, intended_content="x", intended_emotion="neutral"),
            _TurnPlanOutput(turn_index=1, speaker_index=1, intended_content="y", intended_emotion="neutral"),
        ]
    )
    monkeypatch.setattr(agent_module.llm_client, "build_generation_agent", lambda *a, **k: _FakeAgent(plan))

    conversation = agent_module.generate_conversation(_scenario(category=Category.BENIGN), _personas())

    assert conversation.injected_threat_turn_indices == []


def test_llm_marked_injections_are_preserved(monkeypatch):
    plan = _ConversationPlanOutput(
        turns=[
            _TurnPlanOutput(turn_index=0, speaker_index=0, intended_content="x", intended_emotion="neutral"),
            _TurnPlanOutput(
                turn_index=1, speaker_index=0, intended_content="threat", intended_emotion="angry",
                is_threat_injection=True,
            ),
        ]
    )
    monkeypatch.setattr(agent_module.llm_client, "build_generation_agent", lambda *a, **k: _FakeAgent(plan))

    conversation = agent_module.generate_conversation(
        _scenario(category=Category.THREAT_OF_VIOLENCE, severity=Severity.HIGH), _personas()
    )

    assert conversation.injected_threat_turn_indices == [1]


def test_non_benign_scenario_falls_back_to_middle_turn_if_llm_marks_none(monkeypatch):
    plan = _ConversationPlanOutput(
        turns=[
            _TurnPlanOutput(turn_index=0, speaker_index=0, intended_content="a", intended_emotion="neutral"),
            _TurnPlanOutput(turn_index=1, speaker_index=1, intended_content="b", intended_emotion="neutral"),
            _TurnPlanOutput(turn_index=2, speaker_index=0, intended_content="c", intended_emotion="neutral"),
        ]
    )
    monkeypatch.setattr(agent_module.llm_client, "build_generation_agent", lambda *a, **k: _FakeAgent(plan))

    conversation = agent_module.generate_conversation(
        _scenario(category=Category.FRAUD_SOCIAL_ENGINEERING, severity=Severity.LOW), _personas()
    )

    # No turn was marked by the "LLM" -- fallback picks the middle turn (index 1).
    assert conversation.injected_threat_turn_indices == [1]
