"""
Unit tests for the Transcript Generator Agent.

llm_client.build_generation_agent is monkeypatched to a fake PydanticAI-shaped
agent, so these tests never make a real LLM call.
"""

from types import SimpleNamespace

from src.agents import transcript_generator_agent as agent_module
from src.agents.transcript_generator_agent import _RenderedTurnOutput, _TranscriptOutput
from src.schemas import Conversation, ConversationTurnPlan, Persona


class _FakeAgent:
    def __init__(self, output: _TranscriptOutput):
        self._output = output

    def run_sync(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(output=self._output)


def _conversation() -> Conversation:
    return Conversation(
        conversation_id="conv-1",
        scenario_id="scenario-1",
        turns=[
            ConversationTurnPlan(
                turn_index=0, speaker_persona_id="p-caller", intended_content="greets the agent", intended_emotion="neutral"
            ),
            ConversationTurnPlan(
                turn_index=1, speaker_persona_id="p-agent", intended_content="asks how to help", intended_emotion="professional"
            ),
        ],
        injected_threat_turn_indices=[],
    )


def _personas() -> list[Persona]:
    return [
        Persona(persona_id="p-caller", role="caller", voice_traits={"name": "Alex"}, emotional_baseline="calm"),
        Persona(persona_id="p-agent", role="agent", voice_traits={"name": "Sam"}, emotional_baseline="professional"),
    ]


def test_rendered_text_is_matched_to_the_correct_turn(monkeypatch):
    output = _TranscriptOutput(
        turns=[
            _RenderedTurnOutput(turn_index=0, text="Hi there!"),
            _RenderedTurnOutput(turn_index=1, text="How can I help you today?"),
        ]
    )
    monkeypatch.setattr(agent_module.llm_client, "build_generation_agent", lambda *a, **k: _FakeAgent(output))

    turns = agent_module.generate_transcript(_conversation(), _personas())

    assert turns[0].text == "Hi there!"
    assert turns[0].speaker_persona_id == "p-caller"
    assert turns[1].text == "How can I help you today?"
    assert turns[1].speaker_persona_id == "p-agent"


def test_missing_turn_index_in_llm_output_falls_back_to_intended_content(monkeypatch):
    # LLM only returned turn 0 -- turn 1 is missing from its response.
    output = _TranscriptOutput(turns=[_RenderedTurnOutput(turn_index=0, text="Hi there!")])
    monkeypatch.setattr(agent_module.llm_client, "build_generation_agent", lambda *a, **k: _FakeAgent(output))

    turns = agent_module.generate_transcript(_conversation(), _personas())

    assert len(turns) == 2  # still one ConversationTurn per planned turn
    assert turns[1].text == "asks how to help"  # fell back to intended_content


def test_prompt_includes_persona_names_not_just_ids(monkeypatch):
    captured_agent = _FakeAgent(_TranscriptOutput(turns=[]))
    monkeypatch.setattr(agent_module.llm_client, "build_generation_agent", lambda *a, **k: captured_agent)

    agent_module.generate_transcript(_conversation(), _personas())

    assert "Alex" in captured_agent.last_prompt
    assert "Sam" in captured_agent.last_prompt
