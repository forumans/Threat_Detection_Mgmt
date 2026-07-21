"""
Unit tests for the Translation Engine Agent.

llm_client.build_generation_agent is monkeypatched to a fake PydanticAI-shaped
agent, so these tests never make a real LLM call.
"""

from types import SimpleNamespace

from src.agents import translation_engine_agent as agent_module
from src.agents.translation_engine_agent import _TranslatedTurnOutput, _TranslationOutput
from src.schemas import ConversationTurn


class _FakeAgent:
    def __init__(self, output: _TranslationOutput):
        self._output = output

    def run_sync(self, prompt):
        self.last_prompt = prompt
        return SimpleNamespace(output=self._output)


def _turns() -> list[ConversationTurn]:
    return [
        ConversationTurn(turn_index=0, speaker_persona_id="p-caller", text="Hello", intended_emotion="neutral"),
        ConversationTurn(turn_index=1, speaker_persona_id="p-agent", text="Hi there", intended_emotion="professional"),
    ]


def test_matching_locales_is_a_pure_pass_through(monkeypatch):
    # No agent should even be built for a same-locale request.
    monkeypatch.setattr(
        agent_module.llm_client,
        "build_generation_agent",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")),
    )

    result = agent_module.translate_transcript(_turns(), source_locale="en-US", target_locale="en-US")

    assert result == _turns()


def test_different_locale_translates_each_turn(monkeypatch):
    output = _TranslationOutput(
        turns=[
            _TranslatedTurnOutput(turn_index=0, text="Hola"),
            _TranslatedTurnOutput(turn_index=1, text="Hola, ¿en qué puedo ayudarte?"),
        ]
    )
    monkeypatch.setattr(agent_module.llm_client, "build_generation_agent", lambda *a, **k: _FakeAgent(output))

    result = agent_module.translate_transcript(_turns(), source_locale="en-US", target_locale="es-MX")

    assert result[0].text == "Hola"
    assert result[1].text == "Hola, ¿en qué puedo ayudarte?"
    assert result[0].speaker_persona_id == "p-caller"  # speaker/turn structure preserved


def test_missing_translation_falls_back_to_source_text(monkeypatch):
    # LLM only translated turn 0.
    output = _TranslationOutput(turns=[_TranslatedTurnOutput(turn_index=0, text="Hola")])
    monkeypatch.setattr(agent_module.llm_client, "build_generation_agent", lambda *a, **k: _FakeAgent(output))

    result = agent_module.translate_transcript(_turns(), source_locale="en-US", target_locale="es-MX")

    assert len(result) == 2
    assert result[1].text == "Hi there"  # fell back to the original English text
