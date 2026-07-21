"""
Unit tests for the Persona Generator Agent.

llm_client.generate is monkeypatched so these tests never make a real LLM
call. Faker and the seeded RNG run for real (lightweight, deterministic).
"""

import pytest

from src.agents import persona_generator_agent
from src.schemas import Category, ChannelQuality, Scenario, Severity


def _scenario(**overrides) -> Scenario:
    defaults = dict(
        scenario_id="scenario-1",
        category=Category.VERBAL_ABUSE,
        severity=Severity.HIGH,
        setting="a collections call",
        locale="en-US",
        channel_quality=ChannelQuality.CLEAN,
        num_speakers=2,
    )
    defaults.update(overrides)
    return Scenario(**defaults)


def test_two_party_scenario_yields_one_caller_and_one_agent(monkeypatch):
    monkeypatch.setattr(persona_generator_agent.llm_client, "generate", lambda *a, **k: "calm baseline")

    personas = persona_generator_agent.generate_personas(_scenario(num_speakers=2))

    assert [p.role for p in personas] == ["caller", "agent"]


def test_multi_party_scenario_yields_extra_callers(monkeypatch):
    monkeypatch.setattr(persona_generator_agent.llm_client, "generate", lambda *a, **k: "calm baseline")

    personas = persona_generator_agent.generate_personas(_scenario(num_speakers=4))

    assert [p.role for p in personas] == ["caller", "agent", "caller", "caller"]


def test_same_scenario_id_reproduces_identical_personas(monkeypatch):
    monkeypatch.setattr(persona_generator_agent.llm_client, "generate", lambda *a, **k: "calm baseline")

    first = persona_generator_agent.generate_personas(_scenario())
    second = persona_generator_agent.generate_personas(_scenario())

    assert [p.voice_traits for p in first] == [p.voice_traits for p in second]


def test_voice_traits_include_a_name_and_locale_accent(monkeypatch):
    monkeypatch.setattr(persona_generator_agent.llm_client, "generate", lambda *a, **k: "calm baseline")

    personas = persona_generator_agent.generate_personas(_scenario(locale="en-US"))

    for persona in personas:
        assert persona.voice_traits["name"]
        assert persona.voice_traits["accent"] == "General American"


def test_unsupported_locale_falls_back_without_raising(monkeypatch):
    monkeypatch.setattr(persona_generator_agent.llm_client, "generate", lambda *a, **k: "calm baseline")

    personas = persona_generator_agent.generate_personas(_scenario(locale="xx-ZZ"))

    assert personas[0].voice_traits["name"]
    assert personas[0].voice_traits["accent"] == "Neutral"


def test_scenario_schema_rejects_single_speaker():
    # num_speakers < 2 is a schema-level constraint (see schemas.Scenario), not
    # something this agent needs to re-validate itself.
    with pytest.raises(ValueError):
        _scenario(num_speakers=1)
