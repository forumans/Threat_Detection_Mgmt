"""
Unit tests for the Scenario Generator Agent.

llm_client.generate is monkeypatched so these tests never make a real LLM
call -- the deterministic, seeded categorical-choice logic is what's under
test here.
"""

from src.agents import scenario_generator_agent
from src.schemas import Category, Configuration


def _configuration(**overrides) -> Configuration:
    defaults = dict(
        config_id="cfg-1",
        category_distribution={"benign": 1.0},
        sample_count=10,
        locales=["en-US", "es-MX"],
        seed=123,
    )
    defaults.update(overrides)
    return Configuration(**defaults)


def test_same_config_and_index_reproduces_identical_scenario(monkeypatch):
    monkeypatch.setattr(scenario_generator_agent.llm_client, "generate", lambda *a, **k: "a setting")
    config = _configuration()

    first = scenario_generator_agent.generate_scenario(config, sample_index=3)
    second = scenario_generator_agent.generate_scenario(config, sample_index=3)

    assert first.category == second.category
    assert first.severity == second.severity
    assert first.locale == second.locale
    assert first.channel_quality == second.channel_quality
    assert first.num_speakers == second.num_speakers


def test_benign_category_has_no_severity(monkeypatch):
    monkeypatch.setattr(scenario_generator_agent.llm_client, "generate", lambda *a, **k: "a setting")
    config = _configuration(category_distribution={"benign": 1.0})

    scenario = scenario_generator_agent.generate_scenario(config, sample_index=0)

    assert scenario.category == Category.BENIGN
    assert scenario.severity is None


def test_non_benign_category_always_has_severity(monkeypatch):
    monkeypatch.setattr(scenario_generator_agent.llm_client, "generate", lambda *a, **k: "a setting")
    config = _configuration(category_distribution={"verbal_abuse": 1.0})

    scenario = scenario_generator_agent.generate_scenario(config, sample_index=0)

    assert scenario.category == Category.VERBAL_ABUSE
    assert scenario.severity is not None


def test_locale_is_drawn_from_configured_locales(monkeypatch):
    monkeypatch.setattr(scenario_generator_agent.llm_client, "generate", lambda *a, **k: "a setting")
    config = _configuration(locales=["fr-FR"])

    scenario = scenario_generator_agent.generate_scenario(config, sample_index=0)

    assert scenario.locale == "fr-FR"


def test_setting_generation_receives_category_severity_locale_context(monkeypatch):
    captured = {}

    def fake_generate(prompt, *, system_prompt=None):
        captured["prompt"] = prompt
        return "a generated setting"

    monkeypatch.setattr(scenario_generator_agent.llm_client, "generate", fake_generate)
    config = _configuration(category_distribution={"threat_of_violence": 1.0}, locales=["en-US"])

    scenario = scenario_generator_agent.generate_scenario(config, sample_index=0)

    assert scenario.setting == "a generated setting"
    assert "threat_of_violence" in captured["prompt"]
    assert "en-US" in captured["prompt"]
