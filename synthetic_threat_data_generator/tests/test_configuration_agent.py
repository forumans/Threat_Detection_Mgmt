"""
Unit tests for the Configuration Agent.

Pure validation logic, no mocking needed -- these tests exercise the real
function directly.
"""

import pytest

from src.agents.configuration_agent import build_configuration


def test_valid_request_is_normalized_and_assigned_a_config_id():
    config = build_configuration(
        category_distribution={"benign": 0.5, "verbal_abuse": 0.5},
        sample_count=100,
        locales=["en-US"],
        seed=7,
    )

    assert config.config_id
    assert config.sample_count == 100
    assert config.locales == ["en-US"]
    assert config.seed == 7
    assert config.category_distribution == {"benign": 0.5, "verbal_abuse": 0.5}


def test_distribution_is_normalized_when_close_to_but_not_exactly_one():
    config = build_configuration(
        category_distribution={"benign": 0.4, "threat_of_violence": 0.5},  # sums to 0.9
        sample_count=10,
        locales=["en-US"],
    )

    assert sum(config.category_distribution.values()) == pytest.approx(1.0)
    assert config.category_distribution["benign"] == pytest.approx(0.4 / 0.9)


def test_missing_seed_defaults_from_settings():
    config = build_configuration(
        category_distribution={"benign": 1.0}, sample_count=1, locales=["en-US"]
    )

    assert config.seed is not None


def test_unknown_category_is_rejected():
    with pytest.raises(ValueError, match="Unknown categories"):
        build_configuration(
            category_distribution={"not_a_real_category": 1.0}, sample_count=1, locales=["en-US"]
        )


def test_distribution_summing_far_from_one_is_rejected():
    with pytest.raises(ValueError, match="expected ~1.0"):
        build_configuration(
            category_distribution={"benign": 50, "verbal_abuse": 50},  # sums to 100 -- looks like percentages
            sample_count=1,
            locales=["en-US"],
        )


def test_non_positive_sample_count_is_rejected():
    with pytest.raises(ValueError, match="sample_count must be positive"):
        build_configuration(category_distribution={"benign": 1.0}, sample_count=0, locales=["en-US"])


def test_empty_locales_is_rejected():
    with pytest.raises(ValueError, match="at least one locale"):
        build_configuration(category_distribution={"benign": 1.0}, sample_count=1, locales=[])
