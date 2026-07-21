"""
Scenario Generator Agent
==========================
Why this agent is needed:
    Every sample starts from a scenario spec: which threat category it
    represents, how severe, in what setting, in what locale, over what channel
    quality, with how many speakers. This agent turns one `Configuration`
    (a whole batch's requirements) into one concrete `Scenario` per sample,
    hitting the requested category mix while varying everything else so
    samples don't feel templated.

What it does, step by step:
    1. Picks a category via weighted random choice against
       `Configuration.category_distribution`, using a seed derived from the
       configuration's seed + this sample's index so the same (config, index)
       pair always reproduces the same scenario.
    2. Picks a severity (skipped -- stays None -- for `benign`).
    3. Picks a locale from `Configuration.locales` and a channel_quality
       (clean/degraded).
    4. Picks a speaker count (2-party by default, occasionally multi-party).
    5. Asks the LLM for a short, concrete setting description (e.g. "a
       collections call about a missed auto loan payment") matching the
       category/severity/locale, so settings don't repeat across samples.
    6. Packages everything into a `Scenario`.

Tools used:
    Python's `random.Random` (seeded, deterministic) for every categorical
    choice; the LLM (via llm_client.generate, defaulting to OpenAI -- see
    src/config.py) only for the free-text `setting` description, since that's
    the one field where template repetition would be obvious to a reader.

Other details:
    The severity/channel_quality/speaker-count weightings below are simple
    placeholders. They're a reasonable starting distribution, not a tuned one --
    revisit once real benchmark releases show what mix Project 2 actually needs.
"""

from __future__ import annotations

import random
import uuid

from .. import llm_client
from ..schemas import Category, ChannelQuality, Configuration, Scenario, Severity

_SEVERITIES = [Severity.LOW, Severity.MEDIUM, Severity.HIGH, Severity.CRITICAL]

# Placeholder weightings -- see module docstring "Other details".
_CHANNEL_QUALITY_WEIGHTS = {ChannelQuality.CLEAN: 0.7, ChannelQuality.DEGRADED: 0.3}
_NUM_SPEAKERS_WEIGHTS = {2: 0.85, 3: 0.15}

_SETTING_SYSTEM_PROMPT = (
    "You write one-sentence call-center scenario settings for a synthetic benchmark dataset. "
    "Given a threat category, severity, and locale, describe a concrete, realistic situation "
    "for the call (e.g. the type of business, the caller's stated reason for calling). Output "
    "only the one-sentence description, no preamble."
)


def _weighted_choice(rng: random.Random, weights: dict) -> object:
    """Pick one key from a {value: weight} dict using the given RNG."""
    values = list(weights.keys())
    return rng.choices(values, weights=list(weights.values()), k=1)[0]


def generate_scenario(configuration: Configuration, sample_index: int = 0) -> Scenario:
    """Generate one concrete Scenario for one sample of the given Configuration."""
    # Step 1: deterministic per-sample RNG, so (configuration, sample_index) always
    # reproduces the same scenario -- required by the architecture doc's
    # "version everything" / reproducibility principle.
    rng = random.Random(configuration.seed * 1000 + sample_index)
    category = Category(_weighted_choice(rng, configuration.category_distribution))

    # Step 2: benign scenarios carry no severity.
    severity = None if category == Category.BENIGN else rng.choice(_SEVERITIES)

    # Step 3: locale + channel quality.
    locale = rng.choice(configuration.locales)
    channel_quality = _weighted_choice(rng, _CHANNEL_QUALITY_WEIGHTS)

    # Step 4: speaker count.
    num_speakers = _weighted_choice(rng, _NUM_SPEAKERS_WEIGHTS)

    # Step 5: LLM-generated setting description, so it doesn't feel templated.
    severity_text = severity.value if severity else "n/a"
    setting = llm_client.generate(
        f"Category: {category.value}\nSeverity: {severity_text}\nLocale: {locale}",
        system_prompt=_SETTING_SYSTEM_PROMPT,
    )

    # Step 6: package into the Scenario contract.
    return Scenario(
        scenario_id=str(uuid.uuid4()),
        category=category,
        severity=severity,
        setting=setting,
        locale=locale,
        channel_quality=channel_quality,
        num_speakers=num_speakers,
    )
