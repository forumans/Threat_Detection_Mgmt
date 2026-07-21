"""
Configuration Agent
=====================
Why this agent is needed:
    Every other agent in the pipeline trusts that the generation request it's
    working from is internally consistent (percentages that add up, at least
    one locale, a positive sample count). Rather than every downstream agent
    re-checking those basics, this agent is the single gate a raw request
    passes through once, producing a validated `Configuration` object the rest
    of the pipeline can trust without re-checking.

What it does, step by step:
    1. Validates every category key in the requested distribution is a real
       taxonomy category (including "benign").
    2. Normalizes the category distribution so it sums to 1.0 (small rounding
       drift is corrected silently; a distribution that's off by more than a
       small tolerance is rejected outright rather than silently "fixed").
    3. Validates sample_count is positive and at least one locale was given.
    4. Fills in a default seed (from Settings.default_seed) when none was
       given, so every generation run is reproducible by default -- per the
       "version everything" principle in the architecture doc.
    5. Assigns a config_id and returns the validated `Configuration`.

Tools used:
    None -- pure validation/defaulting logic. No LLM call, no external
    service, which makes this agent trivially fast and deterministic.

Other details:
    This agent deliberately does no generation of its own; it only certifies
    the request that every other agent will act on.
"""

from __future__ import annotations

import uuid

from ..config import get_settings
from ..schemas import Category, Configuration

# How far a category_distribution's sum may drift from 1.0 before we reject it
# outright instead of silently normalizing (catches genuine mistakes, like a
# distribution that sums to 10.0 because it was given as percentages).
_MAX_SUM_DRIFT = 0.25


def build_configuration(
    category_distribution: dict[str, float],
    sample_count: int,
    locales: list[str],
    seed: int | None = None,
) -> Configuration:
    """Validate and normalize a raw generation request into a `Configuration`."""
    # Step 1: every requested category must be a real taxonomy value.
    valid_categories = {c.value for c in Category}
    unknown = set(category_distribution) - valid_categories
    if unknown:
        raise ValueError(f"Unknown categories in distribution: {sorted(unknown)}")

    # Step 2: normalize the distribution to sum to 1.0, but reject anything
    # too far off to be a rounding error -- that's more likely a real mistake.
    total = sum(category_distribution.values())
    if total <= 0:
        raise ValueError("category_distribution must sum to a positive value")
    if abs(total - 1.0) > _MAX_SUM_DRIFT:
        raise ValueError(f"category_distribution sums to {total}, expected ~1.0")
    normalized_distribution = {category: weight / total for category, weight in category_distribution.items()}

    # Step 3: basic request sanity.
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    if not locales:
        raise ValueError("at least one locale is required")

    # Step 4: default the seed for reproducibility if the caller didn't supply one.
    resolved_seed = seed if seed is not None else get_settings().default_seed

    # Step 5: package into the validated Configuration.
    return Configuration(
        config_id=str(uuid.uuid4()),
        category_distribution=normalized_distribution,
        sample_count=sample_count,
        locales=locales,
        seed=resolved_seed,
    )
