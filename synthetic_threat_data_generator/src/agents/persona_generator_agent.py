"""
Persona Generator Agent
=========================
Why this agent is needed:
    A scenario alone doesn't say who's on the call. This agent creates one
    concrete persona per speaker slot in the scenario -- a name, a voice
    profile, and an emotional baseline -- so the Conversation Generator has
    someone specific to write dialogue for, and the TTS Engine has a voice to
    render.

What it does, step by step:
    1. Assigns a role to each speaker slot (caller/agent for a 2-party call;
       extra callers for multi-party ones).
    2. Generates a realistic name per persona via Faker, localized to the
       scenario's locale where Faker has a matching locale provider (falls
       back to a generic locale otherwise).
    3. Picks voice traits (pitch_range, pace, accent, timbre) via seeded
       random choice, so the same scenario always produces the same-sounding
       personas.
    4. Asks the LLM for a short emotional-baseline description appropriate to
       the persona's role and the scenario's category/severity (e.g. an agent
       stays calm/professional; a caller in a high-severity scenario starts
       out already agitated).
    5. Returns one `Persona` per speaker.

Tools used:
    Faker (see docs/architecture/synthetic_data_gen_architecture-plan.md §4 --
    Data Generation) for names; Python's seeded `random.Random` for voice
    traits; the LLM (via llm_client.generate, defaulting to OpenAI) only for
    the emotional_baseline free text.

Other details:
    The voice-trait categories below are placeholders for whatever the
    eventual TTSClient implementation (Piper by default, see config.py)
    actually exposes as tunable parameters -- expected to be revisited once a
    real voice library is wired in.
"""

from __future__ import annotations

import random
import uuid

from faker import Faker

from .. import llm_client
from ..schemas import Persona, Scenario

_PITCH_RANGES = ["low", "medium", "high"]
_PACES = ["slow", "medium", "fast"]
_TIMBRES = ["warm", "nasal", "breathy", "gravelly", "bright"]

_ACCENT_BY_LOCALE = {
    "en-US": "General American",
    "en-GB": "British RP",
    "es-MX": "Mexican Spanish",
    "es-ES": "Castilian Spanish",
    "fr-FR": "Standard French",
    "de-DE": "Standard German",
}
_DEFAULT_ACCENT = "Neutral"

_EMOTIONAL_BASELINE_SYSTEM_PROMPT = (
    "You write one-sentence emotional-baseline descriptions for synthetic call-center "
    "personas. Given a persona's role, the call's threat category, and its severity, "
    "describe how this person is likely to sound at the START of the call -- their baseline "
    "demeanor, before anything escalates. Output only the one-sentence description, no "
    "preamble."
)


def _assign_roles(num_speakers: int) -> list[str]:
    """
    Step 1: one agent, everyone else is a caller (covers 2-party and
    multi-party). `Scenario.num_speakers` is schema-constrained to >= 2
    (see schemas.py), so no lower-bound check is needed here.
    """
    return ["caller", "agent"] + ["caller"] * (num_speakers - 2)


def _build_faker(locale: str, seed: int) -> Faker:
    """Step 2 setup: a locale-matched Faker instance, falling back to a default locale."""
    try:
        faker = Faker(locale.replace("-", "_"))
    except AttributeError:
        faker = Faker()  # unsupported locale -- fall back to Faker's default (en_US)
    faker.seed_instance(seed)
    return faker


def _build_voice_traits(rng: random.Random, name: str, locale: str) -> dict:
    """Step 3: seeded, categorical voice traits for one persona."""
    return {
        "name": name,
        "pitch_range": rng.choice(_PITCH_RANGES),
        "pace": rng.choice(_PACES),
        "accent": _ACCENT_BY_LOCALE.get(locale, _DEFAULT_ACCENT),
        "timbre": rng.choice(_TIMBRES),
    }


def generate_personas(scenario: Scenario) -> list[Persona]:
    """Generate one Persona per speaker required by the given Scenario."""
    # Step 1: role assignment.
    roles = _assign_roles(scenario.num_speakers)

    # Deterministic per-scenario seed, so the same scenario always produces
    # the same personas.
    seed = abs(hash(scenario.scenario_id)) % (2**32)
    rng = random.Random(seed)
    faker = _build_faker(scenario.locale, seed)

    severity_text = scenario.severity.value if scenario.severity else "n/a"

    personas: list[Persona] = []
    for role in roles:
        # Step 2: locale-appropriate name.
        name = faker.name()

        # Step 3: seeded voice traits.
        voice_traits = _build_voice_traits(rng, name, scenario.locale)

        # Step 4: LLM-generated emotional baseline for this role/scenario.
        emotional_baseline = llm_client.generate(
            f"Role: {role}\nCategory: {scenario.category.value}\nSeverity: {severity_text}",
            system_prompt=_EMOTIONAL_BASELINE_SYSTEM_PROMPT,
        )

        # Step 5: package into the Persona contract.
        personas.append(
            Persona(
                persona_id=str(uuid.uuid4()),
                role=role,
                voice_traits=voice_traits,
                emotional_baseline=emotional_baseline,
            )
        )

    return personas
