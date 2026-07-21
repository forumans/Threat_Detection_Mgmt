"""
Metadata Generator Agent
===========================
Why this agent is needed:
    Every sample needs call-level bookkeeping -- when it happened, how long it
    ran, what channel it was carried on, who was on the line -- independent of
    the transcript or audio content itself. Keeping that independent (rather
    than inferring it from the rendered transcript) means metadata generation
    never has to wait on, or agree with, a sibling branch of the pipeline.

What it does, step by step:
    1. Reuses `Conversation.conversation_id` as `call_id` -- one conversation
       is one call is one benchmark sample, so no separate ID is minted here.
    2. Generates a realistic start_timestamp via Faker, seeded from the
       conversation ID so metadata is reproducible.
    3. Estimates duration_ms from the turn count (a placeholder average
       seconds-per-turn, not real timing -- the Audio Generator produces the
       real, authoritative duration once audio is actually rendered).
    4. Derives channel_info (codec, sample rate, bitrate) from the scenario's
       channel_quality.
    5. Generates one Faker participant_id per unique speaker referenced in the
       conversation's turns.
    6. Returns the `CallMetadata`.

Tools used:
    Faker (see docs/architecture/synthetic_data_gen_architecture-plan.md §4 --
    Data Generation) for the timestamp and participant IDs. No LLM call --
    call metadata doesn't need creative generation, only realistic-looking
    structured values.

Other details:
    `duration_ms` here is an *estimate*. The Dataset Exporter overwrites it
    with the real `AudioFile.duration_ms` once the Audio Generator branch
    produces actual rendered audio, right before writing metadata.json --
    those two branches only converge at Dataset Exporter (see the pipeline
    diagram in §3), so this agent has no way to know the real duration yet.
"""

from __future__ import annotations

import random

from faker import Faker

from ..schemas import CallMetadata, ChannelQuality, Conversation, Scenario

# Placeholder timing estimate -- see module docstring "Other details".
_AVG_TURN_DURATION_MS = 4000
_TURN_DURATION_JITTER_MS = 1000

_CHANNEL_INFO_BY_QUALITY: dict[ChannelQuality, dict] = {
    ChannelQuality.CLEAN: {"codec": "pcm_s16le", "sample_rate_hz": 16000, "bitrate_kbps": 256},
    ChannelQuality.DEGRADED: {"codec": "g711", "sample_rate_hz": 8000, "bitrate_kbps": 64},
}


def generate_metadata(conversation: Conversation, scenario: Scenario) -> CallMetadata:
    """Generate call-level metadata for one sample."""
    seed = abs(hash(conversation.conversation_id)) % (2**32)
    rng = random.Random(seed)
    faker = Faker()
    faker.seed_instance(seed)

    # Step 2: reproducible, realistic start time.
    start_timestamp = faker.date_time_between(start_date="-1y", end_date="now")

    # Step 3: rough duration estimate from turn count.
    duration_ms = sum(
        _AVG_TURN_DURATION_MS + rng.randint(-_TURN_DURATION_JITTER_MS, _TURN_DURATION_JITTER_MS)
        for _ in conversation.turns
    )
    duration_ms = max(duration_ms, 0)

    # Step 4: channel info from the scenario's channel quality.
    channel_info = _CHANNEL_INFO_BY_QUALITY[scenario.channel_quality]

    # Step 5: one participant ID per unique speaker, in order of first appearance.
    seen_personas: list[str] = []
    for turn in conversation.turns:
        if turn.speaker_persona_id not in seen_personas:
            seen_personas.append(turn.speaker_persona_id)
    participant_ids = [faker.phone_number() for _ in seen_personas]

    # Step 6: package into the CallMetadata contract.
    return CallMetadata(
        call_id=conversation.conversation_id,
        scenario_id=scenario.scenario_id,
        start_timestamp=start_timestamp,
        duration_ms=duration_ms,
        channel_info=channel_info,
        participant_ids=participant_ids,
        locale=scenario.locale,
    )
