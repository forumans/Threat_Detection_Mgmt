"""
TTS Engine Agent
==================
Why this agent is needed:
    Everything upstream is text. This agent is where a sample first becomes
    audio -- rendering each turn's (possibly translated) dialogue into speech
    using a voice consistent with that turn's persona, so the Audio Generator
    has real per-turn clips to stitch and mix. Distinct speakers need
    distinct-sounding voices -- a call where the caller and agent sound
    identical isn't a useful benchmark sample.

What it does, step by step:
    1. For the call's full persona list, assigns each persona a gender
       ("male"/"female"), alternating across personas in order -- so two
       speakers in the same call are never assigned the same gender pool --
       with the starting gender randomized per call (seeded from the
       personas' IDs) so callers/agents aren't systematically the same
       gender across an entire generated dataset.
    2. For each turn, deterministically maps its speaker to one voice within
       their assigned gender's pool (the same persona always gets the same
       voice, so a caller doesn't change voices mid-call).
    3. Lazily loads that voice's Piper model (cached after first load --
       voice models are expensive to construct).
    4. Synthesizes the turn's text through Piper, producing raw PCM audio.
    5. Converts the raw audio into a normalized float32 mono array.
    6. Returns one `TurnAudioClip` per turn, in turn order, ready for the
       Audio Generator to stitch together.

Tools used:
    Piper (see docs/architecture/synthetic_data_gen_architecture-plan.md §4 --
    TTS): a lightweight, local, ONNX-based TTS engine -- no API key, no GPU
    required. Runs fully offline once voice models are downloaded.

Other details:
    The gendered pools below are deliberately restricted to Piper voices
    whose gender is unambiguous: `hfc_male`/`hfc_female` are explicitly
    gender-labeled by the Piper project itself (from the Hi-Fi Captain
    dataset), and `amy`/`ryan` are well-established, widely-used voices
    consistent with their labeled gender in the wider Piper community. Other
    catalog voices (e.g. `lessac`) were left out here because their gender
    isn't authoritatively documented -- better to have a smaller, correct
    pool than a larger, potentially mislabeled one. This is still a
    placeholder pending a real per-locale voice library (see "Voice
    diversity/accent coverage strategy" in the architecture doc's Open
    Questions) -- it doesn't yet vary by locale or by the
    pitch/pace/timbre fields in `Persona.voice_traits`.

    `TurnAudioClip` is a plain dataclass, not a Pydantic schema, since it
    carries a raw numpy array (an internal, non-serialized transport type
    between this agent and the Audio Generator, unlike the persisted schemas
    in src/schemas.py).
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..schemas import ConversationTurn, Persona
from ._stable_hash import stable_seed

# See module docstring "Other details" for why these specific voices.
_MALE_VOICES = ["en_US-ryan-high", "en_US-hfc_male-medium"]
_FEMALE_VOICES = ["en_US-amy-medium", "en_US-hfc_female-medium"]
_ALL_VOICES = _MALE_VOICES + _FEMALE_VOICES


@dataclass
class TurnAudioClip:
    turn_index: int
    speaker_persona_id: str
    samples: np.ndarray  # mono float32 PCM, range [-1.0, 1.0]
    sample_rate_hz: int


def _assign_genders(personas: list[Persona]) -> dict[str, str]:
    """
    Step 1: alternate male/female across this call's personas, with the
    starting gender randomized (but deterministic per call, from the
    personas' own IDs) so gender isn't systematically tied to speaker order
    (e.g. "caller" isn't always male) across a whole generated dataset.
    """
    seed = stable_seed(*(p.persona_id for p in personas))
    starting_gender = random.Random(seed).choice(["male", "female"])
    other_gender = "female" if starting_gender == "male" else "male"
    return {p.persona_id: (starting_gender if i % 2 == 0 else other_gender) for i, p in enumerate(personas)}


def _select_voice_name(persona_id: str, gender: str) -> str:
    """Step 2: deterministically map a persona to one voice within their gender's pool."""
    pool = _MALE_VOICES if gender == "male" else _FEMALE_VOICES
    index = stable_seed(persona_id) % len(pool)
    return pool[index]


@lru_cache(maxsize=len(_ALL_VOICES))
def _load_voice(voice_name: str):
    """Load (and cache) one named Piper voice model from PIPER_VOICES_DIR."""
    # Imported lazily: piper-tts pulls in onnxruntime, a heavy, optional
    # dependency that unit tests (which monkeypatch this function) never need
    # to actually install or import.
    from piper import PiperVoice

    voices_dir = Path(get_settings().piper_voices_dir)
    model_path = voices_dir / f"{voice_name}.onnx"
    return PiperVoice.load(str(model_path))


def _synthesize_to_array(voice, text: str) -> tuple[np.ndarray, int]:
    """Step 4 + 5: run Piper synthesis, concatenating its chunks into one array.

    `PiperVoice.synthesize()` yields `AudioChunk`s, each already carrying a
    normalized float32 `audio_float_array` plus the model's `sample_rate` --
    no manual int16-to-float conversion needed.
    """
    chunks = list(voice.synthesize(text))
    if not chunks:
        return np.array([], dtype=np.float32), voice.config.sample_rate
    samples = np.concatenate([c.audio_float_array for c in chunks]).astype(np.float32)
    return samples, chunks[0].sample_rate


def synthesize_turns(turns: list[ConversationTurn], personas: list[Persona]) -> list[TurnAudioClip]:
    """Render every turn's text to speech using a voice consistent per persona."""
    # Step 1: decide each persona's gender once for this whole call.
    genders = _assign_genders(personas)

    clips: list[TurnAudioClip] = []
    for turn in turns:
        # Step 2: pick this speaker's voice within their assigned gender.
        gender = genders.get(turn.speaker_persona_id, "male")
        voice_name = _select_voice_name(turn.speaker_persona_id, gender)

        # Step 3: load (or reuse the cached) voice model.
        voice = _load_voice(voice_name)

        # Step 4 + 5: synthesize and normalize.
        samples, sample_rate_hz = _synthesize_to_array(voice, turn.text)

        # Step 6: package into the transport type Audio Generator consumes.
        clips.append(
            TurnAudioClip(
                turn_index=turn.turn_index,
                speaker_persona_id=turn.speaker_persona_id,
                samples=samples,
                sample_rate_hz=sample_rate_hz,
            )
        )

    return clips
