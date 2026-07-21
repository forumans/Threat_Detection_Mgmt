"""
TTS Engine Agent
==================
Why this agent is needed:
    Everything upstream is text. This agent is where a sample first becomes
    audio -- rendering each turn's (possibly translated) dialogue into speech
    using a voice consistent with that turn's persona, so the Audio Generator
    has real per-turn clips to stitch and mix.

What it does, step by step:
    1. For each turn, looks up the speaking persona and deterministically maps
       it to one voice from a small local voice pool (the same persona always
       gets the same voice, so a caller doesn't change voices mid-call).
    2. Lazily loads that voice's Piper model (cached after first load --
       voice models are expensive to construct).
    3. Synthesizes the turn's text through Piper, producing raw PCM audio.
    4. Converts the raw audio into a normalized float32 mono array.
    5. Returns one `TurnAudioClip` per turn, in turn order, ready for the
       Audio Generator to stitch together.

Tools used:
    Piper (see docs/architecture/synthetic_data_gen_architecture-plan.md §4 --
    TTS): a lightweight, local, ONNX-based TTS engine -- no API key, no GPU
    required. Runs fully offline once voice models are downloaded.

Other details:
    Piper voices are discrete pretrained models, not continuously tunable by
    the pitch/pace/timbre fields in `Persona.voice_traits` -- the pool below
    is a placeholder until a real per-locale voice library is wired in (see
    the "Voice diversity/accent coverage strategy" item in the architecture
    doc's Open Questions). `TurnAudioClip` is a plain dataclass, not a
    Pydantic schema, since it carries a raw numpy array (an internal,
    non-serialized transport type between this agent and the Audio Generator,
    unlike the persisted schemas in src/schemas.py).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..config import get_settings
from ..schemas import ConversationTurn, Persona

# Placeholder voice pool -- see module docstring "Other details".
_VOICE_POOL = ["en_US-amy-medium", "en_US-ryan-high", "en_US-lessac-medium"]


@dataclass
class TurnAudioClip:
    turn_index: int
    speaker_persona_id: str
    samples: np.ndarray  # mono float32 PCM, range [-1.0, 1.0]
    sample_rate_hz: int


def _select_voice_name(persona_id: str) -> str:
    """Step 1: deterministically map a persona to one voice from the pool."""
    index = abs(hash(persona_id)) % len(_VOICE_POOL)
    return _VOICE_POOL[index]


@lru_cache(maxsize=len(_VOICE_POOL))
def _load_voice(voice_name: str):
    # Imported lazily: piper-tts pulls in onnxruntime, a heavy, optional
    # dependency that unit tests (which monkeypatch this function) never need
    # to actually install or import.
    from piper import PiperVoice

    voices_dir = Path(get_settings().piper_voices_dir)
    model_path = voices_dir / f"{voice_name}.onnx"
    return PiperVoice.load(str(model_path))


def _synthesize_to_array(voice, text: str) -> tuple[np.ndarray, int]:
    """Step 3 + 4: run Piper synthesis and normalize the result to float32 PCM."""
    raw_pcm = b"".join(voice.synthesize_stream_raw(text))
    samples = np.frombuffer(raw_pcm, dtype=np.int16).astype(np.float32) / 32768.0
    return samples, voice.config.sample_rate


def synthesize_turns(turns: list[ConversationTurn], personas: list[Persona]) -> list[TurnAudioClip]:
    """Render every turn's text to speech using a voice consistent per persona."""
    clips: list[TurnAudioClip] = []
    for turn in turns:
        # Step 1: pick this speaker's voice.
        voice_name = _select_voice_name(turn.speaker_persona_id)

        # Step 2: load (or reuse the cached) voice model.
        voice = _load_voice(voice_name)

        # Step 3 + 4: synthesize and normalize.
        samples, sample_rate_hz = _synthesize_to_array(voice, turn.text)

        # Step 5: package into the transport type Audio Generator consumes.
        clips.append(
            TurnAudioClip(
                turn_index=turn.turn_index,
                speaker_persona_id=turn.speaker_persona_id,
                samples=samples,
                sample_rate_hz=sample_rate_hz,
            )
        )

    return clips
