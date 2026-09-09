"""
Speaker Diarization Agent
==========================
Why this agent is needed:
    Raw transcription tells you WHAT was said, not WHO said it. Without knowing
    which speaker is talking at each moment, downstream agents can't tell a
    caller's threatening statement from an agent calmly repeating it back, and
    the Prosody/Emotion agents can't isolate one speaker's voice from another's.
    This agent answers "who spoke when" so the rest of the pipeline can attribute
    everything correctly.

What it does, step by step:
    1. Lazily loads a pretrained pyannote.audio diarization pipeline (a gated
       Hugging Face model -- requires accepting its license and a
       HUGGINGFACE_TOKEN, see src/config.py).
    2. Runs the pipeline against the call's audio file.
    3. Iterates the pipeline's output tracks (each a time span + speaker label)
       and converts them into our shared DiarizationSegment records.
    4. Returns the segments sorted by start time.

Tools used:
    pyannote.audio (see docs/architecture/tech_stack_guidelines.md -- Diarization), specifically
    the pretrained "speaker-diarization-3.1" pipeline. This is a real neural
    diarization model, not a heuristic -- it requires a HUGGINGFACE_TOKEN and a
    one-time acceptance of the model's license on huggingface.co.

Other details:
    The pipeline is cached after first load (it's expensive to construct), and
    the loader is a separate function so unit tests can monkeypatch it without
    needing pyannote.audio (and its torch dependency) installed at all.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from ..schemas import AudioInput, DiarizationSegment


@lru_cache(maxsize=1)
def _load_pipeline(model_name: str, hf_token: str | None):
    """Load (and cache, via lru_cache) the pretrained pyannote.audio diarization pipeline."""
    # Imported lazily: pyannote.audio pulls in torch, a heavy, optional
    # dependency that unit tests (which monkeypatch this function) never need
    # to actually install or import.
    from pyannote.audio import Pipeline

    return Pipeline.from_pretrained(model_name, use_auth_token=hf_token)


def diarize(audio_input: AudioInput) -> list[DiarizationSegment]:
    """Identify "who spoke when" in one call's audio via pyannote.audio."""
    settings = get_settings()

    # Step 1: get (or lazily create) the shared diarization pipeline.
    pipeline = _load_pipeline(settings.diarization_model, settings.huggingface_token)

    # Step 2: run the pipeline. pyannote returns an Annotation object.
    annotation = pipeline(audio_input.audio_path)

    # Step 3: convert each (time span, speaker) track into our shared schema.
    # pyannote doesn't expose a per-track confidence score, so we use 1.0 --
    # the pipeline already thresholds internally before emitting a track.
    segments = [
        DiarizationSegment(
            speaker_label=str(speaker),
            start_ms=int(turn.start * 1000),
            end_ms=int(turn.end * 1000),
            confidence=1.0,
        )
        for turn, _, speaker in annotation.itertracks(yield_label=True)
    ]

    # Step 4: return in chronological order, which downstream merging with the
    # transcript relies on.
    return sorted(segments, key=lambda s: s.start_ms)
