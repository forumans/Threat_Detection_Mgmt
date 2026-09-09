"""
Speech-to-Text Agent
=====================
Why this agent is needed:
    Every transcript-intelligence agent (Verbal Abuse, Threat, Fraud & Social
    Engineering, Compliance) operates on text, not raw audio. This agent is the
    single place where audio becomes text, so every downstream agent can stay
    audio-agnostic and never touch an audio file directly.

What it does, step by step:
    1. Lazily loads a local faster-whisper model (weights are downloaded and
       cached by the library on first use).
    2. Runs the model against the call's audio file, requesting word-level
       timestamps.
    3. Flattens the model's segment/word output into a flat list of STTWord
       records.
    4. Joins all words into a single full_text string for convenience.
    5. Returns a Transcript (words + full_text + detected language).

Tools used:
    faster-whisper (CTranslate2-based local Whisper inference; see
    docs/architecture/tech_stack_guidelines.md -- STT). Runs fully locally with no API key,
    which keeps this agent independent of whichever LLM provider is configured
    for the reasoning agents elsewhere in this package.

Other details:
    Model size/device are configurable via Settings.whisper_model_size /
    whisper_device (src/config.py), so a small model can be used for fast local
    dev and a larger one in production. The loaded model is cached per
    (size, device) so repeated calls don't reload it from disk every time.
"""

from __future__ import annotations

from functools import lru_cache

from ..config import get_settings
from ..schemas import AudioInput, STTWord, Transcript


@lru_cache(maxsize=4)
def _load_model(model_size: str, device: str):
    """Load (and cache, via lru_cache) the faster-whisper speech-to-text model."""
    # Imported lazily: faster-whisper pulls in ctranslate2, a heavy, optional
    # dependency that unit tests (which monkeypatch this function) never need
    # to actually install or import.
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device)


def transcribe(audio_input: AudioInput) -> Transcript:
    """Transcribe one call's audio into a word-level Transcript via faster-whisper."""
    settings = get_settings()

    # Step 1: get (or lazily create) the shared Whisper model instance.
    model = _load_model(settings.whisper_model_size, settings.whisper_device)

    # Step 2: run inference with word-level timestamps enabled.
    segments, info = model.transcribe(audio_input.audio_path, word_timestamps=True)

    # Step 3: flatten every segment's words into a single flat list, converting
    # faster-whisper's float-seconds timestamps into the millisecond ints used
    # throughout our shared schemas.
    words: list[STTWord] = []
    for segment in segments:
        for word in segment.words or []:
            words.append(
                STTWord(
                    start_ms=int(word.start * 1000),
                    end_ms=int(word.end * 1000),
                    text=word.word.strip(),
                    confidence=float(word.probability),
                )
            )

    # Step 4: build the joined full-text transcript for convenience (used by
    # agents that just need the plain text, not per-word timing).
    full_text = " ".join(w.text for w in words)

    # Step 5: package into the shared Transcript contract.
    return Transcript(words=words, full_text=full_text, language=info.language)
