"""
Background Audio Detection Agent
===================================
Why this agent is needed:
    A threatening statement made in an otherwise quiet room reads differently
    than one made against loud background noise, crowd sounds, or another
    voice in the background (which might indicate coercion, a public setting,
    or an unaccounted-for third party). This agent flags when significant
    non-speech audio energy is present, so the Audio Correlation Agent can
    factor "what else is going on in this recording" into its assessment.

What it does, step by step:
    1. Runs Silero VAD (Voice Activity Detection) against the call's audio to
       find which time ranges contain speech.
    2. Computes the inverse: the non-speech ("background") time ranges, dropping
       gaps too short to be meaningful (normal micro-pauses between words).
    3. For each background range, measures RMS energy via librosa. A background
       range with energy well above near-silence indicates notable background
       noise, as opposed to an ordinary quiet pause.
    4. Emits one AgentFinding per notable background range; near-silent gaps
       are not reported at all (they're normal, not a signal).

Tools used:
    Silero VAD (see docs/architecture/tech_stack_guidelines.md -- Voice Activity Detection)
    to locate speech vs. non-speech regions; librosa for the energy measurement
    within each non-speech region.

Other details:
    The energy threshold below is a simple placeholder pending calibration
    against Project 1's labeled benchmark dataset (see
    docs/architecture/threat_detection_architecture-plan.md §5). category is always None --
    like Prosody, this is a pure audio signal, not itself a threat-taxonomy
    classification.
"""

from __future__ import annotations

from functools import lru_cache

import librosa
import numpy as np

from ..schemas import AgentFinding, AudioInput, Domain, Evidence, Severity

# Minimum length of a non-speech gap worth analyzing at all -- short gaps
# between words are normal and not informative.
_MIN_GAP_MS = 500

# Placeholder energy threshold -- see module docstring "Other details".
_BACKGROUND_ENERGY_NOTABLE = 0.03


@lru_cache(maxsize=1)
def _load_vad_model():
    # Imported lazily: silero-vad pulls in torch/onnxruntime, a heavy, optional
    # dependency that unit tests (which monkeypatch this function) never need
    # to install.
    from silero_vad import load_silero_vad

    return load_silero_vad()


def _get_speech_timestamps_ms(model, audio_path: str, sr: int) -> list[tuple[int, int]]:
    """Run Silero VAD and return speech ranges in milliseconds."""
    from silero_vad import get_speech_timestamps, read_audio

    wav = read_audio(audio_path, sampling_rate=sr)
    timestamps = get_speech_timestamps(wav, model, sampling_rate=sr, return_seconds=True)
    return [(int(t["start"] * 1000), int(t["end"] * 1000)) for t in timestamps]


def _invert_to_gaps(speech_ranges: list[tuple[int, int]], total_duration_ms: int) -> list[tuple[int, int]]:
    """Turn a list of speech ranges into the complementary list of non-speech gaps."""
    gaps: list[tuple[int, int]] = []
    cursor = 0
    for start_ms, end_ms in sorted(speech_ranges):
        if start_ms - cursor >= _MIN_GAP_MS:
            gaps.append((cursor, start_ms))
        cursor = max(cursor, end_ms)
    if total_duration_ms - cursor >= _MIN_GAP_MS:
        gaps.append((cursor, total_duration_ms))
    return gaps


def detect_background_audio(audio_input: AudioInput) -> list[AgentFinding]:
    """Flag non-speech audio ranges with notable energy (background noise/events)."""
    # Step 1: load audio once, both for VAD (Silero expects a fixed sample
    # rate) and for the energy measurement below.
    y, sr = librosa.load(audio_input.audio_path, sr=16000)
    total_duration_ms = int(len(y) / sr * 1000)

    model = _load_vad_model()
    speech_ranges = _get_speech_timestamps_ms(model, audio_input.audio_path, sr)

    # Step 2: the gaps between/around speech are the candidate background windows.
    gaps = _invert_to_gaps(speech_ranges, total_duration_ms)

    findings: list[AgentFinding] = []
    for start_ms, end_ms in gaps:
        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        window = y[start_sample:end_sample]
        if window.size == 0:
            continue

        # Step 3: measure how "loud" this supposedly-silent gap actually is.
        energy = float(np.mean(librosa.feature.rms(y=window)))
        if energy < _BACKGROUND_ENERGY_NOTABLE:
            continue  # near-silence -- a normal pause, not a finding.

        # Step 4: emit a finding for this notable background window.
        findings.append(
            AgentFinding(
                agent_name="background_audio",
                domain=Domain.AUDIO,
                category=None,
                severity=Severity.LOW if energy < _BACKGROUND_ENERGY_NOTABLE * 2 else Severity.MEDIUM,
                confidence=round(min(energy / _BACKGROUND_ENERGY_NOTABLE, 1.0), 3),
                evidence=Evidence(start_ms=start_ms, end_ms=end_ms),
                summary=f"non-speech energy={energy:.3f} during a {end_ms - start_ms}ms gap",
            )
        )

    return findings
