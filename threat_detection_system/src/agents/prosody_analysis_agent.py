"""
Prosody Analysis Agent
========================
Why this agent is needed:
    Two people can say the exact same words in very different emotional states.
    Prosody -- pitch and loudness, independent of the words themselves -- is
    often the first signal that a conversation is escalating, well before the
    language itself becomes explicitly abusive or threatening. This agent gives
    the Audio Correlation Agent an early, word-independent stress signal.

What it does, step by step:
    1. Loads the call's audio waveform.
    2. For each analysis window (one per diarized speaker segment, or the whole
       clip if no diarization is available), extracts:
         - fundamental frequency / pitch variability, via the pYIN algorithm
         - RMS energy (loudness)
    3. Scores each window against fixed thresholds for "elevated pitch
       variability + elevated loudness" -- a simple, well-established proxy for
       vocal stress/agitation.
    4. Emits one AgentFinding per analyzed window. A finding only carries a
       severity when its window crossed a threshold; category is always None,
       since prosody is a pure audio signal, not a threat-taxonomy classifier
       on its own (see schemas.AgentFinding.category).

Tools used:
    librosa (see docs/architecture/tech_stack_guidelines.md -- Audio Processing) for pitch
    extraction (pyin) and RMS energy. This is pure signal processing -- no ML
    model weights to download and no LLM call, so this agent is fast and fully
    deterministic/local.

Other details:
    The pitch/energy thresholds below are deliberately simple placeholders.
    They should be recalibrated against Project 1's labeled benchmark dataset
    (see docs/architecture/threat_detection_architecture-plan.md §5, Risk Scoring Model)
    once real calibration data is available, rather than trusted as-is.
"""

from __future__ import annotations

import librosa
import numpy as np

from ..schemas import AgentFinding, AudioInput, DiarizationSegment, Domain, Evidence, Severity

# Typical fundamental-frequency range for human speech, used to bound pYIN's search.
_PITCH_FMIN_HZ = 65.0
_PITCH_FMAX_HZ = 300.0

# Placeholder severity thresholds -- see module docstring "Other details".
_PITCH_STD_MEDIUM_HZ = 25.0
_PITCH_STD_HIGH_HZ = 40.0
_ENERGY_MEDIUM = 0.04
_ENERGY_HIGH = 0.08


def _extract_pitch_and_energy(y: np.ndarray, sr: int) -> tuple[float, float, float]:
    """Return (pitch_std_hz, energy_mean, voiced_ratio) for one audio window."""
    # pYIN returns per-frame f0 estimates plus a boolean "is this frame voiced" flag.
    f0, voiced_flag, _voiced_prob = librosa.pyin(y, fmin=_PITCH_FMIN_HZ, fmax=_PITCH_FMAX_HZ, sr=sr)
    voiced_f0 = f0[voiced_flag] if voiced_flag is not None else np.array([])

    pitch_std = float(np.std(voiced_f0)) if voiced_f0.size > 0 else 0.0
    voiced_ratio = float(voiced_flag.mean()) if voiced_flag is not None and voiced_flag.size > 0 else 0.0

    rms = librosa.feature.rms(y=y)
    energy_mean = float(np.mean(rms))

    return pitch_std, energy_mean, voiced_ratio


def _score_severity(pitch_std: float, energy_mean: float) -> Severity | None:
    """Simple threshold-based severity scoring -- see module docstring."""
    if pitch_std >= _PITCH_STD_HIGH_HZ and energy_mean >= _ENERGY_HIGH:
        return Severity.HIGH
    if pitch_std >= _PITCH_STD_MEDIUM_HZ or energy_mean >= _ENERGY_MEDIUM:
        return Severity.MEDIUM
    return None


def analyze_prosody(
    audio_input: AudioInput, diarization: list[DiarizationSegment] | None = None
) -> list[AgentFinding]:
    """Analyze pitch/energy per speaker segment (or the whole clip) for vocal stress."""
    # Step 1: load the full waveform once (sr=None preserves the file's native rate).
    y, sr = librosa.load(audio_input.audio_path, sr=None)

    # Step 2: decide the analysis windows -- one per diarized segment, or the
    # whole clip as a single window if no diarization was supplied.
    if diarization:
        windows = [(seg.start_ms, seg.end_ms) for seg in diarization]
    else:
        duration_ms = int(len(y) / sr * 1000)
        windows = [(0, duration_ms)]

    findings: list[AgentFinding] = []
    for start_ms, end_ms in windows:
        start_sample = int(start_ms / 1000 * sr)
        end_sample = int(end_ms / 1000 * sr)
        window_audio = y[start_sample:end_sample]
        if window_audio.size == 0:
            continue

        # Step 3: extract features and score this window.
        pitch_std, energy_mean, voiced_ratio = _extract_pitch_and_energy(window_audio, sr)
        severity = _score_severity(pitch_std, energy_mean)

        # Step 4: emit a finding for this window. Confidence reflects how much
        # reliably-voiced signal the assessment was based on, not the severity
        # itself -- a window that's mostly silence gives a low-confidence read.
        findings.append(
            AgentFinding(
                agent_name="prosody",
                domain=Domain.AUDIO,
                category=None,
                severity=severity,
                confidence=round(voiced_ratio, 3),
                evidence=Evidence(start_ms=start_ms, end_ms=end_ms),
                summary=(
                    f"pitch_std={pitch_std:.1f}Hz, energy_mean={energy_mean:.3f}"
                    + (f" -- elevated ({severity.value})" if severity else " -- within normal range")
                ),
            )
        )

    return findings
