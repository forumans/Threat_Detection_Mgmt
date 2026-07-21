"""
Emotion Detection Agent
=========================
Why this agent is needed:
    Prosody measures raw vocal stress, but doesn't name the emotion behind it.
    Knowing specifically that a speaker sounds angry (vs. sad, vs. calm) is a
    stronger, more specific signal for the Audio Correlation Agent to weigh --
    anger correlates with escalation risk in a way sadness or neutrality don't.

What it does, step by step:
    1. Lazily loads a pretrained SpeechBrain emotion-recognition classifier
       (wav2vec2-based, trained on IEMOCAP).
    2. For each analysis window (one per diarized speaker segment, or the whole
       clip if no diarization is available), extracts the audio for that window
       to a short-lived temporary WAV file (the classifier's API takes a file
       path, not raw samples).
    3. Runs the classifier, which returns a predicted emotion label and its
       confidence score.
    4. Maps the predicted label to a severity (anger/fear are treated as high
       severity; sadness as low; happiness/neutral as no signal) and emits one
       AgentFinding per window.

Tools used:
    SpeechBrain (see docs/architecture/tech_stack_guidelines.md -- Emotion Recognition),
    specifically the "speechbrain/emotion-recognition-wav2vec2-IEMOCAP"
    pretrained model from Hugging Face. Downloads model weights on first use.

Other details:
    The label-to-severity mapping is a simple placeholder pending calibration
    against Project 1's labeled benchmark dataset (see
    docs/architecture/threat_detection_architecture-plan.md §5). Temporary per-segment WAV
    files live only inside a TemporaryDirectory and are cleaned up immediately
    after classification.
"""

from __future__ import annotations

import tempfile
from functools import lru_cache
from pathlib import Path

import soundfile as sf

from ..config import get_settings
from ..schemas import AgentFinding, AudioInput, DiarizationSegment, Domain, Evidence, Severity

# IEMOCAP-style short labels -> our severity scale. Any label not listed here
# (e.g. "hap", "neu") is treated as no notable severity.
_EMOTION_SEVERITY: dict[str, Severity] = {
    "ang": Severity.HIGH,
    "fea": Severity.HIGH,
    "sad": Severity.LOW,
}


@lru_cache(maxsize=1)
def _load_classifier(model_name: str):
    # Imported lazily: speechbrain pulls in torch, a heavy, optional dependency
    # that unit tests (which monkeypatch this function) never need to install.
    from speechbrain.inference.interfaces import foreign_class

    return foreign_class(
        source=model_name,
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
    )


def _classify_file(classifier, path: str) -> tuple[str, float]:
    """Run the classifier on one audio file, returning (label, confidence)."""
    _out_prob, score, _index, text_lab = classifier.classify_file(path)
    label = text_lab[0] if isinstance(text_lab, (list, tuple)) else str(text_lab)
    confidence = float(score[0]) if hasattr(score, "__getitem__") else float(score)
    return label, confidence


def _build_finding(label: str, confidence: float, start_ms: int, end_ms: int) -> AgentFinding:
    """Step 4: map a raw emotion label + confidence into our shared AgentFinding shape."""
    severity = _EMOTION_SEVERITY.get(label)
    return AgentFinding(
        agent_name="emotion",
        domain=Domain.AUDIO,
        category=None,
        severity=severity,
        confidence=round(confidence, 3),
        evidence=Evidence(start_ms=start_ms, end_ms=end_ms),
        summary=f"predicted_emotion={label}" + (f" -- elevated ({severity.value})" if severity else ""),
    )


def detect_emotion(
    audio_input: AudioInput, diarization: list[DiarizationSegment] | None = None
) -> list[AgentFinding]:
    """Classify the dominant emotion per speaker segment (or the whole clip)."""
    settings = get_settings()

    # Step 1: get (or lazily create) the shared classifier instance.
    classifier = _load_classifier(settings.emotion_model)

    if not diarization:
        # Step 2 (whole-clip case): classify the original file directly --
        # no slicing needed.
        label, confidence = _classify_file(classifier, audio_input.audio_path)
        duration_ms = int(sf.info(audio_input.audio_path).duration * 1000)
        return [_build_finding(label, confidence, 0, duration_ms)]

    # Step 2 (per-segment case): read the waveform once, then classify each
    # diarized segment via a short-lived temporary WAV file.
    y, sr = sf.read(audio_input.audio_path)
    findings: list[AgentFinding] = []
    with tempfile.TemporaryDirectory() as tmp_dir:
        for segment in diarization:
            start_sample = int(segment.start_ms / 1000 * sr)
            end_sample = int(segment.end_ms / 1000 * sr)
            window = y[start_sample:end_sample]
            if len(window) == 0:
                continue

            segment_path = str(Path(tmp_dir) / f"segment_{segment.start_ms}_{segment.end_ms}.wav")
            sf.write(segment_path, window, sr)

            # Step 3: classify this one segment.
            label, confidence = _classify_file(classifier, segment_path)
            findings.append(_build_finding(label, confidence, segment.start_ms, segment.end_ms))

    return findings
