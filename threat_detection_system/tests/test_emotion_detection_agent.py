"""
Unit tests for the Emotion Detection Agent.

The SpeechBrain classifier loader is monkeypatched entirely, so these tests
run fast and deterministically without downloading model weights or having
speechbrain/torch installed at all. Real (tiny) WAV files are still used for
the diarized case, since the agent genuinely slices and re-writes audio.
"""

import numpy as np
import soundfile as sf

from src.agents import emotion_detection_agent
from src.schemas import AudioInput, DiarizationSegment

_SR = 16000


class _FakeClassifier:
    def __init__(self, label: str, confidence: float):
        self._label = label
        self._confidence = confidence

    def classify_file(self, path: str):
        return None, [self._confidence], None, [self._label]


def _write_wav(tmp_path, name: str, seconds: float = 2.0) -> str:
    y = np.zeros(int(_SR * seconds), dtype="float32")
    path = tmp_path / name
    sf.write(str(path), y, _SR)
    return str(path)


def test_whole_clip_angry_label_maps_to_high_severity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        emotion_detection_agent, "_load_classifier", lambda model_name: _FakeClassifier("ang", 0.87)
    )
    audio_path = _write_wav(tmp_path, "angry.wav")

    findings = emotion_detection_agent.detect_emotion(AudioInput(call_id="call-1", audio_path=audio_path))

    assert len(findings) == 1
    assert findings[0].severity.value == "high"
    assert findings[0].confidence == 0.87
    assert findings[0].evidence.start_ms == 0
    assert findings[0].evidence.end_ms == 2000


def test_whole_clip_neutral_label_has_no_severity(tmp_path, monkeypatch):
    monkeypatch.setattr(
        emotion_detection_agent, "_load_classifier", lambda model_name: _FakeClassifier("neu", 0.6)
    )
    audio_path = _write_wav(tmp_path, "neutral.wav")

    findings = emotion_detection_agent.detect_emotion(AudioInput(call_id="call-2", audio_path=audio_path))

    assert findings[0].severity is None


def test_diarized_segments_are_classified_individually(tmp_path, monkeypatch):
    monkeypatch.setattr(
        emotion_detection_agent, "_load_classifier", lambda model_name: _FakeClassifier("sad", 0.7)
    )
    audio_path = _write_wav(tmp_path, "two_speakers.wav")
    diarization = [
        DiarizationSegment(speaker_label="SPEAKER_00", start_ms=0, end_ms=1000, confidence=1.0),
        DiarizationSegment(speaker_label="SPEAKER_01", start_ms=1000, end_ms=2000, confidence=1.0),
    ]

    findings = emotion_detection_agent.detect_emotion(
        AudioInput(call_id="call-3", audio_path=audio_path), diarization=diarization
    )

    assert len(findings) == 2
    assert findings[0].evidence.start_ms == 0 and findings[0].evidence.end_ms == 1000
    assert findings[1].evidence.start_ms == 1000 and findings[1].evidence.end_ms == 2000
    assert all(f.severity.value == "low" for f in findings)
