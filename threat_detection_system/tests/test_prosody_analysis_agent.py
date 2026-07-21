"""
Unit tests for the Prosody Analysis Agent.

Unlike the ML-backed agents, this one is pure signal processing (librosa), so
these tests use real synthetic audio -- a steady tone for the "calm" case and
a wide, loud frequency sweep for the "elevated" case -- rather than mocks.
"""

import librosa
import soundfile as sf

from src.agents.prosody_analysis_agent import analyze_prosody
from src.schemas import AudioInput, DiarizationSegment

_SR = 16000


def _write_wav(tmp_path, name: str, y) -> str:
    path = tmp_path / name
    sf.write(str(path), y, _SR)
    return str(path)


def test_calm_steady_tone_yields_no_severity(tmp_path):
    # A quiet, constant-pitch tone: minimal pitch variance, low energy.
    y = librosa.tone(150, sr=_SR, duration=2.0) * 0.02
    audio_path = _write_wav(tmp_path, "calm.wav", y)

    findings = analyze_prosody(AudioInput(call_id="call-1", audio_path=audio_path))

    assert len(findings) == 1
    assert findings[0].severity is None
    assert findings[0].category is None
    assert findings[0].agent_name == "prosody"


def test_loud_wide_pitch_sweep_yields_elevated_severity(tmp_path):
    # A loud, wide frequency sweep: high pitch variance and high energy --
    # simulates a raised, agitated voice.
    y = librosa.chirp(fmin=100, fmax=250, sr=_SR, duration=2.0) * 0.3
    audio_path = _write_wav(tmp_path, "stressed.wav", y)

    findings = analyze_prosody(AudioInput(call_id="call-2", audio_path=audio_path))

    assert len(findings) == 1
    assert findings[0].severity is not None
    assert "elevated" in findings[0].summary


def test_analyzes_one_window_per_diarization_segment(tmp_path):
    y = librosa.tone(150, sr=_SR, duration=2.0) * 0.02
    audio_path = _write_wav(tmp_path, "two_speakers.wav", y)
    diarization = [
        DiarizationSegment(speaker_label="SPEAKER_00", start_ms=0, end_ms=1000, confidence=1.0),
        DiarizationSegment(speaker_label="SPEAKER_01", start_ms=1000, end_ms=2000, confidence=1.0),
    ]

    findings = analyze_prosody(AudioInput(call_id="call-3", audio_path=audio_path), diarization=diarization)

    assert len(findings) == 2
    assert findings[0].evidence.start_ms == 0
    assert findings[0].evidence.end_ms == 1000
    assert findings[1].evidence.start_ms == 1000
    assert findings[1].evidence.end_ms == 2000
