"""
Unit tests for the Background Audio Detection Agent.

The Silero VAD model loader and speech-timestamp lookup are monkeypatched
entirely, so these tests run fast and deterministically without silero-vad
installed. The energy measurement itself is real (librosa on synthetic audio).
"""

import numpy as np
import soundfile as sf

from src.agents import background_audio_detection_agent as bg_agent
from src.schemas import AudioInput

_SR = 16000


def _write_wav(tmp_path, name: str, y: np.ndarray) -> str:
    path = tmp_path / name
    sf.write(str(path), y.astype("float32"), _SR)
    return str(path)


def test_loud_non_speech_gap_is_flagged(tmp_path, monkeypatch):
    # Layout: [0-1s silence] [1-2s "speech", excluded via mocked VAD] [2-3s loud noise].
    silence = np.zeros(_SR * 1, dtype="float32")
    placeholder_speech = np.zeros(_SR * 1, dtype="float32")
    rng = np.random.default_rng(0)
    loud_noise = (rng.uniform(-1, 1, _SR * 1) * 0.5).astype("float32")
    y = np.concatenate([silence, placeholder_speech, loud_noise])
    audio_path = _write_wav(tmp_path, "mixed.wav", y)

    monkeypatch.setattr(bg_agent, "_load_vad_model", lambda: object())
    monkeypatch.setattr(
        bg_agent, "_get_speech_timestamps_ms", lambda model, path, sr: [(1000, 2000)]
    )

    findings = bg_agent.detect_background_audio(AudioInput(call_id="call-1", audio_path=audio_path))

    assert len(findings) == 1
    assert findings[0].evidence.start_ms == 2000
    assert findings[0].evidence.end_ms == 3000
    assert findings[0].severity is not None
    assert findings[0].category is None


def test_quiet_gaps_are_not_flagged(tmp_path, monkeypatch):
    y = np.zeros(_SR * 3, dtype="float32")
    audio_path = _write_wav(tmp_path, "silent.wav", y)

    monkeypatch.setattr(bg_agent, "_load_vad_model", lambda: object())
    monkeypatch.setattr(bg_agent, "_get_speech_timestamps_ms", lambda model, path, sr: [])

    findings = bg_agent.detect_background_audio(AudioInput(call_id="call-2", audio_path=audio_path))

    assert findings == []


def test_short_gaps_below_minimum_length_are_ignored(tmp_path, monkeypatch):
    y = np.zeros(_SR * 3, dtype="float32")
    audio_path = _write_wav(tmp_path, "mostly_speech.wav", y)

    monkeypatch.setattr(bg_agent, "_load_vad_model", lambda: object())
    # Speech covers almost the whole clip, leaving only a 200ms tail gap --
    # below _MIN_GAP_MS, so it should be ignored regardless of energy.
    monkeypatch.setattr(bg_agent, "_get_speech_timestamps_ms", lambda model, path, sr: [(0, 2800)])

    findings = bg_agent.detect_background_audio(AudioInput(call_id="call-3", audio_path=audio_path))

    assert findings == []
