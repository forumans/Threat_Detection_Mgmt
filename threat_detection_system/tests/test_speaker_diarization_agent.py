"""
Unit tests for the Speaker Diarization Agent.

The pyannote.audio pipeline loader is monkeypatched entirely, so these tests
run fast and deterministically without a HUGGINGFACE_TOKEN, model download, or
pyannote.audio/torch installed at all.
"""

from src.agents import speaker_diarization_agent
from src.schemas import AudioInput


class _FakeTurn:
    def __init__(self, start: float, end: float):
        self.start = start
        self.end = end


class _FakeAnnotation:
    def __init__(self, tracks):
        self._tracks = tracks

    def itertracks(self, yield_label=True):
        return iter(self._tracks)


class _FakePipeline:
    def __init__(self, tracks):
        self._tracks = tracks

    def __call__(self, audio_path):
        return _FakeAnnotation(self._tracks)


def test_diarize_converts_and_sorts_tracks_by_start_time(monkeypatch):
    # Deliberately out of order, to verify the agent sorts its output.
    tracks = [
        (_FakeTurn(5.0, 8.0), None, "SPEAKER_01"),
        (_FakeTurn(0.0, 4.5), None, "SPEAKER_00"),
    ]
    monkeypatch.setattr(
        speaker_diarization_agent, "_load_pipeline", lambda model_name, hf_token: _FakePipeline(tracks)
    )

    segments = speaker_diarization_agent.diarize(AudioInput(call_id="call-1", audio_path="fake.wav"))

    assert [s.speaker_label for s in segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert segments[0].start_ms == 0
    assert segments[0].end_ms == 4500
    assert segments[1].start_ms == 5000
    assert segments[1].end_ms == 8000
    assert all(s.confidence == 1.0 for s in segments)


def test_diarize_returns_empty_list_for_no_tracks(monkeypatch):
    monkeypatch.setattr(speaker_diarization_agent, "_load_pipeline", lambda model_name, hf_token: _FakePipeline([]))

    segments = speaker_diarization_agent.diarize(AudioInput(call_id="call-2", audio_path="fake.wav"))

    assert segments == []
