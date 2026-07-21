"""
Unit tests for the Speech-to-Text Agent.

The faster-whisper model loader is monkeypatched entirely, so these tests run
fast and deterministically without downloading model weights or touching real
audio -- and without needing faster-whisper installed at all.
"""

from types import SimpleNamespace

from src.agents import speech_to_text_agent
from src.schemas import AudioInput


class _FakeWord:
    def __init__(self, start: float, end: float, word: str, probability: float):
        self.start = start
        self.end = end
        self.word = word
        self.probability = probability


class _FakeSegment:
    def __init__(self, words):
        self.words = words


class _FakeModel:
    def __init__(self, segments, language: str):
        self._segments = segments
        self._language = language

    def transcribe(self, audio_path, word_timestamps=True):
        return self._segments, SimpleNamespace(language=self._language)


def test_transcribe_flattens_words_and_builds_full_text(monkeypatch):
    fake_segments = [
        _FakeSegment(words=[_FakeWord(0.0, 0.5, " Hello", 0.98), _FakeWord(0.5, 1.0, " there", 0.95)]),
        _FakeSegment(words=[_FakeWord(1.2, 1.6, " friend", 0.9)]),
    ]
    fake_model = _FakeModel(fake_segments, language="en")
    monkeypatch.setattr(speech_to_text_agent, "_load_model", lambda size, device: fake_model)

    result = speech_to_text_agent.transcribe(AudioInput(call_id="call-1", audio_path="fake.wav"))

    assert result.language == "en"
    assert result.full_text == "Hello there friend"
    assert len(result.words) == 3
    assert result.words[0].start_ms == 0
    assert result.words[0].end_ms == 500
    assert result.words[0].confidence == 0.98


def test_transcribe_handles_segment_with_no_words(monkeypatch):
    fake_segments = [_FakeSegment(words=None)]
    fake_model = _FakeModel(fake_segments, language="en")
    monkeypatch.setattr(speech_to_text_agent, "_load_model", lambda size, device: fake_model)

    result = speech_to_text_agent.transcribe(AudioInput(call_id="call-2", audio_path="fake.wav"))

    assert result.words == []
    assert result.full_text == ""
