"""
Unit tests for the TTS Engine Agent.

The Piper voice loader is monkeypatched entirely, so these tests run fast and
deterministically without downloading voice models or having piper-tts
installed at all.
"""

from types import SimpleNamespace

import numpy as np

from src.agents import tts_engine_agent
from src.schemas import ConversationTurn


class _FakeVoice:
    def __init__(self, sample_rate: int = 22050):
        self.config = SimpleNamespace(sample_rate=sample_rate)
        self._sample_rate = sample_rate

    def synthesize(self, text: str):
        # Two fake AudioChunk-shaped objects, matching PiperVoice's real API:
        # already-normalized float32 samples plus a sample_rate per chunk.
        yield SimpleNamespace(
            audio_float_array=np.array([0.1, -0.1], dtype=np.float32), sample_rate=self._sample_rate
        )
        yield SimpleNamespace(
            audio_float_array=np.array([0.2, -0.2], dtype=np.float32), sample_rate=self._sample_rate
        )


def test_one_clip_is_produced_per_turn(monkeypatch):
    monkeypatch.setattr(tts_engine_agent, "_load_voice", lambda voice_name: _FakeVoice())
    turns = [
        ConversationTurn(turn_index=0, speaker_persona_id="p-caller", text="hi", intended_emotion="neutral"),
        ConversationTurn(turn_index=1, speaker_persona_id="p-agent", text="hello", intended_emotion="neutral"),
    ]

    clips = tts_engine_agent.synthesize_turns(turns, personas=[])

    assert len(clips) == 2
    assert clips[0].turn_index == 0
    assert clips[0].speaker_persona_id == "p-caller"
    assert clips[1].turn_index == 1


def test_samples_are_normalized_to_float32_range(monkeypatch):
    monkeypatch.setattr(tts_engine_agent, "_load_voice", lambda voice_name: _FakeVoice())
    turns = [ConversationTurn(turn_index=0, speaker_persona_id="p-caller", text="hi", intended_emotion="neutral")]

    clips = tts_engine_agent.synthesize_turns(turns, personas=[])

    assert clips[0].samples.dtype == np.float32
    assert np.all(np.abs(clips[0].samples) <= 1.0)
    assert clips[0].sample_rate_hz == 22050


def test_same_persona_always_selects_the_same_voice():
    first = tts_engine_agent._select_voice_name("persona-abc")
    second = tts_engine_agent._select_voice_name("persona-abc")

    assert first == second
    assert first in tts_engine_agent._VOICE_POOL
