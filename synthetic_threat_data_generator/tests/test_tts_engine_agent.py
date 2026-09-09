"""
Unit tests for the TTS Engine Agent.

The Piper voice loader is monkeypatched entirely, so these tests run fast and
deterministically without downloading voice models or having piper-tts
installed at all.
"""

from types import SimpleNamespace

import numpy as np

from src.agents import tts_engine_agent
from src.schemas import ConversationTurn, Persona


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


def _persona(persona_id: str) -> Persona:
    return Persona(persona_id=persona_id, role="caller", voice_traits={"name": "X"}, emotional_baseline="calm")


def test_one_clip_is_produced_per_turn(monkeypatch):
    monkeypatch.setattr(tts_engine_agent, "_load_voice", lambda voice_name: _FakeVoice())
    turns = [
        ConversationTurn(turn_index=0, speaker_persona_id="p-caller", text="hi", intended_emotion="neutral"),
        ConversationTurn(turn_index=1, speaker_persona_id="p-agent", text="hello", intended_emotion="neutral"),
    ]
    personas = [_persona("p-caller"), _persona("p-agent")]

    clips = tts_engine_agent.synthesize_turns(turns, personas)

    assert len(clips) == 2
    assert clips[0].turn_index == 0
    assert clips[0].speaker_persona_id == "p-caller"
    assert clips[1].turn_index == 1


def test_samples_are_normalized_to_float32_range(monkeypatch):
    monkeypatch.setattr(tts_engine_agent, "_load_voice", lambda voice_name: _FakeVoice())
    turns = [ConversationTurn(turn_index=0, speaker_persona_id="p-caller", text="hi", intended_emotion="neutral")]

    clips = tts_engine_agent.synthesize_turns(turns, personas=[_persona("p-caller")])

    assert clips[0].samples.dtype == np.float32
    assert np.all(np.abs(clips[0].samples) <= 1.0)
    assert clips[0].sample_rate_hz == 22050


def test_same_persona_always_selects_the_same_voice():
    first = tts_engine_agent._select_voice_name("persona-abc", "male")
    second = tts_engine_agent._select_voice_name("persona-abc", "male")

    assert first == second
    assert first in tts_engine_agent._MALE_VOICES


def test_select_voice_name_respects_gender():
    assert tts_engine_agent._select_voice_name("persona-abc", "male") in tts_engine_agent._MALE_VOICES
    assert tts_engine_agent._select_voice_name("persona-abc", "female") in tts_engine_agent._FEMALE_VOICES


def test_two_personas_in_one_call_are_assigned_different_genders():
    genders = tts_engine_agent._assign_genders([_persona("p-caller"), _persona("p-agent")])

    assert genders["p-caller"] != genders["p-agent"]
    assert set(genders.values()) == {"male", "female"}


def test_gender_assignment_alternates_across_more_than_two_personas():
    personas = [_persona("p-1"), _persona("p-2"), _persona("p-3"), _persona("p-4")]

    genders = tts_engine_agent._assign_genders(personas)

    assert genders["p-1"] == genders["p-3"]
    assert genders["p-2"] == genders["p-4"]
    assert genders["p-1"] != genders["p-2"]


def test_same_persona_set_always_gets_the_same_gender_assignment():
    personas = [_persona("p-caller"), _persona("p-agent")]

    first = tts_engine_agent._assign_genders(personas)
    second = tts_engine_agent._assign_genders(personas)

    assert first == second


def test_starting_gender_varies_across_different_persona_sets():
    # Not every call should start with the same gender (e.g. caller always
    # male) -- across many different persona-ID pairs, both orderings should
    # occur at least once.
    starting_genders = {
        tts_engine_agent._assign_genders([_persona(f"p-{i}-caller"), _persona(f"p-{i}-agent")])[f"p-{i}-caller"]
        for i in range(20)
    }

    assert starting_genders == {"male", "female"}


def test_two_speaker_call_gets_distinct_voices_end_to_end(monkeypatch):
    monkeypatch.setattr(tts_engine_agent, "_load_voice", lambda voice_name: _FakeVoice())
    turns = [
        ConversationTurn(turn_index=0, speaker_persona_id="p-caller", text="hi", intended_emotion="neutral"),
        ConversationTurn(turn_index=1, speaker_persona_id="p-agent", text="hello", intended_emotion="neutral"),
    ]
    personas = [_persona("p-caller"), _persona("p-agent")]
    genders = tts_engine_agent._assign_genders(personas)
    expected_voice_by_persona = {
        p.persona_id: tts_engine_agent._select_voice_name(p.persona_id, genders[p.persona_id]) for p in personas
    }

    loaded_voice_names = []
    monkeypatch.setattr(
        tts_engine_agent, "_load_voice", lambda voice_name: (loaded_voice_names.append(voice_name), _FakeVoice())[1]
    )

    tts_engine_agent.synthesize_turns(turns, personas)

    assert loaded_voice_names == [expected_voice_by_persona["p-caller"], expected_voice_by_persona["p-agent"]]
    assert loaded_voice_names[0] != loaded_voice_names[1]
