"""
Unit tests for the Audio Generator Agent.

Real pydub mixing on small synthetic clips -- no mocking needed, since this
agent has no external service dependency (unlike TTS Engine, it only
consumes already-synthesized TurnAudioClips).
"""

import numpy as np

from src.agents.audio_generator_agent import generate_audio
from src.agents.tts_engine_agent import TurnAudioClip
from src.schemas import ChannelQuality, GroundTruthLabel, Scenario, Category, Severity

_SR = 16000


def _clip(turn_index: int, speaker: str, seconds: float = 0.5) -> TurnAudioClip:
    samples = (np.sin(np.linspace(0, 440 * 2 * np.pi * seconds, int(_SR * seconds))) * 0.3).astype(np.float32)
    return TurnAudioClip(turn_index=turn_index, speaker_persona_id=speaker, samples=samples, sample_rate_hz=_SR)


def _scenario(channel_quality=ChannelQuality.CLEAN) -> Scenario:
    return Scenario(
        scenario_id="scenario-1",
        category=Category.THREAT_OF_VIOLENCE,
        severity=Severity.HIGH,
        setting="a call",
        locale="en-US",
        channel_quality=channel_quality,
        num_speakers=2,
    )


def test_exports_a_real_wav_file(tmp_path):
    clips = [_clip(0, "p-caller"), _clip(1, "p-agent")]
    output_path = tmp_path / "audio.wav"

    audio_file, _ = generate_audio(clips, _scenario(), sample_id="sample-1", ground_truth_labels=[], output_path=output_path)

    assert output_path.exists()
    assert output_path.stat().st_size > 0
    assert audio_file.path == str(output_path)
    assert audio_file.sample_rate_hz == _SR


def test_duration_accounts_for_clips_and_inter_turn_pauses(tmp_path):
    clips = [_clip(0, "p-caller", seconds=0.5), _clip(1, "p-agent", seconds=0.5)]

    audio_file, _ = generate_audio(
        clips, _scenario(), sample_id="sample-1", ground_truth_labels=[], output_path=tmp_path / "audio.wav"
    )

    # 2 clips of 500ms + 1 pause (300ms, see _INTER_TURN_PAUSE_MS) = ~1300ms.
    assert 1250 <= audio_file.duration_ms <= 1350


def test_ground_truth_timing_is_reconciled_from_real_stitched_positions(tmp_path):
    clips = [_clip(0, "p-caller", seconds=0.5), _clip(1, "p-agent", seconds=0.5)]
    label = GroundTruthLabel(
        sample_id="sample-1", turn_index=1, span=None, category=Category.THREAT_OF_VIOLENCE,
        severity=Severity.HIGH, speaker_persona_id="p-agent", audio_start_ms=None, audio_end_ms=None,
        expected_emotion="angry", expected_background_event=None, expected_prosody_notes=None,
    )

    _, reconciled = generate_audio(
        clips, _scenario(), sample_id="sample-1", ground_truth_labels=[label], output_path=tmp_path / "audio.wav"
    )

    assert reconciled[0].audio_start_ms is not None
    assert reconciled[0].audio_end_ms is not None
    # Turn 1 starts after turn 0 (~500ms) plus the inter-turn pause (~300ms).
    assert reconciled[0].audio_start_ms >= 750


def test_original_ground_truth_labels_are_not_mutated(tmp_path):
    clips = [_clip(0, "p-caller", seconds=0.2)]
    label = GroundTruthLabel(
        sample_id="sample-1", turn_index=0, span=None, category=Category.THREAT_OF_VIOLENCE,
        severity=Severity.HIGH, speaker_persona_id="p-caller", audio_start_ms=None, audio_end_ms=None,
        expected_emotion="angry", expected_background_event=None, expected_prosody_notes=None,
    )

    generate_audio(clips, _scenario(), sample_id="sample-1", ground_truth_labels=[label], output_path=tmp_path / "audio.wav")

    assert label.audio_start_ms is None  # the original object passed in is untouched


def test_degraded_channel_quality_is_reflected_in_output(tmp_path):
    clips = [_clip(0, "p-caller"), _clip(1, "p-agent")]

    audio_file, _ = generate_audio(
        clips, _scenario(channel_quality=ChannelQuality.DEGRADED), sample_id="sample-1",
        ground_truth_labels=[], output_path=tmp_path / "audio.wav",
    )

    assert audio_file.channel_quality == ChannelQuality.DEGRADED
