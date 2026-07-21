"""
Audio Generator Agent
========================
Why this agent is needed:
    The TTS Engine produces isolated per-turn clips; a real call is one
    continuous track with natural pacing and, for degraded-channel scenarios,
    background noise. This agent does that final mix -- and, just as
    importantly, it's the only place that knows each turn's *real* position
    in the finished audio, so it's responsible for reconciling that timing
    back into the ground truth labels the Ground Truth Generator produced
    without knowing it yet.

What it does, step by step:
    1. Converts each turn's raw samples into a pydub AudioSegment and
       concatenates them in turn order, inserting a short pause between turns
       so the result doesn't sound like clips glued together with no breath.
    2. Tracks each turn's real (start_ms, end_ms) as it builds the track --
       this is the authoritative timing no earlier stage could have known.
    3. For `degraded` channel-quality scenarios, overlays low-level noise and
       reduces overall gain slightly, simulating a lower-quality line.
    4. Exports the finished mix to `output_path` and returns it as an
       `AudioFile`.
    5. Reconciles the ground-truth labels: for every label whose turn_index
       matches a turn in this audio, fills in the real `audio_start_ms` /
       `audio_end_ms` from step 2 (returns new label objects -- the inputs are
       never mutated).

Tools used:
    pydub (see docs/architecture/synthetic_data_gen_architecture-plan.md §4 --
    Audio Processing) for concatenation, silence, gain, and noise overlay, and
    for the final WAV export. No LLM, no ML model -- purely deterministic
    signal assembly.

Other details:
    The inter-turn pause and noise/gain parameters below are simple
    placeholders, not perceptually tuned -- reasonable starting points, not a
    final answer (same caveat as the severity/channel-quality weightings in
    the Scenario Generator).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from pydub import AudioSegment

from ..schemas import AudioFile, ChannelQuality, GroundTruthLabel, Scenario
from .tts_engine_agent import TurnAudioClip

# Placeholder mixing parameters -- see module docstring "Other details".
_INTER_TURN_PAUSE_MS = 300
_DEGRADED_NOISE_AMPLITUDE = 0.01
_DEGRADED_GAIN_REDUCTION_DB = 3.0


def _clip_to_segment(clip: TurnAudioClip) -> AudioSegment:
    """Convert one TurnAudioClip's float32 samples into a pydub AudioSegment."""
    int16_samples = np.clip(clip.samples * 32767, -32768, 32767).astype(np.int16)
    return AudioSegment(data=int16_samples.tobytes(), sample_width=2, frame_rate=clip.sample_rate_hz, channels=1)


def _stitch_clips(clips: list[TurnAudioClip]) -> tuple[AudioSegment, dict[int, tuple[int, int]]]:
    """Step 1 + 2: concatenate every clip with pauses, tracking each turn's real timing."""
    track = _clip_to_segment(clips[0])
    timing: dict[int, tuple[int, int]] = {clips[0].turn_index: (0, len(track))}

    for clip in clips[1:]:
        track = track + AudioSegment.silent(duration=_INTER_TURN_PAUSE_MS, frame_rate=clip.sample_rate_hz)
        start_ms = len(track)
        track = track + _clip_to_segment(clip)
        timing[clip.turn_index] = (start_ms, len(track))

    return track, timing


def _apply_channel_degradation(track: AudioSegment) -> AudioSegment:
    """Step 3: simulate a lower-quality line -- background noise + reduced gain."""
    noise_samples = (np.random.default_rng(0).uniform(-1, 1, int(track.frame_count())) * 32767 * _DEGRADED_NOISE_AMPLITUDE)
    noise_segment = AudioSegment(
        data=noise_samples.astype(np.int16).tobytes(),
        sample_width=2,
        frame_rate=track.frame_rate,
        channels=1,
    )
    return track.overlay(noise_segment) - _DEGRADED_GAIN_REDUCTION_DB


def generate_audio(
    clips: list[TurnAudioClip],
    scenario: Scenario,
    sample_id: str,
    ground_truth_labels: list[GroundTruthLabel],
    output_path: str | Path,
) -> tuple[AudioFile, list[GroundTruthLabel]]:
    """Mix per-turn TTS clips into the final call audio and reconcile ground-truth timing."""
    # Steps 1 + 2: stitch clips, recording each turn's real position.
    track, timing = _stitch_clips(clips)

    # Step 3: degrade the channel if the scenario calls for it.
    if scenario.channel_quality == ChannelQuality.DEGRADED:
        track = _apply_channel_degradation(track)

    # Step 4: export and package the AudioFile.
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    track.export(str(output_path), format="wav")
    audio_file = AudioFile(
        sample_id=sample_id,
        path=str(output_path),
        duration_ms=len(track),
        sample_rate_hz=track.frame_rate,
        channel_quality=scenario.channel_quality,
    )

    # Step 5: reconcile ground-truth timing without mutating the inputs.
    reconciled_labels = [
        label.model_copy(update={"audio_start_ms": timing[label.turn_index][0], "audio_end_ms": timing[label.turn_index][1]})
        if label.turn_index in timing
        else label
        for label in ground_truth_labels
    ]

    return audio_file, reconciled_labels
