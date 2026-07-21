"""
Unit tests for the Dataset Exporter Agent.

Real file I/O against tmp_path -- no mocking needed, since this agent has no
external service dependency.
"""

import json

import pytest

from src.agents.dataset_exporter_agent import SampleValidationError, export_sample, write_manifest
from src.schemas import (
    AudioFile,
    CallMetadata,
    Category,
    ChannelQuality,
    ConversationTurn,
    GroundTruthLabel,
    Scenario,
    Severity,
)


def _audio_file(tmp_path, duration_ms=2000) -> AudioFile:
    source = tmp_path / "source_audio.wav"
    source.write_bytes(b"RIFF....WAVEfmt ")  # placeholder bytes -- content doesn't matter for these tests
    return AudioFile(sample_id="sample-1", path=str(source), duration_ms=duration_ms, sample_rate_hz=16000, channel_quality=ChannelQuality.CLEAN)


def _transcript() -> list[ConversationTurn]:
    return [
        ConversationTurn(turn_index=0, speaker_persona_id="p-caller", text="hi", intended_emotion="neutral"),
        ConversationTurn(turn_index=1, speaker_persona_id="p-agent", text="hello", intended_emotion="neutral"),
    ]


def _ground_truth(**overrides) -> GroundTruthLabel:
    defaults = dict(
        sample_id="sample-1", turn_index=1, span=None, category=Category.THREAT_OF_VIOLENCE,
        severity=Severity.HIGH, speaker_persona_id="p-agent", audio_start_ms=500, audio_end_ms=1500,
        expected_emotion="angry", expected_background_event=None, expected_prosody_notes=None,
    )
    defaults.update(overrides)
    return GroundTruthLabel(**defaults)


def _metadata(duration_ms=999) -> CallMetadata:
    return CallMetadata(
        call_id="sample-1", scenario_id="scenario-1", start_timestamp="2024-01-01T00:00:00",
        duration_ms=duration_ms, channel_info={}, participant_ids=["p1", "p2"], locale="en-US",
    )


def test_valid_sample_is_written_with_all_four_files(tmp_path):
    ref = export_sample(
        "sample-1", _audio_file(tmp_path), _transcript(), [_ground_truth()], _metadata(), output_dir=tmp_path / "out"
    )

    assert (tmp_path / "out" / "sample-1" / "audio.wav").exists()
    assert (tmp_path / "out" / "sample-1" / "transcript.json").exists()
    assert (tmp_path / "out" / "sample-1" / "ground_truth.json").exists()
    assert (tmp_path / "out" / "sample-1" / "metadata.json").exists()
    assert ref.sample_id == "sample-1"


def test_metadata_duration_is_reconciled_with_real_audio_duration(tmp_path):
    export_sample(
        "sample-1", _audio_file(tmp_path, duration_ms=2500), _transcript(), [_ground_truth()],
        _metadata(duration_ms=999), output_dir=tmp_path / "out",
    )

    written = json.loads((tmp_path / "out" / "sample-1" / "metadata.json").read_text())
    assert written["duration_ms"] == 2500  # not the original estimate of 999


def test_ground_truth_referencing_missing_turn_is_rejected(tmp_path):
    bad_label = _ground_truth(turn_index=99)

    with pytest.raises(SampleValidationError, match="missing turn_index"):
        export_sample("sample-1", _audio_file(tmp_path), _transcript(), [bad_label], _metadata(), output_dir=tmp_path / "out")

    assert not (tmp_path / "out" / "sample-1").exists()  # nothing written on rejection


def test_speaker_mismatch_between_transcript_and_ground_truth_is_rejected(tmp_path):
    bad_label = _ground_truth(turn_index=1, speaker_persona_id="p-caller")  # turn 1 is actually p-agent

    with pytest.raises(SampleValidationError, match="speaker mismatch"):
        export_sample("sample-1", _audio_file(tmp_path), _transcript(), [bad_label], _metadata(), output_dir=tmp_path / "out")


def test_audio_end_ms_beyond_audio_duration_is_rejected(tmp_path):
    bad_label = _ground_truth(audio_end_ms=99999)

    with pytest.raises(SampleValidationError, match="exceeds"):
        export_sample(
            "sample-1", _audio_file(tmp_path, duration_ms=2000), _transcript(), [bad_label], _metadata(),
            output_dir=tmp_path / "out",
        )


def test_non_ascii_dialogue_round_trips_correctly(tmp_path):
    # Regression test: Path.write_text() defaults to the OS locale encoding on
    # Windows (often cp1252), which would silently mangle non-ASCII text like
    # curly quotes or accented characters unless encoding="utf-8" is explicit.
    transcript = [
        ConversationTurn(turn_index=0, speaker_persona_id="p-caller", text="There’s a problem, señor.", intended_emotion="neutral"),
        ConversationTurn(turn_index=1, speaker_persona_id="p-agent", text="hello", intended_emotion="neutral"),
    ]
    label = _ground_truth(turn_index=0, speaker_persona_id="p-caller")

    export_sample("sample-1", _audio_file(tmp_path), transcript, [label], _metadata(), output_dir=tmp_path / "out")

    written = json.loads((tmp_path / "out" / "sample-1" / "transcript.json").read_text(encoding="utf-8"))
    assert written[0]["text"] == "There’s a problem, señor."


def test_write_manifest_produces_manifest_and_coverage_report(tmp_path):
    ref = export_sample(
        "sample-1", _audio_file(tmp_path), _transcript(), [_ground_truth()], _metadata(), output_dir=tmp_path / "out"
    )
    scenarios = [
        Scenario(scenario_id="s1", category=Category.THREAT_OF_VIOLENCE, severity=Severity.HIGH, setting="x", locale="en-US", channel_quality=ChannelQuality.CLEAN, num_speakers=2),
        Scenario(scenario_id="s2", category=Category.BENIGN, severity=None, setting="y", locale="en-US", channel_quality=ChannelQuality.CLEAN, num_speakers=2),
    ]

    manifest = write_manifest("v1", [ref], scenarios, output_dir=tmp_path / "out")

    manifest_path = tmp_path / "out" / "manifest.jsonl"
    coverage_path = tmp_path / "out" / "coverage_report.json"
    assert manifest_path.exists()
    assert coverage_path.exists()
    assert manifest.coverage_report["total_samples"] == 2
    assert manifest.coverage_report["category_counts"]["threat_of_violence"] == 1
    assert manifest.coverage_report["category_counts"]["benign"] == 1
    assert "low" not in manifest.coverage_report["severity_counts"]
