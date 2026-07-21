"""
Dataset Exporter Agent
=========================
Why this agent is needed:
    Every other agent produces one piece of a sample. This agent is where
    those pieces (audio, transcript, ground truth, metadata) either prove
    they're consistent with each other and get written out as a real sample,
    or get rejected before they can corrupt a benchmark release. It's also
    the only place that assembles many samples into one versioned release.

What it does, step by step:
    1. Cross-checks one sample's artifacts against each other: every ground
       truth label's turn_index must exist in the transcript, the speaker on
       that label must match the speaker of that transcript turn, and any
       audio timing on a label must fall within the actual audio's duration.
    2. Raises `SampleValidationError` (rather than writing anything) if any
       check fails -- callers are expected to quarantine that one sample and
       continue the batch, not let one bad sample abort the whole export.
    3. Reconciles `CallMetadata.duration_ms` (an estimate, per the Metadata
       Generator) with the real `AudioFile.duration_ms`, since this is the
       first point in the pipeline where both are available together.
    4. Writes audio.wav, transcript.json, ground_truth.json, and metadata.json
       into `<output_dir>/<sample_id>/`, per the layout in architecture doc §8.
    5. Separately, `write_manifest` aggregates every sample's `SampleRef` plus
       a coverage report (counts per category/severity) into manifest.jsonl +
       coverage_report.json for one dataset release.

Tools used:
    Standard library only (json, shutil, pathlib) -- this agent is pure I/O
    and validation, no LLM, no ML model.

Other details:
    `export_sample` never partially writes a sample: validation happens
    before any file is touched, so a rejected sample leaves no directory
    behind to accidentally get picked up by a later manifest pass.
"""

from __future__ import annotations

import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from ..schemas import (
    AudioFile,
    CallMetadata,
    ConversationTurn,
    DatasetManifest,
    GroundTruthLabel,
    SampleRef,
    Scenario,
)


class SampleValidationError(ValueError):
    """Raised when a sample's artifacts are inconsistent with each other -- quarantine, don't export."""


def _validate_sample(
    transcript: list[ConversationTurn], ground_truth: list[GroundTruthLabel], audio_file: AudioFile
) -> None:
    """Step 1: cross-check every ground truth label against the transcript and audio."""
    transcript_by_turn = {t.turn_index: t for t in transcript}
    errors: list[str] = []

    for label in ground_truth:
        turn = transcript_by_turn.get(label.turn_index)
        if turn is None:
            errors.append(f"ground truth references missing turn_index {label.turn_index}")
            continue
        if turn.speaker_persona_id != label.speaker_persona_id:
            errors.append(
                f"turn {label.turn_index}: speaker mismatch "
                f"(transcript={turn.speaker_persona_id}, ground_truth={label.speaker_persona_id})"
            )
        if label.audio_end_ms is not None and label.audio_end_ms > audio_file.duration_ms:
            errors.append(
                f"turn {label.turn_index}: audio_end_ms {label.audio_end_ms} exceeds "
                f"audio duration {audio_file.duration_ms}"
            )

    # Step 2: reject the whole sample rather than writing something inconsistent.
    if errors:
        raise SampleValidationError("; ".join(errors))


def export_sample(
    sample_id: str,
    audio_file: AudioFile,
    transcript: list[ConversationTurn],
    ground_truth: list[GroundTruthLabel],
    metadata: CallMetadata,
    output_dir: str | Path,
) -> SampleRef:
    """Validate and write one sample's artifacts, returning its manifest entry."""
    # Step 1 + 2: validate before writing anything.
    _validate_sample(transcript, ground_truth, audio_file)

    # Step 3: reconcile the metadata's estimated duration with the real audio.
    metadata = metadata.model_copy(update={"duration_ms": audio_file.duration_ms})

    # Step 4: write every artifact into the sample's folder.
    sample_dir = Path(output_dir) / sample_id
    sample_dir.mkdir(parents=True, exist_ok=True)

    audio_dest = sample_dir / "audio.wav"
    shutil.copyfile(audio_file.path, audio_dest)

    # NOTE: encoding="utf-8" is explicit everywhere below -- Path.write_text()
    # defaults to the OS locale encoding on Windows (often cp1252), which would
    # silently corrupt non-ASCII dialogue (curly quotes, accented characters,
    # any non-English locale's translated text).
    transcript_path = sample_dir / "transcript.json"
    transcript_path.write_text(
        json.dumps([t.model_dump(mode="json") for t in transcript], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    ground_truth_path = sample_dir / "ground_truth.json"
    ground_truth_path.write_text(
        json.dumps([g.model_dump(mode="json") for g in ground_truth], indent=2, ensure_ascii=False), encoding="utf-8"
    )

    metadata_path = sample_dir / "metadata.json"
    metadata_path.write_text(metadata.model_dump_json(indent=2), encoding="utf-8")

    return SampleRef(
        sample_id=sample_id,
        audio_path=str(audio_dest),
        transcript_path=str(transcript_path),
        ground_truth_path=str(ground_truth_path),
        metadata_path=str(metadata_path),
    )


def _build_coverage_report(scenarios: list[Scenario]) -> dict:
    """Simple counts per category/severity -- see architecture doc §8."""
    category_counts: dict[str, int] = {}
    severity_counts: dict[str, int] = {}
    for scenario in scenarios:
        category_counts[scenario.category.value] = category_counts.get(scenario.category.value, 0) + 1
        if scenario.severity:
            severity_counts[scenario.severity.value] = severity_counts.get(scenario.severity.value, 0) + 1
    return {"total_samples": len(scenarios), "category_counts": category_counts, "severity_counts": severity_counts}


def write_manifest(
    dataset_version: str, sample_refs: list[SampleRef], scenarios: list[Scenario], output_dir: str | Path
) -> DatasetManifest:
    """Step 5: aggregate every exported sample into one versioned release manifest."""
    coverage_report = _build_coverage_report(scenarios)
    manifest = DatasetManifest(
        dataset_version=dataset_version,
        generated_at=datetime.now(timezone.utc),
        samples=sample_refs,
        coverage_report=coverage_report,
    )

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    manifest_path = output_dir / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as f:
        for ref in sample_refs:
            f.write(ref.model_dump_json() + "\n")

    coverage_path = output_dir / "coverage_report.json"
    coverage_path.write_text(json.dumps(coverage_report, indent=2), encoding="utf-8")

    return manifest
