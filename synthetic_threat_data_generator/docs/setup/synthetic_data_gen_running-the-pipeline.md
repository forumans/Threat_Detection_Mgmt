# Running the Pipeline

How to run the 11 agents together, now that they're wired into a graph (see
`src/orchestration/graph.py`).

## Run order

1. Configuration validates the batch request once (not per sample).
2. For each sample: Scenario Generator -> Persona Generator -> Conversation Generator, run in a strict chain.
3. Ground Truth Generator -> Transcript Generator -> Translation Engine -> TTS Engine -> Audio Generator, run in sequence.
4. Metadata Generator runs independently of step 3.
5. Dataset Exporter joins steps 3 and 4, validates everything, and writes the sample to disk.
6. Once every sample is done, Dataset Exporter writes the release manifest.

## Run it

```python
from src.orchestration.graph import generate_dataset

manifest = generate_dataset(
    category_distribution={"benign": 0.5, "verbal_abuse": 0.5},
    sample_count=10,
    locales=["en-US"],
    dataset_version="v1",
    output_dir="datasets/v1",
)

print(manifest.coverage_report)
```

Requires `OPENAI_API_KEY` (for the LLM-based agents) and Piper installed with
a voice model available (TTS Engine) -- see `.env.example`.

Output lands under `output_dir`: one folder per sample (`audio.wav`,
`transcript.json`, `ground_truth.json`, `metadata.json`), plus
`manifest.jsonl` and `coverage_report.json` for the release as a whole --
this is exactly what `threat_detection_system`'s evaluation harness will read
once that's built.
