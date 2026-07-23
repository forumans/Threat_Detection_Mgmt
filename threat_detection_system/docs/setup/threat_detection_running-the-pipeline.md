# Running the Pipeline

How to run the 12 agents together, now that they're wired into a graph (see
`src/orchestration/graph.py`). For environment setup, see
`threat_detection_local-setup.md` first.

## Run order

1. Speech-to-Text, Speaker Diarization, Background Audio Detection -- run in parallel off the raw audio.
2. Prosody Analysis, Emotion Detection -- run in parallel once diarization is done.
3. Transcript is merged from Speech-to-Text + Speaker Diarization.
4. Verbal Abuse, Threat, Fraud & Social Engineering, Compliance detection -- run in parallel off that transcript.
5. Audio Correlation (joins step 1-2's findings) and Transcript Correlation (joins step 4's findings) -- run in parallel.
6. Threat Correlation & Decision -- joins both, produces the final result.

## Run it

```python
from src.orchestration.graph import run_pipeline

assessment = run_pipeline(call_id="call-1", audio_path="path/to/call.wav")

print(assessment.risk_score, assessment.alert_decision, assessment.decision_summary)
```

Requires `OPENAI_API_KEY` (for the LLM-based agents) and the audio-ML
dependencies installed with real model weights available (Speech-to-Text,
Speaker Diarization, Emotion Detection) -- see `.env.example`.

`run_pipeline` returns a `ThreatAssessment` (see `src/schemas.py`) with the
overall risk score, category/severity, alert decision, and both domain
scores -- each of which still carries every individual finding that fed it,
for traceability back to the specific agent and evidence behind the result.
