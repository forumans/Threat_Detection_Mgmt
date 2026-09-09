# Threat Detection System — Sequence Diagram

This diagram traces one call through `run_pipeline(call_id, audio_path)` in
[`src/orchestration/graph.py`](../../src/orchestration/graph.py), showing exactly which
agent function is invoked, in what order, and which calls run in parallel.

A high-resolution, infinitely-zoomable standalone SVG rendering of the same
diagram (generated via `mermaid-cli`) is available at
[`threat_detection_sequence-diagram.svg`](threat_detection_sequence-diagram.svg) —
open it directly in a browser and zoom to any level without quality loss.
The Mermaid source below renders natively in GitHub and VS Code.

## Diagram

```mermaid
sequenceDiagram
    autonumber
    participant Caller as Caller
    participant Graph as Orchestration Graph<br/>(run_pipeline)
    participant STT as Speech-to-Text<br/>Agent
    participant Diar as Speaker Diarization<br/>Agent
    participant Pros as Prosody Analysis<br/>Agent
    participant Emo as Emotion Detection<br/>Agent
    participant BG as Background Audio<br/>Detection Agent
    participant AudioCorr as Audio Correlation<br/>Agent
    participant VA as Verbal Abuse<br/>Detection Agent
    participant Thr as Threat Detection<br/>Agent
    participant Fraud as Fraud & Social<br/>Engineering Agent
    participant Comp as Compliance<br/>Detection Agent
    participant TransCorr as Transcript<br/>Correlation Agent
    participant Dec as Threat Correlation<br/>& Decision Agent

    Caller->>+Graph: run_pipeline(call_id, audio_path)

    rect rgb(235, 245, 255)
    Note over Graph,BG: Audio Intelligence domain
    par Speech-to-Text and Diarization run off the raw audio
        Graph->>+STT: transcribe(audio_input)
    and
        Graph->>+Diar: diarize(audio_input)
    end
    Diar-->>-Graph: diarization segments
    par Prosody, Emotion, Background Audio all wait on diarization<br/>(equal graph depth avoids a LangGraph fan-in bug — see docs)
        Graph->>+Pros: analyze_prosody(audio_input, diarization)
        Pros-->>-Graph: prosody findings
    and
        Graph->>+Emo: detect_emotion(audio_input, diarization)
        Emo-->>-Graph: emotion findings
    and
        Graph->>+BG: detect_background_audio(audio_input)
        BG-->>-Graph: background findings
    end
    end

    rect rgb(255, 248, 235)
    Note over Graph,Comp: Transcript Intelligence domain
    STT-->>-Graph: transcript (word-level)
    Graph->>Graph: merge_transcript_turns(transcript, diarization)
    par Verbal Abuse, Threat, Fraud, Compliance all fan out<br/>off the merged, diarized transcript
        Graph->>+VA: detect_verbal_abuse(turns)
        VA-->>-Graph: verbal abuse findings
    and
        Graph->>+Thr: detect_threats(turns)
        Thr-->>-Graph: threat findings
    and
        Graph->>+Fraud: detect_fraud_and_social_engineering(turns)
        Fraud-->>-Graph: fraud findings
    and
        Graph->>+Comp: detect_compliance_violations(turns)
        Comp-->>-Graph: compliance findings
    end
    end

    rect rgb(238, 255, 238)
    Note over Graph,Dec: Enterprise Decision domain
    Graph->>+AudioCorr: correlate_audio_findings(prosody + emotion + background)
    AudioCorr-->>-Graph: audio_domain_score
    Graph->>+TransCorr: correlate_transcript_findings(verbal_abuse + threat + fraud + compliance)
    TransCorr-->>-Graph: transcript_domain_score
    Graph->>+Dec: make_threat_assessment(audio_domain_score, transcript_domain_score)
    Note right of Dec: defer=True — this is the one node that waits<br/>for every other pending task in the graph first,<br/>since its two inputs arrive at unequal depth
    Dec-->>-Graph: ThreatAssessment
    end

    Graph-->>-Caller: ThreatAssessment
```

## Reading notes

- **Three color-coded phases** mirror the three domains in the architecture doc:
  Audio Intelligence (blue), Transcript Intelligence (peach), and the Enterprise
  Decision layer (green).
- **`par ... and ... end` blocks** are genuine parallelism in the real
  `StateGraph` — not illustrative simplification. Prosody, Emotion, and
  Background Audio Detection are deliberately routed through Speaker
  Diarization (not `START` directly) so all three sit at equal graph depth;
  this sidesteps a LangGraph fan-in gotcha where a default join fires as soon
  as any one predecessor completes rather than waiting for all of them (see
  the root [`README.md`](../../README.md) for the full writeup).
- **`Decision` is the only `defer=True` node** in this graph — it is the single
  terminal barrier that waits for every other pending task, which is the one
  scenario `defer=True` is actually safe for.
- **`merge_transcript_turns`** is real orchestration logic (not a passthrough):
  it groups Speech-to-Text's word-level output into diarized turns using the
  Speaker Diarization Agent's segments, falling back to a single "unknown"
  speaker turn if diarization produced nothing.
