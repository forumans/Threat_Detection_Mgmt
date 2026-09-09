# Synthetic Threat Data Generator — Sequence Diagram

This diagram traces one `generate.py` CLI run through `generate_dataset(...)` in
[`src/orchestration/graph.py`](../../src/orchestration/graph.py): building the
`Configuration` once, then running the per-sample graph (`generate_sample`) in a
loop, and finally writing the dataset manifest once.

A high-resolution, infinitely-zoomable standalone SVG rendering of the same
diagram (generated via `mermaid-cli`) is available at
[`synthetic_data_gen_sequence-diagram.svg`](synthetic_data_gen_sequence-diagram.svg) —
open it directly in a browser and zoom to any level without quality loss.
The Mermaid source below renders natively in GitHub and VS Code.

## Diagram

```mermaid
sequenceDiagram
    autonumber
    participant CLI as generate.py<br/>(CLI)
    participant ConfigAgent as Configuration<br/>Agent
    participant Graph as Per-Sample<br/>Orchestration Graph
    participant Scenario as Scenario<br/>Generator
    participant Persona as Persona<br/>Generator
    participant Conv as Conversation<br/>Generator
    participant GT as Ground Truth<br/>Generator
    participant Trans as Transcript<br/>Generator
    participant Translate as Translation<br/>Engine
    participant TTS as TTS Engine
    participant Audio as Audio<br/>Generator
    participant Meta as Metadata<br/>Generator
    participant Export as Dataset<br/>Exporter

    CLI->>+ConfigAgent: build_configuration(category_distribution, sample_count, locales, seed)
    ConfigAgent-->>-CLI: Configuration

    loop for sample_index in 0..sample_count-1
        CLI->>+Graph: generate_sample(configuration, sample_index)

        rect rgb(235, 245, 255)
        Note over Graph,Conv: Scenario to Persona to Conversation (strict chain)
        Graph->>+Scenario: generate_scenario(configuration, sample_index)
        Scenario-->>-Graph: Scenario
        Graph->>+Persona: generate_personas(scenario)
        Persona-->>-Graph: [Persona]
        Graph->>+Conv: generate_conversation(scenario, personas)
        Conv-->>-Graph: Conversation
        end

        rect rgb(255, 248, 235)
        Note over Graph,Meta: Ground Truth then Transcript then Translation then TTS then Audio<br/>run as one sequential chain, deliberately not parallel, see README<br/>for the LangGraph defer=True limitation this works around.<br/>Metadata runs independently alongside this chain.
        par Ground Truth to Audio chain runs alongside Metadata
            Graph->>+GT: generate_ground_truth(conversation, scenario)
            GT-->>-Graph: [GroundTruthLabel]
            Graph->>+Trans: generate_transcript(conversation, personas)
            Trans-->>-Graph: [ConversationTurn]
            Graph->>+Translate: translate_transcript(transcript, source_locale, target_locale)
            Translate-->>-Graph: [ConversationTurn] (translated)
            Graph->>+TTS: synthesize_turns(translated_turns, personas)
            Note right of TTS: gendered voice pools, alternating<br/>assignment per call (see docs)
            TTS-->>-Graph: [TurnAudioClip]
            Graph->>+Audio: generate_audio(clips, scenario, ground_truth)
            Audio-->>-Graph: AudioFile + reconciled ground truth
        and
            Graph->>+Meta: generate_metadata(conversation, scenario)
            Meta-->>-Graph: CallMetadata
        end
        end

        rect rgb(238, 255, 238)
        Graph->>+Export: export_sample(audio_file, transcript, ground_truth, metadata)
        Note right of Export: validates before writing,<br/>rejects inconsistent samples,<br/>moves not copies the staged audio
        Export-->>-Graph: SampleRef
        end

        Graph-->>-CLI: SampleRef, Scenario
        CLI->>CLI: on_sample_done(completed, total, scenario)
    end

    CLI->>+Export: write_manifest(dataset_version, sample_refs, scenarios)
    Export-->>-CLI: DatasetManifest
```

## Reading notes

- **`build_configuration` and `write_manifest` are not graph nodes.**
  `generate_dataset` calls `configuration_agent.build_configuration(...)` once
  before the loop and `dataset_exporter_agent.write_manifest(...)` once after
  it — only the per-sample work inside the loop runs through the LangGraph
  `StateGraph`.
- **The Ground Truth → Transcript → Translation → TTS → Audio chain is
  deliberately sequential**, not parallel, even though Ground Truth
  Generation's own logic only reads `conversation`/`scenario` and never touches
  transcript data. This was a fix for a LangGraph limitation: `defer=True` is
  documented as "defer until the run is about to end," which works for exactly
  one terminal barrier node (`export`) — using it on two chained joins (`audio`
  and `export`) let `export` fire before `audio` had actually run. See the
  root [`README.md`](../../README.md) for the full writeup.
- **`Metadata` is the one branch that genuinely runs in parallel** with that
  chain, since it depends only on `conversation`.
- **Gendered TTS voice assignment** (alternating male/female per call, seeded
  deterministically via `stable_seed`, never both speakers landing on
  indistinguishable voices) happens inside `synthesize_turns` — see
  [`src/agents/tts_engine_agent.py`](../../src/agents/tts_engine_agent.py).
- **`on_sample_done`** is the CLI's progress callback, invoked once per
  completed sample so `generate.py --count N` prints live progress instead of
  appearing to hang during a multi-sample run.
