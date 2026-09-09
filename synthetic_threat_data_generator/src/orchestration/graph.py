"""
Synthetic Dataset Generation Pipeline Graph
==============================================
Why this module is needed:
    Every agent in src/agents/ is independently callable and tested, but none
    of them know about each other -- something has to run them in the right
    order, fan work out into the two branches described in the architecture
    doc's pipeline diagram (§3), and join them at export. This module is that
    "something": it wires 10 of the 11 agents into a single LangGraph pipeline
    that produces one sample, plus a thin batch-level function that runs it
    once per requested sample and assembles the final release manifest.

What it does, step by step:
    1. Runs Scenario Generator -> Persona Generator -> Conversation Generator
       -> Ground Truth Generator -> Transcript Generator -> Translation
       Engine -> TTS Engine -> Audio Generator as one strict chain (see
       "Other details" for why Ground Truth Generator sits in this chain
       rather than running in true parallel with the transcript branch, as
       the architecture doc's diagram depicts).
    2. Runs Metadata Generator off the canonical Conversation, independently
       of that whole chain.
    3. Dataset Exporter joins the finished audio (end of the chain) with the
       metadata, validates everything is consistent, and writes the sample
       to disk.
    4. `generate_dataset` (not part of the graph) runs steps 1-3 once per
       requested sample, then calls Dataset Exporter's batch-level
       `write_manifest` once at the end -- see "Other details" for why
       Configuration and manifest-writing sit outside the per-sample graph.

Tools used:
    LangGraph (see docs/architecture/tech_stack_guidelines.md -- Agent
    Orchestration) for the per-sample graph. Every node is a thin wrapper
    calling exactly one agent function from src/agents/ -- this module
    contains no generation/validation logic of its own, only wiring.

Other details:
    Configuration Agent isn't a graph node: it validates a whole BATCH
    request once (see docs/architecture/synthetic_data_gen_architecture-plan.md
    §3.1), not once per sample, so re-running it inside a per-sample graph
    would be redundant. Similarly, Dataset Exporter's `write_manifest` (as
    opposed to its per-sample `export_sample`, which IS a graph node)
    aggregates every sample produced by a full `generate_dataset` run, so it
    can only happen after the per-sample graph has run to completion for
    every sample -- also outside the graph.

    Ground Truth Generator runs sequentially before Transcript Generator,
    not in parallel as the architecture doc's diagram shows it. This is a
    deliberate, LangGraph-specific adaptation, not a change in what Ground
    Truth Generator depends on: it still only reads `conversation`/`scenario`
    (see ground_truth_generator_agent.py), never the transcript text. Two
    things forced this: (1) LangGraph's default fan-in fires a join as soon
    as ANY ONE incoming edge completes, not once ALL of them have, so a join
    whose predecessors sit at different depths from `conversation` (the
    1-hop Ground Truth Generator vs. the 3-hop Transcript/Translation/TTS
    chain) fires prematurely; (2) the usual fix for that, `add_node(...,
    defer=True)`, is documented as deferring a node "until the run is about
    to end" -- it's built for exactly one true terminal barrier, not a chain
    of intermediate joins, and using it on both "audio" and "export" caused
    "export" to fire before "audio" had even run (reproduced in this
    project's LangGraph smoke-testing; see prompts/agent/ for that trail).
    Folding Ground Truth Generator into the sequential chain removes the
    join entirely, at the cost of a few milliseconds of otherwise-parallel
    work -- negligible next to the LLM/TTS calls elsewhere in the chain.
    `export` is the one remaining join (audio chain + Metadata Generator),
    and it IS the true terminal node here, so `defer=True` is the correct,
    well-supported tool for it.
"""

from __future__ import annotations

import tempfile
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

from langgraph.graph import END, START, StateGraph

from ..agents import (
    configuration_agent,
    conversation_generator_agent,
    dataset_exporter_agent,
    ground_truth_generator_agent,
    metadata_generator_agent,
    persona_generator_agent,
    scenario_generator_agent,
    transcript_generator_agent,
    translation_engine_agent,
    tts_engine_agent,
)
from ..agents import audio_generator_agent as audio_gen
from ..schemas import Configuration, DatasetManifest, SampleRef, Scenario
from .state import GenerationState

# The language Conversation Generator/Transcript Generator draft in; Translation
# Engine adapts from here to whatever locale Scenario Generator picked for this
# sample (a no-op when that locale is already this one -- see
# translation_engine_agent.py).
_CANONICAL_SOURCE_LOCALE = "en-US"


# --- Configuration -> Scenario -> Persona -> Conversation chain -------------


def _scenario_node(state: GenerationState) -> dict:
    """Graph node: run Scenario Generator for this sample's index within the batch."""
    scenario = scenario_generator_agent.generate_scenario(state["configuration"], sample_index=state["sample_index"])
    return {"scenario": scenario}


def _personas_node(state: GenerationState) -> dict:
    """Graph node: run Persona Generator for the scenario produced by _scenario_node."""
    return {"personas": persona_generator_agent.generate_personas(state["scenario"])}


def _conversation_node(state: GenerationState) -> dict:
    """Graph node: run Conversation Generator, producing the canonical per-sample Conversation."""
    conversation = conversation_generator_agent.generate_conversation(state["scenario"], state["personas"])
    return {"conversation": conversation}


# --- Sequential chain: ground truth -> text -> translation -> speech -> audio -


def _ground_truth_node(state: GenerationState) -> dict:
    """Graph node: run Ground Truth Generator directly off the canonical Conversation."""
    ground_truth = ground_truth_generator_agent.generate_ground_truth(state["conversation"], state["scenario"])
    return {"ground_truth": ground_truth}


def _transcript_node(state: GenerationState) -> dict:
    """Graph node: run Transcript Generator, rendering the conversation plan into dialogue text."""
    transcript = transcript_generator_agent.generate_transcript(state["conversation"], state["personas"])
    return {"transcript": transcript}


def _translation_node(state: GenerationState) -> dict:
    """Graph node: run Translation Engine from the canonical source locale to the scenario's locale."""
    translated = translation_engine_agent.translate_transcript(
        state["transcript"], source_locale=_CANONICAL_SOURCE_LOCALE, target_locale=state["scenario"].locale
    )
    return {"translated_transcript": translated}


def _tts_node(state: GenerationState) -> dict:
    """Graph node: run TTS Engine, synthesizing each (possibly translated) turn to speech."""
    clips = tts_engine_agent.synthesize_turns(state["translated_transcript"], state["personas"])
    return {"audio_clips": clips}


def _audio_node(state: GenerationState) -> dict:
    """Graph node: run Audio Generator, mixing the TTS clips and reconciling ground-truth timing."""
    sample_id = state["conversation"].conversation_id
    # A staging path for the mixed track -- export_sample() moves it into the
    # final samples/<sample_id>/audio.wav, so nothing lingers here afterward.
    staging_dir = Path(tempfile.mkdtemp(prefix="audio_gen_"))
    audio_file, reconciled = audio_gen.generate_audio(
        state["audio_clips"], state["scenario"], sample_id, state["ground_truth"], staging_dir / "audio.wav"
    )
    return {"audio_file": audio_file, "reconciled_ground_truth": reconciled}


# --- Independent branch: metadata --------------------------------------------


def _metadata_node(state: GenerationState) -> dict:
    """Graph node: run Metadata Generator, independently of the audio/ground-truth chain."""
    metadata = metadata_generator_agent.generate_metadata(state["conversation"], state["scenario"])
    return {"metadata": metadata}


def _export_node(state: GenerationState) -> dict:
    """Graph node: run Dataset Exporter, joining the finished audio with the metadata."""
    sample_id = state["conversation"].conversation_id
    sample_ref = dataset_exporter_agent.export_sample(
        sample_id,
        state["audio_file"],
        state["translated_transcript"],
        state["reconciled_ground_truth"],
        state["metadata"],
        state["output_dir"],
    )
    return {"sample_ref": sample_ref}


@lru_cache(maxsize=1)
def build_graph():
    """Build and compile the per-sample pipeline graph once; cached for reuse."""
    builder = StateGraph(GenerationState)

    builder.add_node("scenario", _scenario_node)
    builder.add_node("personas", _personas_node)
    builder.add_node("conversation", _conversation_node)
    builder.add_node("ground_truth", _ground_truth_node)
    builder.add_node("transcript", _transcript_node)
    builder.add_node("translation", _translation_node)
    builder.add_node("tts", _tts_node)
    builder.add_node("audio", _audio_node)
    builder.add_node("metadata", _metadata_node)
    # defer=True: see module docstring "Other details" -- "export" is the
    # one true terminal join in this graph (audio chain + Metadata
    # Generator), which is exactly what defer is designed for.
    builder.add_node("export", _export_node, defer=True)

    # Step 1: strict chain up to the canonical Conversation, then through
    # Ground Truth Generator, Transcript Generator, Translation Engine, TTS
    # Engine, and Audio Generator -- see module docstring "Other details" for
    # why Ground Truth Generator is sequential here rather than parallel.
    builder.add_edge(START, "scenario")
    builder.add_edge("scenario", "personas")
    builder.add_edge("personas", "conversation")
    builder.add_edge("conversation", "ground_truth")
    builder.add_edge("ground_truth", "transcript")
    builder.add_edge("transcript", "translation")
    builder.add_edge("translation", "tts")
    builder.add_edge("tts", "audio")

    # Step 2: Metadata Generator runs independently of that whole chain.
    builder.add_edge("conversation", "metadata")

    # Step 3: Dataset Exporter joins the finished audio with the metadata.
    builder.add_edge("audio", "export")
    builder.add_edge("metadata", "export")
    builder.add_edge("export", END)

    return builder.compile()


def generate_sample(configuration: Configuration, sample_index: int, output_dir: str | Path) -> tuple[SampleRef, Scenario]:
    """Run the per-sample graph once, returning its manifest entry and scenario."""
    graph = build_graph()
    final_state = graph.invoke(
        {"configuration": configuration, "sample_index": sample_index, "output_dir": str(output_dir)}
    )
    return final_state["sample_ref"], final_state["scenario"]


def generate_dataset(
    category_distribution: dict[str, float],
    sample_count: int,
    locales: list[str],
    dataset_version: str,
    output_dir: str | Path,
    seed: int | None = None,
    on_sample_done: Callable[[int, int, Scenario], None] | None = None,
) -> DatasetManifest:
    """
    Step 5: validate the batch request once (Configuration Agent), generate
    every requested sample through the per-sample graph, then aggregate them
    into one versioned release manifest (Dataset Exporter's write_manifest).

    `on_sample_done`, if given, is called after each sample finishes with
    (completed_count, total_count, scenario) -- each sample runs several LLM
    and TTS calls, so a multi-sample batch can take minutes with no other
    feedback otherwise. `generate.py` (the CLI) uses this to print progress.
    """
    configuration = configuration_agent.build_configuration(category_distribution, sample_count, locales, seed)

    sample_refs: list[SampleRef] = []
    scenarios: list[Scenario] = []
    for sample_index in range(configuration.sample_count):
        sample_ref, scenario = generate_sample(configuration, sample_index, output_dir)
        sample_refs.append(sample_ref)
        scenarios.append(scenario)
        if on_sample_done:
            on_sample_done(sample_index + 1, configuration.sample_count, scenario)

    return dataset_exporter_agent.write_manifest(dataset_version, sample_refs, scenarios, output_dir)
