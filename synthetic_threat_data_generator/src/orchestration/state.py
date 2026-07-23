"""
Shared graph state for the per-sample generation pipeline (architecture doc §3).

A plain TypedDict, not a Pydantic model: LangGraph merges each node's returned
dict into this state key-by-key between steps, and a TypedDict is what its
StateGraph expects for that merge behavior. Every field here is written by
exactly one node (see graph.py) -- no two parallel nodes ever write the same
key in the same step, which is what lets the two post-Conversation-Generator
branches (audio-bound vs. ground-truth-bound) run concurrently without custom
reducers for conflicting writes.
"""

from __future__ import annotations

from typing import TypedDict

from ..agents.tts_engine_agent import TurnAudioClip
from ..schemas import (
    AudioFile,
    CallMetadata,
    Configuration,
    Conversation,
    ConversationTurn,
    GroundTruthLabel,
    Persona,
    SampleRef,
    Scenario,
)


class GenerationState(TypedDict, total=False):
    # Inputs
    configuration: Configuration
    sample_index: int
    output_dir: str

    # Configuration -> Scenario -> Persona -> Conversation
    scenario: Scenario
    personas: list[Persona]
    conversation: Conversation

    # Branch A: text -> translation -> speech -> mixed audio
    transcript: list[ConversationTurn]
    translated_transcript: list[ConversationTurn]
    audio_clips: list[TurnAudioClip]
    audio_file: AudioFile

    # Branch B: ground truth (language-independent) + metadata
    ground_truth: list[GroundTruthLabel]
    metadata: CallMetadata

    # Join: Audio Generator reconciles ground-truth timing once real audio exists
    reconciled_ground_truth: list[GroundTruthLabel]

    # Dataset Exporter's output
    sample_ref: SampleRef
