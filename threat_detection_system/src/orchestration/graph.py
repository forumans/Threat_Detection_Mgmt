"""
Threat Detection Pipeline Graph
==================================
Why this module is needed:
    Every agent in src/agents/ is independently callable and tested, but none
    of them know about each other -- something has to run them in the right
    order, fan work out across the Audio/Transcript domains in parallel, and
    join their outputs at the right correlation/decision points. This module
    is that "something": it wires all 12 agents into the single LangGraph
    pipeline described in docs/architecture/threat_detection_architecture-plan.md §4.

What it does, step by step:
    1. Runs Speech-to-Text and Speaker Diarization in parallel off the raw
       audio -- neither depends on the other's output.
    2. Once diarization completes, runs Prosody Analysis, Emotion Detection,
       and Background Audio Detection in parallel. Prosody/Emotion use the
       diarization data (when present) to analyze per-speaker rather than the
       whole mixed track; Background Audio Detection doesn't use it at all,
       but still waits on this same step -- see "Other details" below for why.
    3. Merges the Speech-to-Text word list and the diarization segments into a
       diarized transcript (`TranscriptTurn` list) -- this exact merge step
       isn't one of the 12 named agents, but the architecture doc's graph
       diagram assumes it happens between Speech-to-Text/Diarization and the
       transcript-detection agents, so it lives here as orchestration glue.
    4. Runs Verbal Abuse, Threat, Fraud & Social Engineering, and Compliance
       detection in parallel off that diarized transcript.
    5. Joins the three audio-signal findings at Audio Correlation, and the
       four transcript findings at Transcript Correlation.
    6. Joins both domain scores at the Threat Correlation & Decision Agent,
       producing the final `ThreatAssessment`.

Tools used:
    LangGraph (see docs/architecture/tech_stack_guidelines.md -- Agent
    Orchestration) for the graph itself. Every node is a thin wrapper calling
    exactly one agent function from src/agents/ -- this module contains no
    detection/scoring logic of its own, only wiring and the transcript-merge
    step described above.

Other details:
    Every field in CallState (see state.py) is written by exactly one node,
    so parallel branches never conflict writing to the same key -- see
    state.py's docstring for why that matters to LangGraph.

    Background Audio Detection's graph edge comes from Speaker Diarization,
    not START, even though the agent itself never reads diarization data.
    That's deliberate: LangGraph's default join only waits for a node's
    direct predecessors to fire in the current step, not the full transitive
    closure, so a join whose predecessors sit at different depths from START
    (e.g. one 1 hop away, two others 2 hops away) fires prematurely -- and
    then fires again per remaining predecessor -- instead of running once
    with everyone's output present. Routing this edge through Speaker
    Diarization puts all three of Audio Correlation's predecessors at equal
    depth, which sidesteps that bug without the heavier `defer=True` fix
    (see the "decision" node below, where the same problem is unavoidable
    and defer=True is the right tool).
"""

from __future__ import annotations

from functools import lru_cache

from langgraph.graph import END, START, StateGraph

from ..agents import (
    audio_correlation_agent,
    background_audio_detection_agent,
    compliance_detection_agent,
    emotion_detection_agent,
    fraud_social_engineering_agent,
    prosody_analysis_agent,
    speaker_diarization_agent,
    speech_to_text_agent,
    threat_correlation_decision_agent,
    threat_detection_agent,
    transcript_correlation_agent,
    verbal_abuse_detection_agent,
)
from ..schemas import AudioInput, ThreatAssessment, Transcript, TranscriptTurn
from .state import CallState


def _merge_transcript_turns(transcript: Transcript, diarization: list) -> list[TranscriptTurn]:
    """
    Step 3: combine word-level STT output with diarization segments into one
    TranscriptTurn per diarized segment -- the words whose midpoint falls
    inside a segment become that turn's text. Falls back to a single
    "unknown speaker" turn covering the whole transcript if diarization
    produced no segments, so the transcript-detection agents always have at
    least one turn to analyze.
    """
    if not diarization:
        end_ms = transcript.words[-1].end_ms if transcript.words else 0
        return [TranscriptTurn(turn_index=0, speaker_label="unknown", start_ms=0, end_ms=end_ms, text=transcript.full_text)]

    turns: list[TranscriptTurn] = []
    for turn_index, segment in enumerate(sorted(diarization, key=lambda s: s.start_ms)):
        words_in_segment = [
            w for w in transcript.words if segment.start_ms <= (w.start_ms + w.end_ms) / 2 < segment.end_ms
        ]
        turns.append(
            TranscriptTurn(
                turn_index=turn_index,
                speaker_label=segment.speaker_label,
                start_ms=segment.start_ms,
                end_ms=segment.end_ms,
                text=" ".join(w.text for w in words_in_segment),
            )
        )
    return turns


# --- Audio Intelligence domain nodes -----------------------------------------


def _speech_to_text_node(state: CallState) -> dict:
    """Graph node: run the Speech-to-Text Agent on the call's raw audio."""
    audio_input = AudioInput(call_id=state["call_id"], audio_path=state["audio_path"])
    return {"transcript": speech_to_text_agent.transcribe(audio_input)}


def _speaker_diarization_node(state: CallState) -> dict:
    """Graph node: run the Speaker Diarization Agent on the call's raw audio."""
    audio_input = AudioInput(call_id=state["call_id"], audio_path=state["audio_path"])
    return {"diarization": speaker_diarization_agent.diarize(audio_input)}


def _background_audio_node(state: CallState) -> dict:
    """Graph node: run the Background Audio Detection Agent on the call's raw audio."""
    audio_input = AudioInput(call_id=state["call_id"], audio_path=state["audio_path"])
    return {"background_findings": background_audio_detection_agent.detect_background_audio(audio_input)}


def _prosody_node(state: CallState) -> dict:
    """Graph node: run the Prosody Analysis Agent, using diarization if available."""
    audio_input = AudioInput(call_id=state["call_id"], audio_path=state["audio_path"])
    findings = prosody_analysis_agent.analyze_prosody(audio_input, diarization=state.get("diarization"))
    return {"prosody_findings": findings}


def _emotion_node(state: CallState) -> dict:
    """Graph node: run the Emotion Detection Agent, using diarization if available."""
    audio_input = AudioInput(call_id=state["call_id"], audio_path=state["audio_path"])
    findings = emotion_detection_agent.detect_emotion(audio_input, diarization=state.get("diarization"))
    return {"emotion_findings": findings}


def _merge_transcript_node(state: CallState) -> dict:
    """Graph node: merge Speech-to-Text output with Speaker Diarization into a diarized transcript."""
    turns = _merge_transcript_turns(state["transcript"], state["diarization"])
    return {"transcript_turns": turns}


def _audio_correlation_node(state: CallState) -> dict:
    """Graph node: run the Audio Correlation Agent, joining Prosody/Emotion/Background Audio findings."""
    findings = [
        *state.get("prosody_findings", []),
        *state.get("emotion_findings", []),
        *state.get("background_findings", []),
    ]
    return {"audio_domain_score": audio_correlation_agent.correlate_audio_findings(findings)}


# --- Transcript Intelligence domain nodes ------------------------------------


def _verbal_abuse_node(state: CallState) -> dict:
    """Graph node: run the Verbal Abuse Detection Agent on the diarized transcript."""
    return {"verbal_abuse_findings": verbal_abuse_detection_agent.detect_verbal_abuse(state["transcript_turns"])}


def _threat_node(state: CallState) -> dict:
    """Graph node: run the Threat Detection Agent on the diarized transcript."""
    return {"threat_findings": threat_detection_agent.detect_threats(state["transcript_turns"])}


def _fraud_node(state: CallState) -> dict:
    """Graph node: run the Fraud & Social Engineering Agent on the diarized transcript."""
    findings = fraud_social_engineering_agent.detect_fraud_and_social_engineering(state["transcript_turns"])
    return {"fraud_findings": findings}


def _compliance_node(state: CallState) -> dict:
    """Graph node: run the Compliance Detection Agent on the diarized transcript."""
    findings = compliance_detection_agent.detect_compliance_violations(state["transcript_turns"])
    return {"compliance_findings": findings}


def _transcript_correlation_node(state: CallState) -> dict:
    """Graph node: run the Transcript Correlation Agent, joining all four transcript-detection findings."""
    findings = [
        *state.get("verbal_abuse_findings", []),
        *state.get("threat_findings", []),
        *state.get("fraud_findings", []),
        *state.get("compliance_findings", []),
    ]
    return {"transcript_domain_score": transcript_correlation_agent.correlate_transcript_findings(findings)}


# --- Enterprise Decision domain node -----------------------------------------


def _decision_node(state: CallState) -> dict:
    """Graph node: run the Threat Correlation & Decision Agent, joining both domain scores."""
    assessment = threat_correlation_decision_agent.make_threat_assessment(
        state["audio_domain_score"], state["transcript_domain_score"]
    )
    return {"final_assessment": assessment}


@lru_cache(maxsize=1)
def build_graph():
    """Build and compile the full pipeline graph once; cached for reuse across calls."""
    builder = StateGraph(CallState)

    builder.add_node("speech_to_text", _speech_to_text_node)
    builder.add_node("speaker_diarization", _speaker_diarization_node)
    builder.add_node("background_audio", _background_audio_node)
    builder.add_node("merge_transcript", _merge_transcript_node)
    builder.add_node("prosody", _prosody_node)
    builder.add_node("emotion", _emotion_node)
    builder.add_node("verbal_abuse", _verbal_abuse_node)
    builder.add_node("threat", _threat_node)
    builder.add_node("fraud_social_engineering", _fraud_node)
    builder.add_node("compliance", _compliance_node)
    builder.add_node("audio_correlation", _audio_correlation_node)
    builder.add_node("transcript_correlation", _transcript_correlation_node)
    # defer=True: LangGraph's default fan-in fires a node as soon as ANY of
    # its incoming edges completes, not once ALL of them have -- fine when a
    # join's predecessors sit at equal depth (every other join below), but
    # "decision" depends on audio_correlation (3 hops from START) and
    # transcript_correlation (4 hops, since the transcript branch has one
    # extra merge_transcript step) -- an irreducible depth mismatch. Without
    # defer, decision would run prematurely off audio_correlation alone and
    # crash on the still-missing transcript_domain_score (this actually
    # happened -- see test_orchestration_graph.py). defer=True instead waits
    # for every other pending task in the graph, which is exactly correct
    # here since nothing else runs after decision anyway.
    builder.add_node("decision", _decision_node, defer=True)

    # Step 1: fan out from the raw audio. background_audio is routed through
    # speaker_diarization (rather than directly off START) purely so it
    # reaches audio_correlation at the same depth as prosody/emotion --
    # avoiding the same premature-fan-in problem decision has, without
    # needing defer=True there too (which would otherwise force
    # audio_correlation to wait on the unrelated transcript branch as well).
    builder.add_edge(START, "speech_to_text")
    builder.add_edge(START, "speaker_diarization")

    # Step 2: prosody/emotion/background_audio all run once diarization is
    # available (background_audio ignores the diarization data itself --
    # see module docstring "Other details" above for why the edge still
    # routes through here).
    builder.add_edge("speaker_diarization", "prosody")
    builder.add_edge("speaker_diarization", "emotion")
    builder.add_edge("speaker_diarization", "background_audio")

    # Step 3: merge waits for both Speech-to-Text and Speaker Diarization
    # (both exactly 1 hop from START, so a plain join fires correctly here).
    builder.add_edge("speech_to_text", "merge_transcript")
    builder.add_edge("speaker_diarization", "merge_transcript")

    # Step 4: transcript-detection agents fan out off the merged transcript.
    builder.add_edge("merge_transcript", "verbal_abuse")
    builder.add_edge("merge_transcript", "threat")
    builder.add_edge("merge_transcript", "fraud_social_engineering")
    builder.add_edge("merge_transcript", "compliance")

    # Step 5: join at each domain's correlation agent -- every predecessor
    # of both joins below sits at equal depth, so no defer needed here.
    builder.add_edge("prosody", "audio_correlation")
    builder.add_edge("emotion", "audio_correlation")
    builder.add_edge("background_audio", "audio_correlation")

    builder.add_edge("verbal_abuse", "transcript_correlation")
    builder.add_edge("threat", "transcript_correlation")
    builder.add_edge("fraud_social_engineering", "transcript_correlation")
    builder.add_edge("compliance", "transcript_correlation")

    # Step 6: join at the final decision.
    builder.add_edge("audio_correlation", "decision")
    builder.add_edge("transcript_correlation", "decision")
    builder.add_edge("decision", END)

    return builder.compile()


def run_pipeline(call_id: str, audio_path: str) -> ThreatAssessment:
    """Run the full pipeline for one call's audio and return its ThreatAssessment."""
    graph = build_graph()
    final_state = graph.invoke({"call_id": call_id, "audio_path": audio_path})
    return final_state["final_assessment"]
