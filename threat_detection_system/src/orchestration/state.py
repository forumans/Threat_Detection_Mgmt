"""
Shared graph state for the threat detection pipeline (architecture doc §4.1: CallState).

A plain TypedDict, not a Pydantic model: LangGraph merges each node's returned
dict into this state key-by-key between steps, and a TypedDict is what its
StateGraph expects for that merge behavior. Every field here is written by
exactly one node (see graph.py) -- no two parallel nodes ever write the same
key in the same step, which is what lets the audio/transcript branches below
run concurrently without needing custom reducers for conflicting writes.
"""

from __future__ import annotations

from typing import TypedDict

from ..schemas import (
    AgentFinding,
    DiarizationSegment,
    DomainScore,
    ThreatAssessment,
    Transcript,
    TranscriptTurn,
)


class CallState(TypedDict, total=False):
    call_id: str
    audio_path: str

    # Audio Intelligence domain
    transcript: Transcript
    diarization: list[DiarizationSegment]
    transcript_turns: list[TranscriptTurn]  # merged/diarized transcript, see graph.py
    prosody_findings: list[AgentFinding]
    emotion_findings: list[AgentFinding]
    background_findings: list[AgentFinding]
    audio_domain_score: DomainScore

    # Transcript Intelligence domain
    verbal_abuse_findings: list[AgentFinding]
    threat_findings: list[AgentFinding]
    fraud_findings: list[AgentFinding]
    compliance_findings: list[AgentFinding]
    transcript_domain_score: DomainScore

    # Enterprise Decision domain
    final_assessment: ThreatAssessment
