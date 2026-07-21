"""
Verbal Abuse Detection Agent
==============================
Why this agent is needed:
    Verbal abuse -- insults, harassment, or degrading language directed at a
    person on the call -- is one of the four core threat categories this
    platform exists to catch. It's distinct from a threat of violence (see
    Threat Detection Agent): the caller isn't threatening harm, they're
    actively demeaning someone right now.

What it does, step by step:
    1. Takes the call's merged/diarized transcript turns.
    2. Sends them to an LLM (via the shared _shared_detection helper) with a
       system prompt describing exactly what counts as verbal abuse and what
       doesn't (e.g. ordinary complaints/frustration are not abuse).
    3. The LLM returns zero or more flagged turns, each with a severity,
       confidence, and short explanation.
    4. Each flagged turn is converted into an AgentFinding tagged with the
       verbal_abuse category, ready for the Transcript Correlation Agent.

Tools used:
    PydanticAI (structured LLM output) backed by OpenAI by default for local
    development and testing (see src/config.py, src/llm_client.py). The actual
    LLM-calling plumbing lives in _shared_detection.py -- this file only
    supplies the category and the system prompt.

Other details:
    See docs/architecture/threat_detection_architecture-plan.md §3.2 for this agent's place
    in the Transcript Intelligence domain.
"""

from __future__ import annotations

from ..schemas import AgentFinding, ThreatCategory, TranscriptTurn
from ._shared_detection import run_transcript_detection

_SYSTEM_PROMPT = (
    "You are a content-safety classifier for call-center transcripts. Read the transcript "
    "turns and identify every turn that contains verbal abuse: insults, harassment, or "
    "degrading language directed at a specific person on the call. For each one, report the "
    "turn index, a severity (low/medium/high/critical) reflecting how severe the abuse is, "
    "your confidence (0-1), and a one-sentence explanation. If nothing qualifies, return no "
    "spans. Do not flag ordinary frustration, complaints about a product or service, or "
    "profanity that is not directed at a person."
)


def detect_verbal_abuse(turns: list[TranscriptTurn]) -> list[AgentFinding]:
    """Flag transcript turns containing verbal abuse."""
    return run_transcript_detection(
        turns,
        agent_name="verbal_abuse",
        category=ThreatCategory.VERBAL_ABUSE,
        system_prompt=_SYSTEM_PROMPT,
    )
