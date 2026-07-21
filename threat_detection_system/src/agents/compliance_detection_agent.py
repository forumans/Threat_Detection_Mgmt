"""
Compliance Detection Agent
=============================
Why this agent is needed:
    Not every issue this platform must catch is adversarial. Regulatory and
    policy violations -- exposing PII/PCI data insecurely, skipping a required
    disclosure, deviating from a mandated script -- create real legal and
    business risk even when no one on the call intends any harm. This agent
    covers that distinct, compliance-driven category.

What it does, step by step:
    1. Takes the call's merged/diarized transcript turns.
    2. Sends them to an LLM (via the shared _shared_detection helper) with a
       system prompt describing both flavors of compliance violation: a turn
       where something was said that shouldn't have been (e.g. reading a card
       number aloud), and a turn where something required was missing (e.g. no
       recording disclosure at call start).
    3. The LLM returns zero or more flagged turns, each with a severity,
       confidence, and short explanation.
    4. Each flagged turn is converted into an AgentFinding tagged with the
       compliance_violation category, ready for the Transcript Correlation
       Agent.

Tools used:
    PydanticAI (structured LLM output) backed by OpenAI by default for local
    development and testing (see src/config.py, src/llm_client.py). The actual
    LLM-calling plumbing lives in _shared_detection.py -- this file only
    supplies the category and the system prompt.

Other details:
    See docs/architecture/threat_detection_architecture-plan.md §3.2 for this agent's place
    in the Transcript Intelligence domain. The specific set of required
    disclosures/policies is organization-specific; the prompt below covers
    common, general-purpose cases and is expected to be extended per deployment
    (e.g. via retrieval over an org's policy documents -- see the Vector DB
    section of docs/database/threat_detection_database-model.md).
"""

from __future__ import annotations

from ..schemas import AgentFinding, ThreatCategory, TranscriptTurn
from ._shared_detection import run_transcript_detection

_SYSTEM_PROMPT = (
    "You are a compliance classifier for call-center transcripts. Read the transcript turns "
    "and identify every turn with a compliance violation: sensitive data (full card numbers, "
    "SSNs, passwords) spoken aloud insecurely, or the clear absence of a required disclosure "
    "at the point in the call where it should have occurred (e.g. a recording/monitoring "
    "notice near the start of the call). For each one, report the turn index, a severity "
    "(low/medium/high/critical) reflecting the regulatory/business risk, your confidence "
    "(0-1), and a one-sentence explanation. If nothing qualifies, return no spans."
)


def detect_compliance_violations(turns: list[TranscriptTurn]) -> list[AgentFinding]:
    """Flag transcript turns containing compliance violations."""
    return run_transcript_detection(
        turns,
        agent_name="compliance",
        category=ThreatCategory.COMPLIANCE_VIOLATION,
        system_prompt=_SYSTEM_PROMPT,
    )
