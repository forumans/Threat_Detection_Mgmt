"""
Threat Detection Agent
========================
Why this agent is needed:
    Explicit or implicit threats of physical harm are the most safety-critical
    category this platform detects -- these are the findings most likely to
    justify an immediate supervisor alert (see the Threat Correlation &
    Decision Agent). This agent is dedicated solely to that one category so
    its prompt can stay narrowly focused and its findings unambiguous.

What it does, step by step:
    1. Takes the call's merged/diarized transcript turns.
    2. Sends them to an LLM (via the shared _shared_detection helper) with a
       system prompt describing what counts as a threat of violence, including
       implicit/veiled threats, not just explicit statements.
    3. The LLM returns zero or more flagged turns, each with a severity,
       confidence, and short explanation.
    4. Each flagged turn is converted into an AgentFinding tagged with the
       threat_of_violence category, ready for the Transcript Correlation Agent.

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
    "You are a safety classifier for call-center transcripts. Read the transcript turns and "
    "identify every turn that contains a threat of physical harm -- explicit statements of "
    "intent to hurt someone, as well as implicit/veiled threats (e.g. 'you'll regret this', "
    "'I know where you live'). For each one, report the turn index, a severity "
    "(low/medium/high/critical) reflecting how credible and specific the threat is, your "
    "confidence (0-1), and a one-sentence explanation. If nothing qualifies, return no spans. "
    "Do not flag hyperbolic/figurative language with no genuine intent to harm (e.g. "
    "'this is killing me')."
)


def detect_threats(turns: list[TranscriptTurn]) -> list[AgentFinding]:
    """Flag transcript turns containing threats of violence."""
    return run_transcript_detection(
        turns,
        agent_name="threat",
        category=ThreatCategory.THREAT_OF_VIOLENCE,
        system_prompt=_SYSTEM_PROMPT,
    )
