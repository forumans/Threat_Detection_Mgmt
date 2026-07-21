"""
Fraud & Social Engineering Agent
===================================
Why this agent is needed:
    Not every threat is confrontational -- scams, impersonation, and
    manipulation to extract money, credentials, or personal information are
    just as damaging and often far more subtle than verbal abuse or explicit
    threats. This agent specializes in recognizing manipulation tactics, which
    require different judgment than the other three transcript categories.

What it does, step by step:
    1. Takes the call's merged/diarized transcript turns.
    2. Sends them to an LLM (via the shared _shared_detection helper) with a
       system prompt describing common fraud/social-engineering patterns
       (urgency pressure, impersonating an authority, requesting credentials
       or payment via unusual channels, etc.).
    3. The LLM returns zero or more flagged turns, each with a severity,
       confidence, and short explanation.
    4. Each flagged turn is converted into an AgentFinding tagged with the
       fraud_social_engineering category, ready for the Transcript Correlation
       Agent.

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
    "You are a fraud classifier for call-center transcripts. Read the transcript turns and "
    "identify every turn that shows signs of fraud or social engineering: impersonating an "
    "authority (bank, government, tech support), creating false urgency or fear, requesting "
    "credentials/OTP codes/gift cards/wire transfers, or otherwise manipulating the other "
    "party into an action against their interest. For each one, report the turn index, a "
    "severity (low/medium/high/critical) reflecting how manipulative and consequential the "
    "tactic is, your confidence (0-1), and a one-sentence explanation. If nothing qualifies, "
    "return no spans. Do not flag legitimate business requests for identity verification made "
    "through normal, expected channels."
)


def detect_fraud_and_social_engineering(turns: list[TranscriptTurn]) -> list[AgentFinding]:
    """Flag transcript turns containing fraud or social-engineering tactics."""
    return run_transcript_detection(
        turns,
        agent_name="fraud_social_engineering",
        category=ThreatCategory.FRAUD_SOCIAL_ENGINEERING,
        system_prompt=_SYSTEM_PROMPT,
    )
