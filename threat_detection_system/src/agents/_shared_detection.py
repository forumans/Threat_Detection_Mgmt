"""
Shared helper for the four transcript-detection agents (Verbal Abuse, Threat,
Fraud & Social Engineering, Compliance). Not itself one of the 12 agents --
the leading underscore marks it as internal plumbing.

All four agents share the exact same shape: given the call's transcript turns,
ask an LLM to find zero or more spans matching one specific threat category,
and return them as AgentFindings. The only things that differ between the four
are the category and the system prompt describing what to look for. Rather
than duplicating that plumbing four times, it lives here once; each agent file
is a short wrapper that supplies its own category + prompt.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from .. import llm_client
from ..schemas import AgentFinding, Domain, Evidence, Severity, ThreatCategory, TranscriptTurn


class _DetectedSpan(BaseModel):
    """One LLM-identified instance of the target category in the transcript."""

    turn_index: int = Field(description="Index of the transcript turn this finding is about.")
    severity: Severity
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = Field(description="One short sentence explaining why this turn was flagged.")


class _DetectionResult(BaseModel):
    """The full structured output the LLM must return: zero or more spans."""

    spans: list[_DetectedSpan] = Field(default_factory=list)


def _format_transcript(turns: list[TranscriptTurn]) -> str:
    """Render transcript turns as a numbered, speaker-labeled block for the prompt."""
    return "\n".join(f"[turn {t.turn_index}] {t.speaker_label}: {t.text}" for t in turns)


def run_transcript_detection(
    turns: list[TranscriptTurn],
    *,
    agent_name: str,
    category: ThreatCategory,
    system_prompt: str,
) -> list[AgentFinding]:
    """
    Run one category's LLM-based detection over a transcript and return the
    findings in our shared AgentFinding shape.

    Step 1: skip the LLM call entirely for an empty transcript.
    Step 2: build a PydanticAI agent (defaults to OpenAI, see src/config.py)
            constrained to return our _DetectionResult schema.
    Step 3: run it against the formatted transcript.
    Step 4: map each detected span into our shared AgentFinding shape.
    """
    # Step 1.
    if not turns:
        return []

    # Step 2.
    detection_agent = llm_client.build_detection_agent(_DetectionResult, system_prompt)

    # Step 3.
    result = detection_agent.run_sync(_format_transcript(turns))

    # Step 4.
    return [
        AgentFinding(
            agent_name=agent_name,
            domain=Domain.TRANSCRIPT,
            category=category,
            severity=span.severity,
            confidence=span.confidence,
            evidence=Evidence(turn_index=span.turn_index),
            summary=span.explanation,
        )
        for span in result.output.spans
    ]
