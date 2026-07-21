"""
Audio Correlation Agent
=========================
Why this agent is needed:
    Three separate audio-signal agents (Prosody, Emotion, Background Audio)
    each look at one narrow slice of the call's audio. None of them alone
    determines whether the call is a threat -- someone can sound angry
    (Emotion) about something completely benign, or a loud background
    (Background Audio) can just be a busy office. This agent is the first
    place those independent signals get combined into a single audio-domain
    verdict.

What it does, step by step:
    1. Takes the findings emitted by the Prosody, Emotion, and Background Audio
       agents for one call (Speech-to-Text and Diarization produce raw data,
       not severity-bearing findings, so they aren't scored here).
    2. Converts each finding's severity into a numeric weight and combines them
       into a single 0-100 audio domain score, via a deterministic
       weighted-max formula (see _score_findings for the exact rule).
    3. Asks the configured LLM for a short, human-readable summary of what
       drove the score, given the findings as context (via llm_client.generate,
       which defaults to OpenAI -- see src/config.py).
    4. Returns a DomainScore(domain="audio") carrying the score, the summary,
       and the full list of contributing findings for traceability.

Tools used:
    No ML model of its own -- this agent is a deterministic aggregator over
    other agents' outputs, plus one LLM call (via src/llm_client.py) purely for
    the natural-language summary. Keeping the score itself deterministic (not
    LLM-generated) makes it reproducible and easy to unit test.

Other details:
    The severity weights below are placeholders pending calibration against
    Project 1's labeled benchmark dataset (see
    docs/architecture/threat_detection_architecture-plan.md §5, Risk Scoring Model).
"""

from __future__ import annotations

from .. import llm_client
from ..schemas import AgentFinding, Domain, DomainScore, Severity

_SEVERITY_WEIGHT: dict[Severity, float] = {
    Severity.LOW: 25.0,
    Severity.MEDIUM: 50.0,
    Severity.HIGH: 75.0,
    Severity.CRITICAL: 100.0,
}

_SUMMARY_SYSTEM_PROMPT = (
    "You are summarizing audio-analysis findings from a call-threat-detection system for a "
    "human supervisor. Given a list of findings (each with an agent name, optional severity, "
    "and a short technical summary), write ONE concise sentence describing what stood out in "
    "the audio. If there are no notable findings, say the audio showed no notable signals."
)


def _score_findings(findings: list[AgentFinding]) -> float:
    """
    Deterministic 0-100 score: the single highest-severity finding dominates
    (a strong signal should not get diluted by averaging against several
    unremarkable ones), with a small bonus when more than one agent
    corroborates an elevated severity.
    """
    weights = [_SEVERITY_WEIGHT[f.severity] for f in findings if f.severity is not None]
    if not weights:
        return 0.0

    base_score = max(weights)
    corroboration_bonus = min(len(weights) - 1, 3) * 5.0  # up to +15 for multiple corroborating signals
    return min(base_score + corroboration_bonus, 100.0)


def correlate_audio_findings(findings: list[AgentFinding]) -> DomainScore:
    """Combine Prosody/Emotion/Background-Audio findings into one audio DomainScore."""
    # Step 1 (caller's responsibility): `findings` is assumed to already be
    # scoped to this one call's Prosody/Emotion/Background Audio findings.

    # Step 2: compute the deterministic score.
    score = _score_findings(findings)

    # Step 3: ask the LLM for a short natural-language summary of the findings.
    findings_text = (
        "\n".join(
            f"- agent={f.agent_name}, severity={f.severity.value if f.severity else 'none'}, "
            f"confidence={f.confidence}, summary={f.summary}"
            for f in findings
        )
        or "(no findings)"
    )
    summary = llm_client.generate(findings_text, system_prompt=_SUMMARY_SYSTEM_PROMPT)

    # Step 4: package into the shared DomainScore contract.
    return DomainScore(domain=Domain.AUDIO, score=score, summary=summary, contributing_findings=findings)
