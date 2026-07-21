"""
Transcript Correlation Agent
==============================
Why this agent is needed:
    Four separate detection agents (Verbal Abuse, Threat, Fraud & Social
    Engineering, Compliance) each scan the transcript for one specific
    category. A single call can trigger more than one of them, or several
    lower-confidence findings that only matter in combination. This agent is
    the first place those independent transcript signals get combined into a
    single transcript-domain verdict, mirroring what the Audio Correlation
    Agent does for the audio side.

What it does, step by step:
    1. Takes the findings emitted by the four transcript-detection agents for
       one call.
    2. Converts each finding's severity into a numeric weight and combines them
       into a single 0-100 transcript domain score, via the same deterministic
       weighted-max formula used by the Audio Correlation Agent.
    3. Asks the configured LLM for a short, human-readable summary of what
       drove the score (via llm_client.generate, which defaults to OpenAI --
       see src/config.py).
    4. Returns a DomainScore(domain="transcript") carrying the score, the
       summary, and the full list of contributing findings for traceability.

Tools used:
    No ML model of its own -- this agent is a deterministic aggregator over
    other agents' outputs, plus one LLM call (via src/llm_client.py) purely for
    the natural-language summary. Keeping the score itself deterministic (not
    LLM-generated) makes it reproducible and easy to unit test.

Other details:
    The severity weights are shared with the Audio Correlation Agent's
    formula and are placeholders pending calibration against Project 1's
    labeled benchmark dataset (see
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
    "You are summarizing transcript-analysis findings from a call-threat-detection system for "
    "a human supervisor. Given a list of findings (each with an agent name, threat category, "
    "optional severity, and a short technical summary), write ONE concise sentence describing "
    "what stood out in the transcript. If there are no notable findings, say the transcript "
    "showed no notable signals."
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


def correlate_transcript_findings(findings: list[AgentFinding]) -> DomainScore:
    """Combine the four transcript-detection agents' findings into one DomainScore."""
    # Step 1 (caller's responsibility): `findings` is assumed to already be
    # scoped to this one call's transcript-detection findings.

    # Step 2: compute the deterministic score.
    score = _score_findings(findings)

    # Step 3: ask the LLM for a short natural-language summary of the findings.
    findings_text = (
        "\n".join(
            f"- agent={f.agent_name}, category={f.category.value if f.category else 'none'}, "
            f"severity={f.severity.value if f.severity else 'none'}, confidence={f.confidence}, "
            f"summary={f.summary}"
            for f in findings
        )
        or "(no findings)"
    )
    summary = llm_client.generate(findings_text, system_prompt=_SUMMARY_SYSTEM_PROMPT)

    # Step 4: package into the shared DomainScore contract.
    return DomainScore(domain=Domain.TRANSCRIPT, score=score, summary=summary, contributing_findings=findings)
