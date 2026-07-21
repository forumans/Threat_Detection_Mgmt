"""
Threat Correlation & Decision Agent
======================================
Why this agent is needed:
    This is the single place in the whole platform where "what did the audio
    show" and "what did the transcript show" get combined into one verdict.
    Everything upstream -- 11 other agents across two domains -- exists to
    feed this final decision. Without it, a supervisor would have to manually
    reconcile two separate, potentially contradictory domain scores for every
    call.

What it does, step by step:
    1. Takes the audio DomainScore (from Audio Correlation) and the transcript
       DomainScore (from Transcript Correlation) for one call.
    2. Computes a deterministic 0-100 overall risk_score: the higher of the two
       domain scores, plus a corroboration bonus when BOTH domains show at
       least a moderate signal (two domains agreeing independently is a
       stronger signal than either alone -- see architecture doc §5).
    3. Applies a fixed risk_score threshold to decide alert_decision
       (True/False) -- whether this call should raise a supervisor alert.
    4. Asks an LLM (via PydanticAI, defaulting to OpenAI) to pick the single
       most representative final_category/final_severity for the call and
       write a one-paragraph decision_summary a supervisor can read at a
       glance, given both domains' scores and summaries as context.
    5. Returns a ThreatAssessment bundling all of the above plus both input
       DomainScores, for full traceability back to every contributing finding.

Tools used:
    PydanticAI (structured LLM output) backed by OpenAI by default for local
    development and testing (see src/config.py, src/llm_client.py), used only
    for the qualitative category/severity/summary -- the risk_score and
    alert_decision themselves are deterministic and reproducible.

Other details:
    The corroboration-bonus rule and the alert threshold are placeholders
    pending calibration against Project 1's labeled benchmark dataset (see
    docs/architecture/threat_detection_architecture-plan.md §5, Risk Scoring Model).
"""

from __future__ import annotations

from pydantic import BaseModel

from .. import llm_client
from ..schemas import DomainScore, Severity, ThreatAssessment, ThreatCategory

# Corroboration only counts when BOTH domains clear this bar -- a single-domain
# finding shouldn't get inflated by a near-zero score in the other domain.
_CORROBORATION_MIN_SCORE = 40.0
_CORROBORATION_WEIGHT = 0.2

# A call is escalated to a supervisor alert once its overall risk_score
# reaches this threshold.
_ALERT_THRESHOLD = 60.0

_DECISION_SYSTEM_PROMPT = (
    "You are the final decision-maker in a call-threat-detection system. You are given the "
    "audio-domain and transcript-domain analysis summaries and scores for one call. Pick the "
    "single threat category (verbal_abuse, threat_of_violence, fraud_social_engineering, "
    "compliance_violation) that best represents the dominant issue in this call, or none if "
    "there isn't one. Pick a single overall severity (low/medium/high/critical), or none if "
    "there isn't one. Then write a one-paragraph decision_summary a supervisor can read in a "
    "few seconds to understand what happened and why it matters."
)


class _DecisionOutput(BaseModel):
    final_category: ThreatCategory | None = None
    final_severity: Severity | None = None
    decision_summary: str


def _compute_risk_score(audio_score: float, transcript_score: float) -> float:
    """Deterministic 0-100 overall risk score -- see module docstring step 2."""
    base = max(audio_score, transcript_score)
    if audio_score >= _CORROBORATION_MIN_SCORE and transcript_score >= _CORROBORATION_MIN_SCORE:
        bonus = min(audio_score, transcript_score) * _CORROBORATION_WEIGHT
    else:
        bonus = 0.0
    return min(base + bonus, 100.0)


def make_threat_assessment(
    audio_domain_score: DomainScore, transcript_domain_score: DomainScore
) -> ThreatAssessment:
    """Combine both domains' scores into the final, call-level ThreatAssessment."""
    # Step 2: deterministic risk score.
    risk_score = _compute_risk_score(audio_domain_score.score, transcript_domain_score.score)

    # Step 3: threshold-based alert decision.
    alert_decision = risk_score >= _ALERT_THRESHOLD

    # Step 4: ask the LLM to pick a representative category/severity and write
    # the human-readable decision summary, given both domains as context.
    decision_agent = llm_client.build_detection_agent(_DecisionOutput, _DECISION_SYSTEM_PROMPT)
    prompt = (
        f"Audio domain -- score: {audio_domain_score.score}, summary: {audio_domain_score.summary}\n"
        f"Transcript domain -- score: {transcript_domain_score.score}, summary: {transcript_domain_score.summary}\n"
        f"Computed overall risk_score: {risk_score} (alert_decision={alert_decision})"
    )
    decision = decision_agent.run_sync(prompt).output

    # Step 5: package into the final, traceable ThreatAssessment.
    return ThreatAssessment(
        audio_domain_score=audio_domain_score,
        transcript_domain_score=transcript_domain_score,
        risk_score=risk_score,
        final_category=decision.final_category,
        final_severity=decision.final_severity,
        alert_decision=alert_decision,
        decision_summary=decision.decision_summary,
    )
