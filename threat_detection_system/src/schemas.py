"""
Shared typed contracts used by every agent (see docs/architecture/threat_detection_architecture-plan.md).

Every agent accepts and returns one of the Pydantic models defined here, per the
"typed contracts everywhere" rule in docs/architecture/tech_stack_guidelines.md. Keeping these
in a single flat module (rather than one file per model) means any agent file can
depend on the full shared vocabulary with a single import, without a maze of
cross-imports between agent modules.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Domain(str, Enum):
    """Which of the two intelligence domains a finding/score belongs to."""

    AUDIO = "audio"
    TRANSCRIPT = "transcript"


class ThreatCategory(str, Enum):
    """The threat taxonomy. Must stay in sync with Project 1's synthetic benchmark labels."""

    VERBAL_ABUSE = "verbal_abuse"
    THREAT_OF_VIOLENCE = "threat_of_violence"
    FRAUD_SOCIAL_ENGINEERING = "fraud_social_engineering"
    COMPLIANCE_VIOLATION = "compliance_violation"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class AudioInput(BaseModel):
    """What every audio-domain agent takes as input: a reference to one call's audio."""

    call_id: str
    audio_path: str = Field(description="Local file path or object-store URI to the audio file.")


class STTWord(BaseModel):
    """A single word-level transcription result, as produced by the Speech-to-Text Agent."""

    start_ms: int
    end_ms: int
    text: str
    confidence: float = Field(ge=0.0, le=1.0)


class Transcript(BaseModel):
    """Full output of the Speech-to-Text Agent: word-level timings plus the joined text."""

    words: list[STTWord]
    full_text: str
    language: str | None = None


class DiarizationSegment(BaseModel):
    """A single "who spoke when" segment, as produced by the Speaker Diarization Agent."""

    speaker_label: str
    start_ms: int
    end_ms: int
    confidence: float = Field(ge=0.0, le=1.0)


class TranscriptTurn(BaseModel):
    """
    One turn of the merged/diarized transcript: a contiguous span of speech from a
    single speaker, with text. This is what the transcript-intelligence agents read.
    """

    turn_index: int
    speaker_label: str
    start_ms: int
    end_ms: int
    text: str


class Evidence(BaseModel):
    """
    Pointer back to the exact source material a finding is based on. Either a
    transcript location (turn_index + character offsets) or an audio location
    (millisecond offsets), depending on which kind of agent produced the finding.
    """

    turn_index: int | None = None
    char_start: int | None = None
    char_end: int | None = None
    start_ms: int | None = None
    end_ms: int | None = None


class AgentFinding(BaseModel):
    """
    The single, uniform output shape used by every detection/analysis agent
    (all audio-signal and transcript-detection agents alike). Correlation
    agents consume lists of these generically, regardless of which agent
    produced them.
    """

    agent_name: str
    domain: Domain
    category: ThreatCategory | None = Field(
        default=None,
        description="None for pure audio-signal agents (e.g. prosody) that don't classify into the threat taxonomy.",
    )
    severity: Severity | None = None
    confidence: float = Field(ge=0.0, le=1.0)
    evidence: Evidence
    summary: str


class DomainScore(BaseModel):
    """Output of a Correlation Agent: one domain's findings reduced to a single score."""

    domain: Domain
    score: float = Field(ge=0.0, le=100.0)
    summary: str
    contributing_findings: list[AgentFinding] = Field(default_factory=list)


class ThreatAssessment(BaseModel):
    """Output of the Threat Correlation & Decision Agent: the final, call-level verdict."""

    audio_domain_score: DomainScore
    transcript_domain_score: DomainScore
    risk_score: float = Field(ge=0.0, le=100.0)
    final_category: ThreatCategory | None = None
    final_severity: Severity | None = None
    alert_decision: bool
    decision_summary: str
