"""
Shared typed contracts used by every agent (see docs/architecture/synthetic_data_gen_architecture-plan.md §5).

Every agent accepts and returns one of the Pydantic models defined here, matching
the "typed contracts everywhere" convention shared with Project 2. Keeping them in
one flat module (rather than one file per model) means any agent file depends on
the full shared vocabulary with a single import, with no cross-imports between
agent modules.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


class Category(str, Enum):
    """The threat taxonomy, plus `benign` for negative-control samples. Must stay
    in sync with Project 2's ThreatCategory (verbal_abuse/threat_of_violence/
    fraud_social_engineering/compliance_violation) -- `benign` only exists here,
    since Project 2 never needs to classify "no threat" as its own category."""

    VERBAL_ABUSE = "verbal_abuse"
    THREAT_OF_VIOLENCE = "threat_of_violence"
    FRAUD_SOCIAL_ENGINEERING = "fraud_social_engineering"
    COMPLIANCE_VIOLATION = "compliance_violation"
    BENIGN = "benign"


class Severity(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ChannelQuality(str, Enum):
    CLEAN = "clean"
    DEGRADED = "degraded"


class Configuration(BaseModel):
    """A validated generation request -- output of the Configuration Agent."""

    config_id: str
    category_distribution: dict[str, float] = Field(
        description="Requested mix of taxonomy categories, including 'benign'. Must sum to ~1.0."
    )
    sample_count: int = Field(gt=0)
    locales: list[str] = Field(min_length=1)
    seed: int | None = None


class Scenario(BaseModel):
    """One sample's scenario spec -- output of the Scenario Generator."""

    scenario_id: str
    category: Category
    severity: Severity | None = Field(default=None, description="None for benign scenarios.")
    setting: str = Field(description='e.g. "customer support call", "collections call"')
    locale: str
    channel_quality: ChannelQuality
    num_speakers: int = Field(ge=2)


class Persona(BaseModel):
    """One speaker's persona -- output of the Persona Generator."""

    persona_id: str
    role: str = Field(description='e.g. "caller", "agent"')
    voice_traits: dict = Field(description="pitch_range, pace, accent, timbre")
    emotional_baseline: str


class ConversationTurnPlan(BaseModel):
    """
    One turn's plan, before it's phrased as dialogue: what this turn should
    communicate and how it should feel, not yet the verbatim words.
    """

    turn_index: int
    speaker_persona_id: str
    intended_content: str
    intended_emotion: str


class Conversation(BaseModel):
    """
    The canonical, language-independent conversation plan -- output of the
    Conversation Generator, and the single source of truth every downstream
    agent derives from (see architecture doc §3).
    """

    conversation_id: str
    scenario_id: str
    turns: list[ConversationTurnPlan]
    injected_threat_turn_indices: list[int] = Field(
        default_factory=list, description="Which turns carry the scenario's threat indicators."
    )


class ConversationTurn(BaseModel):
    """One turn's rendered, verbatim dialogue text -- output of the Transcript Generator."""

    turn_index: int
    speaker_persona_id: str
    text: str
    intended_emotion: str


class GroundTruthLabel(BaseModel):
    """
    One ground-truth label -- output of the Ground Truth Generator. This is the
    contract Project 2's evaluation harness reads directly, so it must stay in
    sync with `agent_findings`/`GroundTruthLabel` as documented in Project 2's
    docs/database/threat_detection_database-model.md.
    """

    sample_id: str
    turn_index: int
    span: tuple[int, int] | None = Field(
        default=None, description="Char offsets within the turn's rendered text; None if turn-level only."
    )
    category: Category
    severity: Severity | None = None
    speaker_persona_id: str
    audio_start_ms: int | None = Field(default=None, description="Populated by Audio Generator after mixing.")
    audio_end_ms: int | None = None
    expected_emotion: str | None = None
    expected_background_event: str | None = None
    expected_prosody_notes: str | None = None


class CallMetadata(BaseModel):
    """Call-level metadata -- output of the Metadata Generator."""

    call_id: str
    scenario_id: str
    start_timestamp: datetime
    duration_ms: int
    channel_info: dict
    participant_ids: list[str]
    locale: str


class AudioFile(BaseModel):
    """The final mixed call-audio track -- output of the Audio Generator."""

    sample_id: str
    path: str = Field(description="audio.wav location within the sample folder, see architecture doc §8.")
    duration_ms: int
    sample_rate_hz: int
    channel_quality: ChannelQuality


class SampleRef(BaseModel):
    """One sample's file paths, as recorded in the dataset manifest."""

    sample_id: str
    audio_path: str
    transcript_path: str
    ground_truth_path: str
    metadata_path: str


class DatasetManifest(BaseModel):
    """The versioned benchmark release manifest -- output of the Dataset Exporter."""

    dataset_version: str
    generated_at: datetime
    samples: list[SampleRef]
    coverage_report: dict
