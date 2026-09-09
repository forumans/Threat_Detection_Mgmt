"""
Ground Truth Generator Agent
===============================
Why this agent is needed:
    This agent produces the one artifact the whole project exists to create:
    the labels Project 2's evaluation harness scores itself against. Getting
    it right -- and keeping it independent of wording and language -- matters
    more than any other single output in this pipeline.

What it does, step by step:
    1. Reads `Conversation.injected_threat_turn_indices` directly -- the exact
       turns the Conversation Generator already decided should carry the
       scenario's threat indicator. It does NOT look at rendered dialogue text
       at all: labels come from the conversation *plan*, not from re-parsing
       words a later stage wrote.
    2. For each injected turn, looks up that turn's plan to get the speaker
       and the emotion it was planned to convey.
    3. Derives `expected_prosody_notes` from the scenario's severity (a
       `critical` scenario implies clearly elevated pitch/pace; a `low` one
       implies only subtle tension) -- a hint for evaluating the Prosody
       Analysis Agent in Project 2.
    4. Leaves `span` as None (turn-level only -- this agent runs before any
       transcript text exists to have offsets into) and leaves
       `audio_start_ms`/`audio_end_ms` as None (filled in later by the Audio
       Generator, once mixing determines real timing).
    5. Returns one `GroundTruthLabel` per injected turn; a `benign` scenario
       (no injected turns) returns an empty list -- correct, since a negative
       control should have nothing flagged.

Tools used:
    None -- pure, deterministic derivation from data the Conversation
    Generator already produced. No LLM call: the whole point of generating
    ground truth this way is that it can't be second-guessed or drift from
    what was actually planned.

Other details:
    `sample_id` is the `Conversation.conversation_id` -- one conversation is
    one benchmark sample, so no separate sample-ID minting step exists in this
    pipeline. `expected_background_event` stays None for now; there's no
    background-event planning step yet to derive it from (see Open Questions
    in the architecture doc).
"""

from __future__ import annotations

from ..schemas import Conversation, ConversationTurnPlan, GroundTruthLabel, Scenario, Severity

# Placeholder severity -> prosody-hint mapping -- see module docstring "Other details".
_PROSODY_NOTES_BY_SEVERITY: dict[Severity, str] = {
    Severity.LOW: "subtle tension, slightly clipped speech",
    Severity.MEDIUM: "raised volume, noticeably faster pace",
    Severity.HIGH: "elevated pitch and volume, rapid pace",
    Severity.CRITICAL: "sharply elevated pitch and volume, escalating pace",
}


def _find_turn_plan(conversation: Conversation, turn_index: int) -> ConversationTurnPlan:
    """Look up one turn's plan by index, raising if the conversation has no such turn."""
    for turn in conversation.turns:
        if turn.turn_index == turn_index:
            return turn
    raise ValueError(f"injected_threat_turn_indices references missing turn_index {turn_index}")


def generate_ground_truth(conversation: Conversation, scenario: Scenario) -> list[GroundTruthLabel]:
    """Derive ground-truth labels directly from the conversation's injection plan."""
    sample_id = conversation.conversation_id  # one conversation == one benchmark sample

    labels: list[GroundTruthLabel] = []
    for turn_index in conversation.injected_threat_turn_indices:
        # Step 2: look up the planned turn for speaker + intended emotion.
        turn_plan = _find_turn_plan(conversation, turn_index)

        # Step 3: severity-derived prosody hint.
        prosody_notes = _PROSODY_NOTES_BY_SEVERITY.get(scenario.severity) if scenario.severity else None

        # Step 4 + 5: build the label, leaving span/audio timing for later stages.
        labels.append(
            GroundTruthLabel(
                sample_id=sample_id,
                turn_index=turn_index,
                span=None,
                category=scenario.category,
                severity=scenario.severity,
                speaker_persona_id=turn_plan.speaker_persona_id,
                audio_start_ms=None,
                audio_end_ms=None,
                expected_emotion=turn_plan.intended_emotion,
                expected_background_event=None,
                expected_prosody_notes=prosody_notes,
            )
        )

    return labels
