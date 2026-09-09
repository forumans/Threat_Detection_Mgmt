"""
Conversation Generator Agent
==============================
Why this agent is needed:
    This is where the scenario and personas become an actual call. Its output
    -- the canonical `Conversation` -- is the single source of truth the
    architecture doc calls for: every downstream agent (Transcript, Ground
    Truth, Metadata, and ultimately Audio) derives from this one object rather
    than from each other, so they can never drift out of sync.

What it does, step by step:
    1. Describes the scenario and personas (by index, role, and emotional
       baseline) to the LLM and asks it to plan -- not write verbatim dialogue
       for -- an 8-14 turn conversation: who speaks, what each turn should
       communicate, and how the speaker feels.
    2. For non-benign scenarios, asks the LLM to mark which specific turns
       should carry the scenario's threat indicator -- one subtle turn for
       `low` severity, several escalating turns for `critical`.
    3. Maps the LLM's speaker_index (an index into the personas list) back to
       real `persona_id`s, since roles like "caller" aren't unique in
       multi-party scenarios.
    4. Falls back to marking the middle turn as the injection point if a
       non-benign scenario somehow comes back with none marked, so this
       invariant never silently breaks the pipeline even on an off LLM response.
    5. Returns the canonical `Conversation` (turn plans + injection indices).

Tools used:
    PydanticAI (structured LLM output, via llm_client.build_generation_agent,
    defaulting to OpenAI -- see src/config.py). This is the single most
    LLM-dependent agent in the pipeline, since planning a coherent multi-turn
    conversation isn't something templates or randomization do well.

Other details:
    Threat indicators are injected at specific, controllable turns rather than
    spread uniformly -- this is what makes the span-level ground truth that
    the Ground Truth Generator produces next actually meaningful.
"""

from __future__ import annotations

import uuid

from pydantic import BaseModel, Field

from .. import llm_client
from ..schemas import Category, Conversation, ConversationTurnPlan, Persona, Scenario

_CONVERSATION_SYSTEM_PROMPT = (
    "You plan synthetic call-center conversations for a benchmark dataset. You do NOT write "
    "verbatim dialogue -- only the plan: who speaks, what each turn should communicate "
    "(one sentence, not exact words), and how the speaker feels. Produce 8 to 14 turns, "
    "numbered from 0, alternating speakers naturally (not necessarily strictly alternating). "
    "If the scenario's category is not 'benign', mark exactly the turns that should carry the "
    "threat indicator as is_threat_injection=true: use exactly 1 subtle/borderline turn for "
    "'low' severity, 1-2 clear turns for 'medium'/'high', and 2-3 escalating turns building to "
    "a clear climax for 'critical'. For a 'benign' scenario, no turn should be marked."
)


class _TurnPlanOutput(BaseModel):
    turn_index: int
    speaker_index: int = Field(description="Index into the provided personas list.")
    intended_content: str
    intended_emotion: str
    is_threat_injection: bool = False


class _ConversationPlanOutput(BaseModel):
    turns: list[_TurnPlanOutput]


def _describe_personas(personas: list[Persona]) -> str:
    """Render the personas as an indexed, LLM-readable list for the planning prompt."""
    lines = [
        f"{i}: role={p.role}, name={p.voice_traits.get('name', '?')}, baseline={p.emotional_baseline}"
        for i, p in enumerate(personas)
    ]
    return "\n".join(lines)


def generate_conversation(scenario: Scenario, personas: list[Persona]) -> Conversation:
    """Plan the canonical, language-independent conversation for one sample."""
    # Step 1 + 2: ask the LLM to plan the conversation, including injection points.
    severity_text = scenario.severity.value if scenario.severity else "n/a"
    prompt = (
        f"Scenario -- category: {scenario.category.value}, severity: {severity_text}, "
        f"setting: {scenario.setting}\n\nPersonas:\n{_describe_personas(personas)}"
    )
    planning_agent = llm_client.build_generation_agent(_ConversationPlanOutput, _CONVERSATION_SYSTEM_PROMPT)
    plan = planning_agent.run_sync(prompt).output

    # Step 3: map speaker_index -> persona_id. Clamp out-of-range indices
    # (via modulo) rather than raising, so one hallucinated index doesn't
    # crash an otherwise-usable conversation plan.
    turns = [
        ConversationTurnPlan(
            turn_index=t.turn_index,
            speaker_persona_id=personas[t.speaker_index % len(personas)].persona_id,
            intended_content=t.intended_content,
            intended_emotion=t.intended_emotion,
        )
        for t in plan.turns
    ]
    injected_indices = [t.turn_index for t in plan.turns if t.is_threat_injection]

    # Step 4: guarantee a non-benign scenario always has at least one injection
    # point, even if the LLM's response missed the instruction.
    if scenario.category != Category.BENIGN and not injected_indices and turns:
        injected_indices = [turns[len(turns) // 2].turn_index]

    # Step 5: package into the canonical Conversation.
    return Conversation(
        conversation_id=str(uuid.uuid4()),
        scenario_id=scenario.scenario_id,
        turns=turns,
        injected_threat_turn_indices=injected_indices,
    )
