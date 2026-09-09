"""
Transcript Generator Agent
=============================
Why this agent is needed:
    The Conversation Generator plans WHAT happens in a call; this agent
    decides HOW it's phrased -- the actual, verbatim words a speaker says. It's
    a rendering step, not a planning step, which keeps the "what happens"
    decision (including where threat indicators land) independent of wording,
    so it survives translation and audio rendering unchanged.

What it does, step by step:
    1. Describes each planned turn (speaker name/role, intended content,
       intended emotion) to the LLM, in order.
    2. Asks the LLM to write natural, conversational verbatim dialogue for
       every turn -- matching the intended content and emotion, not just
       summarizing it.
    3. Matches each returned line of dialogue back to its turn_index.
    4. Falls back to the turn's own `intended_content` as the rendered text
       for any turn_index the LLM's response happens to omit, so the output
       always has exactly one `ConversationTurn` per planned turn -- downstream
       agents (Ground Truth Generator, Dataset Exporter) depend on that
       one-to-one match.
    5. Returns the rendered `ConversationTurn` list (transcript.json).

Tools used:
    PydanticAI (structured LLM output, via llm_client.build_generation_agent,
    defaulting to OpenAI -- see src/config.py).

Other details:
    The architecture doc's §3.5 lists this agent's input as just the
    `Conversation` object; in practice it also needs the `Persona` list to
    resolve speaker names for natural-sounding dialogue (a `Conversation`
    turn only carries a `speaker_persona_id`, not a name) -- both are passed
    in below.
"""

from __future__ import annotations

from pydantic import BaseModel

from .. import llm_client
from ..schemas import Conversation, ConversationTurn, Persona

_TRANSCRIPT_SYSTEM_PROMPT = (
    "You write verbatim, natural-sounding dialogue for a synthetic call-center conversation "
    "benchmark. You're given an ordered list of turn plans (speaker name, intended content, "
    "intended emotion). For EACH turn, write the actual words that speaker would say -- "
    "matching the intended content and emotion, using natural conversational phrasing "
    "(contractions, the occasional filler word), not a stilted paraphrase. Return exactly one "
    "output turn per input turn plan, preserving turn_index exactly."
)


class _RenderedTurnOutput(BaseModel):
    turn_index: int
    text: str


class _TranscriptOutput(BaseModel):
    turns: list[_RenderedTurnOutput]


def _describe_turn_plans(conversation: Conversation, persona_by_id: dict[str, Persona]) -> str:
    """Render each turn's plan (speaker name, intended content/emotion) for the rendering prompt."""
    lines = []
    for turn in conversation.turns:
        persona = persona_by_id.get(turn.speaker_persona_id)
        name = persona.voice_traits.get("name", turn.speaker_persona_id) if persona else turn.speaker_persona_id
        lines.append(
            f"[turn {turn.turn_index}] {name}: intent='{turn.intended_content}', emotion={turn.intended_emotion}"
        )
    return "\n".join(lines)


def generate_transcript(conversation: Conversation, personas: list[Persona]) -> list[ConversationTurn]:
    """Render the canonical Conversation's turn plans into verbatim dialogue text."""
    persona_by_id = {p.persona_id: p for p in personas}

    # Step 1 + 2: ask the LLM to render every planned turn as verbatim dialogue.
    prompt = _describe_turn_plans(conversation, persona_by_id)
    transcript_agent = llm_client.build_generation_agent(_TranscriptOutput, _TRANSCRIPT_SYSTEM_PROMPT)
    rendered = transcript_agent.run_sync(prompt).output

    # Step 3: index the LLM's output by turn_index for lookup.
    text_by_turn_index = {t.turn_index: t.text for t in rendered.turns}

    # Step 4 + 5: build one ConversationTurn per planned turn, falling back to
    # the plan's intended_content if the LLM happened to skip a turn_index.
    return [
        ConversationTurn(
            turn_index=plan.turn_index,
            speaker_persona_id=plan.speaker_persona_id,
            text=text_by_turn_index.get(plan.turn_index, plan.intended_content),
            intended_emotion=plan.intended_emotion,
        )
        for plan in conversation.turns
    ]
