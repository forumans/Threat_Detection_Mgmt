"""
Translation Engine Agent
===========================
Why this agent is needed:
    A benchmark dataset that's only ever in one language doesn't exercise
    Project 2's multilingual robustness. This agent lets one canonical
    conversation produce samples in every locale `Configuration.locales`
    requested, without regenerating the conversation (and therefore the
    ground truth) from scratch per language.

What it does, step by step:
    1. If the target locale matches the source locale, returns the transcript
       unchanged -- translation is conditional, not mandatory for every
       sample.
    2. Otherwise, describes each source turn (speaker, text) to the LLM and
       asks for a natural translation into the target locale, preserving
       turn_index and speaker.
    3. Matches each returned translation back to its turn_index.
    4. Falls back to the untranslated source text for any turn_index the
       LLM's response happens to omit, so translation failures degrade
       gracefully (mixed-language transcript) rather than dropping a turn
       outright.
    5. Returns the target-locale `ConversationTurn` list.

Tools used:
    PydanticAI (structured LLM output, via llm_client.build_generation_agent,
    defaulting to OpenAI -- see src/config.py). The architecture doc's tech
    stack (§4) also lists a dedicated MT model (NLLB/MarianMT) as optional --
    not implemented here, but swappable behind this same function signature
    later if LLM-based translation proves too slow/expensive at scale.

Other details:
    Ground truth never needs re-deriving here: `GroundTruthLabel` is produced
    once, upstream, directly from the language-independent `Conversation`
    object (see Ground Truth Generator) -- it's correct for every locale this
    agent produces without any changes.
"""

from __future__ import annotations

from pydantic import BaseModel

from .. import llm_client
from ..schemas import ConversationTurn

_TRANSLATION_SYSTEM_PROMPT = (
    "You translate call-center dialogue for a synthetic benchmark dataset. You're given an "
    "ordered list of turns (speaker, source-language text) and a target locale. Translate "
    "each turn's text naturally into the target locale -- preserve the meaning and tone "
    "(including any tension/emotion in the wording), not just a literal word-for-word "
    "translation. Return exactly one output turn per input turn, preserving turn_index."
)


class _TranslatedTurnOutput(BaseModel):
    turn_index: int
    text: str


class _TranslationOutput(BaseModel):
    turns: list[_TranslatedTurnOutput]


def translate_transcript(
    turns: list[ConversationTurn], source_locale: str, target_locale: str
) -> list[ConversationTurn]:
    """Translate a rendered transcript into target_locale, or pass it through unchanged."""
    # Step 1: no-op when the scenario's locale already matches the target.
    if source_locale == target_locale:
        return turns

    # Step 2: ask the LLM to translate every turn.
    prompt_lines = [f"[turn {t.turn_index}] {t.speaker_persona_id}: {t.text}" for t in turns]
    prompt = f"Target locale: {target_locale}\n\n" + "\n".join(prompt_lines)
    translation_agent = llm_client.build_generation_agent(_TranslationOutput, _TRANSLATION_SYSTEM_PROMPT)
    translated = translation_agent.run_sync(prompt).output

    # Step 3: index the LLM's output by turn_index for lookup.
    text_by_turn_index = {t.turn_index: t.text for t in translated.turns}

    # Step 4 + 5: rebuild the turn list, falling back to the original text for
    # any turn the LLM's response happened to omit.
    return [
        ConversationTurn(
            turn_index=turn.turn_index,
            speaker_persona_id=turn.speaker_persona_id,
            text=text_by_turn_index.get(turn.turn_index, turn.text),
            intended_emotion=turn.intended_emotion,
        )
        for turn in turns
    ]
