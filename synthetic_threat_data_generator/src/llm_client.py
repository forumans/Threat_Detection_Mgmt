"""
LLM abstraction layer (architecture doc §6: the LLMClient interface).

Two access patterns, matching the two shapes of LLM usage across the agents
(see the "LLM Abstraction: LiteLLM" vs. "Agent Framework: PydanticAI" rows in
this project's own §4 tech stack table):

1. generate(prompt) -> str
   A minimal, provider-agnostic text-completion call, backed by LiteLLM. Used
   wherever an agent just needs unstructured text back (e.g. a short setting
   description) with no schema to validate against.

2. build_generation_agent(output_type, system_prompt) -> pydantic_ai.Agent
   A factory for structured-output agents, backed by PydanticAI. Used by every
   agent that must return a typed Pydantic model (or list of them) directly
   from the LLM's response, with schema validation built in -- Scenario Generator,
   Persona Generator, Conversation Generator, Transcript Generator, and
   (optionally) Translation Engine.

Both default to OpenAI (see config.py) for local development and testing.
Swapping providers later is a one-line change to Settings.llm_model /
llm_text_model -- no agent code changes, since no agent imports OpenAI (or any
other provider SDK) directly; they only ever go through this module.
"""

from __future__ import annotations

from typing import TypeVar

import litellm
from pydantic import BaseModel
from pydantic_ai import Agent

from .config import get_settings

T = TypeVar("T", bound=BaseModel)


def generate(prompt: str, *, system_prompt: str | None = None) -> str:
    """Plain text completion for agents that don't need structured output."""
    settings = get_settings()

    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    call_kwargs: dict[str, object] = {"model": settings.llm_text_model, "messages": messages}
    if settings.openai_api_key:
        call_kwargs["api_key"] = settings.openai_api_key

    response = litellm.completion(**call_kwargs)
    return response["choices"][0]["message"]["content"].strip()


def build_generation_agent(output_type: type[T], system_prompt: str) -> Agent:
    """
    Factory for a structured-output PydanticAI agent. Every LLM-based
    generation agent builds its Agent instance through this function, so the
    model choice and API key wiring live in exactly one place.
    """
    settings = get_settings()
    return Agent(
        model=settings.llm_model,
        output_type=output_type,
        system_prompt=system_prompt,
    )
