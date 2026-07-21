"""
LLM abstraction layer (architecture doc §10: the LLMClient interface).

Two access patterns are provided, matching the two shapes of LLM usage across
the agents (see docs/architecture/tech_stack_guidelines.md: "LLM Abstraction: LiteLLM" vs.
"Agent Development: PydanticAI"):

1. generate(prompt) -> str
   A minimal, provider-agnostic text-completion call, backed by LiteLLM. Used by
   the two Correlation Agents, which only need a short natural-language summary
   of already-computed findings -- no structured output, no tool use, so the
   lightest possible interface is the right fit.

2. build_detection_agent(output_type, system_prompt) -> pydantic_ai.Agent
   A factory for structured-output agents, backed by PydanticAI. Used by every
   agent that must return a typed Pydantic model (AgentFinding, ThreatAssessment)
   directly from the LLM's response, with schema validation built in.

Both default to OpenAI (see config.py) for local development and testing.
Swapping providers later is a one-line change to Settings.llm_model /
llm_summary_model -- no agent code changes, since no agent imports OpenAI (or
any other provider SDK) directly; they only ever go through this module.
"""

from __future__ import annotations

from typing import TypeVar

import litellm
from pydantic import BaseModel
from pydantic_ai import Agent

from .config import get_settings

T = TypeVar("T", bound=BaseModel)


def generate(prompt: str, *, system_prompt: str | None = None) -> str:
    """Plain text completion. Used by the Correlation Agents for summary generation."""
    settings = get_settings()

    # Build the chat-style message list LiteLLM expects.
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})

    call_kwargs: dict[str, object] = {"model": settings.llm_summary_model, "messages": messages}
    if settings.openai_api_key:
        call_kwargs["api_key"] = settings.openai_api_key

    response = litellm.completion(**call_kwargs)
    return response["choices"][0]["message"]["content"].strip()


def build_detection_agent(output_type: type[T], system_prompt: str) -> Agent:
    """
    Factory for a structured-output PydanticAI agent. Every LLM-based detection
    agent (Verbal Abuse, Threat, Fraud & Social Engineering, Compliance, and the
    Decision agent) builds its Agent instance through this function, so the model
    choice and API key wiring live in exactly one place.
    """
    settings = get_settings()
    return Agent(
        model=settings.llm_model,
        output_type=output_type,
        system_prompt=system_prompt,
    )
