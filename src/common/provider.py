"""Shared Provider enum for LLM providers."""

from enum import Enum


class Provider(Enum):
    """Enumeration of supported LLM providers."""

    OPENROUTER = "openrouter"
    OPENAI = "openai"
    GEMINI = "gemini"
