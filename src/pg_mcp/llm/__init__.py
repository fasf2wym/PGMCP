"""Shared LLM infrastructure for OpenAI-backed services.

This package centralizes the pieces that ``SQLGenerator`` and
``ResultValidator`` previously duplicated: constructing the ``AsyncOpenAI``
client and translating OpenAI SDK exceptions into the pg-mcp error hierarchy.
"""

from pg_mcp.llm.client import create_openai_client
from pg_mcp.llm.errors import translate_openai_error

__all__ = ["create_openai_client", "translate_openai_error"]
