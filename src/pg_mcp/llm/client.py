"""Shared OpenAI client factory.

Both LLM-backed services (``SQLGenerator`` and ``ResultValidator``) obtain
their ``AsyncOpenAI`` client from here so client construction lives in one
place. Services that need different effective timeouts share one client and
pass a per-request ``timeout`` on each call.
"""

from openai import AsyncOpenAI

from pg_mcp.config.settings import OpenAIConfig


def create_openai_client(openai_config: OpenAIConfig, timeout: float) -> AsyncOpenAI:
    """Create an ``AsyncOpenAI`` client from configuration.

    Args:
        openai_config: OpenAI configuration providing the API key and base URL.
        timeout: Request timeout in seconds applied as the client default.

    Returns:
        AsyncOpenAI: Configured async client instance.

    Example:
        >>> client = create_openai_client(settings.openai, timeout=30.0)
    """
    return AsyncOpenAI(
        api_key=openai_config.api_key.get_secret_value(),
        base_url=openai_config.base_url,
        timeout=timeout,
    )
