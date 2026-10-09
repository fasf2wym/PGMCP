"""OpenAI exception translation into the pg-mcp LLM error hierarchy.

Historically ``SQLGenerator`` and ``ResultValidator`` each classified OpenAI
errors by string matching on ``str(exc)``. This module owns the mapping in
one place: typed OpenAI exceptions are classified first (the production
path), and plain exceptions fall back to the historical message-substring
matching, which keeps the documented error semantics for compatible API
gateways that raise generic errors.
"""

from openai import (
    APIConnectionError,
    APITimeoutError,
    AuthenticationError,
    RateLimitError,
)

from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError

_AUTH_MESSAGE = "OpenAI API authentication failed - check API key"
_RATE_LIMIT_MESSAGE = "OpenAI API rate limit exceeded"


def translate_openai_error(exc: Exception, context_message: str) -> LLMError:
    """Translate an OpenAI SDK exception into the pg-mcp LLM error hierarchy.

    Args:
        exc: The exception raised around an OpenAI SDK call.
        context_message: Short context used to compose generic messages,
            e.g. ``"OpenAI API request"`` or ``"Result validation"``.

    Returns:
        LLMError: The matching LLM error subclass instance.

    Example:
        >>> try:
        ...     await client.chat.completions.create(...)
        ... except Exception as e:
        ...     raise translate_openai_error(e, "OpenAI API request") from e
    """
    # Typed OpenAI exceptions first. Note: APITimeoutError subclasses
    # APIConnectionError, so the timeout check must come first.
    if isinstance(exc, APITimeoutError):
        return LLMTimeoutError(f"{context_message} timed out", details={"error": str(exc)})
    if isinstance(exc, AuthenticationError):
        return LLMUnavailableError(_AUTH_MESSAGE, details={"error": str(exc)})
    if isinstance(exc, RateLimitError):
        return LLMUnavailableError(_RATE_LIMIT_MESSAGE, details={"error": str(exc)})
    if isinstance(exc, APIConnectionError):
        return LLMUnavailableError(f"{context_message} failed: {exc}", details={"error": str(exc)})

    # Fall back to the historical message-substring classification.
    error_msg = str(exc)
    lowered = error_msg.lower()
    if "authentication" in lowered or "api_key" in lowered:
        return LLMUnavailableError(_AUTH_MESSAGE, details={"error": error_msg})
    if "rate_limit" in lowered:
        return LLMUnavailableError(_RATE_LIMIT_MESSAGE, details={"error": error_msg})
    return LLMError(f"{context_message} failed: {error_msg}", details={"error": error_msg})
