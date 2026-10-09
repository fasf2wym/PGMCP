"""Unit tests for the shared LLM client factory and error translation."""

import httpx
import pytest
from openai import AsyncOpenAI

from pg_mcp.config.settings import OpenAIConfig
from pg_mcp.llm import create_openai_client, translate_openai_error
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError

_REQUEST = httpx.Request("POST", "https://api.openai.com/v1/chat/completions")

AUTH_MESSAGE = "OpenAI API authentication failed - check API key"
RATE_LIMIT_MESSAGE = "OpenAI API rate limit exceeded"


def _api_error_with_status(status_code: int, message: str) -> httpx.Response:
    return httpx.Response(status_code, request=_REQUEST)


class TestTranslateTypedErrors:
    """Typed OpenAI exceptions are classified before string fallback."""

    def test_api_timeout_error(self) -> None:
        from openai import APITimeoutError

        exc = APITimeoutError(request=_REQUEST)
        result = translate_openai_error(exc, "OpenAI API request")
        assert isinstance(result, LLMTimeoutError)
        assert "timed out" in str(result)
        assert result.details == {"error": str(exc)}

    def test_authentication_error(self) -> None:
        from openai import AuthenticationError

        exc = AuthenticationError(
            "Invalid API key", response=_api_error_with_status(401, "Invalid"), body=None
        )
        result = translate_openai_error(exc, "OpenAI API request")
        assert isinstance(result, LLMUnavailableError)
        assert str(result) == AUTH_MESSAGE
        assert result.details == {"error": str(exc)}

    def test_rate_limit_error(self) -> None:
        from openai import RateLimitError

        exc = RateLimitError(
            "Too many requests", response=_api_error_with_status(429, "Slow down"), body=None
        )
        result = translate_openai_error(exc, "Result validation")
        assert isinstance(result, LLMUnavailableError)
        assert str(result) == RATE_LIMIT_MESSAGE

    def test_api_connection_error(self) -> None:
        from openai import APIConnectionError

        exc = APIConnectionError(message="Connection reset by peer", request=_REQUEST)
        result = translate_openai_error(exc, "OpenAI API request")
        assert isinstance(result, LLMUnavailableError)
        assert str(result) == f"OpenAI API request failed: {exc}"

    def test_timeout_checked_before_connection(self) -> None:
        # APITimeoutError subclasses APIConnectionError; the timeout branch
        # must win so timeouts are not misclassified as connection failures.
        from openai import APITimeoutError

        exc = APITimeoutError(request=_REQUEST)
        result = translate_openai_error(exc, "OpenAI API request")
        assert isinstance(result, LLMTimeoutError)
        assert not isinstance(result, LLMUnavailableError)


class TestTranslateStringFallback:
    """Plain exceptions keep the historical message-substring semantics."""

    def test_authentication_substring(self) -> None:
        result = translate_openai_error(
            Exception("Authentication failed - invalid api_key"), "OpenAI API request"
        )
        assert isinstance(result, LLMUnavailableError)
        assert str(result) == AUTH_MESSAGE
        assert result.details == {"error": "Authentication failed - invalid api_key"}

    def test_rate_limit_substring(self) -> None:
        result = translate_openai_error(Exception("rate_limit exceeded"), "OpenAI API request")
        assert isinstance(result, LLMUnavailableError)
        assert str(result) == RATE_LIMIT_MESSAGE

    def test_generic_error_uses_context(self) -> None:
        result = translate_openai_error(Exception("Unknown error occurred"), "OpenAI API request")
        assert isinstance(result, LLMError)
        assert not isinstance(result, LLMUnavailableError)
        assert str(result) == "OpenAI API request failed: Unknown error occurred"
        assert result.details == {"error": "Unknown error occurred"}

    def test_connection_reset_is_generic_llm_error(self) -> None:
        # Historical behavior: connection errors raised as plain exceptions
        # fall through to LLMError (only typed APIConnectionError maps to
        # LLMUnavailableError).
        result = translate_openai_error(Exception("connection reset"), "Result validation")
        assert isinstance(result, LLMError)
        assert not isinstance(result, LLMUnavailableError)
        assert str(result) == "Result validation failed: connection reset"


class TestCreateOpenAIClient:
    """Smoke tests for the shared client factory."""

    def test_returns_async_openai_client(self) -> None:
        # Explicit base_url so the assertion is independent of the
        # OPENAI_BASE_URL environment variable.
        config = OpenAIConfig(
            api_key="sk-test-key-12345", base_url="https://api.example-gateway.com/v1"
        )
        client = create_openai_client(config, timeout=15.0)
        assert isinstance(client, AsyncOpenAI)
        assert "api.example-gateway.com" in str(client.base_url)
        assert client.timeout == 15.0

    def test_custom_base_url(self) -> None:
        config = OpenAIConfig(
            api_key="sk-test-key-12345", base_url="https://gateway.example.com/v1"
        )
        client = create_openai_client(config, timeout=5.0)
        assert "gateway.example.com" in str(client.base_url)
        assert client.timeout == 5.0


@pytest.mark.parametrize(
    ("exc_factory", "expected_type"),
    [
        (lambda: Exception("Authentication failed - invalid api_key"), LLMUnavailableError),
        (lambda: Exception("rate_limit exceeded"), LLMUnavailableError),
        (lambda: Exception("connection reset"), LLMError),
    ],
)
def test_fallback_exception_types(exc_factory, expected_type: type) -> None:
    result = translate_openai_error(exc_factory(), "OpenAI API request")
    assert isinstance(result, expected_type)
    assert isinstance(result, LLMError)
