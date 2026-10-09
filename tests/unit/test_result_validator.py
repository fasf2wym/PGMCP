"""Unit tests for the result validation service.

These tests mock the OpenAI client to verify response parsing, confidence
boundaries, and error handling without network access.
"""

import json
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import OpenAIConfig, ValidationConfig
from pg_mcp.models.errors import LLMError, LLMTimeoutError, LLMUnavailableError
from pg_mcp.services.result_validator import ResultValidator


class TestResultValidator:
    """Tests for ResultValidator."""

    def _make_validator(self, validation_config: ValidationConfig | None = None) -> ResultValidator:
        """Create a validator with a test OpenAI config."""
        return ResultValidator(
            openai_config=OpenAIConfig(api_key="sk-test"),
            validation_config=validation_config or ValidationConfig(),
        )

    @staticmethod
    def _mock_response(content: str | None) -> MagicMock:
        """Build a mock ChatCompletion response."""
        response = MagicMock()
        choice = MagicMock()
        choice.message.content = content
        response.choices = [choice]
        return response

    @staticmethod
    def _patch_create(
        validator: ResultValidator,
        response: Any = None,
        side_effect: Any = None,
    ) -> Any:
        """Patch the OpenAI chat completion call on the validator's client."""
        return patch.object(
            validator.client.chat.completions,
            "create",
            new=AsyncMock(return_value=response, side_effect=side_effect),
        )

    @pytest.mark.asyncio
    async def test_disabled_returns_high_confidence(self) -> None:
        """Test that disabled validation short-circuits with confidence 100."""
        validator = self._make_validator(ValidationConfig(enabled=False))

        result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

        assert result.confidence == 100
        assert result.is_acceptable is True

    @pytest.mark.asyncio
    async def test_valid_json_response(self) -> None:
        """Test parsing a well-formed LLM JSON response."""
        validator = self._make_validator()
        content = json.dumps({"confidence": 85, "explanation": "matches", "suggestion": None})

        with self._patch_create(validator, self._mock_response(content)):
            result = await validator.validate(
                question="Count users",
                sql="SELECT COUNT(*) FROM users",
                results=[{"count": 42}],
                row_count=1,
            )

        assert result.confidence == 85
        assert result.explanation == "matches"
        assert result.is_acceptable is True  # >= default threshold 70

    @pytest.mark.asyncio
    async def test_confidence_below_threshold_not_acceptable(self) -> None:
        """Test that low confidence is marked not acceptable."""
        validator = self._make_validator(ValidationConfig(confidence_threshold=70))
        content = json.dumps({"confidence": 40, "explanation": "mismatch"})

        with self._patch_create(validator, self._mock_response(content)):
            result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

        assert result.confidence == 40
        assert result.is_acceptable is False

    @pytest.mark.asyncio
    async def test_invalid_json_returns_moderate_confidence(self) -> None:
        """Test that unparseable LLM output yields moderate confidence."""
        validator = self._make_validator()

        with self._patch_create(validator, self._mock_response("not json at all")):
            result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

        assert result.confidence == 60
        assert result.is_acceptable is False
        assert "parsing failed" in result.explanation.lower()

    @pytest.mark.asyncio
    async def test_confidence_clamped_to_bounds(self) -> None:
        """Test that out-of-range confidence values are clamped to 0-100."""
        validator = self._make_validator()
        content = json.dumps({"confidence": 150, "explanation": "over"})

        with self._patch_create(validator, self._mock_response(content)):
            result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

        assert result.confidence == 100

    @pytest.mark.asyncio
    async def test_non_numeric_confidence_defaults(self) -> None:
        """Test that non-numeric confidence defaults to 50."""
        validator = self._make_validator()
        content = json.dumps({"confidence": "high", "explanation": "?"})

        with self._patch_create(validator, self._mock_response(content)):
            result = await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

        assert result.confidence == 50

    @pytest.mark.asyncio
    async def test_empty_choices_raises_llm_error(self) -> None:
        """Test that an empty choices list raises LLMError."""
        validator = self._make_validator()
        response = MagicMock()
        response.choices = []

        with self._patch_create(validator, response), pytest.raises(LLMError):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_none_content_raises_llm_error(self) -> None:
        """Test that None message content raises LLMError."""
        validator = self._make_validator()

        with (
            self._patch_create(validator, self._mock_response(None)),
            pytest.raises(LLMError),
        ):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_timeout_raises_llm_timeout(self) -> None:
        """Test that timeouts are converted to LLMTimeoutError."""
        validator = self._make_validator()

        with (
            self._patch_create(validator, side_effect=TimeoutError("timed out")),
            pytest.raises(LLMTimeoutError),
        ):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_auth_error_raises_llm_unavailable(self) -> None:
        """Test that authentication errors raise LLMUnavailableError."""
        validator = self._make_validator()

        with (
            self._patch_create(validator, side_effect=Exception("invalid api_key provided")),
            pytest.raises(LLMUnavailableError),
        ):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_generic_error_raises_llm_error(self) -> None:
        """Test that generic errors raise LLMError."""
        validator = self._make_validator()

        with (
            self._patch_create(validator, side_effect=Exception("connection reset")),
            pytest.raises(LLMError),
        ):
            await validator.validate(question="q", sql="SELECT 1", results=[], row_count=0)

    @pytest.mark.asyncio
    async def test_sample_rows_limits_payload(self) -> None:
        """Test that only sample_rows rows are sent to the LLM."""
        validator = self._make_validator(ValidationConfig(sample_rows=2))
        content = json.dumps({"confidence": 90, "explanation": "ok"})

        mock_create = AsyncMock(return_value=self._mock_response(content))
        with patch.object(validator.client.chat.completions, "create", new=mock_create):
            await validator.validate(
                question="q",
                sql="SELECT 1",
                results=[{"i": i} for i in range(10)],
                row_count=10,
            )

        # The user prompt should mention only the 2 sampled rows
        user_prompt = mock_create.call_args.kwargs["messages"][1]["content"]
        assert "showing 2 of 10 rows" in user_prompt


@pytest.mark.asyncio
async def test_non_standard_response_degrades_gracefully() -> None:
    """A non-ChatCompletion body (e.g. HTML from a wrong base_url) degrades to confidence 60."""
    validator = ResultValidator(
        openai_config=OpenAIConfig(api_key="sk-test"),
        validation_config=ValidationConfig(),
    )
    mock_response = MagicMock(content="not a completion object", spec=["content"])

    with patch.object(
        validator.client.chat.completions, "create", new=AsyncMock(return_value=mock_response)
    ):
        result = await validator.validate(
            question="Count users",
            sql="SELECT COUNT(*) FROM users",
            results=[{"count": 1}],
            row_count=1,
        )

    assert result.confidence == 60
    assert result.is_acceptable is False
    assert "non-standard" in result.explanation or "OPENAI_BASE_URL" in result.explanation
