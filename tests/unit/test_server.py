"""Unit tests for the MCP server query tool.

These tests exercise the `query` tool function in isolation by injecting
mock orchestrator/settings into the server module's global state. The
lifespan (which requires real database connections) is covered by the
e2e/integration suites.
"""

from typing import Any
from unittest.mock import AsyncMock

import pytest

import pg_mcp.server as server_module
from pg_mcp.config.settings import (
    OpenAIConfig,
    Settings,
    ValidationConfig,
)
from pg_mcp.models.query import QueryResponse, ValidationResult


class TestQueryToolNotInitialized:
    """Test behavior when the server is not initialized."""

    @pytest.mark.asyncio
    async def test_returns_server_not_initialized(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that queries fail gracefully when orchestrator is missing."""
        monkeypatch.setattr(server_module, "_orchestrator", None)

        result = await server_module.query(question="How many users?")

        assert result["success"] is False
        assert result["error"]["code"] == "SERVER_NOT_INITIALIZED"


class TestQueryToolParameterValidation:
    """Test parameter validation in the query tool."""

    @pytest.fixture
    def mock_orchestrator(self, monkeypatch: pytest.MonkeyPatch) -> AsyncMock:
        """Inject a mock orchestrator."""
        orchestrator = AsyncMock()
        monkeypatch.setattr(server_module, "_orchestrator", orchestrator)
        return orchestrator

    @pytest.mark.asyncio
    async def test_invalid_return_type(self, mock_orchestrator: AsyncMock) -> None:
        """Test that invalid return_type is rejected."""
        result = await server_module.query(question="Test", return_type="drop")

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_PARAMETER"
        mock_orchestrator.execute_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_question_too_long(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that questions exceeding the configured limit are rejected."""
        monkeypatch.setattr(server_module, "_orchestrator", AsyncMock())
        monkeypatch.setattr(
            server_module,
            "_settings",
            Settings(
                openai=OpenAIConfig(api_key="sk-test"),
                validation=ValidationConfig(max_question_length=10),
            ),
        )

        result = await server_module.query(question="x" * 11)

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_REQUEST"
        assert "maximum length" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_empty_question_rejected(self, mock_orchestrator: AsyncMock) -> None:
        """Test that whitespace-only questions are rejected by the model."""
        result = await server_module.query(question="   ")

        assert result["success"] is False
        assert result["error"]["code"] == "INVALID_REQUEST"
        mock_orchestrator.execute_query.assert_not_awaited()


class TestQueryToolExecution:
    """Test query tool execution paths."""

    @pytest.mark.asyncio
    async def test_success_response_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that a successful response has the documented structure."""
        orchestrator = AsyncMock()
        orchestrator.execute_query.return_value = QueryResponse(
            success=True,
            generated_sql="SELECT 1;",
            validation=ValidationResult(is_valid=True, is_select=True),
            data=None,
            error=None,
            confidence=100,
            tokens_used=None,
        )
        monkeypatch.setattr(server_module, "_orchestrator", orchestrator)

        result = await server_module.query(question="Test", return_type="sql")

        assert result["success"] is True
        assert result["generated_sql"] == "SELECT 1;"
        # to_dict normalizes tokens_used to 0 when None
        assert result["tokens_used"] == 0
        orchestrator.execute_query.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_error_response_structure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Test that error responses preserve the error structure."""
        orchestrator = AsyncMock()

        def _build_response(*args: Any, **kwargs: Any) -> QueryResponse:
            # Construct via model_construct to bypass the data/error
            # mutual-exclusion field validators for this test.
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error={"code": "security_violation", "message": "DELETE not allowed"},
                confidence=0,
                tokens_used=None,
            )

        orchestrator.execute_query.side_effect = _build_response
        monkeypatch.setattr(server_module, "_orchestrator", orchestrator)

        result = await server_module.query(question="Test")

        assert result["success"] is False
        assert result["error"]["code"] == "security_violation"
        assert result["tokens_used"] == 0

    @pytest.mark.asyncio
    async def test_orchestrator_exception_returns_internal_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Test that unexpected orchestrator exceptions are contained."""
        orchestrator = AsyncMock()
        orchestrator.execute_query.side_effect = RuntimeError("boom")
        monkeypatch.setattr(server_module, "_orchestrator", orchestrator)

        result = await server_module.query(question="Test")

        assert result["success"] is False
        assert result["error"]["code"] == "INTERNAL_ERROR"
        assert result["tokens_used"] == 0
