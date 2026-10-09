"""SQL generation service using OpenAI for natural language to SQL conversion.

This module provides the SQLGenerator class that uses OpenAI's LLM to convert
natural language questions into valid PostgreSQL SQL queries.
"""

import json
import re
from typing import TYPE_CHECKING

from openai import AsyncOpenAI

from pg_mcp.config.settings import OpenAIConfig
from pg_mcp.llm import create_openai_client, translate_openai_error
from pg_mcp.models.errors import LLMError, LLMTimeoutError
from pg_mcp.prompts.sql_generation import SQL_GENERATION_SYSTEM_PROMPT, build_user_prompt

if TYPE_CHECKING:
    from openai.types.chat import ChatCompletion

    from pg_mcp.models.schema import DatabaseSchema


class SQLGenerator:
    """SQL generator using OpenAI for natural language to SQL conversion.

    This class handles the interaction with OpenAI's API to generate SQL queries
    from natural language questions. It includes robust error handling, SQL extraction
    from various response formats, and support for retry scenarios with error feedback.

    Example:
        >>> config = OpenAIConfig(api_key="sk-...", model="gpt-4")
        >>> generator = SQLGenerator(config)
        >>> sql = await generator.generate(
        ...     question="How many users registered today?",
        ...     schema=db_schema
        ... )
    """

    def __init__(self, config: OpenAIConfig, client: AsyncOpenAI | None = None) -> None:
        """Initialize SQL generator with OpenAI configuration.

        Args:
            config: OpenAI configuration including API key and model settings.
            client: Optional pre-configured ``AsyncOpenAI`` client to share
                across services. A per-request timeout is applied on every
                call, so a shared client still honors ``config.timeout``.
                Defaults to a client built from ``config``.
        """
        self.config = config
        self.client = client if client is not None else create_openai_client(config, config.timeout)

    async def generate(
        self,
        question: str,
        schema: "DatabaseSchema",
        context: str | None = None,
        previous_attempt: str | None = None,
        error_feedback: str | None = None,
    ) -> str:
        """Generate SQL statement from natural language question.

        This method sends the question and database schema to OpenAI's API
        and extracts the generated SQL query from the response. It supports
        retry scenarios by accepting previous failed attempts and error feedback.

        Args:
            question: User's natural language question.
            schema: Database schema information for context.
            context: Optional additional context to guide generation.
            previous_attempt: Previously generated SQL that failed (for retry).
            error_feedback: Error message from previous attempt (for retry).

        Returns:
            str: Generated SQL query (without trailing semicolon).

        Raises:
            LLMError: If generation fails or response is invalid.
            LLMTimeoutError: If the API request times out.
            LLMUnavailableError: If the API is unavailable or authentication fails.

        Example:
            >>> # Initial generation
            >>> sql = await generator.generate(
            ...     question="Count all active users",
            ...     schema=db_schema
            ... )
            >>> # Retry with error feedback
            >>> sql = await generator.generate(
            ...     question="Count all active users",
            ...     schema=db_schema,
            ...     previous_attempt="SELECT COUNT(*) FROM user",
            ...     error_feedback='relation "user" does not exist'
            ... )
        """
        user_prompt = build_user_prompt(
            question=question,
            schema=schema,
            context=context,
            previous_attempt=previous_attempt,
            error_feedback=error_feedback,
        )

        try:
            response: ChatCompletion = await self.client.chat.completions.create(
                model=self.config.model,
                messages=[
                    {"role": "system", "content": SQL_GENERATION_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=self.config.temperature,
                max_tokens=self.config.max_tokens,
                response_format={"type": "json_object"},  # Structured SQL output
                timeout=self.config.timeout,
            )
        except TimeoutError as e:
            raise LLMTimeoutError(
                message=f"OpenAI API request timed out after {self.config.timeout}s",
                details={"timeout": self.config.timeout},
            ) from e
        except Exception as e:
            # Classify via the shared OpenAI error translation (typed
            # exceptions first, message-substring fallback).
            raise translate_openai_error(e, "OpenAI API request") from e

        # Extract SQL from response
        if not hasattr(response, "choices"):
            # Misconfigured base_url or incompatible gateway: the SDK hands
            # back the raw body (e.g. an HTML page as a plain string).
            raise LLMError(
                message=(
                    "OpenAI returned a non-standard response - "
                    "check that OPENAI_BASE_URL points to the API root (e.g. /v1)"
                ),
                details={"response_type": type(response).__name__},
            )

        if not response.choices:
            raise LLMError(
                message="OpenAI returned empty response",
                details={"response": response.model_dump()},
            )

        content = response.choices[0].message.content
        if not content:
            raise LLMError(
                message="OpenAI returned empty message content",
                details={"response": response.model_dump()},
            )

        sql = self._sql_from_json(content) or self._extract_sql(content)
        if not sql:
            raise LLMError(
                message="Failed to extract SQL from OpenAI response",
                details={"content": content},
            )

        return sql

    def _sql_from_json(self, content: str) -> str | None:
        """Extract the SQL query from a structured JSON response.

        The generation call requests ``response_format={"type": "json_object"}``,
        so well-behaved responses are JSON objects like ``{"sql": "..."}``.
        Anything else (plain SQL, markdown fences) falls back to
        :meth:`_extract_sql`.

        Args:
            content: Raw content from LLM response.

        Returns:
            str | None: The SQL query with the same trailing-semicolon
                normalization as ``_extract_sql``, or None if the content is
                not a JSON object holding a non-empty "sql" string.

        Example:
            >>> generator._sql_from_json('{"sql": "SELECT 1"}')
            'SELECT 1;'
            >>> generator._sql_from_json("SELECT 1;")  # Not JSON
            None
        """
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            return None

        if not isinstance(data, dict):
            return None

        sql = data.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            return None

        return sql.strip().rstrip(";") + ";"

    def _extract_sql(self, content: str) -> str | None:
        """Extract SQL query from LLM response content.

        This method implements a multi-strategy approach to extract SQL from
        various response formats that LLMs might generate:

        1. Try to match ```sql ... ``` code blocks (preferred format)
        2. Try to match generic ``` ... ``` code blocks
        3. Try to find SELECT/WITH statements in plain text
        4. Check if entire content looks like SQL

        Args:
            content: Raw content from LLM response.

        Returns:
            str | None: Extracted SQL query, or None if extraction fails.

        Example:
            >>> generator._extract_sql("```sql\\nSELECT 1;\\n```")
            'SELECT 1;'
            >>> generator._extract_sql("SELECT * FROM users;")
            'SELECT * FROM users;'
        """
        if not content:
            return None

        content = content.strip()

        # Strategy 1: Match ```sql ... ``` or ``` ... ``` code blocks
        code_block_pattern = r"```(?:sql)?\s*\n?(.*?)\n?```"
        matches: list[str] = re.findall(code_block_pattern, content, re.DOTALL | re.IGNORECASE)

        if matches:
            sql = matches[0].strip()
            # Remove trailing semicolon for consistency
            return sql.rstrip(";") + ";"

        # Strategy 2: Find SELECT/WITH statements in plain text
        sql_pattern = r"((?:WITH|SELECT)\s+.*?)(?:;|$)"
        matches = re.findall(sql_pattern, content, re.DOTALL | re.IGNORECASE)

        if matches:
            sql = matches[0].strip()
            return sql.rstrip(";") + ";"

        # Strategy 3: Check if entire content looks like SQL
        if content.upper().startswith(("SELECT", "WITH")):
            return content.rstrip(";") + ";"

        return None
