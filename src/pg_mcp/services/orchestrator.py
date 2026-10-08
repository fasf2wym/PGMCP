"""Query orchestrator for coordinating the complete query flow.

This module provides the QueryOrchestrator class that coordinates all components
of the query processing pipeline: SQL generation, validation, execution, and result
validation. It implements retry logic with error feedback and exponential backoff,
rate limiting, Prometheus metrics collection, request tracing, and multi-database
executor routing.
"""

import asyncio
import logging
import time
from typing import Any

from asyncpg import Pool

from pg_mcp.cache.schema_cache import SchemaCache
from pg_mcp.config.settings import ResilienceConfig, ValidationConfig
from pg_mcp.models.errors import (
    DatabaseError,
    ErrorCode,
    LLMError,
    PgMcpError,
    RateLimitExceededError,
    SchemaLoadError,
    SecurityViolationError,
    SQLParseError,
)
from pg_mcp.models.query import (
    ErrorDetail,
    QueryRequest,
    QueryResponse,
    QueryResult,
    ReturnType,
    ValidationResult,
)
from pg_mcp.observability.metrics import MetricsCollector
from pg_mcp.observability.tracing import request_context
from pg_mcp.resilience.circuit_breaker import CircuitBreaker
from pg_mcp.resilience.rate_limiter import MultiRateLimiter
from pg_mcp.services.result_validator import ResultValidator
from pg_mcp.services.sql_executor import SQLExecutor
from pg_mcp.services.sql_generator import SQLGenerator
from pg_mcp.services.sql_validator import SQLValidator

logger = logging.getLogger(__name__)


class QueryOrchestrator:
    """Orchestrates the complete query processing pipeline.

    This class coordinates SQL generation, validation, execution, and result
    validation across multiple databases. It implements retry logic with error
    feedback and exponential backoff, circuit breaker pattern for fault tolerance,
    rate limiting for both queries and LLM calls, Prometheus metrics, and
    request tracing via context variables.

    Example:
        >>> orchestrator = QueryOrchestrator(
        ...     sql_generator=generator,
        ...     sql_validator=validator,
        ...     sql_executors={"main": executor, "analytics": executor2},
        ...     result_validator=result_validator,
        ...     schema_cache=cache,
        ...     pools={"main": pool, "analytics": pool2},
        ...     resilience_config=resilience_config,
        ...     validation_config=validation_config,
        ... )
        >>> response = await orchestrator.execute_query(QueryRequest(
        ...     question="How many users?",
        ...     database="analytics"
        ... ))
    """

    def __init__(
        self,
        sql_generator: SQLGenerator,
        sql_validator: SQLValidator,
        sql_executors: dict[str, SQLExecutor],
        result_validator: ResultValidator,
        schema_cache: SchemaCache,
        pools: dict[str, Pool],
        resilience_config: ResilienceConfig,
        validation_config: ValidationConfig,
        rate_limiter: MultiRateLimiter | None = None,
        metrics: MetricsCollector | None = None,
    ) -> None:
        """Initialize query orchestrator.

        Args:
            sql_generator: SQL generation service.
            sql_validator: SQL validation service.
            sql_executors: SQL execution services keyed by database name.
            result_validator: Result validation service.
            schema_cache: Schema cache instance.
            pools: Dictionary mapping database names to connection pools.
            resilience_config: Resilience configuration for retries, backoff,
                circuit breaker, and rate limits.
            validation_config: Validation configuration including thresholds.
            rate_limiter: Optional rate limiter for query and LLM concurrency control.
            metrics: Optional Prometheus metrics collector.
        """
        self.sql_generator = sql_generator
        self.sql_validator = sql_validator
        self.sql_executors = sql_executors
        self.result_validator = result_validator
        self.schema_cache = schema_cache
        self.pools = pools
        self.resilience_config = resilience_config
        self.validation_config = validation_config
        self.rate_limiter = rate_limiter
        self.metrics = metrics

        # Create circuit breaker for LLM calls
        self.circuit_breaker = CircuitBreaker(
            failure_threshold=resilience_config.circuit_breaker_threshold,
            recovery_timeout=resilience_config.circuit_breaker_timeout,
        )

    async def execute_query(self, request: QueryRequest) -> QueryResponse:
        """Execute complete query flow from question to results.

        This method orchestrates the entire pipeline:
        1. Open a request tracing context (request_id propagated to all logs)
        2. Acquire a query rate-limit slot (reject with RATE_LIMIT_EXCEEDED if
           the wait times out)
        3. Resolve and validate database name
        4. Load schema from cache
        5. Generate and validate SQL with retry/backoff logic
        6. Execute SQL on the executor for the resolved database
        7. Validate results (optional, non-blocking)
        8. Record metrics and release the rate-limit slot

        Args:
            request: Query request containing question and parameters.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.

        Example:
            >>> response = await orchestrator.execute_query(
            ...     QueryRequest(question="Count all users", return_type="result")
            ... )
            >>> if response.success:
            ...     print(f"Found {response.data.row_count} rows")
        """
        async with request_context() as request_id:
            return await self._execute_with_rate_limit(request, request_id)

    async def _execute_with_rate_limit(
        self,
        request: QueryRequest,
        request_id: str,
    ) -> QueryResponse:
        """Execute the query pipeline under the query rate limiter.

        Args:
            request: Query request containing question and parameters.
            request_id: Request ID for tracing.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.
        """
        acquired = False
        if self.rate_limiter is not None:
            acquired = await self.rate_limiter.query_limiter.acquire(
                timeout=self.resilience_config.rate_limit_timeout
            )
            if not acquired:
                logger.warning(
                    "Query rejected: rate limit exceeded",
                    extra={"request_id": request_id},
                )
                return QueryResponse(
                    success=False,
                    generated_sql=None,
                    validation=None,
                    data=None,
                    error=ErrorDetail(
                        code=ErrorCode.RATE_LIMIT_EXCEEDED.value,
                        message="Too many concurrent queries, please retry later",
                        details={
                            "query_limit": self.rate_limiter.query_limiter.max_concurrent,
                        },
                    ),
                    confidence=0,
                    tokens_used=None,
                )

        start = time.monotonic()
        try:
            response = await self._process_query(request, request_id)
            if self.metrics is not None:
                status = (
                    "success"
                    if response.success
                    else str(response.error.code)
                    if response.error
                    else "error"
                )
                # Label by the requested database; auto-selected single
                # databases are attributed to "default".
                self.metrics.increment_query_request(
                    status=status, database=request.database or "default"
                )
                self.metrics.query_duration.observe(time.monotonic() - start)
            return response
        finally:
            if self.rate_limiter is not None and acquired:
                self.rate_limiter.query_limiter.release()

    async def _process_query(
        self,
        request: QueryRequest,
        request_id: str,
    ) -> QueryResponse:
        """Run the query pipeline steps after admission control.

        Args:
            request: Query request containing question and parameters.
            request_id: Request ID for tracing.

        Returns:
            QueryResponse: Complete response with SQL, results, or error information.
        """
        logger.info(
            "Starting query execution",
            extra={"request_id": request_id, "question": request.question[:100]},
        )

        try:
            # Step 1: Resolve database name
            database_name = self._resolve_database(request.database)
            logger.debug(
                "Resolved database",
                extra={"request_id": request_id, "database": database_name},
            )

            # Step 2: Get schema from cache
            schema = self.schema_cache.get(database_name)
            if schema is None:
                # Schema not in cache, load it
                pool = self.pools.get(database_name)
                if pool is None:
                    raise DatabaseError(
                        message=f"No connection pool available for database '{database_name}'",
                        details={"database": database_name},
                    )
                try:
                    schema = await self.schema_cache.load(database_name, pool)
                except Exception as e:
                    raise SchemaLoadError(
                        message=f"Failed to load schema for database '{database_name}': {e!s}",
                        details={"database": database_name, "error": str(e)},
                    ) from e

            logger.debug(
                "Schema loaded",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "tables": len(schema.tables),
                },
            )

            # Step 3: Generate and validate SQL with retry logic
            generated_sql, validation_result, tokens_used = await self._generate_sql_with_retry(
                question=request.question,
                schema=schema,
                request_id=request_id,
            )

            # Step 4: If return_type is SQL, return early
            if request.return_type == ReturnType.SQL:
                logger.info(
                    "Returning SQL only",
                    extra={"request_id": request_id, "sql_length": len(generated_sql)},
                )
                return QueryResponse(
                    success=True,
                    generated_sql=generated_sql,
                    validation=validation_result,
                    data=None,
                    error=None,
                    confidence=100,
                    tokens_used=tokens_used,
                )

            # Step 5: Execute SQL on the executor for the resolved database
            executor = self.sql_executors.get(database_name)
            if executor is None:
                raise DatabaseError(
                    message=f"No SQL executor available for database '{database_name}'",
                    details={"database": database_name},
                )

            logger.debug("Executing SQL", extra={"request_id": request_id})
            exec_start = time.monotonic()

            results, total_count = await executor.execute(generated_sql)

            execution_time_ms = (time.monotonic() - exec_start) * 1000.0
            if self.metrics is not None:
                self.metrics.observe_db_query_duration(execution_time_ms / 1000.0)

            logger.info(
                "SQL executed successfully",
                extra={
                    "request_id": request_id,
                    "database": database_name,
                    "row_count": total_count,
                    "execution_time_ms": execution_time_ms,
                },
            )

            # Step 6: Validate results (non-blocking, failures don't fail the request)
            result_confidence = await self._validate_results_safely(
                question=request.question,
                sql=generated_sql,
                results=results,
                row_count=total_count,
                request_id=request_id,
            )

            # Step 7: Build successful response
            query_result = QueryResult(
                columns=list(results[0].keys()) if results else [],
                rows=results,
                row_count=len(results),  # Limited row count (after max_rows applied)
                execution_time_ms=execution_time_ms,
            )

            return QueryResponse(
                success=True,
                generated_sql=generated_sql,
                validation=validation_result,
                data=query_result,
                error=None,
                confidence=result_confidence,
                tokens_used=tokens_used,
            )

        except PgMcpError as e:
            # Handle known application errors
            logger.warning(
                "Query execution failed with known error",
                extra={
                    "request_id": request_id,
                    "error_code": e.code,
                    "error_message": str(e),
                },
            )
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=e.code.value,
                    message=e.message,
                    details=e.details,
                ),
                confidence=0,
                tokens_used=None,
            )
        except Exception as e:
            # Handle unexpected errors
            logger.exception(
                "Query execution failed with unexpected error",
                extra={"request_id": request_id},
            )
            return QueryResponse(
                success=False,
                generated_sql=None,
                validation=None,
                data=None,
                error=ErrorDetail(
                    code=ErrorCode.INTERNAL_ERROR.value,
                    message=f"Internal server error: {e!s}",
                    details={"error_type": type(e).__name__},
                ),
                confidence=0,
                tokens_used=None,
            )

    def _resolve_database(self, database: str | None) -> str:
        """Resolve database name from request or auto-select.

        If database is specified, validate it exists.
        If not specified and only one database available, auto-select it.

        Args:
            database: Database name from request (optional).

        Returns:
            str: Resolved database name.

        Raises:
            DatabaseError: If database is invalid or cannot be auto-selected.

        Example:
            >>> name = orchestrator._resolve_database("mydb")  # Validates "mydb" exists
            >>> name = orchestrator._resolve_database(None)  # Auto-selects if only one DB
        """
        if database is not None:
            # Validate specified database exists
            if database not in self.pools:
                raise DatabaseError(
                    message=f"Database '{database}' not found",
                    details={
                        "requested_database": database,
                        "available_databases": list(self.pools.keys()),
                    },
                )
            return database

        # Auto-select if only one database available
        available_dbs = list(self.pools.keys())
        if len(available_dbs) == 0:
            raise DatabaseError(
                message="No databases configured",
                details={},
            )
        if len(available_dbs) == 1:
            return available_dbs[0]

        # Multiple databases, must specify
        raise DatabaseError(
            message="Multiple databases available, please specify which to query",
            details={"available_databases": available_dbs},
        )

    async def _generate_sql_with_retry(
        self,
        question: str,
        schema: Any,
        request_id: str,
    ) -> tuple[str, ValidationResult, int | None]:
        """Generate and validate SQL with retry logic on validation failures.

        This method implements a retry loop that:
        1. Checks circuit breaker state
        2. Generates SQL using LLM (under the LLM rate limiter)
        3. Validates the generated SQL
        4. On validation failure, waits with exponential backoff and retries
           with error feedback
        5. Records success/failure to circuit breaker and metrics

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            request_id: Request ID for tracking.

        Returns:
            tuple: (generated_sql, validation_result, tokens_used)

        Raises:
            LLMError: If circuit breaker is open or generation fails.
            RateLimitExceededError: If the LLM rate limit slot cannot be acquired.
            SecurityViolationError: If SQL fails validation after all retries.
            SQLParseError: If SQL cannot be parsed.

        Example:
            >>> sql, validation, tokens = await orchestrator._generate_sql_with_retry(
            ...     question="Count users",
            ...     schema=db_schema,
            ...     request_id="123",
            ... )
        """
        # Check circuit breaker
        if not self.circuit_breaker.allow_request():
            raise LLMError(
                message="SQL generation service is temporarily unavailable (circuit breaker open)",
                details={
                    "circuit_state": self.circuit_breaker.state,
                    "failure_count": self.circuit_breaker.failure_count,
                },
            )

        previous_sql: str | None = None
        error_feedback: str | None = None
        max_retries = self.resilience_config.max_retries
        tokens_used: int | None = None

        for attempt in range(max_retries + 1):
            try:
                logger.debug(
                    "Generating SQL",
                    extra={
                        "request_id": request_id,
                        "attempt": attempt + 1,
                        "max_retries": max_retries + 1,
                    },
                )

                # Generate SQL (under LLM rate limiter, with metrics)
                generated_sql = await self._generate_with_resilience(
                    question=question,
                    schema=schema,
                    previous_attempt=previous_sql,
                    error_feedback=error_feedback,
                )

                # Note: tokens_used would come from OpenAI response metadata if available
                # For now, we don't extract it, but it can be added later

                logger.debug(
                    "SQL generated",
                    extra={
                        "request_id": request_id,
                        "sql_length": len(generated_sql),
                    },
                )

                # Validate SQL
                try:
                    self.sql_validator.validate_or_raise(generated_sql)
                except (SecurityViolationError, SQLParseError) as validation_error:
                    if self.metrics is not None:
                        self.metrics.increment_sql_rejected(str(validation_error.code))

                    if attempt < max_retries:
                        # Record as failure and retry with feedback
                        logger.warning(
                            "SQL validation failed, retrying with feedback",
                            extra={
                                "request_id": request_id,
                                "attempt": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        previous_sql = generated_sql
                        error_feedback = str(validation_error)
                        await self._wait_before_retry(attempt)
                        continue
                    else:
                        # Out of retries, record failure and raise
                        self.circuit_breaker.record_failure()
                        logger.error(
                            "SQL validation failed after all retries",
                            extra={
                                "request_id": request_id,
                                "attempts": attempt + 1,
                                "error": str(validation_error),
                            },
                        )
                        raise

                # Validation successful
                self.circuit_breaker.record_success()
                logger.info(
                    "SQL generated and validated successfully",
                    extra={
                        "request_id": request_id,
                        "attempts": attempt + 1,
                    },
                )

                # Build validation result
                validation_result = ValidationResult(
                    is_valid=True,
                    is_select=True,
                    allows_data_modification=False,
                    uses_blocked_functions=[],
                    error_message=None,
                )

                return generated_sql, validation_result, tokens_used

            except (LLMError, RateLimitExceededError, SecurityViolationError, SQLParseError):
                # Re-raise known errors
                raise
            except Exception as e:
                # Unexpected error during generation
                self.circuit_breaker.record_failure()
                logger.exception(
                    "Unexpected error during SQL generation",
                    extra={"request_id": request_id},
                )
                raise LLMError(
                    message=f"SQL generation failed unexpectedly: {e!s}",
                    details={"error_type": type(e).__name__},
                ) from e

        # Should not reach here, but just in case
        self.circuit_breaker.record_failure()
        raise LLMError(
            message="SQL generation failed after all retry attempts",
            details={"max_retries": max_retries},
        )

    async def _generate_with_resilience(
        self,
        question: str,
        schema: Any,
        previous_attempt: str | None,
        error_feedback: str | None,
    ) -> str:
        """Call the SQL generator under the LLM rate limiter with metrics.

        Args:
            question: User's natural language question.
            schema: Database schema for context.
            previous_attempt: Previous SQL attempt (for retry feedback).
            error_feedback: Validation error from the previous attempt.

        Returns:
            str: The generated SQL query.

        Raises:
            RateLimitExceededError: If no LLM slot becomes available in time.
        """
        llm_acquired = False
        if self.rate_limiter is not None:
            llm_acquired = await self.rate_limiter.llm_limiter.acquire(
                timeout=self.resilience_config.rate_limit_timeout
            )
            if not llm_acquired:
                logger.warning("LLM call rejected: rate limit exceeded")
                raise RateLimitExceededError(
                    message="LLM rate limit exceeded, please retry later",
                    details={
                        "llm_limit": self.rate_limiter.llm_limiter.max_concurrent,
                    },
                )

        try:
            llm_start = time.monotonic()
            generated_sql = await self.sql_generator.generate(
                question=question,
                schema=schema,
                previous_attempt=previous_attempt,
                error_feedback=error_feedback,
            )
            if self.metrics is not None:
                self.metrics.increment_llm_call("generate_sql")
                self.metrics.observe_llm_latency("generate_sql", time.monotonic() - llm_start)
            return generated_sql
        finally:
            if self.rate_limiter is not None and llm_acquired:
                self.rate_limiter.llm_limiter.release()

    async def _wait_before_retry(self, attempt: int) -> None:
        """Wait with exponential backoff before the next retry attempt.

        The delay grows as ``retry_delay * backoff_factor ** attempt``.

        Args:
            attempt: Zero-based index of the attempt that just failed.
        """
        delay = self.resilience_config.retry_delay * (
            self.resilience_config.backoff_factor**attempt
        )
        if delay > 0:
            logger.debug(
                "Backing off before retry",
                extra={"delay_seconds": delay, "next_attempt": attempt + 2},
            )
            await asyncio.sleep(delay)

    async def _validate_results_safely(
        self,
        question: str,
        sql: str,
        results: list[dict[str, Any]],
        row_count: int,
        request_id: str,
    ) -> int:
        """Validate query results with error handling (non-blocking).

        This method attempts to validate results using LLM, but failures
        don't cause the overall query to fail. Returns a confidence score.

        Args:
            question: User's original question.
            sql: Generated SQL query.
            results: Query results.
            row_count: Total row count.
            request_id: Request ID for tracking.

        Returns:
            int: Confidence score (0-100). Returns 100 if validation disabled/fails.

        Example:
            >>> confidence = await orchestrator._validate_results_safely(
            ...     question="Count users",
            ...     sql="SELECT COUNT(*) FROM users",
            ...     results=[{"count": 42}],
            ...     row_count=1,
            ...     request_id="123",
            ... )
        """
        if not self.validation_config.enabled:
            return 100

        try:
            logger.debug(
                "Validating results",
                extra={"request_id": request_id},
            )

            llm_start = time.monotonic()
            validation_result = await self.result_validator.validate(
                question=question,
                sql=sql,
                results=results,
                row_count=row_count,
            )
            if self.metrics is not None:
                self.metrics.increment_llm_call("validate_result")
                self.metrics.observe_llm_latency("validate_result", time.monotonic() - llm_start)

            logger.info(
                "Result validation completed",
                extra={
                    "request_id": request_id,
                    "confidence": validation_result.confidence,
                    "is_acceptable": validation_result.is_acceptable,
                },
            )

            return validation_result.confidence

        except Exception as e:
            # Log but don't fail the query
            logger.warning(
                "Result validation failed, continuing with default confidence",
                extra={
                    "request_id": request_id,
                    "error": str(e),
                },
            )
            return 100  # Default to high confidence if validation fails
