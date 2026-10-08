"""Unit tests for request tracing.

Covers request ID generation, context propagation, the request_context
manager, trace decorators, and TracingLogger.
"""

import logging
from typing import Any

import pytest

from pg_mcp.observability.tracing import (
    TraceContext,
    TracingLogger,
    clear_request_id,
    generate_request_id,
    get_request_id,
    get_tracing_logger,
    request_context,
    set_request_id,
    trace_async,
    trace_sync,
)


class TestRequestId:
    """Tests for request ID generation and context get/set/clear."""

    def test_generate_request_id_is_uuid(self) -> None:
        req_id = generate_request_id()
        assert isinstance(req_id, str)
        assert len(req_id) == 36
        assert req_id.count("-") == 4

    def test_generate_request_id_unique(self) -> None:
        ids = {generate_request_id() for _ in range(100)}
        assert len(ids) == 100

    def test_get_request_id_default_none(self) -> None:
        clear_request_id()
        assert get_request_id() is None

    def test_set_and_get_request_id(self) -> None:
        set_request_id("req-123")
        assert get_request_id() == "req-123"
        clear_request_id()

    def test_clear_request_id(self) -> None:
        set_request_id("req-123")
        clear_request_id()
        assert get_request_id() is None


class TestRequestContext:
    """Tests for the request_context async context manager."""

    @pytest.mark.asyncio
    async def test_generates_id_when_absent(self) -> None:
        clear_request_id()
        async with request_context() as req_id:
            assert req_id is not None
            assert get_request_id() == req_id
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_uses_provided_id(self) -> None:
        async with request_context("custom-id") as req_id:
            assert req_id == "custom-id"
            assert get_request_id() == "custom-id"
        assert get_request_id() is None

    @pytest.mark.asyncio
    async def test_restores_previous_context_on_exit(self) -> None:
        set_request_id("outer")
        try:
            async with request_context("inner"):
                assert get_request_id() == "inner"
            assert get_request_id() == "outer"
        finally:
            clear_request_id()

    @pytest.mark.asyncio
    async def test_context_propagates_to_child_tasks(self) -> None:
        seen: dict[str, Any] = {}

        async def child() -> None:
            seen["request_id"] = get_request_id()

        async with request_context("parent-id"):
            import asyncio

            await asyncio.create_task(child())

        assert seen["request_id"] == "parent-id"

    @pytest.mark.asyncio
    async def test_reset_on_exception(self) -> None:
        clear_request_id()
        with pytest.raises(RuntimeError):
            async with request_context():
                raise RuntimeError("boom")
        assert get_request_id() is None


class TestTraceContextModel:
    """Tests for the TraceContext model."""

    def test_defaults(self) -> None:
        ctx = TraceContext(request_id="req-1")
        assert ctx.request_id == "req-1"
        assert ctx.parent_id is None
        assert ctx.operation is None
        assert ctx.metadata is None

    def test_full_context(self) -> None:
        ctx = TraceContext(
            request_id="req-1",
            parent_id="parent-1",
            operation="generate_sql",
            metadata={"database": "testdb"},
        )
        assert ctx.operation == "generate_sql"
        assert ctx.metadata == {"database": "testdb"}


class TestTraceDecorators:
    """Tests for trace_async and trace_sync decorators.

    Note: the log record factory swap is process-global, so these tests
    capture a log record inside the decorated function and verify the
    trace attributes; the factory is restored in a finally block by the
    decorator itself.
    """

    @pytest.fixture(autouse=True)
    def _restore_log_factory(self):
        factory = logging.getLogRecordFactory()
        yield
        logging.setLogRecordFactory(factory)

    @staticmethod
    def _capture_record() -> tuple[logging.Logger, list[logging.LogRecord]]:
        records: list[logging.LogRecord] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        logger = logging.getLogger("trace_test_capture")
        logger.setLevel(logging.DEBUG)
        handler = CaptureHandler()
        logger.addHandler(handler)
        logger.propagate = False
        return logger, records

    @pytest.mark.asyncio
    async def test_trace_async_without_context(self) -> None:
        @trace_async()
        async def work(value: int) -> int:
            return value * 2

        assert await work(21) == 42

    @pytest.mark.asyncio
    async def test_trace_async_with_context_stamps_records(self) -> None:
        logger, records = self._capture_record()

        @trace_async(operation="gen_sql")
        async def work() -> None:
            logger.info("hello")

        async with request_context("req-abc"):
            await work()

        assert len(records) == 1
        assert records[0].request_id == "req-abc"
        assert records[0].operation == "gen_sql"

    @pytest.mark.asyncio
    async def test_trace_async_restores_factory_on_error(self) -> None:
        original = logging.getLogRecordFactory()

        @trace_async()
        async def broken() -> None:
            raise ValueError("boom")

        async with request_context("req-err"):
            with pytest.raises(ValueError):
                await broken()

        assert logging.getLogRecordFactory() is original

    @pytest.mark.asyncio
    async def test_trace_async_default_operation_is_function_name(self) -> None:
        logger, records = self._capture_record()

        @trace_async()
        async def my_operation() -> None:
            logger.info("hello")

        async with request_context("req-name"):
            await my_operation()

        assert records[0].operation == "my_operation"

    def test_trace_sync_without_context(self) -> None:
        @trace_sync()
        def work(value: int) -> int:
            return value + 1

        assert work(1) == 2

    def test_trace_sync_with_context_stamps_records(self) -> None:
        logger, records = self._capture_record()

        @trace_sync(operation="validate")
        def work() -> None:
            logger.info("hello")

        set_request_id("req-sync")
        try:
            work()
        finally:
            clear_request_id()

        assert len(records) == 1
        assert records[0].request_id == "req-sync"
        assert records[0].operation == "validate"


class TestTracingLogger:
    """Tests for TracingLogger convenience wrapper."""

    @staticmethod
    def _make_logger() -> tuple[TracingLogger, list[logging.LogRecord]]:
        records: list[logging.LogRecord] = []

        class CaptureHandler(logging.Handler):
            def emit(self, record: logging.LogRecord) -> None:
                records.append(record)

        wrapped = TracingLogger("tracing_logger_test")
        wrapped._logger.setLevel(logging.DEBUG)
        wrapped._logger.addHandler(CaptureHandler())
        wrapped._logger.propagate = False
        return wrapped, records

    @pytest.mark.asyncio
    async def test_all_levels(self) -> None:
        wrapped, records = self._make_logger()
        wrapped.debug("d")
        wrapped.info("i")
        wrapped.warning("w")
        wrapped.error("e")
        wrapped.critical("c")

        levels = [r.levelname for r in records]
        assert levels == ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]

    @pytest.mark.asyncio
    async def test_exception_includes_traceback(self) -> None:
        wrapped, records = self._make_logger()
        try:
            raise ValueError("boom")
        except ValueError:
            wrapped.exception("caught")

        assert records[0].exc_info is not None
        assert records[0].levelno == logging.ERROR

    @pytest.mark.asyncio
    async def test_injects_request_id_into_extra(self) -> None:
        wrapped, records = self._make_logger()

        async with request_context("req-log"):
            wrapped.info("processing")

        assert records[0].request_id == "req-log"

    @pytest.mark.asyncio
    async def test_existing_extra_request_id_not_overwritten(self) -> None:
        wrapped, records = self._make_logger()

        async with request_context("ctx-id"):
            wrapped.info("processing", extra={"request_id": "explicit-id"})

        assert records[0].request_id == "explicit-id"

    @pytest.mark.asyncio
    async def test_message_formatting_args(self) -> None:
        wrapped, records = self._make_logger()
        wrapped.info("value is %s", 42)
        assert records[0].getMessage() == "value is 42"

    def test_get_tracing_logger(self) -> None:
        wrapped = get_tracing_logger("factory_test")
        assert isinstance(wrapped, TracingLogger)
