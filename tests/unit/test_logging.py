"""Unit tests for structured logging.

Covers SensitiveDataFilter sanitization, JSON/Text formatters, and
configure_logging behavior.
"""

import json
import logging
from typing import Any

import pytest

from pg_mcp.observability.logging import (
    JSONFormatter,
    SensitiveDataFilter,
    TextFormatter,
    configure_logging,
    get_logger,
)


def _make_record(
    msg: str = "test message",
    level: int = logging.INFO,
    extra: dict[str, Any] | None = None,
) -> logging.LogRecord:
    """Create a bare LogRecord for formatter/filter tests."""
    record = logging.LogRecord(
        name="test.logger",
        level=level,
        pathname=__file__,
        lineno=42,
        msg=msg,
        args=None,
        exc_info=None,
        func="test_func",
    )
    if extra:
        for key, value in extra.items():
            setattr(record, key, value)
    return record


class TestSensitiveDataFilter:
    """Tests for sensitive data sanitization."""

    def setup_method(self) -> None:
        self.filter = SensitiveDataFilter()

    def test_sensitive_dict_keys_redacted(self) -> None:
        data = {"password": "hunter2", "user": "alice"}
        sanitized = self.filter._sanitize_dict(data)
        assert sanitized["password"] == "***REDACTED***"
        assert sanitized["user"] == "alice"

    def test_all_sensitive_keys_covered(self) -> None:
        for key in SensitiveDataFilter.SENSITIVE_KEYS:
            sanitized = self.filter._sanitize_dict({key: "value"})
            assert sanitized[key] == "***REDACTED***"

    def test_key_matching_is_case_insensitive(self) -> None:
        sanitized = self.filter._sanitize_dict({"API_KEY": "sk-123"})
        assert sanitized["API_KEY"] == "***REDACTED***"

    def test_nested_dicts_sanitized(self) -> None:
        data = {"db": {"password": "hunter2", "host": "localhost"}}
        sanitized = self.filter._sanitize_dict(data)
        assert sanitized["db"]["password"] == "***REDACTED***"
        assert sanitized["db"]["host"] == "localhost"

    def test_lists_and_tuples_sanitized(self) -> None:
        data = {"items": [{"token": "abc"}, "plain"]}
        sanitized = self.filter._sanitize_dict(data)
        assert sanitized["items"][0] == {"token": "***REDACTED***"}
        assert sanitized["items"][1] == "plain"

        tuple_result = self.filter._sanitize_data(({"secret": "x"}, 1))
        assert isinstance(tuple_result, tuple)
        assert tuple_result[0] == {"secret": "***REDACTED***"}

    def test_primitives_pass_through(self) -> None:
        assert self.filter._sanitize_data(42) == 42
        assert self.filter._sanitize_data("text") == "text"
        assert self.filter._sanitize_data(None) is None

    def test_filter_sanitize_args(self) -> None:
        # Two args so logging does not collapse a single mapping arg
        record = logging.LogRecord(
            name="t",
            level=logging.INFO,
            pathname="p",
            lineno=1,
            msg="connecting to %s as %s",
            args=({"password": "x"}, "alice"),
            exc_info=None,
        )
        assert self.filter.filter(record) is True
        assert record.args[0]["password"] == "***REDACTED***"
        assert record.args[1] == "alice"

    def test_filter_sanitize_extra_attributes(self) -> None:
        record = _make_record(extra={"api_key": "sk-live", "detail": {"pwd": "x"}})
        self.filter.filter(record)
        assert record.api_key == "***REDACTED***"
        assert record.detail["pwd"] == "***REDACTED***"

    def test_filter_keeps_non_sensitive_extra(self) -> None:
        record = _make_record(extra={"database": "testdb", "row_count": 5})
        self.filter.filter(record)
        assert record.database == "testdb"
        assert record.row_count == 5


class TestJSONFormatter:
    """Tests for JSON log formatting."""

    def test_basic_fields(self) -> None:
        output = json.loads(JSONFormatter().format(_make_record()))
        assert output["level"] == "INFO"
        assert output["logger"] == "test.logger"
        assert output["message"] == "test message"
        assert output["module"]
        assert output["function"] == "test_func"
        assert output["line"] == 42
        assert "timestamp" in output

    def test_request_id_included_when_present(self) -> None:
        record = _make_record(extra={"request_id": "req-9"})
        output = json.loads(JSONFormatter().format(record))
        assert output["request_id"] == "req-9"

    def test_extra_fields_grouped(self) -> None:
        record = _make_record(extra={"database": "db1", "row_count": 3})
        output = json.loads(JSONFormatter().format(record))
        # taskName etc. may leak in on newer Pythons; assert our keys landed
        assert output["extra"]["database"] == "db1"
        assert output["extra"]["row_count"] == 3

    def test_exception_included(self) -> None:
        try:
            raise ValueError("boom")
        except ValueError:
            import sys

            record = _make_record()
            record.exc_info = sys.exc_info()
        output = json.loads(JSONFormatter().format(record))
        assert "ValueError: boom" in output["exception"]

    def test_output_is_single_line_json(self) -> None:
        output = JSONFormatter().format(_make_record())
        assert json.loads(output)  # valid JSON
        assert "\n" not in output.strip()


class TestTextFormatter:
    """Tests for human-readable text formatting."""

    def test_basic_format(self) -> None:
        output = TextFormatter().format(_make_record())
        assert "[INFO]" in output
        assert "test.logger - test message" in output

    def test_request_id_appended(self) -> None:
        record = _make_record(extra={"request_id": "req-t1"})
        output = TextFormatter().format(record)
        assert "[request_id=req-t1]" in output

    def test_exception_appended(self) -> None:
        try:
            raise RuntimeError("kaput")
        except RuntimeError:
            import sys

            record = _make_record()
            record.exc_info = sys.exc_info()
        output = TextFormatter().format(record)
        assert "RuntimeError: kaput" in output


class TestConfigureLogging:
    """Tests for logging configuration."""

    ROOT_LOGGER = logging.getLogger()

    @pytest.fixture(autouse=True)
    def _restore_logging(self):
        saved_handlers = self.ROOT_LOGGER.handlers[:]
        saved_level = self.ROOT_LOGGER.level
        yield
        self.ROOT_LOGGER.handlers = saved_handlers
        self.ROOT_LOGGER.setLevel(saved_level)

    def _record_count(self) -> int:
        return len(self.ROOT_LOGGER.handlers)

    def test_logs_go_to_stderr_not_stdout(self) -> None:
        """Logs must never touch stdout: stdio transport carries JSON-RPC there."""
        import sys

        configure_logging(level="INFO", log_format="json")
        handler = self.ROOT_LOGGER.handlers[0]
        assert handler.stream is sys.stderr

    def test_json_format_configured(self) -> None:
        configure_logging(level="INFO", log_format="json")
        assert self._record_count() == 1
        assert isinstance(self.ROOT_LOGGER.handlers[0].formatter, JSONFormatter)
        assert self.ROOT_LOGGER.level == logging.INFO

    def test_text_format_configured(self) -> None:
        configure_logging(level="DEBUG", log_format="text")
        assert isinstance(self.ROOT_LOGGER.handlers[0].formatter, TextFormatter)
        assert self.ROOT_LOGGER.level == logging.DEBUG

    def test_level_is_case_insensitive(self) -> None:
        configure_logging(level="warning")
        assert self.ROOT_LOGGER.level == logging.WARNING

    def test_sensitive_filter_added_by_default(self) -> None:
        configure_logging(log_format="json")
        filters = [
            f for f in self.ROOT_LOGGER.handlers[0].filters if isinstance(f, SensitiveDataFilter)
        ]
        assert len(filters) == 1

    def test_sensitive_filter_skipped_when_disabled(self) -> None:
        configure_logging(log_format="json", enable_sensitive_filter=False)
        filters = [
            f for f in self.ROOT_LOGGER.handlers[0].filters if isinstance(f, SensitiveDataFilter)
        ]
        assert not filters

    def test_third_party_loggers_quieted(self) -> None:
        configure_logging()
        try:
            for name in ("asyncpg", "openai", "httpx", "httpcore"):
                assert logging.getLogger(name).level == logging.WARNING
        finally:
            for name in ("asyncpg", "openai", "httpx", "httpcore"):
                logging.getLogger(name).setLevel(logging.NOTSET)

    def test_existing_handlers_replaced(self) -> None:
        sentinel = logging.NullHandler()
        self.ROOT_LOGGER.addHandler(sentinel)
        configure_logging()
        assert self._record_count() == 1
        assert sentinel not in self.ROOT_LOGGER.handlers

    def test_json_log_end_to_end(self) -> None:
        configure_logging(level="INFO", log_format="json")
        logger = get_logger("e2e_test")
        logger.info("hello %s", "world", extra={"request_id": "req-e2e"})

    def test_get_logger(self) -> None:
        logger = get_logger("some.module")
        assert isinstance(logger, logging.Logger)
        assert logger.name == "some.module"
