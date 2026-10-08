"""Unit tests for the Prometheus metrics collector."""

from unittest.mock import patch

import pytest
from prometheus_client import REGISTRY

from pg_mcp.observability.metrics import MetricsCollector, metrics


class TestMetricsCollectorSingleton:
    """Tests for singleton behavior."""

    def test_same_instance_returned(self) -> None:
        assert MetricsCollector() is MetricsCollector()
        assert MetricsCollector() is metrics

    def test_metrics_registered_in_registry(self) -> None:
        names = set(REGISTRY._names_to_collectors.keys())
        expected = {
            "pg_mcp_query_requests_total",
            "pg_mcp_query_duration_seconds",
            "pg_mcp_llm_calls_total",
            "pg_mcp_llm_latency_seconds",
            "pg_mcp_llm_tokens_used",
            "pg_mcp_sql_rejected_total",
            "pg_mcp_db_connections_active",
            "pg_mcp_db_query_duration_seconds",
            "pg_mcp_schema_cache_age_seconds",
        }
        assert expected <= names


class TestMetricUpdates:
    """Tests for metric helper methods (unique labels avoid cross-test collisions)."""

    def test_increment_query_request(self) -> None:
        before = REGISTRY.get_sample_value(
            "pg_mcp_query_requests_total",
            {"status": "unit_success", "database": "unitdb"},
        )
        metrics.increment_query_request(status="unit_success", database="unitdb")
        after = REGISTRY.get_sample_value(
            "pg_mcp_query_requests_total",
            {"status": "unit_success", "database": "unitdb"},
        )
        assert after == (before or 0) + 1

    def test_increment_llm_call(self) -> None:
        metrics.increment_llm_call("unit_generate")
        value = REGISTRY.get_sample_value("pg_mcp_llm_calls_total", {"operation": "unit_generate"})
        assert value == 1

    def test_observe_llm_latency(self) -> None:
        metrics.observe_llm_latency("unit_latency", 0.123)
        count = REGISTRY.get_sample_value(
            "pg_mcp_llm_latency_seconds_count", {"operation": "unit_latency"}
        )
        assert count == 1

    def test_increment_llm_tokens(self) -> None:
        metrics.increment_llm_tokens("unit_tokens", 77)
        # prometheus_client appends _total to Counter sample names
        value = REGISTRY.get_sample_value(
            "pg_mcp_llm_tokens_used_total", {"operation": "unit_tokens"}
        )
        assert value == 77

    def test_increment_sql_rejected(self) -> None:
        metrics.increment_sql_rejected("unit_ddl_detected")
        value = REGISTRY.get_sample_value(
            "pg_mcp_sql_rejected_total", {"reason": "unit_ddl_detected"}
        )
        assert value == 1

    def test_set_db_connections_active(self) -> None:
        metrics.set_db_connections_active("unitdb", 7)
        value = REGISTRY.get_sample_value("pg_mcp_db_connections_active", {"database": "unitdb"})
        assert value == 7

    def test_observe_db_query_duration(self) -> None:
        metrics.observe_db_query_duration(0.25)
        count = REGISTRY.get_sample_value("pg_mcp_db_query_duration_seconds_count")
        assert count >= 1

    def test_set_schema_cache_age(self) -> None:
        metrics.set_schema_cache_age("unitdb", 120.5)
        value = REGISTRY.get_sample_value("pg_mcp_schema_cache_age_seconds", {"database": "unitdb"})
        assert value == 120.5

    def test_reset_all_metrics(self) -> None:
        metrics.increment_llm_call("reset_probe")
        metrics.reset_all_metrics()
        # After re-initialization the counter starts fresh.
        metrics.increment_llm_call("reset_probe")
        value = REGISTRY.get_sample_value("pg_mcp_llm_calls_total", {"operation": "reset_probe"})
        assert value == 1


class TestMetricsServer:
    """Tests for the metrics HTTP server helper."""

    def test_start_metrics_server_delegates(self) -> None:
        with patch("pg_mcp.observability.metrics.start_http_server") as start:
            metrics.start_metrics_server(9999)
        start.assert_called_once_with(9999)


@pytest.fixture(autouse=True)
def _isolated_labels():
    """Ensure per-test labels don't leak counters across assertions."""
    yield
    metrics.reset_all_metrics()
