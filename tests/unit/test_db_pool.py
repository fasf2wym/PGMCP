"""Unit tests for database connection pool management.

asyncpg.create_pool is monkeypatched so no real database is required.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from pg_mcp.config.settings import DatabaseConfig
from pg_mcp.db.pool import close_pools, create_pool, create_pools


def _make_config(name: str = "testdb") -> DatabaseConfig:
    return DatabaseConfig(name=name, host="localhost", user="test")


def _make_fake_pool() -> MagicMock:
    pool = MagicMock()
    pool.close = AsyncMock()
    pool.terminate = MagicMock()
    return pool


class TestCreatePool:
    """Tests for single pool creation."""

    @pytest.mark.asyncio
    async def test_creates_pool_with_config(self) -> None:
        fake_pool = _make_fake_pool()
        with patch(
            "pg_mcp.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=fake_pool)
        ) as factory:
            pool = await create_pool(_make_config("mydb"))

        assert pool is fake_pool
        kwargs = factory.call_args.kwargs
        assert kwargs["database"] == "mydb"
        assert kwargs["host"] == "localhost"
        assert kwargs["user"] == "test"
        assert kwargs["min_size"] == 5
        assert kwargs["max_size"] == 20

    @pytest.mark.asyncio
    async def test_none_pool_raises_runtime_error(self) -> None:
        with (
            patch("pg_mcp.db.pool.asyncpg.create_pool", new=AsyncMock(return_value=None)),
            pytest.raises(RuntimeError, match="testdb"),
        ):
            await create_pool(_make_config("testdb"))

    @pytest.mark.asyncio
    async def test_connection_failure_propagates(self) -> None:
        with (
            patch(
                "pg_mcp.db.pool.asyncpg.create_pool",
                new=AsyncMock(side_effect=OSError("connection refused")),
            ),
            pytest.raises(OSError, match="connection refused"),
        ):
            await create_pool(_make_config("testdb"))


class TestCreatePools:
    """Tests for multi-database pool creation."""

    @pytest.mark.asyncio
    async def test_creates_pool_per_database(self) -> None:
        pool_a, pool_b = _make_fake_pool(), _make_fake_pool()
        factory = AsyncMock(side_effect=[pool_a, pool_b])

        with patch("pg_mcp.db.pool.asyncpg.create_pool", new=factory):
            pools = await create_pools([_make_config("db_a"), _make_config("db_b")])

        assert set(pools.keys()) == {"db_a", "db_b"}
        assert pools["db_a"] is pool_a
        assert pools["db_b"] is pool_b
        assert factory.await_count == 2

    @pytest.mark.asyncio
    async def test_empty_config_list(self) -> None:
        assert await create_pools([]) == {}


class TestClosePools:
    """Tests for graceful pool shutdown."""

    @pytest.mark.asyncio
    async def test_graceful_close(self) -> None:
        pool = _make_fake_pool()
        await close_pools({"db": pool})
        pool.close.assert_awaited_once()
        pool.terminate.assert_not_called()

    @pytest.mark.asyncio
    async def test_timeout_forces_termination(self) -> None:
        import asyncio

        pool = _make_fake_pool()

        async def slow_close() -> None:
            await asyncio.sleep(30)

        pool.close = slow_close

        await close_pools({"db": pool}, timeout=0.01)
        pool.terminate.assert_called_once()

    @pytest.mark.asyncio
    async def test_error_forces_termination_and_continues(self) -> None:
        pool_bad = _make_fake_pool()
        pool_bad.close = AsyncMock(side_effect=RuntimeError("close failed"))
        pool_good = _make_fake_pool()

        await close_pools({"bad": pool_bad, "good": pool_good})

        pool_bad.terminate.assert_called_once()
        pool_good.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_empty_pool_dict(self) -> None:
        await close_pools({})
