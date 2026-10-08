"""Unit tests for schema introspection.

Uses a fake asyncpg connection that dispatches on SQL markers so the full
introspection flow runs without a real database.
"""

from contextlib import asynccontextmanager
from typing import Any

import pytest

from pg_mcp.db.introspection import SchemaIntrospector

VERSION_STRING = "PostgreSQL 16.2 (Homebrew), 64-bit"


class FakeConnection:
    """Fake asyncpg connection dispatching fetch/fetchval by SQL substring."""

    def __init__(self, responses: dict[str, Any]):
        self._responses = responses
        self.queries: list[str] = []

    def _match(self, query: str) -> Any:
        self.queries.append(query)
        for marker, value in self._responses.items():
            if marker in query:
                return value
        raise AssertionError(f"Unexpected query: {query[:120]!r}")

    async def fetch(self, query: str, *args: Any) -> list[dict[str, Any]]:
        value = self._match(query)
        if callable(value):
            value = value(*args)
        return value

    async def fetchval(self, query: str, *args: Any) -> Any:
        value = self._match(query)
        if callable(value):
            value = value(*args)
        return value


class FakePool:
    def __init__(self, conn: FakeConnection):
        self._conn = conn

    @asynccontextmanager
    async def acquire(self):
        yield self._conn


def _responses() -> dict[str, Any]:
    return {
        "SELECT version()": VERSION_STRING,
        "relkind = 'r'": [
            {"schema_name": "public", "table_name": "users", "comment": "User accounts"},
        ],
        "relkind = 'v'": [
            {"schema_name": "public", "table_name": "active_users", "comment": None},
        ],
        "typtype = 'e'": [
            {
                "schema_name": "public",
                "type_name": "user_status",
                "values": ["active", "inactive"],
            },
        ],
        "format_type": [
            {
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": False,
                "default_value": None,
                "comment": None,
            },
            {
                "column_name": "email",
                "data_type": "character varying",
                "is_nullable": True,
                "default_value": None,
                "comment": "User email",
            },
        ],
        "contype = 'u'": True,
        "NOT idx.indisprimary": [
            {
                "index_name": "idx_users_email",
                "is_unique": True,
                "index_type": "btree",
                "columns": ["email"],
            },
        ],
        "indisprimary": [{"column_name": "id"}],
        "contype = 'f'": [
            {
                "constraint_name": "fk_orders_user",
                "column_name": "user_id",
                "referenced_table": "users",
                "referenced_column": "id",
            },
        ],
        "reltuples": 1234,
    }


@pytest.fixture
def introspector() -> SchemaIntrospector:
    conn = FakeConnection(_responses())
    return SchemaIntrospector(pool=FakePool(conn), database_name="testdb")  # type: ignore[arg-type]


class TestIntrospect:
    """Tests for the full introspection flow."""

    @pytest.mark.asyncio
    async def test_version_parsed(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        assert schema.version == "PostgreSQL 16.2 (Homebrew)"

    @pytest.mark.asyncio
    async def test_database_name(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        assert schema.database_name == "testdb"

    @pytest.mark.asyncio
    async def test_tables_and_views_combined(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        names = {t.table_name for t in schema.tables}
        assert names == {"users", "active_users"}

    @pytest.mark.asyncio
    async def test_table_comment_preserved(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        users = schema.get_table("users")
        assert users is not None
        assert users.comment == "User accounts"

    @pytest.mark.asyncio
    async def test_columns_enriched(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        users = schema.get_table("users")
        assert users is not None
        assert [c.name for c in users.columns] == ["id", "email"]
        assert users.columns[0].data_type == "integer"
        assert users.columns[1].comment == "User email"

    @pytest.mark.asyncio
    async def test_unique_column_flag(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        users = schema.get_table("users")
        assert users is not None
        # The fake unique-check returns True for every column.
        assert all(c.is_unique for c in users.columns)

    @pytest.mark.asyncio
    async def test_primary_keys_marked(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        users = schema.get_table("users")
        assert users is not None
        by_name = {c.name: c for c in users.columns}
        assert by_name["id"].is_primary_key is True
        assert by_name["email"].is_primary_key is False

    @pytest.mark.asyncio
    async def test_foreign_keys_attached(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        users = schema.get_table("users")
        assert users is not None
        assert users.foreign_keys[0].referenced_table == "users"
        assert users.foreign_keys[0].column_name == "user_id"

    @pytest.mark.asyncio
    async def test_indexes_attached(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        users = schema.get_table("users")
        assert users is not None
        assert users.indexes[0].name == "idx_users_email"
        assert users.indexes[0].columns == ["email"]
        assert users.indexes[0].is_unique

    @pytest.mark.asyncio
    async def test_row_count_estimate(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        users = schema.get_table("users")
        assert users is not None
        assert users.row_count_estimate == 1234

    @pytest.mark.asyncio
    async def test_enum_types(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        assert len(schema.enum_types) == 1
        assert schema.enum_types[0].type_name == "user_status"
        assert schema.enum_types[0].values == ["active", "inactive"]


class TestEdgeCases:
    """Tests for edge cases in individual helpers."""

    @pytest.mark.asyncio
    async def test_empty_database(self) -> None:
        conn = FakeConnection(
            {
                "SELECT version()": VERSION_STRING,
                "relkind = 'r'": [],
                "relkind = 'v'": [],
                "typtype = 'e'": [],
            }
        )
        introspector = SchemaIntrospector(pool=FakePool(conn), database_name="empty")
        schema = await introspector.introspect()
        assert schema.tables == []
        assert schema.enum_types == []

    @pytest.mark.asyncio
    async def test_version_none_when_unavailable(self) -> None:
        conn = FakeConnection(
            {
                "SELECT version()": None,
                "relkind = 'r'": [],
                "relkind = 'v'": [],
                "typtype = 'e'": [],
            }
        )
        introspector = SchemaIntrospector(pool=FakePool(conn), database_name="db")
        schema = await introspector.introspect()
        assert schema.version is None

    @pytest.mark.asyncio
    async def test_unique_check_none_returns_false(self) -> None:
        introspector = SchemaIntrospector.__new__(SchemaIntrospector)
        introspector.pool = None
        introspector.database_name = "db"
        conn = FakeConnection({"contype = 'u'": None})
        result = await introspector._is_column_unique(conn, "users", "public", "email")  # type: ignore[arg-type]
        assert result is False

    @pytest.mark.asyncio
    async def test_row_count_none_returns_zero(self) -> None:
        introspector = SchemaIntrospector.__new__(SchemaIntrospector)
        introspector.pool = None
        introspector.database_name = "db"
        conn = FakeConnection({"reltuples": None})
        result = await introspector._get_row_count_estimate(conn, "users", "public")  # type: ignore[arg-type]
        assert result == 0
