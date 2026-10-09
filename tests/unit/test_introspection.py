"""Unit tests for schema introspection.

Uses a fake asyncpg connection that dispatches on the ``-- pg_mcp:<name>``
comment markers embedded in the catalog queries, so the full introspection
flow runs without a real database.
"""

from contextlib import asynccontextmanager
from typing import Any

import pytest

from pg_mcp.db.introspection import SchemaIntrospector

VERSION_STRING = "PostgreSQL 16.2 (Homebrew), 64-bit"

# Every catalog query except "SELECT version()" starts with one of these
# markers; the fake dispatches on them (substring match, no collisions).
QUERY_MARKERS = (
    "-- pg_mcp:relations",
    "-- pg_mcp:columns",
    "-- pg_mcp:unique_columns",
    "-- pg_mcp:primary_keys",
    "-- pg_mcp:foreign_keys",
    "-- pg_mcp:indexes",
    "-- pg_mcp:enums",
)


class FakeConnection:
    """Fake asyncpg connection dispatching fetch/fetchval by SQL marker."""

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


def _empty_responses(version: str | None) -> dict[str, Any]:
    """Responses for a database with no user relations."""
    responses: dict[str, Any] = {"SELECT version()": version}
    responses.update({marker: [] for marker in QUERY_MARKERS})
    return responses


def _responses(**overrides: Any) -> dict[str, Any]:
    """Default single-user-table fixtures, overridable per query category.

    ``overrides`` keys are short category names matching the query markers
    (``relations``, ``columns``, ``unique_columns``, ``primary_keys``,
    ``foreign_keys``, ``indexes``, ``enums``).
    """
    responses: dict[str, Any] = {
        "SELECT version()": VERSION_STRING,
        "-- pg_mcp:relations": [
            {
                "schema_name": "public",
                "table_name": "users",
                "relkind": "r",
                "row_count_estimate": 1234,
                "comment": "User accounts",
            },
            {
                "schema_name": "public",
                "table_name": "active_users",
                "relkind": "v",
                "row_count_estimate": 12,
                "comment": None,
            },
        ],
        "-- pg_mcp:enums": [
            {
                "schema_name": "public",
                "type_name": "user_status",
                "values": ["active", "inactive"],
            },
        ],
        "-- pg_mcp:columns": [
            {
                "schema_name": "public",
                "table_name": "users",
                "column_name": "id",
                "data_type": "integer",
                "is_nullable": False,
                "default_value": None,
                "comment": None,
            },
            {
                "schema_name": "public",
                "table_name": "users",
                "column_name": "email",
                "data_type": "character varying",
                "is_nullable": True,
                "default_value": None,
                "comment": "User email",
            },
        ],
        "-- pg_mcp:unique_columns": [
            {"schema_name": "public", "table_name": "users", "column_name": "id"},
            {"schema_name": "public", "table_name": "users", "column_name": "email"},
        ],
        "-- pg_mcp:primary_keys": [
            {"schema_name": "public", "table_name": "users", "column_name": "id"},
        ],
        "-- pg_mcp:foreign_keys": [
            {
                "schema_name": "public",
                "table_name": "users",
                "constraint_name": "fk_orders_user",
                "column_name": "user_id",
                "referenced_table": "users",
                "referenced_column": "id",
            },
        ],
        "-- pg_mcp:indexes": [
            {
                "schema_name": "public",
                "table_name": "users",
                "index_name": "idx_users_email",
                "is_unique": True,
                "index_type": "btree",
                "columns": ["email"],
            },
        ],
    }
    for name, value in overrides.items():
        responses[f"-- pg_mcp:{name}"] = value
    return responses


def _make_introspector(responses: dict[str, Any]) -> tuple[SchemaIntrospector, FakeConnection]:
    conn = FakeConnection(responses)
    introspector = SchemaIntrospector(pool=FakePool(conn), database_name="testdb")  # type: ignore[arg-type]
    return introspector, conn


@pytest.fixture
def introspector() -> SchemaIntrospector:
    introspector, _ = _make_introspector(_responses())
    return introspector


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
        # Both columns are covered by unique constraints in the fake data.
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


class TestBatching:
    """Tests for the set-based query strategy (N+1 elimination)."""

    @pytest.mark.asyncio
    async def test_batch_query_count(self) -> None:
        # 8 queries total regardless of table count: version + 7 categories.
        introspector, conn = _make_introspector(_responses())
        await introspector.introspect()
        assert len(conn.queries) == 8

    @pytest.mark.asyncio
    async def test_tables_before_views(self) -> None:
        # Output ordering is tables first, then views, regardless of the
        # order rows arrive in from the combined relations query.
        responses = _responses(
            relations=[
                {
                    "schema_name": "public",
                    "table_name": "active_users",
                    "relkind": "v",
                    "row_count_estimate": 12,
                    "comment": None,
                },
                {
                    "schema_name": "public",
                    "table_name": "users",
                    "relkind": "r",
                    "row_count_estimate": 1234,
                    "comment": "User accounts",
                },
            ]
        )
        introspector, _ = _make_introspector(responses)
        schema = await introspector.introspect()
        assert [t.table_name for t in schema.tables] == ["users", "active_users"]

    @pytest.mark.asyncio
    async def test_sorted_by_schema_and_table(self) -> None:
        responses = _responses(
            relations=[
                {
                    "schema_name": "public",
                    "table_name": "users",
                    "relkind": "r",
                    "row_count_estimate": 10,
                    "comment": None,
                },
                {
                    "schema_name": "public",
                    "table_name": "accounts",
                    "relkind": "r",
                    "row_count_estimate": 20,
                    "comment": None,
                },
            ]
        )
        introspector, _ = _make_introspector(responses)
        schema = await introspector.introspect()
        assert [t.table_name for t in schema.tables] == ["accounts", "users"]

    @pytest.mark.asyncio
    async def test_view_has_row_count_estimate(self, introspector: SchemaIntrospector) -> None:
        schema = await introspector.introspect()
        view = schema.get_table("active_users")
        assert view is not None
        assert view.row_count_estimate == 12


class TestGrouping:
    """Tests that batched rows are attributed to the right table."""

    @pytest.mark.asyncio
    async def test_columns_dont_leak_between_tables(self) -> None:
        responses = _responses(
            relations=[
                {
                    "schema_name": "public",
                    "table_name": "users",
                    "relkind": "r",
                    "row_count_estimate": 10,
                    "comment": None,
                },
                {
                    "schema_name": "public",
                    "table_name": "orders",
                    "relkind": "r",
                    "row_count_estimate": 20,
                    "comment": None,
                },
            ],
            columns=[
                {
                    "schema_name": "public",
                    "table_name": "users",
                    "column_name": "id",
                    "data_type": "integer",
                    "is_nullable": False,
                    "default_value": None,
                    "comment": None,
                },
                {
                    "schema_name": "public",
                    "table_name": "users",
                    "column_name": "email",
                    "data_type": "character varying",
                    "is_nullable": True,
                    "default_value": None,
                    "comment": None,
                },
                {
                    "schema_name": "public",
                    "table_name": "orders",
                    "column_name": "order_id",
                    "data_type": "integer",
                    "is_nullable": False,
                    "default_value": None,
                    "comment": None,
                },
            ],
        )
        introspector, _ = _make_introspector(responses)
        schema = await introspector.introspect()

        users = schema.get_table("users")
        orders = schema.get_table("orders")
        assert users is not None and orders is not None
        assert [c.name for c in users.columns] == ["id", "email"]
        assert [c.name for c in orders.columns] == ["order_id"]

    @pytest.mark.asyncio
    async def test_indexes_dont_leak_between_tables(self) -> None:
        responses = _responses(
            relations=[
                {
                    "schema_name": "public",
                    "table_name": "users",
                    "relkind": "r",
                    "row_count_estimate": 10,
                    "comment": None,
                },
                {
                    "schema_name": "public",
                    "table_name": "orders",
                    "relkind": "r",
                    "row_count_estimate": 20,
                    "comment": None,
                },
            ],
            indexes=[
                {
                    "schema_name": "public",
                    "table_name": "users",
                    "index_name": "idx_users_email",
                    "is_unique": True,
                    "index_type": "btree",
                    "columns": ["email"],
                },
                {
                    "schema_name": "public",
                    "table_name": "orders",
                    "index_name": "idx_orders_user",
                    "is_unique": False,
                    "index_type": "btree",
                    "columns": ["user_id"],
                },
            ],
        )
        introspector, _ = _make_introspector(responses)
        schema = await introspector.introspect()

        users = schema.get_table("users")
        orders = schema.get_table("orders")
        assert users is not None and orders is not None
        assert [i.name for i in users.indexes] == ["idx_users_email"]
        assert [i.name for i in orders.indexes] == ["idx_orders_user"]

    @pytest.mark.asyncio
    async def test_unique_flag_scoped_to_table(self) -> None:
        # A unique constraint on orders.total must not mark users columns.
        responses = _responses(
            relations=[
                {
                    "schema_name": "public",
                    "table_name": "users",
                    "relkind": "r",
                    "row_count_estimate": 10,
                    "comment": None,
                },
                {
                    "schema_name": "public",
                    "table_name": "orders",
                    "relkind": "r",
                    "row_count_estimate": 20,
                    "comment": None,
                },
            ],
            columns=[
                {
                    "schema_name": "public",
                    "table_name": "users",
                    "column_name": "id",
                    "data_type": "integer",
                    "is_nullable": False,
                    "default_value": None,
                    "comment": None,
                },
                {
                    "schema_name": "public",
                    "table_name": "orders",
                    "column_name": "total",
                    "data_type": "numeric",
                    "is_nullable": True,
                    "default_value": None,
                    "comment": None,
                },
            ],
            unique_columns=[
                {"schema_name": "public", "table_name": "orders", "column_name": "total"},
            ],
            primary_keys=[],
        )
        introspector, _ = _make_introspector(responses)
        schema = await introspector.introspect()

        users = schema.get_table("users")
        orders = schema.get_table("orders")
        assert users is not None and orders is not None
        assert users.columns[0].is_unique is False
        assert orders.columns[0].is_unique is True


class TestEdgeCases:
    """Tests for edge cases in the introspection flow."""

    @pytest.mark.asyncio
    async def test_empty_database(self) -> None:
        introspector, _ = _make_introspector(_empty_responses(VERSION_STRING))
        schema = await introspector.introspect()
        assert schema.tables == []
        assert schema.enum_types == []

    @pytest.mark.asyncio
    async def test_version_none_when_unavailable(self) -> None:
        introspector, _ = _make_introspector(_empty_responses(None))
        schema = await introspector.introspect()
        assert schema.version is None

    @pytest.mark.asyncio
    async def test_row_count_none_defaults_to_zero(self) -> None:
        responses = _responses(
            relations=[
                {
                    "schema_name": "public",
                    "table_name": "users",
                    "relkind": "r",
                    "row_count_estimate": None,
                    "comment": None,
                }
            ]
        )
        introspector, _ = _make_introspector(responses)
        schema = await introspector.introspect()
        users = schema.get_table("users")
        assert users is not None
        assert users.row_count_estimate == 0
