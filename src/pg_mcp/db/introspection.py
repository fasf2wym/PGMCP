"""PostgreSQL schema introspection.

This module provides functionality to introspect PostgreSQL database schemas,
extracting comprehensive metadata about tables, columns, constraints, indexes,
and custom types.

Each metadata category (relations, columns, unique constraints, primary keys,
foreign keys, indexes, enum types) is fetched with a single set-based catalog
query and grouped by table in Python. This avoids the per-table/per-column
N+1 query pattern: startup cost is constant (8 queries total) regardless of
how many tables the database contains.

Every query starts with a unique ``-- pg_mcp:<name>`` comment marker so test
fakes can dispatch on the marker without substring collisions.
"""

from typing import Any

from asyncpg import Pool
from asyncpg.connection import Connection

from pg_mcp.models.schema import (
    ColumnInfo,
    DatabaseSchema,
    EnumTypeInfo,
    ForeignKeyInfo,
    IndexInfo,
    TableInfo,
)

# (schema_name, table_name) identifies a relation across all catalog queries.
TableKey = tuple[str, str]
# (schema_name, table_name, column_name) identifies a column.
ColumnKey = tuple[str, str, str]

TABLES_AND_VIEWS_QUERY = """
    -- pg_mcp:relations
    SELECT
        n.nspname AS schema_name,
        c.relname AS table_name,
        c.relkind AS relkind,
        c.reltuples::bigint AS row_count_estimate,
        obj_description(c.oid, 'pg_class') AS comment
    FROM pg_class c
    JOIN pg_namespace n ON n.oid = c.relnamespace
    WHERE c.relkind IN ('r', 'v')  -- regular tables and views
      AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
"""

COLUMNS_QUERY = """
    -- pg_mcp:columns
    SELECT
        n.nspname AS schema_name,
        c.relname AS table_name,
        a.attname AS column_name,
        pg_catalog.format_type(a.atttypid, a.atttypmod) AS data_type,
        NOT a.attnotnull AS is_nullable,
        pg_get_expr(ad.adbin, ad.adrelid) AS default_value,
        col_description(a.attrelid, a.attnum) AS comment
    FROM pg_attribute a
    JOIN pg_class c ON a.attrelid = c.oid
    JOIN pg_namespace n ON c.relnamespace = n.oid
    LEFT JOIN pg_attrdef ad ON a.attrelid = ad.adrelid AND a.attnum = ad.adnum
    WHERE c.relkind IN ('r', 'v')
      AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
      AND a.attnum > 0
      AND NOT a.attisdropped
    ORDER BY n.nspname, c.relname, a.attnum
"""

UNIQUE_COLUMNS_QUERY = """
    -- pg_mcp:unique_columns
    SELECT DISTINCT
        n.nspname AS schema_name,
        c.relname AS table_name,
        a.attname AS column_name
    FROM pg_constraint con
    JOIN pg_class c ON con.conrelid = c.oid
    JOIN pg_namespace n ON c.relnamespace = n.oid
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(con.conkey)
    WHERE c.relkind IN ('r', 'v')
      AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
      AND con.contype = 'u'  -- unique constraints (primary keys are separate)
"""

PRIMARY_KEYS_QUERY = """
    -- pg_mcp:primary_keys
    SELECT
        n.nspname AS schema_name,
        c.relname AS table_name,
        a.attname AS column_name
    FROM pg_index i
    JOIN pg_class c ON i.indrelid = c.oid
    JOIN pg_namespace n ON c.relnamespace = n.oid
    JOIN pg_attribute a ON a.attrelid = c.oid AND a.attnum = ANY(i.indkey)
    WHERE c.relkind IN ('r', 'v')
      AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
      AND i.indisprimary
    ORDER BY n.nspname, c.relname, array_position(i.indkey, a.attnum)
"""

FOREIGN_KEYS_QUERY = """
    -- pg_mcp:foreign_keys
    SELECT
        n.nspname AS schema_name,
        c.relname AS table_name,
        con.conname AS constraint_name,
        a.attname AS column_name,
        ref_c.relname AS referenced_table,
        ref_a.attname AS referenced_column
    FROM pg_constraint con
    JOIN pg_class c ON con.conrelid = c.oid
    JOIN pg_namespace n ON c.relnamespace = n.oid
    JOIN pg_attribute a
        ON a.attrelid = c.oid AND a.attnum = ANY(con.conkey)
    JOIN pg_class ref_c ON con.confrelid = ref_c.oid
    JOIN pg_attribute ref_a
        ON ref_a.attrelid = ref_c.oid
        AND ref_a.attnum = ANY(con.confkey)
    WHERE c.relkind IN ('r', 'v')
      AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
      AND con.contype = 'f'  -- foreign key
    ORDER BY n.nspname, c.relname, con.conname
"""

INDEXES_QUERY = """
    -- pg_mcp:indexes
    SELECT
        n.nspname AS schema_name,
        c.relname AS table_name,
        i.relname AS index_name,
        idx.indisunique AS is_unique,
        am.amname AS index_type,
        ARRAY(
            SELECT a.attname
            FROM pg_attribute a
            WHERE a.attrelid = idx.indrelid
              AND a.attnum = ANY(idx.indkey)
            ORDER BY array_position(idx.indkey, a.attnum)
        ) AS columns
    FROM pg_index idx
    JOIN pg_class i ON i.oid = idx.indexrelid
    JOIN pg_class c ON c.oid = idx.indrelid
    JOIN pg_namespace n ON c.relnamespace = n.oid
    JOIN pg_am am ON i.relam = am.oid
    WHERE c.relkind IN ('r', 'v')
      AND n.nspname NOT IN ('pg_catalog', 'information_schema', 'pg_toast')
      AND NOT idx.indisprimary  -- exclude primary key indexes
    ORDER BY n.nspname, c.relname, i.relname
"""

ENUM_TYPES_QUERY = """
    -- pg_mcp:enums
    SELECT
        n.nspname AS schema_name,
        t.typname AS type_name,
        ARRAY(
            SELECT e.enumlabel
            FROM pg_enum e
            WHERE e.enumtypid = t.oid
            ORDER BY e.enumsortorder
        ) AS values
    FROM pg_type t
    JOIN pg_namespace n ON t.typnamespace = n.oid
    WHERE t.typtype = 'e'  -- enum types only
      AND n.nspname NOT IN ('pg_catalog', 'information_schema')
    ORDER BY n.nspname, t.typname
"""


class SchemaIntrospector:
    """PostgreSQL schema introspection service.

    This class extracts complete schema metadata from a PostgreSQL database
    using set-based system catalog queries.

    Attributes:
        pool: Database connection pool.
        database_name: Name of the database being introspected.
    """

    def __init__(self, pool: Pool, database_name: str):
        """Initialize schema introspector.

        Args:
            pool: asyncpg connection pool.
            database_name: Name of the database to introspect.
        """
        self.pool = pool
        self.database_name = database_name

    async def introspect(self) -> DatabaseSchema:
        """Execute complete schema introspection.

        This method fetches all schema metadata (tables, views, columns,
        constraints, indexes, custom types) with one set-based query per
        category and assembles the results in Python.

        Returns:
            DatabaseSchema: Complete database schema information.

        Example:
            >>> introspector = SchemaIntrospector(pool, "mydb")
            >>> schema = await introspector.introspect()
            >>> print(f"Found {len(schema.tables)} tables")
        """
        async with self.pool.acquire() as conn:
            # Get PostgreSQL version
            version_result = await conn.fetchval("SELECT version()")
            version = version_result.split(",")[0] if version_result else None

            # Fetch each metadata category with a single query. The queries
            # run sequentially on purpose: asyncpg does not allow concurrent
            # operations on a single connection.
            relations = await conn.fetch(TABLES_AND_VIEWS_QUERY)
            enum_types = await self._get_enum_types(conn)
            columns_by_table = await self._get_columns_by_table(conn)
            unique_columns = await self._get_unique_columns(conn)
            primary_keys = await self._get_primary_keys(conn)
            fks_by_table = await self._get_foreign_keys_by_table(conn)
            indexes_by_table = await self._get_indexes_by_table(conn)

        return DatabaseSchema(
            database_name=self.database_name,
            tables=self._build_tables(
                relations=relations,
                columns_by_table=columns_by_table,
                unique_columns=unique_columns,
                primary_keys=primary_keys,
                fks_by_table=fks_by_table,
                indexes_by_table=indexes_by_table,
            ),
            enum_types=enum_types,
            version=version,
        )

    async def _get_enum_types(self, conn: Connection) -> list[EnumTypeInfo]:
        """Get custom ENUM type definitions.

        Args:
            conn: Database connection.

        Returns:
            list[EnumTypeInfo]: List of enum type information objects.
        """
        rows = await conn.fetch(ENUM_TYPES_QUERY)

        return [
            EnumTypeInfo(
                schema_name=row["schema_name"],
                type_name=row["type_name"],
                values=list(row["values"]),
            )
            for row in rows
        ]

    async def _get_columns_by_table(self, conn: Connection) -> dict[TableKey, list[ColumnInfo]]:
        """Get all columns of all user relations, grouped by table.

        Args:
            conn: Database connection.

        Returns:
            dict: Mapping of (schema_name, table_name) to columns in
                ``attnum`` order. ``is_unique`` is left False here and set
                during assembly from the unique-constraint set.
        """
        rows = await conn.fetch(COLUMNS_QUERY)

        columns: dict[TableKey, list[ColumnInfo]] = {}
        for row in rows:
            key: TableKey = (row["schema_name"], row["table_name"])
            columns.setdefault(key, []).append(
                ColumnInfo(
                    name=row["column_name"],
                    data_type=row["data_type"],
                    is_nullable=row["is_nullable"],
                    default_value=row["default_value"],
                    is_unique=False,
                    comment=row["comment"],
                )
            )
        return columns

    async def _get_unique_columns(self, conn: Connection) -> set[ColumnKey]:
        """Get the set of columns covered by a unique constraint.

        Args:
            conn: Database connection.

        Returns:
            set: (schema_name, table_name, column_name) triples covered by a
                unique constraint (primary keys are tracked separately).
        """
        rows = await conn.fetch(UNIQUE_COLUMNS_QUERY)
        return {(row["schema_name"], row["table_name"], row["column_name"]) for row in rows}

    async def _get_primary_keys(self, conn: Connection) -> set[ColumnKey]:
        """Get the set of primary key columns across all relations.

        Args:
            conn: Database connection.

        Returns:
            set: (schema_name, table_name, column_name) triples.
        """
        rows = await conn.fetch(PRIMARY_KEYS_QUERY)
        return {(row["schema_name"], row["table_name"], row["column_name"]) for row in rows}

    async def _get_foreign_keys_by_table(
        self, conn: Connection
    ) -> dict[TableKey, list[ForeignKeyInfo]]:
        """Get all foreign keys, grouped by referencing table.

        Args:
            conn: Database connection.

        Returns:
            dict: Mapping of (schema_name, table_name) to foreign keys in
                constraint-name order.
        """
        rows = await conn.fetch(FOREIGN_KEYS_QUERY)

        fks: dict[TableKey, list[ForeignKeyInfo]] = {}
        for row in rows:
            key: TableKey = (row["schema_name"], row["table_name"])
            fks.setdefault(key, []).append(
                ForeignKeyInfo(
                    constraint_name=row["constraint_name"],
                    column_name=row["column_name"],
                    referenced_table=row["referenced_table"],
                    referenced_column=row["referenced_column"],
                )
            )
        return fks

    async def _get_indexes_by_table(self, conn: Connection) -> dict[TableKey, list[IndexInfo]]:
        """Get all non-primary-key indexes, grouped by table.

        Args:
            conn: Database connection.

        Returns:
            dict: Mapping of (schema_name, table_name) to indexes in
                index-name order.
        """
        rows = await conn.fetch(INDEXES_QUERY)

        indexes: dict[TableKey, list[IndexInfo]] = {}
        for row in rows:
            key: TableKey = (row["schema_name"], row["table_name"])
            indexes.setdefault(key, []).append(
                IndexInfo(
                    name=row["index_name"],
                    columns=list(row["columns"]),
                    is_unique=row["is_unique"],
                    index_type=row["index_type"],
                )
            )
        return indexes

    def _build_tables(
        self,
        relations: list[Any],
        columns_by_table: dict[TableKey, list[ColumnInfo]],
        unique_columns: set[ColumnKey],
        primary_keys: set[ColumnKey],
        fks_by_table: dict[TableKey, list[ForeignKeyInfo]],
        indexes_by_table: dict[TableKey, list[IndexInfo]],
    ) -> list[TableInfo]:
        """Assemble TableInfo objects from the grouped catalog metadata.

        Tables (relkind ``'r'``) come first, then views (``'v'``); each group
        is sorted by (schema_name, table_name) to match the historical output
        ordering.

        Args:
            relations: Rows from the relations catalog query.
            columns_by_table: Columns grouped by table key.
            unique_columns: Columns covered by a unique constraint.
            primary_keys: Primary key columns.
            fks_by_table: Foreign keys grouped by table key.
            indexes_by_table: Indexes grouped by table key.

        Returns:
            list[TableInfo]: Fully populated tables followed by views.
        """
        pk_names_by_table: dict[TableKey, set[str]] = {}
        for schema_name, table_name, column_name in primary_keys:
            pk_names_by_table.setdefault((schema_name, table_name), set()).add(column_name)

        tables: list[TableInfo] = []
        views: list[TableInfo] = []
        for row in relations:
            key: TableKey = (row["schema_name"], row["table_name"])
            estimate = row["row_count_estimate"]
            table = TableInfo(
                schema_name=row["schema_name"],
                table_name=row["table_name"],
                comment=row["comment"],
                row_count_estimate=int(estimate) if estimate is not None else 0,
            )
            table.columns = table_columns = columns_by_table.get(key, [])
            for column in table_columns:
                column.is_unique = (key[0], key[1], column.name) in unique_columns
                column.is_primary_key = column.name in pk_names_by_table.get(key, set())
            table.foreign_keys = fks_by_table.get(key, [])
            table.indexes = indexes_by_table.get(key, [])

            if row["relkind"] == "r":
                tables.append(table)
            else:
                views.append(table)

        tables.sort(key=lambda t: (t.schema_name, t.table_name))
        views.sort(key=lambda t: (t.schema_name, t.table_name))
        return tables + views
