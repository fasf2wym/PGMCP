"""Composable security rules for SQL validation.

Each rule inspects a parsed sqlglot expression and returns an error message
when it violates the read-only policy, or ``None`` when it passes. Rules are
plain objects satisfying :class:`ValidationRule`, so the validator composes
them as a list (open/closed: adding a check means adding a rule, not editing
the validator).

All rule bodies and error messages are migrated verbatim from the original
``SQLValidator`` check methods.
"""

from typing import Protocol

from sqlglot import exp

# Statement types allowed at the top level (including set operations).
ALLOWED_STATEMENT_TYPES = {exp.Select, exp.Union, exp.Intersect, exp.Except}

# Statement types forbidden anywhere in the query.
FORBIDDEN_STATEMENT_TYPES = {
    exp.Insert,
    exp.Update,
    exp.Delete,
    exp.Drop,
    exp.Create,
    exp.Alter,
    exp.Grant,
    exp.Revoke,
    exp.Set,
    exp.Command,
    exp.Use,
    exp.Merge,
}

# Statement types allowed inside CTE bodies and parenthesized subqueries
# (nested WITH clauses are read-only, matching the top-level policy).
NESTED_ALLOWED_STATEMENT_TYPES = (exp.Select, exp.Union, exp.Intersect, exp.Except, exp.With)


class ValidationRule(Protocol):
    """A single SQL security check.

    Implementations receive both the full parsed statement and the main
    query (the statement with a top-level CTE unwrapped), because the
    statement-type policy applies to the main query only while function,
    table, column, and subquery checks must traverse the full statement
    including CTE bodies.
    """

    def check(self, statement: exp.Expression, main_query: exp.Expression) -> str | None:
        """Return an error message if the rule is violated, else ``None``.

        Args:
            statement: The full parsed SQL statement.
            main_query: The statement with a top-level ``WITH`` unwrapped.

        Returns:
            str | None: Violation message, or None when the check passes.
        """
        ...


class StatementTypeRule:
    """Restrict statements to read-only types (SELECT and set operations).

    Forbidden statement types are reported first; any other non-allowed
    type (e.g. TRUNCATE) falls through to the generic allowed-types message.
    """

    def check(self, statement: exp.Expression, main_query: exp.Expression) -> str | None:
        """Check the main query's statement type against the policy.

        Args:
            statement: The full parsed SQL statement (unused).
            main_query: The statement with a top-level ``WITH`` unwrapped.

        Returns:
            str | None: Violation message, or None when the check passes.
        """
        # Check for forbidden statement types
        for forbidden_type in FORBIDDEN_STATEMENT_TYPES:
            if isinstance(main_query, forbidden_type):
                stmt_name = forbidden_type.__name__.upper()
                return f"{stmt_name} statements are not allowed. Only SELECT queries are permitted."

        # Ensure statement is an allowed type (SELECT or set operations)
        if not isinstance(main_query, tuple(ALLOWED_STATEMENT_TYPES)):
            stmt_type = type(main_query).__name__
            return f"Statement type {stmt_type} is not allowed. Only SELECT queries are permitted."

        return None


class DangerousFunctionRule:
    """Reject queries calling blocked or dangerous functions."""

    def __init__(self, blocked_functions: frozenset[str]) -> None:
        """Initialize the rule.

        Args:
            blocked_functions: Lowercase function names to reject.
        """
        self.blocked_functions = blocked_functions

    def check(self, statement: exp.Expression, main_query: exp.Expression) -> str | None:
        """Check for use of blocked/dangerous functions.

        Args:
            statement: The full parsed SQL statement.
            main_query: The statement with a top-level ``WITH`` unwrapped (unused).

        Returns:
            str | None: Violation message, or None when the check passes.
        """
        # Find all function calls in the query
        for func in statement.find_all(exp.Func):
            func_name = func.name.lower() if func.name else ""

            if func_name in self.blocked_functions:
                return f"Function '{func_name}' is blocked for security reasons"

        return None


class BlockedTableRule:
    """Reject queries referencing configured blocked tables."""

    def __init__(self, blocked_tables: frozenset[str]) -> None:
        """Initialize the rule.

        Args:
            blocked_tables: Lowercase table names to reject.
        """
        self.blocked_tables = blocked_tables

    def check(self, statement: exp.Expression, main_query: exp.Expression) -> str | None:
        """Check for access to blocked tables.

        Args:
            statement: The full parsed SQL statement.
            main_query: The statement with a top-level ``WITH`` unwrapped (unused).

        Returns:
            str | None: Violation message, or None when the check passes.
        """
        if not self.blocked_tables:
            return None

        # Find all table references
        for table in statement.find_all(exp.Table):
            table_name = table.name.lower() if table.name else ""

            if table_name in self.blocked_tables:
                return f"Access to table '{table_name}' is not allowed"

        return None


class BlockedColumnRule:
    """Reject queries referencing configured blocked columns."""

    def __init__(self, blocked_columns: frozenset[str]) -> None:
        """Initialize the rule.

        Args:
            blocked_columns: Lowercase column names to reject.
        """
        self.blocked_columns = blocked_columns

    def check(self, statement: exp.Expression, main_query: exp.Expression) -> str | None:
        """Check for access to blocked columns.

        Args:
            statement: The full parsed SQL statement.
            main_query: The statement with a top-level ``WITH`` unwrapped (unused).

        Returns:
            str | None: Violation message, or None when the check passes.
        """
        if not self.blocked_columns:
            return None

        # Find all column references
        for column in statement.find_all(exp.Column):
            column_name = column.name.lower() if column.name else ""

            # Check for exact match
            if column_name in self.blocked_columns:
                return f"Access to column '{column_name}' is not allowed"

            # Check for qualified column names (table.column)
            if column.table:
                qualified_name = f"{column.table.lower()}.{column_name}"
                if qualified_name in self.blocked_columns:
                    return f"Access to column '{qualified_name}' is not allowed"

        return None


def check_subquery_safety(statement: exp.Expression) -> str | None:
    """Check that all subqueries and CTE bodies contain read-only queries.

    Data-modifying CTEs (e.g. ``WITH d AS (DELETE ...) SELECT * FROM d``)
    parse as a plain Select wrapping a With, so CTE bodies need an
    explicit check in addition to parenthesized subqueries.

    Args:
        statement: Parsed SQL statement.

    Returns:
        str | None: Error message if a check fails, None otherwise.
    """
    # Check all nested queries (CTE bodies and parenthesized subqueries)
    nested = [cte.this for cte in statement.find_all(exp.CTE) if cte.this] + [
        subquery.this for subquery in statement.find_all(exp.Subquery) if subquery.this
    ]

    for inner_stmt in nested:
        # Check if the inner statement is a forbidden type
        for forbidden_type in FORBIDDEN_STATEMENT_TYPES:
            if isinstance(inner_stmt, forbidden_type):
                stmt_name = forbidden_type.__name__.upper()
                return f"{stmt_name} statements in subqueries are not allowed"

        # Ensure it's a read-only query (set operations and nested
        # WITH clauses are read-only, matching the top-level policy)
        if not isinstance(inner_stmt, NESTED_ALLOWED_STATEMENT_TYPES):
            return "Subqueries must contain only SELECT statements"

    return None


class SubquerySafetyRule:
    """Ensure CTE bodies and subqueries contain only read-only queries."""

    def check(self, statement: exp.Expression, main_query: exp.Expression) -> str | None:
        """Check nested queries for read-only compliance.

        Args:
            statement: The full parsed SQL statement.
            main_query: The statement with a top-level ``WITH`` unwrapped (unused).

        Returns:
            str | None: Violation message, or None when the check passes.
        """
        return check_subquery_safety(statement)
