"""Skip e2e tests when required infrastructure is unavailable.

These tests boot the real MCP server lifespan, which needs an OpenAI API
key and a reachable PostgreSQL instance. Autouse fixtures here apply only
to tests in this directory.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _require_infra() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("e2e tests require OPENAI_API_KEY and a live PostgreSQL")
