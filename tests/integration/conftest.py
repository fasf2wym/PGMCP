"""Skip integration tests when required infrastructure is unavailable.

These tests exercise the real server lifespan, which needs an OpenAI API
key and a reachable PostgreSQL instance. Autouse fixtures here apply only
to tests in this directory.
"""

import os

import pytest


@pytest.fixture(autouse=True)
def _require_infra() -> None:
    if not os.environ.get("OPENAI_API_KEY"):
        pytest.skip("integration tests require OPENAI_API_KEY and a live PostgreSQL")
