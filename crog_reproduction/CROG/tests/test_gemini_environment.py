from __future__ import annotations

import os

import pytest

from failure_analysis.gemini_crog_evidence_v1.environment import (
    load_private_env,
    validate_gemini_environment,
)


def test_private_env_loads_only_allowlisted_keys(tmp_path, monkeypatch):
    path = tmp_path / ".env"
    path.write_text(
        "GEMINI_API_KEY=not-a-real-key\n"
        "GEMINI_MAX_SPEND_USD=1000000\n"
        "GEMINI_MAX_CONCURRENCY=1\n"
        "GEMINI_ER2_COST_CAP_PER_REQUEST_USD=1.0\n"
        "UNRELATED=ignored\n",
        encoding="utf-8",
    )
    path.chmod(0o600)
    for key in (
        "GEMINI_API_KEY",
        "GEMINI_MAX_SPEND_USD",
        "GEMINI_MAX_CONCURRENCY",
        "GEMINI_ER2_COST_CAP_PER_REQUEST_USD",
    ):
        monkeypatch.delenv(key, raising=False)
    loaded = load_private_env(path)
    assert set(loaded) == {
        "GEMINI_API_KEY",
        "GEMINI_MAX_SPEND_USD",
        "GEMINI_MAX_CONCURRENCY",
        "GEMINI_ER2_COST_CAP_PER_REQUEST_USD",
    }
    assert "UNRELATED" not in os.environ
    assert validate_gemini_environment()["gemini_max_concurrency"] == 1


def test_private_env_rejects_open_permissions(tmp_path):
    path = tmp_path / ".env"
    path.write_text("GEMINI_API_KEY=x\n", encoding="utf-8")
    path.chmod(0o644)
    with pytest.raises(PermissionError):
        load_private_env(path)


def test_environment_requires_finite_positive_budget_fields():
    with pytest.raises((RuntimeError, ValueError)):
        validate_gemini_environment({})
    with pytest.raises(ValueError):
        validate_gemini_environment(
            {
                "GEMINI_API_KEY": "x",
                "GEMINI_MAX_SPEND_USD": "0",
                "GEMINI_MAX_CONCURRENCY": "1",
                "GEMINI_ER2_COST_CAP_PER_REQUEST_USD": "1",
            }
        )
