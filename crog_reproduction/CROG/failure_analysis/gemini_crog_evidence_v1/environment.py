from __future__ import annotations

import math
import os
import stat
from pathlib import Path
from typing import Mapping


GEMINI_ENV_KEYS = (
    "GEMINI_API_KEY",
    "GEMINI_MAX_SPEND_USD",
    "GEMINI_MAX_CONCURRENCY",
    "GEMINI_ER2_COST_CAP_PER_REQUEST_USD",
)


def load_private_env(path: str | Path, *, override: bool = False) -> dict[str, str]:
    """Load only the four experiment variables from a mode-0600 env file.

    This deliberately implements a tiny non-shell parser: values are never
    expanded, executed, returned in logs, or copied to experiment artifacts.
    The caller may inspect only key presence; the API key value must remain in
    the process environment.
    """

    env_path = Path(path)
    mode = stat.S_IMODE(env_path.stat().st_mode)
    if mode != 0o600:
        raise PermissionError(f"private Gemini env must have mode 0600, observed {mode:04o}")
    allowed = set(GEMINI_ENV_KEYS)
    loaded: dict[str, str] = {}
    for line_number, raw_line in enumerate(env_path.read_text(encoding="utf-8").splitlines(), start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        if "=" not in line:
            raise ValueError(f"invalid env assignment at line {line_number}")
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in allowed:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if not value:
            raise ValueError(f"{key} must not be empty")
        if override or key not in os.environ:
            os.environ[key] = value
        loaded[key] = "SET"
    return loaded


def validate_gemini_environment(values: Mapping[str, str] | None = None) -> dict[str, object]:
    source = os.environ if values is None else values
    missing = [key for key in GEMINI_ENV_KEYS if not str(source.get(key, "")).strip()]
    if missing:
        raise RuntimeError(f"missing required Gemini environment variables: {', '.join(missing)}")
    max_spend = float(source["GEMINI_MAX_SPEND_USD"])
    concurrency = int(source["GEMINI_MAX_CONCURRENCY"])
    er2_cap = float(source["GEMINI_ER2_COST_CAP_PER_REQUEST_USD"])
    if not math.isfinite(max_spend) or not (max_spend > 0.0):
        raise ValueError("GEMINI_MAX_SPEND_USD must be finite and positive")
    if concurrency < 1:
        raise ValueError("GEMINI_MAX_CONCURRENCY must be a positive integer")
    if not math.isfinite(er2_cap) or not (er2_cap > 0.0):
        raise ValueError("GEMINI_ER2_COST_CAP_PER_REQUEST_USD must be finite and positive")
    return {
        "gemini_api_key": "SET",
        "gemini_max_spend_usd": max_spend,
        "gemini_max_concurrency": concurrency,
        "gemini_er2_cost_cap_per_request_usd": er2_cap,
    }
