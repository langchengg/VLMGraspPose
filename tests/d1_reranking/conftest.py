"""D1 test-process native-runtime initialization."""

from __future__ import annotations

import os

# Match the frozen single-thread CPU execution contract before either native
# runtime is initialized by test-module imports.
for _name in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_name, "1")

# Several test modules import PyTorch at collection time, before the P12/R5
# modules are collected.  Production D1 workers initialize LightGBM first for
# the same macOS OpenMP-safety reason; mirror that ordering in the shared test
# process.  LightGBM remains optional for contract-only test environments.
try:  # pragma: no cover - depends on the selected test interpreter
    import lightgbm as _lightgbm  # noqa: F401
except ModuleNotFoundError:  # pragma: no cover
    _lightgbm = None
