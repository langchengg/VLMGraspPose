"""Run memory-heavy VGN/analysis test modules in isolated Python processes.

PyTorch, Open3D, pandas, and scikit-learn retain native allocations after a
test module finishes.  On the reference macOS machine, collecting and running
all modules in one interpreter can therefore be killed by the OS despite each
test passing.  This orchestrator keeps the documented single pytest command
while preserving per-module process isolation and propagating exact failures.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]
TEST_MODULES = (
    "tests/test_vgn_integration.py",
    "tests/test_candidate_multiplicity.py",
    "tests/test_reranking_opportunity.py",
    "tests/test_gt_oracle_bounds.py",
    "tests/test_failure_taxonomy.py",
    "tests/test_analysis_pipeline.py",
)


@pytest.mark.parametrize("test_module", TEST_MODULES)
def test_module_in_isolated_process(test_module: str) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "-q",
            "-c",
            "pytest-vgn.ini",
            test_module,
        ],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    )
    assert completed.returncode == 0, (
        f"isolated pytest failed for {test_module}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

