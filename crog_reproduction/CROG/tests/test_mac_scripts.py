import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def test_mac_helper_scripts_expose_cli_help():
    for relative_path in (
        "scripts/check_mps.py",
        "scripts/inspect_ocid_vlg.py",
        "scripts/monitor_crog_training_diary.py",
        "scripts/visualize_ocid_vlg_sample.py",
    ):
        result = subprocess.run(
            [sys.executable, str(REPO_ROOT / relative_path), "--help"],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0, result.stderr
        assert "usage:" in result.stdout.lower()
