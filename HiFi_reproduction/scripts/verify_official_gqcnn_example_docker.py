#!/usr/bin/env python3
"""Run the untouched official policy example and assert its final q is finite."""

from __future__ import annotations

import math
import re
import runpy
import sys

from visualization import Visualizer2D


captured_titles = []
original_title = Visualizer2D.title


def capture_title(text, *args, **kwargs):
    captured_titles.append(str(text))
    print("OFFICIAL_EXAMPLE_TITLE %s" % text)
    return original_title(text, *args, **kwargs)


Visualizer2D.title = staticmethod(capture_title)
sys.argv = [
    "/opt/gqcnn/examples/policy.py",
    "GQCNN-2.1",
    "--model_dir",
    "/models",
    "--config_filename",
    "/opt/gqcnn/cfg/examples/replication/dex-net_2.1.yaml",
]
runpy.run_path("/opt/gqcnn/examples/policy.py", run_name="__main__")

matches = []
for title in captured_titles:
    match = re.search(r"Q=([-+0-9.eE]+)", title)
    if match:
        matches.append(float(match.group(1)))
if not matches or not all(math.isfinite(value) for value in matches):
    raise RuntimeError("official policy example did not expose a finite final q-value")
print("OFFICIAL_EXAMPLE_FINITE_Q_OK q=%s" % matches[-1])
