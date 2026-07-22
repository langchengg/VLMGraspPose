#!/usr/bin/env python3
"""Check the isolated official GQ-CNN candidate-generation installation."""

from __future__ import annotations

import importlib
import json
import platform
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
GQCNN_ROOT = REPO_ROOT / "third_party" / "gqcnn-official"
sys.path.insert(0, str(GQCNN_ROOT))


def main() -> int:
    checks: dict[str, object] = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "machine": platform.machine(),
        "gqcnn_root": str(GQCNN_ROOT),
        "imports": {},
    }
    failed = False
    for name in ("numpy", "scipy", "autolab_core", "perception", "visualization"):
        try:
            module = importlib.import_module(name)
            checks["imports"][name] = {
                "ok": True,
                "version": getattr(module, "__version__", "unknown"),
            }
        except Exception as error:  # pragma: no cover - diagnostic CLI
            failed = True
            checks["imports"][name] = {
                "ok": False,
                "error": f"{type(error).__name__}: {error}",
            }
    try:
        import gqcnn
        from gqcnn.grasping import AntipodalDepthImageGraspSampler
        from autolab_core import (BinaryImage, CameraIntrinsics, ColorImage,
                                  DepthImage, RgbdImage)

        checks["imports"]["gqcnn"] = {
            "ok": True,
            "version": gqcnn.__version__,
            "official_antipodal_sampler": AntipodalDepthImageGraspSampler.__name__,
            "scoring_import_available": bool(gqcnn.SCORING_IMPORT_AVAILABLE),
        }
        checks["autolab_image_types"] = [
            cls.__name__ for cls in
            (BinaryImage, DepthImage, ColorImage, RgbdImage, CameraIntrinsics)
        ]
    except Exception as error:  # pragma: no cover - diagnostic CLI
        failed = True
        checks["imports"]["gqcnn"] = {
            "ok": False,
            "error": f"{type(error).__name__}: {error}",
        }
    print(json.dumps(checks, indent=2, sort_keys=True))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
