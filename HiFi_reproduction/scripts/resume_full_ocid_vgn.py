#!/usr/bin/env python3
"""Resume the full OCID-VLG VGN runner (resume is also its default)."""

from __future__ import annotations

import sys

from .run_full_ocid_vgn import main


if __name__ == "__main__":
    arguments = list(sys.argv[1:])
    if "--resume" not in arguments and "--no-resume" not in arguments:
        arguments.append("--resume")
    sys.exit(main(arguments))
