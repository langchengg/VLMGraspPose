from __future__ import annotations

import sys

from run_dataset import main


if __name__ == "__main__":
    argv = sys.argv[1:]
    if "--target-source" not in argv:
        argv = ["--target-source", "oracle", *argv]
    main(argv)
