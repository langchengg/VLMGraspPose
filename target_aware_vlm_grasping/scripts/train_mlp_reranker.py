from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from scoring.train_mlp import save_rule_initialized_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a CPU-only MLP reranker checkpoint.")
    parser.add_argument("--output", type=Path, default=Path("outputs/checkpoints/mlp_rule_initialized.npz"))
    args = parser.parse_args()
    path = args.output if args.output.is_absolute() else (ROOT / args.output)
    print(save_rule_initialized_checkpoint(path))


if __name__ == "__main__":
    main()
