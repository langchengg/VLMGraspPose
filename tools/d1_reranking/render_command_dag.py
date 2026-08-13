"""Render or validate the static D1 P0--P17 command DAG."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from d1_reranking.command_dag import audit_catalog, render_markdown  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--format", choices=("markdown", "json"), default="markdown")
    parser.add_argument(
        "--output", type=Path, help="write atomically instead of stdout"
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="validate catalog/tool consistency; blockers are reported but do not fail",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="exit nonzero while any execution blocker or missing CLI remains",
    )
    return parser.parse_args()


def _atomic_text(path: Path, text: str) -> None:
    destination = path.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, destination)


def main() -> int:
    args = parse_args()
    audit = audit_catalog(ROOT)
    text = (
        render_markdown(audit)
        if args.format == "markdown"
        else json.dumps(audit, indent=2, sort_keys=True) + "\n"
    )
    if args.output is None:
        sys.stdout.write(text)
    else:
        _atomic_text(args.output, text)
    if args.strict and audit["status"] != "READY":
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
