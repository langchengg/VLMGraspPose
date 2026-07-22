"""Rebuild the Markdown/HTML report from completed analysis artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.analysis.analysis_report import build_report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-output", type=Path, required=True)
    args = parser.parse_args()
    executive_path = args.analysis_output / "report" / "executive_summary.json"
    if not executive_path.is_file():
        parser.error(f"missing analysis artifact: {executive_path}")
    executive = json.loads(executive_path.read_text(encoding="utf-8"))
    manifest_path = args.analysis_output / "data" / "analysis_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    markdown, html = build_report(
        args.analysis_output,
        integrity=manifest.get("integrity", {}),
        executive=executive,
    )
    print(json.dumps({"report_md": str(markdown), "report_html": str(html)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
