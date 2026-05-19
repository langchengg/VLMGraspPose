from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import yaml


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> dict:
    with open(path, "r") as f:
        data = yaml.safe_load(f) or {}
    return data


def save_json(path: Path, data: Any) -> None:
    ensure_dir(path.parent)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)


def load_json(path: Path) -> Any:
    with open(path, "r") as f:
        return json.load(f)


def append_csv(path: Path, row: dict, fieldnames: list[str] | None = None) -> None:
    ensure_dir(path.parent)
    exists = path.exists()
    fields = fieldnames or list(row.keys())
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow({k: row.get(k) for k in fields})


def write_csv(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("")
        return
    fields = sorted({k for row in rows for k in row.keys()})
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
