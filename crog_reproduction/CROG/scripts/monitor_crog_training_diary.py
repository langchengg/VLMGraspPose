#!/usr/bin/env python3
"""Read-only CROG training monitor for Mac/MPS runs.

This script is intentionally independent from the training loop. It parses
existing log/timing/resource files and writes a human-readable training diary.
It never imports torch, never opens checkpoint files, and never sends signals
except signal 0 for an optional PID existence check.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any


FLOAT_RE = r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?|nan|inf|-inf"

TRAIN_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}).*?"
    rf"Training: Epoch=\[(?P<epoch>\d+)/(?P<epochs>\d+)\]\s+"
    rf"\[\s*(?P<step>\d+)/(?P<total>\d+)\]\s+"
    rf"Batch=(?P<batch>{FLOAT_RE}) \((?P<batch_avg>{FLOAT_RE})\)\s+"
    rf"Data=(?P<data>{FLOAT_RE}) \((?P<data_avg>{FLOAT_RE})\)\s+"
    rf"Lr=(?P<lr>{FLOAT_RE})\s+"
    rf"Loss=(?P<loss>{FLOAT_RE}) \((?P<loss_avg>{FLOAT_RE})\).*?"
    rf"IoU=(?P<iou>{FLOAT_RE}) \((?P<iou_avg>{FLOAT_RE})\)\s+"
    rf"Prec@50=(?P<pr50>{FLOAT_RE}) \((?P<pr50_avg>{FLOAT_RE})\)",
    re.IGNORECASE,
)

EVAL_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}).*?"
    rf"Evaluation: Epoch=\[(?P<epoch>\d+)/(?P<epochs>\d+)\]\s+"
    rf"IoU=(?P<iou>{FLOAT_RE})\s+"
    rf"J_index@1:\s+(?P<j1>{FLOAT_RE})\s+"
    rf"J_index@5:\s+(?P<j5>{FLOAT_RE})\s+"
    rf"Pr@50:\s+(?P<p50>{FLOAT_RE})\s+"
    rf"Pr@60:\s+(?P<p60>{FLOAT_RE})\s+"
    rf"Pr@70:\s+(?P<p70>{FLOAT_RE})\s+"
    rf"Pr@80:\s+(?P<p80>{FLOAT_RE})\s+"
    rf"Pr@90:\s+(?P<p90>{FLOAT_RE})",
    re.IGNORECASE,
)

MID_EPOCH_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}).*?"
    r"Saved mid-epoch checkpoint .*? at epoch (?P<epoch>\d+) "
    r"iteration (?P<step>\d+)/(?P<total>\d+)"
)

MPS_RE = re.compile(
    rf"(?P<ts>\d{{4}}-\d{{2}}-\d{{2}} \d{{2}}:\d{{2}}:\d{{2}}).*?"
    rf"MPS memory step=(?P<step>\d+)/(?P<total>\d+) "
    rf"allocated=(?P<allocated_mb>{FLOAT_RE})MB driver=(?P<driver_mb>{FLOAT_RE})MB",
    re.IGNORECASE,
)

TIMING_RE = re.compile(r"TIMING_SUMMARY\s+(?P<json>\{.*?\})(?:\n|$)")
WARNING_RE = re.compile(
    r"(Traceback|RuntimeError|out of memory|MemoryError|Loss=(?:nan|inf|-inf)|ERROR)",
    re.IGNORECASE,
)


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except FileNotFoundError:
        return ""


def to_number(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def int_or_none(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except ValueError:
        return None


def compact_float(value: Any, digits: int = 3) -> str:
    if value is None:
        return "-"
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return f"{value:.{digits}f}".rstrip("0").rstrip(".")
    return str(value)


def seconds_to_hms(seconds: Any) -> str:
    if not isinstance(seconds, (int, float)):
        return "-"
    seconds = int(round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"


def bytes_to_gib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value / (1024 ** 3):.2f}"


def mb_to_gib(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "-"
    return f"{value / 1024:.2f}"


def file_info(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path), "exists": False}
    stat = path.stat()
    return {
        "path": str(path),
        "exists": True,
        "size_bytes": stat.st_size,
        "size_gib": stat.st_size / (1024 ** 3),
        "mtime": datetime.fromtimestamp(stat.st_mtime).astimezone().isoformat(timespec="seconds"),
    }


def process_is_running(pid: int | None) -> bool | None:
    if pid is None:
        return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True


def parse_match(match: re.Match[str]) -> dict[str, Any]:
    parsed: dict[str, Any] = {}
    for key, value in match.groupdict().items():
        if key == "ts":
            parsed[key] = value
        elif key in {"epoch", "epochs", "step", "total"}:
            parsed[key] = int_or_none(value)
        else:
            parsed[key] = to_number(value)
    return parsed


def parse_console_log(text: str) -> dict[str, Any]:
    train_entries = [parse_match(match) for match in TRAIN_RE.finditer(text)]
    eval_entries = [parse_match(match) for match in EVAL_RE.finditer(text)]
    mid_epoch_entries = [parse_match(match) for match in MID_EPOCH_RE.finditer(text)]
    mps_entries = [parse_match(match) for match in MPS_RE.finditer(text)]
    timing_summaries: list[dict[str, Any]] = []
    for match in TIMING_RE.finditer(text):
        try:
            timing_summaries.append(json.loads(match.group("json")))
        except json.JSONDecodeError:
            continue

    warnings: list[str] = []
    for line in text.splitlines():
        if WARNING_RE.search(line):
            warnings.append(line.strip()[:500])

    return {
        "latest_train": train_entries[-1] if train_entries else None,
        "eval_entries": eval_entries,
        "mid_epoch_entries": mid_epoch_entries,
        "mps_entries": mps_entries,
        "timing_summaries_from_log": timing_summaries,
        "warnings": warnings[-20:],
    }


def load_timing_files(run_dir: Path) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for timing_path in sorted(run_dir.glob("timing_epoch_*.json")):
        try:
            data = json.loads(timing_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        data.setdefault("epoch", int_or_none(timing_path.stem.rsplit("_", 1)[-1]))
        data["path"] = str(timing_path)
        entries.append(data)
    return entries


def parse_cpu_samples(run_dir: Path) -> dict[str, Any]:
    sample_files = sorted(run_dir.glob("cpu_ps_samples_pid*.txt"))
    if not sample_files:
        return {"sample_files": [], "latest": None, "count": 0}

    samples: list[dict[str, Any]] = []
    for sample_file in sample_files:
        for line in read_text(sample_file).splitlines():
            if not line or line.startswith("timestamp "):
                continue
            parts = line.split(maxsplit=8)
            if len(parts) < 9:
                continue
            timestamp, pid, ppid, cpu_pct, mem_pct, rss_kb, vsz_kb, etime, command = parts
            samples.append(
                {
                    "timestamp": timestamp,
                    "pid": int_or_none(pid),
                    "ppid": int_or_none(ppid),
                    "cpu_pct": to_number(cpu_pct),
                    "mem_pct": to_number(mem_pct),
                    "rss_kb": int_or_none(rss_kb),
                    "vsz_kb": int_or_none(vsz_kb),
                    "etime": etime,
                    "command": command,
                    "source": str(sample_file),
                }
            )

    cpu_values = [sample["cpu_pct"] for sample in samples if isinstance(sample.get("cpu_pct"), float)]
    rss_values = [sample["rss_kb"] for sample in samples if isinstance(sample.get("rss_kb"), int)]
    latest = samples[-1] if samples else None
    return {
        "sample_files": [str(path) for path in sample_files],
        "latest": latest,
        "count": len(samples),
        "cpu_avg_pct": sum(cpu_values) / len(cpu_values) if cpu_values else None,
        "cpu_max_pct": max(cpu_values) if cpu_values else None,
        "rss_avg_kb": sum(rss_values) / len(rss_values) if rss_values else None,
        "rss_max_kb": max(rss_values) if rss_values else None,
    }


def collect_state(run_dir: Path, pid: int | None) -> dict[str, Any]:
    console_candidates = sorted(run_dir.glob("*console.log"))
    console_text = "\n".join(read_text(path) for path in console_candidates)
    parsed_console = parse_console_log(console_text)
    timing_entries = load_timing_files(run_dir)

    eval_by_epoch = {
        entry.get("epoch"): entry
        for entry in parsed_console["eval_entries"]
        if entry.get("epoch") is not None
    }
    timing_by_epoch = {
        entry.get("epoch"): entry
        for entry in timing_entries
        if entry.get("epoch") is not None
    }
    completed_epochs = []
    for epoch in sorted(set(eval_by_epoch) | set(timing_by_epoch)):
        completed_epochs.append(
            {
                "epoch": epoch,
                "timing": timing_by_epoch.get(epoch, {}),
                "evaluation": eval_by_epoch.get(epoch, {}),
            }
        )

    checkpoint_names = (
        "last_model.pth",
        "best_iou_model.pth",
        "best_jindex_model.pth",
        "mid_epoch_model.pth",
    )

    latest_mps = parsed_console["mps_entries"][-1] if parsed_console["mps_entries"] else None
    latest_mid_epoch = (
        parsed_console["mid_epoch_entries"][-1] if parsed_console["mid_epoch_entries"] else None
    )

    return {
        "generated_at": now_iso(),
        "run_dir": str(run_dir),
        "pid": pid,
        "pid_running": process_is_running(pid),
        "console_logs": [str(path) for path in console_candidates],
        "latest_train": parsed_console["latest_train"],
        "latest_mps": latest_mps,
        "latest_mid_epoch": latest_mid_epoch,
        "completed_epochs": completed_epochs,
        "timing_files": timing_entries,
        "cpu_samples": parse_cpu_samples(run_dir),
        "checkpoints": {name: file_info(run_dir / name) for name in checkpoint_names},
        "warnings": parsed_console["warnings"],
    }


def latest_progress_line(latest_train: dict[str, Any] | None) -> str:
    if not latest_train:
        return "未解析到训练进度行。"
    epoch = latest_train.get("epoch")
    epochs = latest_train.get("epochs")
    step = latest_train.get("step")
    total = latest_train.get("total")
    percent = "-"
    if isinstance(step, int) and isinstance(total, int) and total:
        percent = f"{step / total * 100:.1f}%"
    return (
        f"Epoch {epoch}/{epochs}, iter {step}/{total} ({percent}), "
        f"loss {compact_float(latest_train.get('loss'))} "
        f"(avg {compact_float(latest_train.get('loss_avg'))}), "
        f"IoU avg {compact_float(latest_train.get('iou_avg'))}, "
        f"Pr@50 avg {compact_float(latest_train.get('pr50_avg'))}, "
        f"lr {compact_float(latest_train.get('lr'), digits=6)}, "
        f"log time {latest_train.get('ts', '-')}"
    )


def render_markdown(state: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# CROG Mac/MPS Training Diary")
    lines.append("")
    lines.append(f"- Last updated: {state['generated_at']}")
    lines.append(f"- Run directory: `{state['run_dir']}`")
    lines.append(f"- Training PID: `{state.get('pid') or '-'}`")
    running = state.get("pid_running")
    lines.append(f"- PID running: `{running if running is not None else 'unknown'}`")
    lines.append(f"- Console logs parsed: `{len(state.get('console_logs', []))}`")
    lines.append("")

    lines.append("## Latest progress")
    lines.append("")
    lines.append(f"- {latest_progress_line(state.get('latest_train'))}")
    latest_mps = state.get("latest_mps")
    if latest_mps:
        lines.append(
            "- Latest MPS log: "
            f"step {latest_mps.get('step')}/{latest_mps.get('total')}, "
            f"allocated {compact_float(latest_mps.get('allocated_mb'))} MB, "
            f"driver {compact_float(latest_mps.get('driver_mb'))} MB, "
            f"log time {latest_mps.get('ts')}"
        )
    latest_mid = state.get("latest_mid_epoch")
    if latest_mid:
        lines.append(
            "- Latest mid-epoch checkpoint log: "
            f"epoch {latest_mid.get('epoch')} iter {latest_mid.get('step')}/"
            f"{latest_mid.get('total')} at {latest_mid.get('ts')}"
        )
    lines.append("")

    lines.append("## Completed epochs")
    lines.append("")
    completed = state.get("completed_epochs", [])
    if not completed:
        lines.append("No completed epoch timing/evaluation records yet.")
    else:
        lines.append(
            "| epoch | train | val | total | sec/iter | finite loss | loss min/max | "
            "IoU | J@1 | J@Any | P50 | P60 | P70 | P80 | P90 | MPS alloc peak GiB | "
            "MPS driver peak GiB | ckpt GiB |"
        )
        lines.append(
            "|---:|---:|---:|---:|---:|:---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"
        )
        for entry in completed:
            timing = entry.get("timing", {})
            evaluation = entry.get("evaluation", {})
            memory = timing.get("memory", {}) if isinstance(timing.get("memory"), dict) else {}
            loss_min = compact_float(timing.get("loss_min"))
            loss_max = compact_float(timing.get("loss_max"))
            lines.append(
                "| "
                f"{entry.get('epoch')} | "
                f"{seconds_to_hms(timing.get('train_seconds'))} | "
                f"{seconds_to_hms(timing.get('validation_seconds'))} | "
                f"{seconds_to_hms(timing.get('total_seconds'))} | "
                f"{compact_float(timing.get('average_seconds_per_iteration'))} | "
                f"{timing.get('loss_finite', '-')} | "
                f"{loss_min}/{loss_max} | "
                f"{compact_float(evaluation.get('iou'))} | "
                f"{compact_float(evaluation.get('j1'))} | "
                f"{compact_float(evaluation.get('j5'))} | "
                f"{compact_float(evaluation.get('p50'))} | "
                f"{compact_float(evaluation.get('p60'))} | "
                f"{compact_float(evaluation.get('p70'))} | "
                f"{compact_float(evaluation.get('p80'))} | "
                f"{compact_float(evaluation.get('p90'))} | "
                f"{bytes_to_gib(memory.get('allocated_peak_bytes'))} | "
                f"{bytes_to_gib(memory.get('driver_peak_bytes'))} | "
                f"{bytes_to_gib(timing.get('checkpoint_size_bytes'))} |"
            )
    lines.append("")

    lines.append("## CPU/RSS samples")
    lines.append("")
    cpu = state.get("cpu_samples", {})
    latest_cpu = cpu.get("latest")
    lines.append(f"- Sample count: `{cpu.get('count', 0)}`")
    if latest_cpu:
        rss_mb = latest_cpu.get("rss_kb") / 1024 if isinstance(latest_cpu.get("rss_kb"), int) else None
        lines.append(
            "- Latest sample: "
            f"{latest_cpu.get('timestamp')}, CPU {compact_float(latest_cpu.get('cpu_pct'))}%, "
            f"MEM {compact_float(latest_cpu.get('mem_pct'))}%, "
            f"RSS {compact_float(rss_mb)} MB, elapsed {latest_cpu.get('etime')}"
        )
    if cpu.get("count"):
        lines.append(
            "- Aggregate: "
            f"CPU avg {compact_float(cpu.get('cpu_avg_pct'))}%, "
            f"CPU max {compact_float(cpu.get('cpu_max_pct'))}%, "
            f"RSS avg {compact_float((cpu.get('rss_avg_kb') or 0) / 1024)} MB, "
            f"RSS max {compact_float((cpu.get('rss_max_kb') or 0) / 1024)} MB"
        )
    lines.append("")

    lines.append("## Checkpoints")
    lines.append("")
    lines.append("| file | exists | size GiB | modified |")
    lines.append("|---|:---:|---:|---|")
    for name, info in state.get("checkpoints", {}).items():
        lines.append(
            f"| `{name}` | {info.get('exists')} | "
            f"{compact_float(info.get('size_gib'))} | {info.get('mtime', '-')} |"
        )
    lines.append("")

    lines.append("## Warnings detected in console log")
    lines.append("")
    warnings = state.get("warnings", [])
    if not warnings:
        lines.append("No warning/error patterns detected.")
    else:
        for warning in warnings:
            lines.append(f"- `{warning}`")
    lines.append("")

    lines.append("## How to monitor")
    lines.append("")
    lines.append("- One-shot refresh:")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python scripts/monitor_crog_training_diary.py "
        f"--run-dir {state['run_dir']} --pid {state.get('pid') or '<PID>'}"
    )
    lines.append("```")
    lines.append("")
    lines.append("- Watch refresh every 5 minutes:")
    lines.append("")
    lines.append("```bash")
    lines.append(
        "python scripts/monitor_crog_training_diary.py "
        f"--run-dir {state['run_dir']} --pid {state.get('pid') or '<PID>'} "
        "--watch-interval 300"
    )
    lines.append("```")
    lines.append("")
    lines.append("This monitor is read-only with respect to the training process and checkpoints.")
    lines.append("")
    return "\n".join(lines)


def write_outputs(state: dict[str, Any], output: Path, json_output: Path | None) -> None:
    def write_text_atomic(path: Path, text: str) -> None:
        tmp_path = path.with_name(f".{path.name}.tmp")
        tmp_path.write_text(text, encoding="utf-8")
        os.replace(tmp_path, path)

    output.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(output, render_markdown(state))
    if json_output:
        json_output.parent.mkdir(parents=True, exist_ok=True)
        write_text_atomic(json_output, json.dumps(state, indent=2, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate a read-only CROG Mac/MPS training diary from existing run artifacts."
    )
    parser.add_argument(
        "--run-dir",
        required=True,
        type=Path,
        help="CROG experiment output directory containing logs, timing JSON, and checkpoints.",
    )
    parser.add_argument("--pid", type=int, default=None, help="Optional training process PID to check.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Markdown diary path. Defaults to <run-dir>/TRAINING_DIARY.md.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Structured JSON output path. Defaults to <run-dir>/TRAINING_DIARY.json.",
    )
    parser.add_argument(
        "--watch-interval",
        type=float,
        default=0,
        help="If > 0, refresh outputs every N seconds until interrupted.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.resolve()
    output = args.output.resolve() if args.output else run_dir / "TRAINING_DIARY.md"
    json_output = args.json_output.resolve() if args.json_output else run_dir / "TRAINING_DIARY.json"

    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    while True:
        state = collect_state(run_dir, args.pid)
        write_outputs(state, output, json_output)
        print(f"[{state['generated_at']}] wrote {output}")
        if args.watch_interval <= 0:
            break
        time.sleep(args.watch_interval)


if __name__ == "__main__":
    main()
