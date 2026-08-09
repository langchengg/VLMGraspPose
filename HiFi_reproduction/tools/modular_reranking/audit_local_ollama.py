#!/usr/bin/env python3
"""Audit the pinned local-only Ollama/Qwen3-VL runtime and save evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import urllib.request
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from src.grasping.reranking_v1.local_vlm import (  # noqa: E402
    SESSION_AUDIT_POLICY_VERSION,
    stable_session_contract,
    stable_session_contract_sha256,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--expected-manifest-sha256", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tmp-root", type=Path, required=True)
    parser.add_argument(
        "--ollama", type=Path, default=Path("/opt/homebrew/bin/ollama")
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def command(*arguments: str) -> str:
    return subprocess.run(
        arguments,
        check=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()


def _log_time(line: str) -> datetime | None:
    match = re.search(r"(?:^|\s)time=([^\s]+)", line)
    if match is None:
        return None
    try:
        return datetime.fromisoformat(match.group(1))
    except ValueError:
        return None


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


def atomic_json(path: Path, value: Any, *, tmp_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_root.mkdir(parents=True, exist_ok=True)
    temporary = (
        tmp_root / f"{path.name}.{os.getpid()}.{uuid.uuid4().hex}.tmp"
    )
    try:
        temporary.write_text(
            json.dumps(
                value,
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
                allow_nan=False,
            )
            + "\n",
            encoding="utf-8",
        )
        # Hard-link publication is atomic and fails if the evidence path
        # already exists.  A session audit referenced by an attempt ledger can
        # therefore never be silently replaced before the runner checks it.
        os.link(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def main() -> int:
    args = parse_args()
    output_path = args.output.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(
            f"local Ollama audit output is immutable: {output_path}"
        )
    if os.environ.get("OLLAMA_NO_CLOUD") != "1":
        raise RuntimeError("OLLAMA_NO_CLOUD=1 is required for this audit")
    home = Path.home()
    server_config = home / ".ollama" / "server.json"
    config = json.loads(server_config.read_text(encoding="utf-8"))
    if config.get("disable_ollama_cloud") is not True:
        raise RuntimeError("Ollama cloud is not disabled in server.json")
    ollama_executable = args.ollama.expanduser().resolve()
    if not ollama_executable.is_file():
        raise FileNotFoundError(ollama_executable)
    model_parts = args.model_name.split(":", 1)
    if len(model_parts) != 2:
        raise ValueError("model-name must be an exact name:tag")
    manifest = (
        home
        / ".ollama"
        / "models"
        / "manifests"
        / "registry.ollama.ai"
        / "library"
        / model_parts[0]
        / model_parts[1]
    )
    manifest_sha = sha256_file(manifest)
    if manifest_sha != args.expected_manifest_sha256:
        raise RuntimeError(
            f"model manifest digest mismatch: {manifest_sha} "
            f"!= {args.expected_manifest_sha256}"
        )
    manifest_payload = json.loads(manifest.read_text(encoding="utf-8"))
    layers = []
    for layer in manifest_payload["layers"]:
        digest = str(layer["digest"])
        blob = home / ".ollama" / "models" / "blobs" / digest.replace(":", "-")
        if not blob.is_file():
            raise FileNotFoundError(blob)
        layers.append(
            {
                **layer,
                "local_path": str(blob),
                "local_size": blob.stat().st_size,
                "local_sha256": sha256_file(blob),
            }
        )
    opener = urllib.request.build_opener(
        urllib.request.ProxyHandler({}), _NoRedirect()
    )
    with opener.open("http://127.0.0.1:11434/api/version", timeout=5) as response:
        api_version = json.load(response)
    listener = command(
        "/usr/sbin/lsof",
        "-nP",
        "-iTCP:11434",
        "-sTCP:LISTEN",
    )
    listener_lines = listener.splitlines()
    if not listener_lines or any(
        "127.0.0.1:11434" not in line for line in listener_lines[1:]
    ):
        raise RuntimeError(f"Ollama listener is not loopback-only:\n{listener}")
    listener_pids = sorted(
        {
            int(line.split()[1])
            for line in listener_lines[1:]
            if len(line.split()) > 1 and line.split()[1].isdigit()
        }
    )
    if len(listener_pids) != 1:
        raise RuntimeError(f"expected one Ollama listener PID, got {listener_pids}")
    listener_pid = listener_pids[0]
    process_started = command(
        "/bin/ps", "-o", "lstart=", "-p", str(listener_pid)
    )
    process_started_time = datetime.strptime(
        process_started, "%a %b %d %H:%M:%S %Y"
    ).replace(tzinfo=datetime.now().astimezone().tzinfo)
    connections = subprocess.run(
        ["/usr/sbin/lsof", "-nP", "-a", "-p", str(listener_pid), "-iTCP"],
        check=False,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    ).stdout.strip()
    remote_established = [
        line
        for line in connections.splitlines()[1:]
        if "(ESTABLISHED)" in line
        and "127.0.0.1:" not in line
        and "[::1]:" not in line
    ]
    if remote_established:
        raise RuntimeError(
            "Ollama has non-loopback established TCP connections:\n"
            + "\n".join(remote_established)
        )
    log_path = Path("/opt/homebrew/var/log/ollama.log")
    log_lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    cloud_evidence = next(
        (
            line
            for line in reversed(log_lines)
            if "Ollama cloud disabled: true" in line
            and _log_time(line) is not None
            and abs(
                (_log_time(line) - process_started_time).total_seconds()
            )
            <= 5.0
        ),
        None,
    )
    listener_evidence = next(
        (
            line
            for line in reversed(log_lines)
            if "Listening on 127.0.0.1:11434" in line
            and str(api_version.get("version")) in line
            and _log_time(line) is not None
            and abs(
                (_log_time(line) - process_started_time).total_seconds()
            )
            <= 5.0
        ),
        None,
    )
    if cloud_evidence is None or listener_evidence is None:
        raise RuntimeError("required local-only evidence is absent from Ollama log")
    if (
        cloud_evidence is not None
        and listener_evidence is not None
        and abs(
            (_log_time(cloud_evidence) - _log_time(listener_evidence)).total_seconds()
        )
        > 5.0
    ):
        raise RuntimeError("Ollama local-only log lines belong to different starts")
    ollama_list = command(str(ollama_executable), "list")
    matching_list_lines = [
        line for line in ollama_list.splitlines() if args.model_name in line
    ]
    if len(matching_list_lines) != 1:
        raise RuntimeError("pinned model is absent or ambiguous in ollama list")
    model_layer = next(
        layer
        for layer in layers
        if layer["mediaType"] == "application/vnd.ollama.image.model"
    )
    license_layers = [
        layer
        for layer in layers
        if layer["mediaType"] == "application/vnd.ollama.image.license"
    ]
    evidence = {
        "schema_version": 1,
        "audited_at_utc": datetime.now(timezone.utc).isoformat(),
        "machine": {
            "architecture": platform.machine(),
            "platform": platform.platform(),
        },
        "ollama": {
            "cli_version": command(str(ollama_executable), "--version"),
            "api_version": api_version,
            "executable_path": str(ollama_executable),
            "executable_sha256": sha256_file(ollama_executable),
            "endpoint": "http://127.0.0.1:11434",
            "listener": listener,
            "listener_pid": listener_pid,
            "listener_process_started": process_started,
            "tcp_connections": connections,
            "remote_established_connections": remote_established,
            "api_transport": "no_proxy_no_redirect_loopback_http",
            "server_config_path": str(server_config),
            "server_config_sha256": sha256_file(server_config),
            "server_config": config,
            "preexisting_server_config": True,
            "cloud_disabled_log_evidence": cloud_evidence,
            "loopback_log_evidence": listener_evidence,
        },
        "model": {
            "exact_name": args.model_name,
            "manifest_path": str(manifest),
            "manifest_sha256": manifest_sha,
            "ollama_list_line": matching_list_lines[0],
            "parameter_scale": "4B",
            "quantization": "Q4_K_M",
            "model_layer_digest": model_layer["digest"],
            "model_layer_bytes": model_layer["size"],
            "layers": layers,
            "license_layers": license_layers,
            "license_metadata": "Qwen model license blob retained locally",
        },
        "local_only": True,
        "remote_api_used": False,
        "official_references": [
            "https://docs.ollama.com/faq",
            "https://docs.ollama.com/api/chat",
            "https://docs.ollama.com/capabilities/structured-outputs",
            "https://docs.ollama.com/capabilities/vision",
            "https://ollama.com/library/qwen3-vl/tags",
            "https://github.com/QwenLM/Qwen3-VL",
        ],
    }
    evidence["stable_session_contract"] = stable_session_contract(evidence)
    evidence["stable_session_contract_sha256"] = (
        stable_session_contract_sha256(evidence)
    )
    evidence["session_audit_policy_version"] = (
        SESSION_AUDIT_POLICY_VERSION
    )
    atomic_json(
        output_path,
        evidence,
        tmp_root=args.tmp_root.expanduser().resolve(),
    )
    print(json.dumps(evidence, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
