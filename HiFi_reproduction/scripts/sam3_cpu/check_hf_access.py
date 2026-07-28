#!/usr/bin/env python3
"""Check official SAM 3 gated access without downloading or exposing a token."""

from __future__ import annotations

import argparse
import json

from huggingface_hub import HfApi, get_hf_file_metadata, hf_hub_url
from huggingface_hub.errors import GatedRepoError, HfHubHTTPError


def check_access(model_id: str) -> tuple[dict, int]:
    api = HfApi()
    try:
        who = api.whoami()
        account = str(who.get("name") or who.get("fullname") or "authenticated")
    except Exception as error:
        return {
            "model_id": model_id,
            "account": None,
            "access": "authentication_required",
            "error_type": type(error).__name__,
        }, 2
    try:
        info = api.model_info(model_id, files_metadata=True)
        metadata = get_hf_file_metadata(
            hf_hub_url(model_id, "config.json", revision=info.sha),
            token=True,
        )
    except GatedRepoError:
        return {
            "model_id": model_id,
            "account": account,
            "access": "denied",
            "reason": "official model conditions are not accepted or access is not granted",
            "http_status": 403,
        }, 2
    except HfHubHTTPError as error:
        return {
            "model_id": model_id,
            "account": account,
            "access": "error",
            "error_type": type(error).__name__,
            "http_status": getattr(error.response, "status_code", None),
        }, 2
    return {
        "model_id": model_id,
        "account": account,
        "access": "granted",
        "resolved_revision": info.sha,
        "gated": info.gated,
        "config_size_bytes": metadata.size,
    }, 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-id", default="facebook/sam3")
    args = parser.parse_args()
    result, status = check_access(args.model_id)
    print(json.dumps(result, indent=2, sort_keys=True))
    return status


if __name__ == "__main__":
    raise SystemExit(main())

