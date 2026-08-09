#!/usr/bin/env python3
"""Isolated google-genai 2.x worker. The API key is read only by the SDK/env."""

from __future__ import annotations

import argparse
import base64
import json
import os
from pathlib import Path
from typing import Any


def _usage(response: Any) -> dict[str, Any]:
    value = getattr(response, "usage", None) or getattr(response, "usage_metadata", None)
    if value is None:
        return {}
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json", exclude_none=True)
    if isinstance(value, dict):
        return value
    return {name: getattr(value, name) for name in dir(value) if name.endswith(("tokens", "token_count")) and not name.startswith("_")}


def _client() -> Any:
    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError("Gemini API credential is unavailable")
    from google import genai
    from google.genai import types
    # Interactions has a generated-client retry layer in addition to the
    # public google-genai HTTP layer.  Disable both SDK-level retry paths so
    # one ledger attempt is exactly one provider transmission; the parent
    # runner owns the bounded, budgeted, auditable retry policy.
    client = genai.Client(
        api_key=key,
        http_options=types.HttpOptions(
            api_version="v1beta",
            retry_options=types.HttpRetryOptions(attempts=1),
        ),
    )
    # google-genai 2.16.0 maps ``attempts=1`` to no retry in the public HTTP
    # client, but its generated Interactions adapter interprets that same
    # field as *one retry*.  Reset the retained option before the lazy
    # Interactions client is constructed.  This compatibility shim is covered
    # by an installed-SDK contract check in the experiment tests.
    client._api_client._http_options.retry_options.attempts = 0
    return client


def _interactions(client: Any, request: dict[str, Any]) -> dict[str, Any]:
    generation = dict(request["generation_config"])
    generation.pop("temperature_policy", None)
    interaction = client.interactions.create(
        model=request["model_id"],
        input=[
            {"type": "text", "text": json.dumps(request["text_payload"], sort_keys=True, separators=(",", ":"), allow_nan=False)},
            {"type": "image", "data": request["overview_png_base64"], "mime_type": "image/png", "resolution": "high"},
            {"type": "image", "data": request["grid_png_base64"], "mime_type": "image/png", "resolution": "high"},
        ],
        stream=False, store=False, background=False, service_tier="standard",
        system_instruction=request["system_prompt"], generation_config=generation,
        # The May-2026 Interactions contract defines response_format as a
        # top-level list of polymorphic output formats.  Passing the legacy
        # object shape is accepted by some SDK builds but can silently leave
        # structured output unenforced.
        response_format=[{"type": "text", "mime_type": "application/json", "schema": request["response_schema"]}],
    )
    return {
        "raw_text": str(getattr(interaction, "output_text", "") or ""),
        "usage": _usage(interaction), "response_model": getattr(interaction, "model", None),
        "request_id": getattr(interaction, "id", None), "endpoint_used": "interactions",
    }


def _generate_content(client: Any, request: dict[str, Any]) -> dict[str, Any]:
    """Explicit compatibility fallback; never selected for ordinary provider errors."""
    from google.genai import types
    generation = dict(request["generation_config"])
    generation.pop("temperature_policy", None)
    thinking_level = generation.pop("thinking_level", None)
    parts = [
        types.Part.from_text(text=json.dumps(request["text_payload"], sort_keys=True, separators=(",", ":"), allow_nan=False)),
        types.Part.from_bytes(data=base64.b64decode(request["overview_png_base64"]), mime_type="image/png"),
        types.Part.from_bytes(data=base64.b64decode(request["grid_png_base64"]), mime_type="image/png"),
    ]
    config = types.GenerateContentConfig(
        system_instruction=request["system_prompt"],
        response_mime_type="application/json", response_json_schema=request["response_schema"],
        thinking_config=None if thinking_level is None else types.ThinkingConfig(thinking_level=thinking_level),
        tools=None, **generation,
    )
    response = client.models.generate_content(
        model=request["model_id"], contents=[types.Content(role="user", parts=parts)], config=config,
    )
    return {
        "raw_text": str(getattr(response, "text", "") or ""), "usage": _usage(response),
        "response_model": getattr(response, "model_version", None),
        "request_id": getattr(response, "response_id", None), "endpoint_used": "generateContent_explicit_fallback",
    }


def invoke(request: dict[str, Any]) -> dict[str, Any]:
    if request.get("model_id") not in {"gemini-robotics-er-2-preview", "gemini-3.6-flash"}:
        raise ValueError("model substitution is forbidden")
    if request.get("endpoint_type") not in {"interactions", "generateContent"}:
        raise ValueError("unexpected endpoint")
    client = _client()
    try:
        return _interactions(client, request) if request["endpoint_type"] == "interactions" else _generate_content(client, request)
    finally:
        client.close()


def model_metadata(model_id: str) -> dict[str, Any]:
    if model_id not in {"gemini-robotics-er-2-preview", "gemini-3.6-flash"}:
        raise ValueError("model substitution is forbidden")
    # Keep the parent Client alive for the complete request. Accessing
    # ``_client().models`` through a temporary object can let Client.__del__
    # close the shared HTTP transport before Models.get sends the request.
    client = _client()
    try:
        value = client.models.get(model=model_id)
        return value.model_dump(mode="json", exclude_none=True) if hasattr(value, "model_dump") else {"name": str(value)}
    finally:
        client.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("invoke", "model-metadata"))
    parser.add_argument("--request", type=Path)
    parser.add_argument("--response", type=Path, required=True)
    parser.add_argument("--model-id")
    args = parser.parse_args()
    if args.command == "invoke":
        if args.request is None:
            raise ValueError("--request is required")
        result = invoke(json.loads(args.request.read_text(encoding="utf-8")))
    else:
        result = model_metadata(str(args.model_id))
    args.response.write_text(json.dumps(result, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
