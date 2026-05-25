from __future__ import annotations

from target.florence2_grounder import Florence2Grounder


def build_vlm_backend(name: str, config: dict | None = None):
    name = name.lower()
    config = config or {}
    if name in {"florence2", "florence2_sam", "florence-2"}:
        florence_cfg = dict(config.get("florence2", config))
        if name == "florence2_sam":
            florence_cfg["sam_enabled"] = True
        return Florence2Grounder(florence_cfg)
    if name in {"groundingdino_sam", "owlvit", "owlv2"}:
        raise RuntimeError(
            f"VLM backend '{name}' is not installed in the Mac-compatible core. "
            "Install the optional backend or use --target-source oracle."
        )
    raise ValueError(f"Unknown VLM backend: {name}")
