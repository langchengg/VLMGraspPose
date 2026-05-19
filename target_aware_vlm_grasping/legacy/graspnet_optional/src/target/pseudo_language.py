def command_for_target(target_id: int | None, label: str | None = None) -> str:
    if label:
        return f"pick {label}"
    if target_id is not None:
        return f"pick object_{target_id:03d}"
    return "pick the target object"
