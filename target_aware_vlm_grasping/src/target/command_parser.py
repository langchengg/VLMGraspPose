from __future__ import annotations

from dataclasses import dataclass
import re


_STOP_PREFIXES = (
    "pick",
    "grab",
    "get",
    "pass",
    "give",
    "take",
    "grasp",
    "hold",
)

_ARTICLES_RE = re.compile(r"\b(the|a|an)\b", re.IGNORECASE)
_SPATIAL_WORDS_RE = re.compile(
    r"\b(leftmost|rightmost|left|right|front|rear|back|behind|center|middle|nearest|farthest)\b",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class ParsedCommand:
    command: str
    target_phrase: str
    target_queries: list[str]
    relation: str | None = None
    reference_phrase: str | None = None
    reference_queries: list[str] | None = None
    ordinal: str | None = None


def parse_command(command: str, target_label: str | None = None) -> ParsedCommand:
    """Extract the object phrase and simple spatial relation from a command.

    This is intentionally conservative. It does not try to solve full natural
    language grounding; it pulls out the pieces that the VLM backend can query
    separately and that the selector can check geometrically.
    """

    command = (command or "").strip()
    cleaned_target = clean_label(target_label) if target_label else ""
    target_phrase = cleaned_target or _guess_target_phrase(command)
    relation, reference = _extract_relation(command)
    ordinal = _extract_ordinal(command)

    target_queries = _dedupe([
        command,
        target_phrase,
        f"the {target_phrase}" if target_phrase else "",
    ])
    reference_queries = _dedupe([
        reference or "",
        f"the {reference}" if reference else "",
    ])
    return ParsedCommand(
        command=command,
        target_phrase=target_phrase,
        target_queries=target_queries,
        relation=relation,
        reference_phrase=reference,
        reference_queries=reference_queries,
        ordinal=ordinal,
    )


def clean_label(label: str | None) -> str:
    label = (label or "").strip().lower().replace("_", " ")
    label = re.sub(r"\s+\d+$", "", label)
    label = re.sub(r"\b\d+\b", "", label)
    label = re.sub(r"\s+", " ", label).strip()
    return label


def _guess_target_phrase(command: str) -> str:
    text = command.strip().lower()
    text = re.sub(r"[^a-z0-9\s_-]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    for prefix in _STOP_PREFIXES:
        text = re.sub(rf"^{prefix}\s+", "", text)
    text = _ARTICLES_RE.sub(" ", text)
    relation_split = re.split(
        r"\b(on|to|at|in)\s+the\s+(left|right|front|rear|back)|\b(left|right)\s+of\b|\bbehind\b|\bin\s+front\s+of\b",
        text,
        maxsplit=1,
    )[0]
    text = _SPATIAL_WORDS_RE.sub(" ", relation_split)
    text = re.sub(r"\s+", " ", text).strip()
    return text or command.strip()


def _extract_ordinal(command: str) -> str | None:
    text = command.lower()
    if "leftmost" in text:
        return "leftmost"
    if "rightmost" in text:
        return "rightmost"
    if re.search(r"\bleft\s+(?!of\b)", text) and not re.search(r"\bleft\s+of\b|\bto\s+the\s+left\s+of\b", text):
        return "leftmost"
    if re.search(r"\bright\s+(?!of\b)", text) and not re.search(r"\bright\s+of\b|\bto\s+the\s+right\s+of\b", text):
        return "rightmost"
    return None


def _extract_relation(command: str) -> tuple[str | None, str | None]:
    text = command.lower()
    normalized = re.sub(r"[^a-z0-9\s_-]", " ", text)
    normalized = re.sub(r"\s+", " ", normalized).strip()

    patterns: list[tuple[str, str]] = [
        ("rear_right_of", r"\b(?:rear|back|behind)\s+right(?:\s+side)?\s+of\s+(?:the\s+)?(.+)$"),
        ("rear_left_of", r"\b(?:rear|back|behind)\s+left(?:\s+side)?\s+of\s+(?:the\s+)?(.+)$"),
        ("front_right_of", r"\bfront\s+right(?:\s+side)?\s+of\s+(?:the\s+)?(.+)$"),
        ("front_left_of", r"\bfront\s+left(?:\s+side)?\s+of\s+(?:the\s+)?(.+)$"),
        ("right_of", r"\b(?:to\s+the\s+right\s+of|right\s+of|right\s+side\s+of|on\s+the\s+right\s+side\s+of)\s+(?:the\s+)?(.+)$"),
        ("left_of", r"\b(?:to\s+the\s+left\s+of|left\s+of|left\s+side\s+of|on\s+the\s+left\s+side\s+of)\s+(?:the\s+)?(.+)$"),
        ("front_of", r"\b(?:in\s+front\s+of|front\s+of|on\s+the\s+front\s+of)\s+(?:the\s+)?(.+)$"),
        ("behind", r"\b(?:behind|rear\s+of|back\s+of|on\s+the\s+rear\s+of|on\s+the\s+back\s+of)\s+(?:the\s+)?(.+)$"),
    ]
    for relation, pattern in patterns:
        match = re.search(pattern, normalized)
        if match:
            reference = _cleanup_phrase(match.group(1))
            return relation, reference or None
    return None, None


def _cleanup_phrase(text: str) -> str:
    text = _ARTICLES_RE.sub(" ", text or "")
    for prefix in _STOP_PREFIXES:
        text = re.sub(rf"^{prefix}\s+", "", text.strip(), flags=re.IGNORECASE)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _dedupe(items: list[str]) -> list[str]:
    seen = set()
    out = []
    for item in items:
        item = re.sub(r"\s+", " ", (item or "").strip())
        key = item.lower()
        if item and key not in seen:
            seen.add(key)
            out.append(item)
    return out
