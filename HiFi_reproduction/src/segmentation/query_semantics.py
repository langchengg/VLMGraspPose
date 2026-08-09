"""Deterministic, query-text-only semantics for OCID-VLG expressions.

The parser deliberately has no annotation or answer-instance loader.  Its
lexicon is a fixed transcription of the public OCID-VLG category, colour,
location, and relation vocabulary, extended only with surface-form synonyms
that occur in the public referring-expression templates.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass
from typing import Final


CATEGORY_SYNONYMS: Final[dict[str, tuple[str, ...]]] = {
    "apple": ("apple",),
    "ball": ("ball with spots", "ball with dots", "rugby ball", "polka ball", "ball"),
    "banana": ("banana",),
    "bell_pepper": ("bell pepper", "pepper"),
    "binder": ("binder",),
    "bowl": ("bowl",),
    "cereal_box": (
        "choco krispies corn flakes box",
        "chocos corn flakes box",
        "mega pack corn flakes box",
        "topas corn flakes box",
        "choco krispies cereal box",
        "mega pack cereal box",
        "topas cereal box",
        "chocos cereal box",
        "choco krispies corn flakes",
        "mega pack corn flakes",
        "topas corn flakes",
        "chocos corn flakes",
        "corn flakes box",
        "corn flakes package",
        "cereal box package",
        "choco krispies box",
        "mega pack cereal",
        "topas cereal",
        "chocos box",
        "cereal box",
        "corn flakes",
        "cereal package",
        "cereal",
    ),
    "coffee_mug": ("coffee mug", "coffee cup", "mug"),
    "flashlight": ("flashlight",),
    "food_bag": (
        "spaghetti penne bag",
        "food bag product",
        "transparent food bag",
        "langkorn rice bag",
        "clever rice bag",
        "bag with lentils",
        "bag with pasta",
        "bag with rice",
        "spaghetti bag",
        "lentil bag",
        "penne bag",
        "rice bag",
        "pasta bag",
        "food bag",
    ),
    "food_box": (
        "box with choco-banana",
        "chocolate banana box",
        "box with oatmeal",
        "box with oats",
        "food box product",
        "choco-bananas",
        "spaghetti box",
        "oatmeal box",
        "barilla box",
        "oat box",
        "tagliatelle",
        "food box",
    ),
    "food_can": (
        "can with breakfast meat",
        "breakfast meat can",
        "tomato pulp can",
        "food can product",
        "exotic food can",
        "can with meat",
        "breakfast meat",
        "tomato can",
        "tomato pulp",
        "can food",
        "food can",
    ),
    "glue_stick": ("glue stick", "glue"),
    "hand_towel": ("hand towel", "towel"),
    "instant_noodles": (
        "alnatura noodles package",
        "shrimp noodles package",
        "instant noodles package",
        "alnatura noodles",
        "shrimp noodles",
        "yumyum noodles",
        "instant noodles",
        "noodles package",
        "noodles",
    ),
    "keyboard": ("keyboard",),
    "kleenex": (
        "cube kleenex tissues package",
        "cube kleenex tissues box",
        "cube keenex box",
        "cuboid kleenex box",
        "lovely kleenex package",
        "lovely kleenex tissues",
        "lovely kleenex box",
        "jembo kleenex package",
        "jembo kleenex box",
        "feh kleenex package",
        "feh kleenex tissues",
        "feh kleenex box",
        "kleenex tissues package",
        "kleenex tissues box",
        "kleenex package",
        "kleenex box",
        "tissues package",
        "tissues box",
        "kleenex tissues",
        "kleenex",
        "tissues",
    ),
    "lemon": ("lemon",),
    "lime": ("lime",),
    "marker": ("marker",),
    "orange": ("orange",),
    "peach": ("peach",),
    "pear": ("pear",),
    "potato": ("potato",),
    "shampoo": ("shampoo bottle", "shampoo product", "shampoo"),
    "soda_can": (
        "orange soda",
        "coca-cola drink",
        "coca-cola can",
        "coke drink",
        "fanta drink",
        "cola drink",
        "coke can",
        "cola can",
        "soft drink can",
        "soft drink",
        "soda drink",
        "soda can",
        "fanta can",
        "coca-cola",
        "fanta",
        "coke",
        "cola",
        "soda",
    ),
    "sponge": ("sponge",),
    "stapler": ("stapler",),
    "tomato": ("tomato",),
    "toothpaste": ("toothpaste product", "toothpaste",),
}

CATEGORY_PROMPTS: Final[dict[str, str]] = {
    name: name.replace("_", " ") for name in CATEGORY_SYNONYMS
}

COLOURS: Final[tuple[str, ...]] = tuple(
    sorted(
        {
            "black and transparent",
            "blue and beige",
            "blue and black",
            "gray and blue",
            "green and blue",
            "green and red",
            "green and yellow",
            "red and white",
            "white and blue",
            "white and green",
            "white and red",
            "white and yellow",
            "yellow and brown",
            "yellow and green",
            "dark blue",
            "light blue",
            "transparent",
            "beige",
            "black",
            "blue",
            "brown",
            "green",
            "orange",
            "pink",
            "red",
            "white",
            "yellow",
        },
        key=lambda value: (-len(value), value),
    )
)

SHAPES: Final[tuple[str, ...]] = (
    "cube",
    "round",
    "rectangular",
    "cylindrical",
    "thin",
)
SIZES: Final[tuple[str, ...]] = ("small", "large", "big", "tiny", "long", "short")
MATERIALS: Final[tuple[str, ...]] = (
    "transparent",
    "metal",
    "metallic",
    "plastic",
    "paper",
    "cardboard",
    "spotted",
)

BRAND_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "alnatura",
        "alverde",
        "aussie",
        "barilla",
        "blend-a-med",
        "choco",
        "chocos",
        "coca-cola",
        "coke",
        "colgate",
        "fanta",
        "feh",
        "jembo",
        "langkorn",
        "lovely",
        "mega",
        "topas",
        "vichy",
        "yumyum",
    }
)

DIRECTIVE_RE: Final[re.Pattern[str]] = re.compile(
    r"^(?P<directive>pick|grasp|grab|get|pass)(?:\s+me)?(?:\s+the)?\s+",
    flags=re.IGNORECASE,
)

RELATION_PATTERNS: Final[tuple[tuple[str, str], ...]] = tuple(
    sorted(
        (
            ("on the front right side of", "front_right"),
            ("to the front right of", "front_right"),
            ("on the front right of", "front_right"),
            ("front right side of", "front_right"),
            ("front right of", "front_right"),
            ("on the front left side of", "front_left"),
            ("to the front left of", "front_left"),
            ("on the front left of", "front_left"),
            ("front left side of", "front_left"),
            ("front left of", "front_left"),
            ("on the rear right side of", "rear_right"),
            ("to the rear right of", "rear_right"),
            ("on the rear right of", "rear_right"),
            ("rear right side of", "rear_right"),
            ("rear right of", "rear_right"),
            ("on the rear left side of", "rear_left"),
            ("to the rear left of", "rear_left"),
            ("on the rear left of", "rear_left"),
            ("rear left side of", "rear_left"),
            ("rear left of", "rear_left"),
            ("on the right side of", "right"),
            ("to the right of", "right"),
            ("right side of", "right"),
            ("right of", "right"),
            ("on the left side of", "left"),
            ("to the left of", "left"),
            ("left side of", "left"),
            ("left of", "left"),
            ("in front of", "front"),
            ("to the front of", "front"),
            ("behind", "behind"),
            ("on top of", "on"),
            ("on", "on"),
        ),
        key=lambda item: (-len(item[0]), item[0]),
    )
)

LOCATION_PATTERNS: Final[tuple[tuple[str, str], ...]] = (
    ("most distant", "furthest"),
    ("furthest", "furthest"),
    ("farthest", "furthest"),
    ("rightmost", "rightmost"),
    ("leftmost", "leftmost"),
    ("closest", "closest"),
    ("nearest", "closest"),
)


@dataclass(frozen=True)
class QuerySemantics:
    query: str
    normalized_query: str
    directive: str | None
    target_category: str | None
    target_instance_phrase: str | None
    target_color: str | None
    target_shape: str | None
    target_size_attribute: str | None
    target_material_or_appearance: str | None
    absolute_location: str | None
    pairwise_relation: str | None
    reference_category: str | None
    reference_attributes: tuple[str, ...]
    query_type: str
    target_category_prompt: str | None
    target_attribute_prompt: str | None
    reference_category_prompt: str | None
    reference_attribute_prompt: str | None
    parser_confidence: float
    parser_version: str = "ocidvlg_lexical_v1"
    uses_answer_instance: bool = False

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _normalise(text: str) -> str:
    value = text.strip().lower().replace("_", " ")
    value = re.sub(r"[^a-z0-9\s-]", " ", value)
    value = re.sub(r"\s+", " ", value).strip()
    return value


def _extract_category(span: str) -> tuple[str | None, str | None]:
    matches: list[tuple[int, int, str, str]] = []
    padded = f" {span} "
    for category, synonyms in CATEGORY_SYNONYMS.items():
        for synonym in synonyms:
            match = re.search(rf"(?<![a-z0-9]){re.escape(synonym)}(?![a-z0-9])", padded)
            if match:
                matches.append((len(synonym), -match.start(), category, synonym))
    if not matches:
        return None, None
    _, _, category, surface = max(matches)
    return category, surface


def _first_attribute(span: str, vocabulary: tuple[str, ...]) -> str | None:
    padded = f" {span} "
    for value in vocabulary:
        if re.search(rf"(?<![a-z0-9]){re.escape(value)}(?![a-z0-9])", padded):
            return value
    return None


def _location(span: str) -> str | None:
    for phrase, canonical in LOCATION_PATTERNS:
        if re.search(rf"(?<![a-z0-9]){re.escape(phrase)}(?![a-z0-9])", span):
            return canonical
    # OCID-VLG location templates also use bare "left" and "right" as
    # superlative selectors (e.g. "Pick the left marker").  Pairwise uses are
    # already split out by the relation parser.
    if re.search(r"\b(?:the\s+)?left\s+[a-z]", span):
        return "leftmost"
    if re.search(r"\b(?:the\s+)?right\s+[a-z]", span):
        return "rightmost"
    return None


def _strip_location(span: str) -> str:
    result = span
    for phrase, _ in LOCATION_PATTERNS:
        result = re.sub(rf"\b{re.escape(phrase)}\b", " ", result)
    result = re.sub(r"\b(left|right)\b(?=\s+[a-z])", " ", result)
    return re.sub(r"\s+", " ", result).strip()


def _attribute_prompt(color: str | None, category_prompt: str | None) -> str | None:
    if category_prompt is None:
        return None
    return category_prompt if color is None else f"{color} {category_prompt}"


def parse_query(query: str) -> QuerySemantics:
    """Parse a natural-language expression without any symbolic answer fields."""

    if not isinstance(query, str) or not query.strip():
        raise ValueError("query must be a non-empty string")
    normalized = _normalise(query)
    directive_match = DIRECTIVE_RE.match(normalized)
    directive = directive_match.group("directive").lower() if directive_match else None
    content = normalized[directive_match.end() :] if directive_match else normalized
    content = re.sub(r"^(the|a|an)\s+", "", content).strip()

    relation: str | None = None
    target_span = content
    reference_span = ""
    for phrase, canonical in RELATION_PATTERNS:
        match = re.search(rf"\b{re.escape(phrase)}\b", content)
        if match and match.start() > 0:
            relation = canonical
            target_span = content[: match.start()].strip()
            reference_span = content[match.end() :].strip()
            reference_span = re.sub(r"^(the|a|an)\s+", "", reference_span)
            break

    target_location = _location(target_span)
    reference_location = _location(reference_span) if reference_span else None
    target_category, target_surface = _extract_category(target_span)
    reference_category, _ = _extract_category(reference_span)
    target_color = _first_attribute(target_span, COLOURS)
    reference_color = _first_attribute(reference_span, COLOURS)
    target_shape = _first_attribute(target_span, SHAPES)
    target_size = _first_attribute(target_span, SIZES)
    target_material = _first_attribute(target_span, MATERIALS)

    target_category_prompt = (
        CATEGORY_PROMPTS[target_category] if target_category is not None else None
    )
    reference_category_prompt = (
        CATEGORY_PROMPTS[reference_category] if reference_category is not None else None
    )
    target_attribute_prompt = _attribute_prompt(target_color, target_category_prompt)
    reference_attribute_prompt = _attribute_prompt(reference_color, reference_category_prompt)

    instance = _strip_location(target_span)
    instance = re.sub(r"\b(?:that|which)\s+is\s*$", "", instance).strip()
    instance = re.sub(r"^(the|a|an)\s+", "", instance).strip()
    brand_surface = instance.replace("-", " ")
    has_brand = any(
        re.search(rf"\b{re.escape(token.replace('-', ' '))}\b", brand_surface)
        for token in BRAND_TOKENS
    )
    generic_phrases = set(CATEGORY_PROMPTS.values()) | {
        "ball",
        "cereal",
        "corn flakes",
        "food bag",
        "food bag product",
        "food box",
        "food box product",
        "food can",
        "food can product",
        "hand towel",
        "instant noodles",
        "kleenex",
        "marker",
        "noodles",
        "soda",
        "soda can",
        "soda drink",
        "soft drink",
        "tissues",
        "towel",
    }
    if (
        not instance
        or instance == target_attribute_prompt
        or instance == target_category_prompt
        or (instance in generic_phrases and not has_brand)
        or (instance == target_surface and not has_brand)
    ):
        instance = None

    semantic_count = sum(
        (
            target_color is not None or target_shape is not None or target_size is not None or target_material is not None,
            target_location is not None or reference_location is not None,
            relation is not None,
        )
    )
    if target_category is None:
        query_type = "unknown"
    elif semantic_count > 1:
        query_type = "mixed"
    elif relation is not None:
        query_type = "relation"
    elif target_location is not None:
        query_type = "location"
    elif target_color is not None or target_shape is not None or target_size is not None or target_material is not None:
        query_type = "attribute"
    else:
        query_type = "name"

    confidence = 0.15
    confidence += 0.50 if target_category is not None else 0.0
    confidence += 0.15 if directive is not None else 0.0
    confidence += 0.10 if relation is None or reference_category is not None else -0.15
    confidence += 0.10 if query_type != "unknown" else 0.0
    confidence = float(min(1.0, max(0.0, confidence)))

    reference_attributes = tuple(
        value for value in (reference_color, reference_location) if value is not None
    )
    return QuerySemantics(
        query=query,
        normalized_query=normalized,
        directive=directive,
        target_category=target_category,
        target_instance_phrase=instance,
        target_color=target_color,
        target_shape=target_shape,
        target_size_attribute=target_size,
        target_material_or_appearance=target_material,
        absolute_location=target_location,
        pairwise_relation=relation,
        reference_category=reference_category,
        reference_attributes=reference_attributes,
        query_type=query_type,
        target_category_prompt=target_category_prompt,
        target_attribute_prompt=target_attribute_prompt,
        reference_category_prompt=reference_category_prompt,
        reference_attribute_prompt=reference_attribute_prompt,
        parser_confidence=confidence,
    )


__all__ = [
    "CATEGORY_PROMPTS",
    "CATEGORY_SYNONYMS",
    "COLOURS",
    "QuerySemantics",
    "parse_query",
]
