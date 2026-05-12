"""Quality utilities for generated benchmark outputs."""

from __future__ import annotations

import json
from collections.abc import Sequence


def parse_json_object(text: str) -> dict[str, object] | None:
    """Parse a JSON object from the full text or the first parseable object substring."""

    decoder = json.JSONDecoder()

    try:
        parsed = json.loads(text)
    except json.JSONDecodeError:
        parsed = None

    if isinstance(parsed, dict):
        return parsed

    for start_index, character in enumerate(text):
        if character != "{":
            continue

        try:
            parsed_substring, _ = decoder.raw_decode(text[start_index:])
        except json.JSONDecodeError:
            continue

        if isinstance(parsed_substring, dict):
            return parsed_substring

    return None


def validate_required_fields(
    payload: dict[str, object],
    required_fields: Sequence[str],
) -> bool:
    """Return whether all required fields are present in the payload."""

    return all(field in payload for field in required_fields)


def score_structured_output(
    text: str,
    required_fields: Sequence[str],
) -> dict[str, object]:
    """Score generated text for JSON validity and required-field completeness."""

    parsed = parse_json_object(text)
    missing_fields = [field for field in required_fields if parsed is None or field not in parsed]

    return {
        "is_valid_json": parsed is not None,
        "has_required_fields": parsed is not None and not missing_fields,
        "missing_fields": missing_fields,
        "parsed": parsed,
    }
