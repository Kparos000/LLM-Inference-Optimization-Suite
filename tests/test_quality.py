from inference_bench.quality import (
    parse_json_object,
    score_structured_output,
    validate_required_fields,
)


def test_valid_json_parses() -> None:
    parsed = parse_json_object('{"category":"helpdesk","answer":"Reset it.","confidence":0.9}')

    assert parsed == {
        "category": "helpdesk",
        "answer": "Reset it.",
        "confidence": 0.9,
    }


def test_embedded_json_object_can_be_extracted() -> None:
    parsed = parse_json_object(
        'Here is the result: {"category":"code","answer":"Check the key.","confidence":0.8}'
    )

    assert parsed == {
        "category": "code",
        "answer": "Check the key.",
        "confidence": 0.8,
    }


def test_invalid_json_returns_none() -> None:
    assert parse_json_object("category: helpdesk, answer: reset it") is None


def test_required_fields_validation_passes() -> None:
    payload: dict[str, object] = {
        "category": "greeting",
        "answer": "Welcome Jordan.",
        "confidence": 0.95,
    }

    assert validate_required_fields(payload, ["category", "answer", "confidence"])


def test_required_fields_validation_fails_with_missing_fields() -> None:
    payload: dict[str, object] = {"category": "greeting", "answer": "Welcome Jordan."}

    assert not validate_required_fields(payload, ["category", "answer", "confidence"])


def test_score_structured_output_returns_flags_and_missing_fields() -> None:
    score = score_structured_output(
        '{"category":"code","answer":"Use dict.get."}',
        ["category", "answer", "confidence"],
    )

    assert score["is_valid_json"] is True
    assert score["has_required_fields"] is False
    assert score["missing_fields"] == ["confidence"]
    assert score["parsed"] == {"category": "code", "answer": "Use dict.get."}


def test_score_structured_output_marks_invalid_json() -> None:
    score = score_structured_output("not json", ["category", "answer"])

    assert score["is_valid_json"] is False
    assert score["has_required_fields"] is False
    assert score["missing_fields"] == ["category", "answer"]
    assert score["parsed"] is None
