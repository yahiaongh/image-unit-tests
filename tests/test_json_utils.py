import pytest

from src.json_utils import parse_constraints


def test_valid_json_parses():
    raw = '[{"id": "c1", "type": "count", "description": "one object", "object_prompt": "chair", "expected_count": 1}]'
    constraints = parse_constraints(raw)
    assert len(constraints) == 1
    assert constraints[0].id == "c1"


def test_markdown_json_parses():
    raw = '```json\n[{"id": "c1", "type": "count", "description": "one object", "object_prompt": "chair", "expected_count": 1}]\n```'
    constraints = parse_constraints(raw)
    assert len(constraints) == 1
    assert constraints[0].id == "c1"


def test_invalid_json_retries_once_then_raises():
    calls = {"count": 0}

    def fake_llm(system_prompt: str, repair_prompt: str) -> str:
        calls["count"] += 1
        return '[{"id": "c1", "type": "count", "description": "one object", "object_prompt": "chair", "expected_count": 1}]'

    raw = '{not valid json}'
    constraints = parse_constraints(raw, llm_callable=fake_llm)
    assert len(constraints) == 1
    assert calls["count"] == 1

    def fake_fail_llm(system_prompt: str, repair_prompt: str) -> str:
        return '{invalid}'

    with pytest.raises(ValueError):
        parse_constraints('{not valid json}', llm_callable=fake_fail_llm)


def test_missing_required_field_raises_value_error():
    raw = '[{"id": "c1", "type": "count", "description": "one object"}]'
    with pytest.raises(ValueError):
        parse_constraints(raw)
