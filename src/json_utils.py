import json
import re
from typing import Callable, Optional

from src.schema import Constraint, ConstraintList

SPARK_SYSTEM_PROMPT = (
    "Return ONLY valid JSON matching the schema: "
    "[{\"id\": \"string\", \"type\": \"count|presence|absence|position|size\", "
    "\"description\": \"string\", \"object_prompt\": \"string\", "
    "\"expected_count\": 0, \"position\": \"left|right|top|bottom|center|top-left|top-right|bottom-left|bottom-right\", "
    "\"min_area_ratio\": 0.0, \"max_area_ratio\": 1.0}] "
    "No markdown, no explanation, no trailing comments."
)


def _strip_code_fences(raw: str) -> str:
    text = raw.strip()
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return text


def _parse_constraint_payload(payload) -> list[Constraint]:
    if isinstance(payload, list):
        data = payload
    elif isinstance(payload, dict) and "constraints" in payload:
        data = payload["constraints"]
    else:
        raise ValueError("Constraint output is not a JSON list or constraint object")

    try:
        parsed = ConstraintList(constraints=data).constraints
    except Exception as exc:  # pragma: no cover - pydantic raises ValidationError
        raise ValueError(f"Constraint JSON is invalid: {exc}") from exc
    return parsed


def parse_constraints(
    raw: str,
    llm_callable: Optional[Callable[[str, str], str]] = None,
) -> list[Constraint]:
    """Parse raw LLM output. Strip code fences if present. Retry once by calling the LLM again with a repair prompt if invalid."""
    cleaned = _strip_code_fences(raw)
    if not cleaned:
        raise ValueError("Constraint output is empty")

    try:
        payload = json.loads(cleaned)
        return _parse_constraint_payload(payload)
    except (TypeError, ValueError) as exc:
        if llm_callable is None:
            raise ValueError(f"Invalid constraint JSON: {exc}") from exc

        repaired = llm_callable(
            SPARK_SYSTEM_PROMPT,
            "Repair the JSON so it matches the given ConstraintList schema exactly.",
        )
        try:
            payload = json.loads(_strip_code_fences(repaired))
            return _parse_constraint_payload(payload)
        except (TypeError, ValueError) as repair_exc:
            raise ValueError(f"Constraint JSON failed repair: {repair_exc}") from repair_exc
