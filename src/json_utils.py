import json
import re
from typing import Callable, Optional

from src.schema import Constraint, ConstraintList

SPARK_SYSTEM_PROMPT = """You are a constraint extraction engine.
You receive a plain-English image brief and return a JSON object with a
"constraints" array describing machine-checkable rules.

ABSOLUTE RULES:
- Output ONLY a single JSON object. No prose. No markdown fences. No
  preamble. No explanation. No image links. No questions.
- Top-level shape is exactly:
    {"constraints": [{"id": str, "type": "count"|"presence"|"absence"|
    "position"|"size", "description": str, "object_prompt": str,
    "expected_count": int|null, "position": "left"|"right"|"top"|
    "bottom"|"center"|"top-left"|"top-right"|"bottom-left"|
    "bottom-right"|null, "min_area_ratio": float|null,
    "max_area_ratio": float|null}, ...]}
- Every constraint has ALL nine fields. Use null for unused fields.
- "object_prompt" is a short noun phrase for a segmentation model:
  "red chair", "beach", "tree". Never a sentence.
- "description" is one short sentence.

Examples:

Input: "exactly 3 red chairs on a beach"
Output: {"constraints":[{"id":"count_chairs","type":"count",
"description":"Exactly 3 red chairs visible","object_prompt":"red chair",
"expected_count":3,"position":null,"min_area_ratio":null,
"max_area_ratio":null},{"id":"presence_beach","type":"presence",
"description":"Beach setting visible","object_prompt":"beach",
"expected_count":null,"position":null,"min_area_ratio":null,
"max_area_ratio":null}]}

Input: "logo in the bottom-right corner, no people"
Output: {"constraints":[{"id":"position_logo","type":"position",
"description":"Logo in bottom-right","object_prompt":"logo",
"expected_count":null,"position":"bottom-right","min_area_ratio":null,
"max_area_ratio":null},{"id":"absence_people","type":"absence",
"description":"No people","object_prompt":"person","expected_count":null,
"position":null,"min_area_ratio":null,"max_area_ratio":null}]}

Input: "a tiny logo in the corner"
Output: {"constraints":[{"id":"size_logo","type":"size",
"description":"Logo is small","object_prompt":"logo","expected_count":null,
"position":null,"min_area_ratio":null,"max_area_ratio":0.05}]}

Start your final answer immediately with '{'. Do not narrate or explain
before the JSON. If you have thought about the answer, your last output
must be the JSON object and nothing else.

Return nothing outside the JSON object."""


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
