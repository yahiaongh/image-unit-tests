from typing import Literal, Optional

from pydantic import BaseModel, field_validator


class Constraint(BaseModel):
    id: str
    type: Literal["count", "presence", "absence", "position", "size"]
    description: str
    object_prompt: str
    expected_count: Optional[int] = None
    position: Optional[
        Literal[
            "left",
            "right",
            "top",
            "bottom",
            "center",
            "top-left",
            "top-right",
            "bottom-left",
            "bottom-right",
        ]
    ] = None
    min_area_ratio: Optional[float] = None
    max_area_ratio: Optional[float] = None

    @field_validator("expected_count")
    @classmethod
    def count_must_be_positive(cls, v):
        if v is not None and v < 0:
            raise ValueError("expected_count must be >= 0")
        return v


class ConstraintList(BaseModel):
    constraints: list[Constraint]
