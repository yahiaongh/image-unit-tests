from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from src.agent import Agent
from src.schema import Constraint


class StubSpark:
    def generate_constraints(self, brief: str):
        return [
            {
                "id": "c1",
                "type": "count",
                "description": "exactly 1 object",
                "object_prompt": "circle",
                "expected_count": 1,
            },
            {
                "id": "c2",
                "type": "position",
                "description": "object is centered",
                "object_prompt": "circle",
                "position": "center",
            },
        ]


class StubImage:
    def __init__(self):
        self.calls = 0

    def generate(self, brief: str) -> bytes:
        image = Image.new("RGB", (100, 100), color=(255, 255, 255))
        buffer = __import__("io").BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()

    def edit(self, image_bytes: bytes, fix_instruction: str) -> bytes:
        self.calls += 1
        image = Image.new("RGB", (100, 100), color=(255, 255, 255))
        buffer = __import__("io").BytesIO()
        image.save(buffer, format="PNG")
        return buffer.getvalue()


class StubSam:
    def __init__(self):
        self.calls = 0

    def segment(self, image_bytes: bytes, text_prompt: str):
        self.calls += 1
        if self.calls == 1:
            left = np.zeros((100, 100), dtype=bool)
            right = np.zeros((100, 100), dtype=bool)
            left[20:80, 10:40] = True
            right[20:80, 60:90] = True
            return [
                {"mask": left, "score": 0.95, "box": [10, 20, 40, 80]},
                {"mask": right, "score": 0.9, "box": [60, 20, 90, 80]},
            ]

        mask = np.zeros((100, 100), dtype=bool)
        mask[35:65, 35:65] = True
        return [{"mask": mask, "score": 0.95, "box": [35, 35, 65, 65]}]


def test_agent_runs_loop_until_constraints_pass_and_logs_rounds(tmp_path):
    spark = StubSpark()
    image = StubImage()
    sam = StubSam()
    config = {"agent": {"max_rounds": 4, "iou_threshold": 0.5, "min_mask_area_ratio": 0.0001, "position_tolerance": 0.10}, "runs_dir": str(tmp_path)}

    agent = Agent(spark, image, sam, config)
    result = agent.run("A single centered circle")

    assert result["final_passed"] is True
    assert len(result["rounds"]) == 2
    assert "exactly 1 object" in result["rounds"][0]["fix_instruction"]
    run_dir = next(tmp_path.iterdir())
    assert (run_dir / "summary.json").exists()
    assert any(path.name.startswith("round_1") for path in run_dir.iterdir())
    assert any(path.name.startswith("round_2") for path in run_dir.iterdir())
