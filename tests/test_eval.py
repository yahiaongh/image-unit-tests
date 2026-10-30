from __future__ import annotations

from pathlib import Path

from src.eval import run_eval
from src.schema import Constraint


class SequenceImage:
    def __init__(self):
        self.calls = 0

    def generate(self, brief: str) -> bytes:
        self.calls += 1
        return f"image-{self.calls}".encode()


def test_pass_at_four_generates_and_verifies_all_four_images(monkeypatch):
    image = SequenceImage()
    seen_images = []

    def fake_verdicts(image_bytes, constraints, sam, config):
        seen_images.append(image_bytes)
        passed = image_bytes == b"image-2"
        return ([{"constraint_id": "count_apples", "passed": passed}], ["green apples"])

    monkeypatch.setattr(run_eval, "_verdicts_for_image", fake_verdicts)
    constraint = Constraint(
        id="count_apples",
        type="count",
        description="exactly 1 apple",
        object_prompt="green apples",
        expected_count=1,
    )

    result = run_eval.run_pass_at_four(
        {"id": "test", "brief": "one apple"},
        {"eval": {"cost_per_image_usd": 0.01, "cost_per_sam_call_usd": 0.0025}},
        (None, image, None),
        [constraint],
    )

    assert result.passed is True
    assert result.attempts == 2
    assert result.images_generated == 4
    assert image.calls == 4
    assert seen_images == [b"image-1", b"image-2", b"image-3", b"image-4"]
    assert result.cost_usd == 0.05


def test_random_retry_repeats_original_prompt_until_pass(monkeypatch):
    image = SequenceImage()
    seen_images = []

    def fake_verdicts(image_bytes, constraints, sam, config):
        seen_images.append(image_bytes)
        passed = image_bytes == b"image-3"
        return ([{"constraint_id": "count_apples", "passed": passed}], ["green apples"])

    monkeypatch.setattr(run_eval, "_verdicts_for_image", fake_verdicts)
    constraint = Constraint(
        id="count_apples",
        type="count",
        description="exactly 1 apple",
        object_prompt="green apples",
        expected_count=1,
    )

    result = run_eval.run_random_retry(
        {"id": "test", "brief": "one apple"},
        {"eval": {"cost_per_image_usd": 0.01, "cost_per_sam_call_usd": 0.0025}},
        (None, image, None),
        [constraint],
    )

    assert result.passed is True
    assert result.attempts == 3
    assert result.images_generated == 3
    assert image.calls == 3
    assert seen_images == [b"image-1", b"image-2", b"image-3"]
    assert abs(result.cost_usd - 0.0375) < 1e-12


def test_summary_has_all_requested_columns_and_average_cost_per_image(tmp_path: Path):
    rows = [
        {
            "mode": "one-shot",
            "passed": True,
            "rounds": 1,
            "images_generated": 1,
            "cost_usd": 0.02,
        },
        {
            "mode": "random_retry",
            "passed": True,
            "rounds": 2,
            "images_generated": 2,
            "cost_usd": 0.04,
        },
        {
            "mode": "pass@4",
            "passed": False,
            "rounds": 4,
            "images_generated": 4,
            "cost_usd": 0.08,
        },
        {
            "mode": "loop",
            "passed": True,
            "rounds": 2,
            "images_generated": 2,
            "cost_usd": 0.04,
        },
    ]

    run_eval._write_summary(tmp_path, rows)

    summary = (tmp_path / "summary.md").read_text()
    assert "| Mode | Pass Rate | Avg Rounds (iterative modes) | Avg Cost per Image* |" in summary
    assert "| one-shot | 1.00 | — | $0.0200 |" in summary
    assert "| random_retry | 1.00 | 2.00 | $0.0200 |" in summary
    assert "| pass@4 | 0.00 | — | $0.0200 |" in summary
    assert "| loop | 1.00 | 2.00 | $0.0200 |" in summary


def test_pass_at_four_cost_reservation_includes_four_images_and_sam_calls():
    config = {
        "eval": {
            "cost_per_image_usd": 0.01,
            "cost_per_sam_call_usd": 0.0025,
        }
    }

    assert run_eval.estimate_cost("pass@4", 2, 4, config) == 0.06


def test_summary_parses_false_values_loaded_from_csv(tmp_path: Path):
    row = {
        "mode": "one-shot",
        "passed": "False",
        "rounds": "1",
        "images_generated": "1",
        "cost_usd": "0.01",
        "error": "",
    }

    run_eval._write_summary(tmp_path, [row])

    summary = (tmp_path / "summary.md").read_text()
    assert "| one-shot | 0.00 | — | $0.0100 |" in summary
