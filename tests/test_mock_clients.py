from io import BytesIO

import numpy as np
from PIL import Image

from src.clients import get_clients
from src.clients.mock import MockImageClient, MockSamClient, MockSparkClient


def test_mock_spark_parses_count_position_and_absence_constraints():
    constraints = MockSparkClient().generate_constraints(
        "Exactly 3 red chairs on the left and no people."
    )

    assert constraints == [
        {
            "id": "count-1",
            "type": "count",
            "description": "exactly 3 red chairs",
            "object_prompt": "red chairs",
            "expected_count": 3,
        },
        {
            "id": "position-1",
            "type": "position",
            "description": "red chairs on the left",
            "object_prompt": "red chairs",
            "position": "left",
        },
        {
            "id": "absence-1",
            "type": "absence",
            "description": "no people",
            "object_prompt": "people",
        },
    ]


def test_mock_spark_falls_back_when_brief_has_no_supported_constraint():
    constraints = MockSparkClient().generate_constraints("A quiet sunset.")

    assert [item["type"] for item in constraints] == ["count", "position"]
    assert constraints[0]["expected_count"] == 1
    assert constraints[1]["position"] == "center"


def test_mock_sam_matches_chair_count_and_can_force_first_round_failure():
    brief = "A photo of exactly 3 chairs."
    image_bytes = MockImageClient().generate(brief)

    sam = MockSamClient(force_fail_first_round=True)
    first_round_masks = sam.segment(image_bytes, "chairs")
    second_round_masks = sam.segment(image_bytes, "chairs")

    assert len(first_round_masks) == 4
    assert len(second_round_masks) == 3
    assert all(mask["mask"].dtype == np.bool_ for mask in first_round_masks)


def test_mock_image_draws_brief_text_and_preserves_it_for_segmentation():
    brief = "Exactly 4 chairs on a beach."
    image_bytes = MockImageClient().generate(brief)

    with Image.open(BytesIO(image_bytes)) as image:
        assert image.info["brief"] == brief
        pixels = np.asarray(image.convert("RGB"))
    assert np.any(np.all(pixels < 100, axis=2))

    _, _, sam = get_clients({"mock": {"enabled": True}})
    assert len(sam.segment(image_bytes, "chairs")) == 4
