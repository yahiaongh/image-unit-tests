import numpy as np
from PIL import Image

from src.ui.app import render_round_overlay


def test_render_round_overlay_numbers_each_mask_at_its_centroid():
    image = Image.new("RGB", (100, 100), "white")
    first_mask = np.zeros((100, 100), dtype=bool)
    first_mask[10:45, 10:45] = True
    second_mask = np.zeros((100, 100), dtype=bool)
    second_mask[55:90, 55:90] = True

    rendered, _ = render_round_overlay(
        image,
        {
            "apples": [
                {"mask": first_mask, "score": 0.9},
                {"mask": second_mask, "score": 0.9},
            ]
        },
        [{"object_prompt": "apples", "passed": True}],
    )

    assert rendered is not None
    for center_x, center_y in ((27, 27), (72, 72)):
        crop = np.asarray(rendered)[center_y - 10:center_y + 11, center_x - 10:center_x + 11]
        assert np.any(np.all(crop == (255, 255, 255), axis=2))
