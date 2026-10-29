import numpy as np

from src.schema import Constraint
from src.verifier import check_constraint, check_position, filter_masks, mask_area_ratio, mask_centroid, verify_all


def test_filter_masks_removes_overlaps_and_keeps_highest_score():
    mask_a = np.ones((10, 10), dtype=bool)
    mask_b = np.ones((10, 10), dtype=bool)
    mask_c = np.zeros((10, 10), dtype=bool)
    mask_c[:5, :5] = True

    masks = [
        {"mask": mask_a, "score": 0.6},
        {"mask": mask_b, "score": 0.9},
        {"mask": mask_c, "score": 0.8},
    ]

    filtered = filter_masks(masks, iou_threshold=0.5)
    assert len(filtered) == 2
    assert filtered[0]["score"] == 0.9


def test_mask_centroid_raises_for_empty_mask():
    empty = np.zeros((5, 5), dtype=bool)
    try:
        mask_centroid(empty)
        assert False, "Expected ValueError"
    except ValueError:
        pass


def test_count_constraint_passes_and_fails():
    constraint_pass = Constraint(
        id="count_ok",
        type="count",
        description="exactly 2 objects",
        object_prompt="chair",
        expected_count=2,
    )
    constraint_fail = Constraint(
        id="count_bad",
        type="count",
        description="exactly 2 objects",
        object_prompt="chair",
        expected_count=2,
    )

    pass_result = check_constraint(constraint_pass, [{"mask": np.ones((10, 10), dtype=bool)}, {"mask": np.ones((10, 10), dtype=bool)}], (100, 100))
    fail_result = check_constraint(constraint_fail, [{"mask": np.ones((10, 10), dtype=bool)}], (100, 100))

    assert pass_result["passed"] is True
    assert fail_result["passed"] is False


def test_position_constraint_passes_when_in_region_and_fails_when_outside():
    constraint = Constraint(
        id="pos",
        type="position",
        description="object is left",
        object_prompt="chair",
        position="left",
    )
    left_mask = np.zeros((20, 20), dtype=bool)
    left_mask[:, :8] = True
    right_mask = np.zeros((20, 20), dtype=bool)
    right_mask[:, 12:] = True

    left_result = check_constraint(constraint, [{"mask": left_mask}], (20, 20))
    right_result = check_constraint(constraint, [{"mask": right_mask}], (20, 20))

    assert left_result["passed"] is True
    assert right_result["passed"] is False


def test_size_constraint_passes_when_area_is_in_bounds_and_fails_when_outside():
    constraint_ok = Constraint(
        id="size_ok",
        type="size",
        description="mask area is acceptable",
        object_prompt="apple",
        min_area_ratio=0.1,
        max_area_ratio=0.5,
    )
    constraint_bad = Constraint(
        id="size_bad",
        type="size",
        description="mask area is too large",
        object_prompt="apple",
        min_area_ratio=0.1,
        max_area_ratio=0.5,
    )

    ok_mask = np.zeros((40, 40), dtype=bool)
    ok_mask[10:30, 10:30] = True
    bad_mask = np.zeros((40, 40), dtype=bool)
    bad_mask[5:35, 5:35] = True

    ok_result = check_constraint(constraint_ok, [{"mask": ok_mask}], (40, 40))
    bad_result = check_constraint(constraint_bad, [{"mask": bad_mask}], (40, 40))

    assert ok_result["passed"] is True
    assert bad_result["passed"] is False


def test_verify_all_returns_one_result_per_constraint_and_does_not_call_clients():
    constraints = [
        Constraint(id="c1", type="count", description="one object", object_prompt="chair", expected_count=1),
        Constraint(id="c2", type="position", description="object is centered", object_prompt="chair", position="center"),
    ]
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:7, 4:7] = True

    results = verify_all(constraints, {"chair": [{"mask": mask, "score": 0.9}]}, (10, 10))

    assert len(results) == 2
    assert {result["constraint_id"] for result in results} == {"c1", "c2"}
