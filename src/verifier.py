from __future__ import annotations

import numpy as np

from src.schema import Constraint


def _iou(mask_a: np.ndarray, mask_b: np.ndarray) -> float:
    if mask_a.size == 0 or mask_b.size == 0:
        return 0.0
    intersection = np.logical_and(mask_a, mask_b).sum()
    union = np.logical_or(mask_a, mask_b).sum()
    return float(intersection / union) if union else 0.0


def filter_masks(masks: list[dict], iou_threshold: float) -> list[dict]:
    """Remove duplicate/overlapping masks using IoU. Keep the highest-score mask in each overlapping cluster."""
    kept: list[dict] = []
    for mask_entry in sorted(masks, key=lambda item: float(item.get("score", 0.0)), reverse=True):
        mask = np.asarray(mask_entry.get("mask", np.zeros((0, 0), dtype=bool)), dtype=bool)
        if np.all(~mask):
            continue
        if any(_iou(np.asarray(existing.get("mask", np.zeros((0, 0), dtype=bool)), dtype=bool), mask) > iou_threshold for existing in kept):
            continue
        kept.append(mask_entry)
    return kept


def mask_area_ratio(mask: np.ndarray) -> float:
    """Return mask.sum() / mask.size."""
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0:
        return 0.0
    return float(arr.sum() / arr.size)


def mask_centroid(mask: np.ndarray) -> tuple[float, float]:
    """Return (cx, cy) normalized to [0, 1]. Raise ValueError if mask is empty."""
    arr = np.asarray(mask, dtype=bool)
    if arr.size == 0 or not arr.any():
        raise ValueError("mask is empty")
    ys, xs = np.where(arr)
    cx = float(xs.mean() / arr.shape[1])
    cy = float(ys.mean() / arr.shape[0])
    return cx, cy


def check_position(centroid: tuple[float, float], position: str, tolerance: float) -> bool:
    """Return True if centroid is inside the named region with tolerance."""
    cx, cy = centroid
    tol = float(tolerance)
    if position == "center":
        return abs(cx - 0.5) <= 0.5 * (1.0 + tol) and abs(cy - 0.5) <= 0.5 * (1.0 + tol)
    if position == "left":
        return cx <= 0.5 + tol and cx < 0.5 + tol
    if position == "right":
        return cx >= 0.5 - tol
    if position == "top":
        return cy <= 0.5 + tol
    if position == "bottom":
        return cy >= 0.5 - tol
    if position == "top-left":
        return cx <= 0.5 + tol and cy <= 0.5 + tol
    if position == "top-right":
        return cx >= 0.5 - tol and cy <= 0.5 + tol
    if position == "bottom-left":
        return cx <= 0.5 + tol and cy >= 0.5 - tol
    if position == "bottom-right":
        return cx >= 0.5 - tol and cy >= 0.5 - tol
    raise ValueError(f"Unsupported position: {position}")


def check_constraint(constraint: Constraint, masks: list[dict], image_shape: tuple[int, int]) -> dict:
    """Return {\"constraint_id\": ..., \"passed\": bool, \"reason\": str, \"masks_used\": [...]}."""
    reason = "" 
    masks_used = []
    prompt_masks = list(masks)

    if constraint.type == "count":
        count = len(prompt_masks)
        passed = count == constraint.expected_count
        reason = f"Expected {constraint.expected_count} objects but found {count}." if not passed else f"Found the expected {count} objects."
        masks_used = [mask.get("mask", None) for mask in prompt_masks]
    elif constraint.type == "presence":
        passed = len(prompt_masks) > 0
        reason = "At least one object is present." if passed else "No matching object was found."
        masks_used = [mask.get("mask", None) for mask in prompt_masks]
    elif constraint.type == "absence":
        passed = len(prompt_masks) == 0
        reason = "No matching objects were detected." if passed else "Matching object detected when it should be absent."
        masks_used = [mask.get("mask", None) for mask in prompt_masks]
    elif constraint.type == "position":
        if not prompt_masks:
            passed = False
            reason = "No masks available to evaluate position."
        else:
            mask = prompt_masks[0]["mask"]
            centroid = mask_centroid(mask)
            passed = check_position(centroid, constraint.position or "center", 0.10)
            reason = (
                f"Centroid {centroid} is within {constraint.position} region."
                if passed
                else f"Centroid {centroid} is outside the {constraint.position} region."
            )
            masks_used = [mask]
    elif constraint.type == "size":
        if not prompt_masks:
            passed = False
            reason = "No masks available to evaluate size."
        else:
            mask = prompt_masks[0]["mask"]
            ratio = mask_area_ratio(mask)
            lower = constraint.min_area_ratio if constraint.min_area_ratio is not None else 0.0
            upper = constraint.max_area_ratio if constraint.max_area_ratio is not None else float("inf")
            passed = lower <= ratio <= upper
            reason = (
                f"Mask area ratio {ratio:.4f} is within [{lower}, {upper}]."
                if passed
                else f"Mask area ratio {ratio:.4f} is outside [{lower}, {upper}]."
            )
            masks_used = [mask]
    else:
        raise ValueError(f"Unsupported constraint type: {constraint.type}")

    return {
        "constraint_id": constraint.id,
        "passed": bool(passed),
        "reason": reason,
        "masks_used": masks_used,
    }


def verify_all(constraints: list[Constraint], masks_by_prompt: dict[str, list[dict]], image_shape) -> list[dict]:
    """Verify every constraint. masks_by_prompt maps object_prompt -> masks returned by SAM."""
    results = []
    for constraint in constraints:
        masks = masks_by_prompt.get(constraint.object_prompt, [])
        results.append(check_constraint(constraint, masks, image_shape))
    return results
