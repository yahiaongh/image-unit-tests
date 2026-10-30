from __future__ import annotations

import json
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image

from src.json_utils import parse_constraints
from src.schema import Constraint
from src.verifier import filter_masks, mask_area_ratio, mask_centroid, verify_all


class Agent:
    def __init__(self, spark, image, sam, config):
        self.spark = spark
        self.image = image
        self.sam = sam
        self.config = config or {}
        agent_cfg = self.config.get("agent", {})
        self.max_rounds = int(agent_cfg.get("max_rounds", 4))
        self.iou_threshold = float(agent_cfg.get("iou_threshold", 0.5))
        self.min_mask_area_ratio = float(agent_cfg.get("min_mask_area_ratio", 0.0001))
        self.position_tolerance = float(agent_cfg.get("position_tolerance", 0.10))
        self.runs_dir = Path(self.config.get("runs_dir", "runs"))

    def _save_round(self, run_dir: Path, round_index: int, image_bytes: bytes, masks_by_prompt: dict, verdicts: list[dict], fix_instruction: str | None) -> None:
        image_path = run_dir / f"round_{round_index}.png"
        if image_bytes:
            with Image.open(BytesIO(image_bytes)).convert("RGB") as image:
                image.save(image_path)

        mask_metadata = []
        mask_index = 0
        for prompt, entries in masks_by_prompt.items():
            for entry in entries:
                mask = np.asarray(entry.get("mask"), dtype=bool)
                if mask.ndim != 2:
                    raise ValueError(f"Expected a 2D mask for prompt {prompt!r}")
                mask_name = f"round_{round_index}_mask_{mask_index}.png"
                Image.fromarray(mask.astype(np.uint8) * 255, mode="L").save(
                    run_dir / mask_name
                )
                box = entry.get("box")
                mask_metadata.append({
                    "prompt": prompt,
                    "score": float(entry.get("score", 0.0)),
                    "box": list(box) if box is not None else None,
                    "area_ratio": mask_area_ratio(mask),
                    "centroid": list(mask_centroid(mask)) if mask.any() else None,
                    "mask_path": mask_name,
                })
                mask_index += 1

        masks_path = run_dir / f"round_{round_index}_masks.json"
        masks_path.write_text(
            json.dumps(mask_metadata, indent=2), encoding="utf-8"
        )

        verdicts_path = run_dir / f"round_{round_index}_verdicts.json"
        compact_verdicts = [
            {key: value for key, value in verdict.items() if key != "masks_used"}
            for verdict in verdicts
        ]
        verdicts_path.write_text(
            json.dumps(compact_verdicts, indent=2), encoding="utf-8"
        )

        fix_path = run_dir / f"round_{round_index}_fix.txt"
        fix_path.write_text(fix_instruction or "", encoding="utf-8")

    def _build_fix_instruction(self, constraints: list, verdicts: list[dict]) -> str:
        failed = []
        verdict_by_id = {item["constraint_id"]: item for item in verdicts}
        for constraint in constraints:
            verdict = verdict_by_id.get(constraint.id)
            if verdict and not verdict["passed"]:
                failed.append((constraint, verdict))

        if not failed:
            return "No fix needed."

        parts = []
        for constraint, verdict in failed:
            reason = verdict.get("reason", "")
            detail = constraint.description
            if constraint.type == "count" and constraint.expected_count is not None:
                found_count = len(verdict.get("masks_used") or [])
                expected_count = constraint.expected_count
                difference = abs(found_count - expected_count)
                object_name = constraint.object_prompt
                plural_name = object_name if object_name.endswith("s") else f"{object_name}s"
                singular_name = object_name[:-1] if object_name.endswith("s") else object_name
                found_name = singular_name if found_count == 1 else plural_name
                delta_name = singular_name if difference == 1 else plural_name
                if found_count > expected_count:
                    correction = (
                        f"The image contains {found_count} {found_name}, which is "
                        f"{difference} too many. You must remove exactly {difference} "
                        f"{delta_name}."
                    )
                else:
                    correction = (
                        f"The image contains {found_count} {found_name}, which is "
                        f"{difference} too few. You must add exactly {difference} "
                        f"more {delta_name}."
                    )
                parts.append(f"Constraint '{detail}' failed because {reason} {correction}")
            elif constraint.type == "position" and constraint.position:
                parts.append(
                    f"Constraint '{detail}' failed because {reason} Reposition the {constraint.object_prompt} to the {constraint.position} side."
                )
            else:
                parts.append(f"Constraint '{detail}' failed because {reason} Adjust the scene to satisfy the requirement.")

        return " ".join(parts)

    def run(self, brief: str, constraints: list[Constraint] | None = None) -> dict:
        if constraints is None:
            raw_constraints = self.spark.generate_constraints(brief)
            constraints = parse_constraints(json.dumps(raw_constraints))
        image_bytes = self.image.generate(brief)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        run_dir = self.runs_dir / timestamp
        run_dir.mkdir(parents=True, exist_ok=True)

        final_passed = False
        all_rounds = []
        image_shape = Image.open(BytesIO(image_bytes)).size

        for round_index in range(1, self.max_rounds + 1):
            masks_by_prompt: dict[str, list[dict]] = {}
            unique_prompts = sorted({constraint.object_prompt for constraint in constraints})
            for prompt in unique_prompts:
                masks = self.sam.segment(image_bytes, prompt)
                masks_by_prompt[prompt] = filter_masks(masks, self.iou_threshold)

            verdicts = verify_all(constraints, masks_by_prompt, image_shape)
            fix_instruction = None
            if all(item["passed"] for item in verdicts):
                final_passed = True
                all_rounds.append({
                    "round": round_index,
                    "passed": True,
                    "verdicts": verdicts,
                    "masks": masks_by_prompt,
                    "fix_instruction": "",
                })
                self._save_round(run_dir, round_index, image_bytes, masks_by_prompt, verdicts, "")
                break

            if round_index >= self.max_rounds:
                final_passed = False
                all_rounds.append({
                    "round": round_index,
                    "passed": False,
                    "verdicts": verdicts,
                    "masks": masks_by_prompt,
                    "fix_instruction": "",
                })
                self._save_round(run_dir, round_index, image_bytes, masks_by_prompt, verdicts, "")
                break

            fix_instruction = self._build_fix_instruction(constraints, verdicts)
            all_rounds.append({
                "round": round_index,
                "passed": False,
                "verdicts": verdicts,
                "masks": masks_by_prompt,
                "fix_instruction": fix_instruction,
            })
            self._save_round(run_dir, round_index, image_bytes, masks_by_prompt, verdicts, fix_instruction)
            image_bytes = self.image.edit(image_bytes, fix_instruction)

        summary = {
            "brief": brief,
            "rounds": len(all_rounds),
            "final_passed": final_passed,
            "constraints": [constraint.model_dump() for constraint in constraints],
            "run_dir": str(run_dir),
        }
        (run_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

        return {
            "brief": brief,
            "rounds": all_rounds,
            "final_image": image_bytes,
            "final_passed": final_passed,
            "constraints": constraints,
        }
