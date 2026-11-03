from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
RESULTS_PATH = ROOT / "runs" / "single_step" / "results.csv"
AUDIT_CSV_PATH = ROOT / "runs" / "audit" / "master_ground_truth.csv"
AUDIT_DIR = ROOT / "runs" / "audit"
PROMPT_MAP_PATH = AUDIT_DIR / "master_prompt_map.json"
EXPORT_DIR = ROOT / "runs" / "export"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _human_audit_by_hash(
    audit_csv_path: Path,
    audit_dir: Path,
) -> dict[str, dict[str, object]]:
    with audit_csv_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    audited = {}
    for row in rows:
        filename = row.get("filename", "").strip()
        human_count = row.get("human_count", "").strip()
        artifact = row.get("visible_artifact", "").strip().lower()
        if not filename or not human_count:
            raise ValueError(f"Incomplete human audit row: {row}")
        if artifact not in {"y", "yes", "n", "no"}:
            raise ValueError(f"Unrecognized artifact verdict for {filename}: {artifact!r}")
        image_path = (audit_dir / filename).resolve()
        if not image_path.is_relative_to(audit_dir.resolve()):
            raise ValueError(f"Audit filename escapes the audit directory: {filename}")
        if not image_path.is_file():
            raise FileNotFoundError(f"Audited image is missing: {image_path}")
        image_hash = _sha256(image_path)
        if image_hash in audited:
            raise ValueError(f"Duplicate audited image content for {filename}")
        audited[image_hash] = {
            "filename": filename,
            "human_count": int(human_count),
            "visible_artifact": artifact in {"y", "yes"},
            "image_path": image_path,
        }
    return audited


def _find_clean_numeric_outcome(
    results_path: Path,
    audit_csv_path: Path,
    audit_dir: Path,
    prompt_map_path: Path,
) -> dict[str, object]:
    audited = _human_audit_by_hash(audit_csv_path, audit_dir)
    with results_path.open(newline="", encoding="utf-8") as handle:
        result_rows = list(csv.DictReader(handle))
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in result_rows:
        grouped[(row["brief_id"], row["seed"])].append(row)

    candidates = []
    for (brief_id, seed), rows in grouped.items():
        replicate_dir = results_path.parent / brief_id / f"seed_{seed}"
        if not replicate_dir.is_dir():
            replicate_dir = results_path.parent / brief_id
        for row in rows:
            if row["condition"].strip().lower() != "numeric_edit":
                continue
            expected = int(row["expected_count"])
            image_path = replicate_dir / "numeric_edit.png"
            if not image_path.is_file():
                raise FileNotFoundError(f"Numeric outcome image is missing: {image_path}")
            audit = audited.get(_sha256(image_path))
            if audit is None:
                raise ValueError(f"Numeric outcome has no human audit: {image_path}")
            if (
                audit["human_count"] == expected
                and not audit["visible_artifact"]
            ):
                candidates.append({
                    "brief_id": brief_id,
                    "seed": seed,
                    "condition": row["condition"].strip(),
                    "expected_count": expected,
                    "sam_after_count": int(row["sam_after"]),
                    "human_count": int(audit["human_count"]),
                    "image_path": image_path,
                    "audit_filename": audit["filename"],
                })

    if len(candidates) != 1:
        raise ValueError(
            "Expected exactly one human-correct, artifact-free Numeric outcome; "
            f"found {len(candidates)}."
        )
    candidate = candidates[0]
    prompt_map = json.loads(prompt_map_path.read_text(encoding="utf-8"))
    audit_filename = candidate["audit_filename"]
    object_name = prompt_map.get(audit_filename)
    if not object_name:
        raise ValueError(f"No object category is mapped for {audit_filename}")
    candidate["category_name"] = str(object_name)
    return candidate


def export_clean_numeric_outcome(
    results_path: Path = RESULTS_PATH,
    audit_csv_path: Path = AUDIT_CSV_PATH,
    audit_dir: Path = AUDIT_DIR,
    prompt_map_path: Path = PROMPT_MAP_PATH,
    export_dir: Path = EXPORT_DIR,
) -> Path:
    candidate = _find_clean_numeric_outcome(
        results_path,
        audit_csv_path,
        audit_dir,
        prompt_map_path,
    )

    image_dir = export_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_name = (
        f"accepted_numeric_{candidate['brief_id']}_seed_{candidate['seed']}.png"
    )
    exported_image = image_dir / image_name
    shutil.copyfile(candidate["image_path"], exported_image)
    with Image.open(exported_image) as image:
        width, height = image.size

    annotation = {
        "id": 1,
        "image_id": 1,
        "category_id": 1,
        "segmentation": [],
        "area": width * height,
        "bbox": [0, 0, width, height],
        "iscrowd": 0,
        "num_instances": candidate["expected_count"],
        "attributes": {
            "human_verified": True,
            "artifact": False,
            "mask_quality": "placeholder_full_frame",
        },
    }
    coco = {
        "info": {
            "description": (
                "Human-audited synthetic image export. The annotation uses a "
                "dummy full-image bounding box because SAM instance masks were "
                "not available in the saved experiment artifacts; it is not an "
                "instance-level training annotation."
            ),
            "version": "1.0",
        },
        "licenses": [],
        "images": [{
            "id": 1,
            "file_name": f"images/{image_name}",
            "width": width,
            "height": height,
            "expected_count": candidate["expected_count"],
            "human_verified_count": candidate["human_count"],
            "human_verified": True,
            "artifact": False,
            "mask_quality": "placeholder_full_frame",
            "source_brief_id": candidate["brief_id"],
            "source_seed": int(candidate["seed"]),
        }],
        "annotations": [annotation],
        "categories": [{
            "id": 1,
            "name": candidate["category_name"],
            "supercategory": "object",
        }],
    }
    annotation_path = export_dir / "coco_annotations.json"
    export_dir.mkdir(parents=True, exist_ok=True)
    annotation_path.write_text(json.dumps(coco, indent=2) + "\n", encoding="utf-8")
    return annotation_path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Export the human-verified clean Numeric outcome in COCO shape."
    )
    parser.add_argument("--results", type=Path, default=RESULTS_PATH)
    parser.add_argument("--audit-csv", type=Path, default=AUDIT_CSV_PATH)
    parser.add_argument("--audit-dir", type=Path, default=AUDIT_DIR)
    parser.add_argument("--prompt-map", type=Path, default=PROMPT_MAP_PATH)
    parser.add_argument("--output", type=Path, default=EXPORT_DIR)
    args = parser.parse_args()
    annotation_path = export_clean_numeric_outcome(
        args.results,
        args.audit_csv,
        args.audit_dir,
        args.prompt_map,
        args.output,
    )
    print(f"Wrote {annotation_path}")
    print(f"Copied image to {args.output / 'images'}")
    print("Annotation uses a dummy full-image bbox, not an instance mask.")


if __name__ == "__main__":
    main()
