from __future__ import annotations

import argparse
import argparse
import csv
import json
import random
import re
import secrets
import shutil
import tempfile
from pathlib import Path

import gradio as gr

ROOT = Path(__file__).resolve().parents[1]
AUDIT_DIR = ROOT / "runs" / "audit"
CSV_PATH = AUDIT_DIR / "master_ground_truth.csv"
RESULTS_PATH = ROOT / "runs" / "single_step" / "results.csv"
BRIEFS_PATH = ROOT / "src" / "eval" / "single_step_briefs.jsonl"
PROMPT_MAP_PATH = AUDIT_DIR / "master_prompt_map.json"
MASTER_COLUMNS = ("filename", "human_count", "visible_artifact")
CONDITIONS = ("numeric_edit", "generic_edit", "marked_edit", "control_edit")


def _read_csv(path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {path}")
        return list(reader), list(reader.fieldnames)


def _object_from_brief(brief: str) -> str:
    match = re.search(
        r"\bexactly\s+\d+\s+(.+?)(?:[,.!?;]|$)",
        brief,
        flags=re.IGNORECASE,
    )
    if not match:
        return "objects"
    phrase = re.split(
        r"\b(?:playing|arranged|in|on|across|with|near|spread|visible|"
        r"distributed|lined|grouped|from|through|by|under|over|at|"
        r"confined|holding|mostly)\b",
        match.group(1),
        maxsplit=1,
        flags=re.IGNORECASE,
    )[0].strip()
    return phrase or "objects"


def prepare_master_audit(
    results_path: Path = RESULTS_PATH,
    briefs_path: Path = BRIEFS_PATH,
    audit_dir: Path | None = None,
) -> int:
    if audit_dir is None:
        audit_dir = AUDIT_DIR
    csv_path = audit_dir / "master_ground_truth.csv"
    prompt_map_path = audit_dir / "master_prompt_map.json"
    images_dir = audit_dir / "master_images"
    if csv_path.is_file() and prompt_map_path.is_file():
        rows, _ = _read_csv(csv_path)
        if len(rows) != 105:
            raise ValueError(
                f"Existing master audit has {len(rows)} rows; expected 105."
            )
        return len(rows)

    result_rows, _ = _read_csv(results_path)
    with briefs_path.open(encoding="utf-8") as handle:
        briefs = {
            item["id"]: item["brief"]
            for line in handle
            if line.strip()
            for item in [json.loads(line)]
        }

    replicates: dict[tuple[str, str], list[dict[str, str]]] = {}
    for row in result_rows:
        key = (row["brief_id"], row["seed"])
        replicates.setdefault(key, []).append(row)
    if len(replicates) != 21:
        raise ValueError(
            f"Expected 21 failed starting images, found {len(replicates)}."
        )

    images: list[tuple[Path, str]] = []
    for (brief_id, seed), rows in sorted(replicates.items()):
        conditions = {row["condition"] for row in rows}
        if conditions != set(CONDITIONS):
            raise ValueError(
                f"{brief_id} seed {seed} does not have all four outcome conditions."
            )
        brief = briefs.get(brief_id)
        if not brief:
            raise ValueError(f"Brief text is unavailable for {brief_id}.")
        object_name = _object_from_brief(brief)
        replicate_dir = results_path.parent / brief_id / f"seed_{seed}"
        original_path = replicate_dir / "original.png"
        if not original_path.is_file():
            raise FileNotFoundError(f"Starting image is missing: {original_path}")
        images.append((original_path, object_name))
        for row in rows:
            outcome_path = replicate_dir / f"{row['condition']}.png"
            if not outcome_path.is_file():
                raise FileNotFoundError(f"Outcome image is missing: {outcome_path}")
            images.append((outcome_path, object_name))

    if len(images) != 105:
        raise ValueError(f"Expected 105 images to audit, found {len(images)}.")

    random.SystemRandom().shuffle(images)
    images_dir.mkdir(parents=True, exist_ok=True)
    audit_dir.mkdir(parents=True, exist_ok=True)
    audit_rows = []
    prompt_map = {}
    for source_path, object_name in images:
        filename = f"master_images/img_{secrets.token_hex(8)}.png"
        shutil.copyfile(source_path, audit_dir / filename)
        audit_rows.append({
            "filename": filename,
            "human_count": "",
            "visible_artifact": "",
        })
        prompt_map[filename] = object_name

    _write_csv(csv_path, MASTER_COLUMNS, audit_rows)
    prompt_map_path.write_text(
        json.dumps(prompt_map, indent=2),
        encoding="utf-8",
    )
    return len(audit_rows)


def _write_csv(path: Path, fieldnames: tuple[str, ...], rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _pending_index(rows: list[dict[str, str]]) -> int | None:
    return next(
        (
            index
            for index, row in enumerate(rows)
            if not row.get("human_count", "").strip()
        ),
        None,
    )


def _image_path(filename: str, audit_dir: Path | None = None) -> str:
    if audit_dir is None:
        audit_dir = AUDIT_DIR
    path = (audit_dir / filename).resolve()
    if not path.is_relative_to(audit_dir.resolve()):
        raise ValueError(f"Audit image path escapes audit directory: {filename}")
    if not path.is_file():
        raise FileNotFoundError(f"Audit image is missing: {path}")
    return str(path)


def load_next():
    rows, _ = _read_csv(CSV_PATH)
    prompt_map = json.loads(PROMPT_MAP_PATH.read_text(encoding="utf-8"))
    index = _pending_index(rows)
    if index is None:
        return (
            None,
            "Audit complete. All images have been reviewed.",
            None,
            gr.update(interactive=False),
            None,
            False,
            "All 105 images are audited.",
        )

    filename = rows[index]["filename"]
    object_name = prompt_map.get(filename)
    if not object_name:
        raise ValueError(f"Object prompt is missing for audit image {filename}.")
    return (
        _image_path(filename),
        f"How many {object_name} do you see?",
        index,
        gr.update(interactive=True),
        None,
        False,
        f"Image {index + 1} of {len(rows)}",
    )


def save_and_next(
    row_index: int | None,
    human_count: float | None,
    visible_artifact: bool,
):
    if row_index is None:
        loaded = load_next()
        return loaded
    if human_count is None:
        raise gr.Error("Enter the number of objects you see.")
    if not float(human_count).is_integer():
        raise gr.Error("The object count must be a whole number.")

    rows, fieldnames = _read_csv(CSV_PATH)
    if row_index < 0 or row_index >= len(rows):
        raise gr.Error("The selected audit row no longer exists.")
    row = rows[row_index]
    if row.get("human_count", "").strip():
        return load_next()

    row["human_count"] = str(int(human_count))
    row["visible_artifact"] = "Y" if visible_artifact else "N"
    with tempfile.NamedTemporaryFile(
        "w",
        newline="",
        encoding="utf-8",
        dir=CSV_PATH.parent,
        delete=False,
    ) as handle:
        temporary_path = Path(handle.name)
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(CSV_PATH)

    loaded = load_next()
    return (
        *loaded[:4],
        None,
        False,
        "Saved. " + loaded[6],
    )


with gr.Blocks(title="Blinded Single-Image Audit") as app:
    gr.Markdown(
        "# Blinded Image Audit\n"
        "Count the requested object in each shuffled image. Images are shown "
        "one at a time; model counts and edit conditions are hidden."
    )
    image = gr.Image(label="Audit image", type="filepath", interactive=False)
    question = gr.Markdown()
    human_count = gr.Number(
        label="Your count",
        precision=0,
        minimum=0,
    )
    visible_artifact = gr.Checkbox(label="Any visible artifacts/seams?")
    with gr.Row():
        save_button = gr.Button("Save & Next", variant="primary")
        status = gr.Markdown()
    row_index = gr.State()

    outputs = [image, question, row_index, save_button, human_count, visible_artifact, status]
    app.load(load_next, outputs=outputs)
    save_button.click(
        save_and_next,
        inputs=[row_index, human_count, visible_artifact],
        outputs=outputs,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Launch the blinded single-image audit UI."
    )
    parser.add_argument("--port", type=int, default=7861)
    args = parser.parse_args()
    total = prepare_master_audit()
    print(f"Prepared {total} shuffled single-image audit records at {CSV_PATH}")
    app.launch(server_port=args.port)


if __name__ == "__main__":
    main()
