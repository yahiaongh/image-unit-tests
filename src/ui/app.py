from __future__ import annotations

import base64
import html
import json
import logging
import os
import re
import textwrap
import traceback
from io import BytesIO
from pathlib import Path

import gradio as gr
import numpy as np
from PIL import Image, ImageDraw, ImageFilter, ImageFont

from src.agent import Agent
from src.clients import get_clients
from src.config import load_config

LOGGER = logging.getLogger(__name__)
MASK_COLORS = {"passed": (34, 197, 94), "failed": (239, 68, 68)}
GREEN = "#22c55e"
RED = "#ef4444"
GREY = "#6b7280"
MODES = (
    "Mock (no API, instant)",
    "Live (real AI, costs money)",
    "Demo replay (saved run, offline)",
)


def _round_number(caption: str | int | None) -> int:
    if isinstance(caption, int):
        return caption
    digits = "".join(character for character in str(caption or "") if character.isdigit())
    return int(digits) if digits else 1


def _load_mask(mask_value) -> np.ndarray | None:
    if isinstance(mask_value, dict):
        mask_value = mask_value.get("mask", mask_value.get("path"))
    if isinstance(mask_value, (str, Path)):
        try:
            with Image.open(mask_value) as mask_image:
                mask_array = np.asarray(mask_image.convert("L")) > 0
        except (OSError, ValueError):
            return None
    else:
        try:
            mask_array = np.asarray(mask_value)
        except (TypeError, ValueError):
            return None
        if mask_array.dtype != np.bool_:
            mask_array = mask_array.astype(bool)
    if mask_array.ndim != 2 or not mask_array.size:
        return None
    return mask_array


def render_round_overlay(
    image_bytes_or_path,
    masks_by_prompt,
    verdicts,
    caption=None,
) -> tuple[Image.Image | None, str]:
    """Render a round and return a Gradio-compatible image/caption pair."""
    round_index = _round_number(caption)
    try:
        if image_bytes_or_path is None:
            return None, f"Round {round_index} — image missing"
        if isinstance(image_bytes_or_path, Image.Image):
            source = image_bytes_or_path.copy()
        elif isinstance(image_bytes_or_path, (str, Path)):
            with Image.open(image_bytes_or_path) as opened:
                source = opened.copy()
        else:
            source = Image.open(BytesIO(image_bytes_or_path)).copy()
        source = source.convert("RGB")

        valid_verdicts = [item for item in verdicts if isinstance(item, dict)]
        passed_count = sum(bool(item.get("passed")) for item in valid_verdicts)
        caption_text = f"Round {round_index} — {passed_count}/{len(valid_verdicts)} passed"
        failed_ids = [
            str(item.get("constraint_id", ""))
            for item in valid_verdicts
            if not item.get("passed")
        ]
        if failed_ids:
            caption_text += f" — failed: {', '.join(failed_ids)}"

        prompt_status: dict[str, str] = {}
        for item in valid_verdicts:
            prompt = item.get("object_prompt")
            if prompt:
                status = "passed" if item.get("passed") else "failed"
                if prompt not in prompt_status or status == "failed":
                    prompt_status[prompt] = status

        overlay = Image.new("RGBA", source.size, (0, 0, 0, 0))
        if not isinstance(masks_by_prompt, dict):
            masks_by_prompt = {}
        for prompt, status in prompt_status.items():
            prompt_masks = masks_by_prompt.get(prompt, [])
            if isinstance(prompt_masks, dict):
                prompt_masks = prompt_masks.get("masks", [prompt_masks])
            if not isinstance(prompt_masks, (list, tuple)):
                prompt_masks = [prompt_masks]
            for mask_entry in prompt_masks:
                mask_array = _load_mask(mask_entry)
                if mask_array is None:
                    continue
                mask_image = Image.fromarray(mask_array.astype(np.uint8) * 255, mode="L")
                if mask_image.size != source.size:
                    mask_image = mask_image.resize(source.size, Image.Resampling.NEAREST)
                    mask_array = np.asarray(mask_image) > 0
                color = MASK_COLORS[status]
                fill = Image.new("RGBA", source.size, (*color, 90))
                overlay.paste(fill, (0, 0), mask_image)
                eroded = mask_image.filter(ImageFilter.MinFilter(5))
                outline = Image.fromarray(
                    np.logical_and(mask_array, np.asarray(eroded) == 0).astype(np.uint8) * 255,
                    mode="L",
                )
                outline_color = Image.new("RGBA", source.size, (*color, 255))
                overlay.paste(outline_color, (0, 0), outline)

        rendered = Image.alpha_composite(source.convert("RGBA"), overlay).convert("RGB")
        return rendered, caption_text
    except Exception:
        LOGGER.error("Failed to render round %s overlay:\n%s", round_index, traceback.format_exc())
        try:
            if isinstance(image_bytes_or_path, Image.Image):
                fallback = image_bytes_or_path.convert("RGB")
            elif isinstance(image_bytes_or_path, (str, Path)):
                with Image.open(image_bytes_or_path) as opened:
                    fallback = opened.convert("RGB")
            elif image_bytes_or_path is not None:
                with Image.open(BytesIO(image_bytes_or_path)) as opened:
                    fallback = opened.convert("RGB")
            else:
                fallback = None
        except Exception:
            fallback = None
        return fallback, f"Round {round_index} — overlay failed"


def _dot_data_uri(color: str) -> str:
    image = Image.new("RGBA", (20, 20), (0, 0, 0, 0))
    ImageDraw.Draw(image).ellipse((2, 2, 18, 18), fill=color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _constraint_sentence(constraint) -> str:
    object_name = constraint.object_prompt
    if constraint.type == "count":
        quantity = int(constraint.expected_count or 0)
        label = object_name if quantity == 1 else _pluralize(object_name)
        return f"Exactly {quantity} {label}"
    if constraint.type == "presence":
        return f"At least one {object_name} visible"
    if constraint.type == "absence":
        return f"No {object_name}"
    if constraint.type == "position":
        if constraint.position == "center":
            return f"{object_name} must be in the center"
        return f"{object_name} must be on the {constraint.position} side"
    if constraint.type == "size":
        area = (
            constraint.min_area_ratio
            if constraint.min_area_ratio is not None
            else constraint.max_area_ratio
        )
        return f"{object_name} must cover {float(area or 0) * 100:.1f}% of the image"
    return "The image must match the requested rule"


def _pluralize(text: str) -> str:
    if re.search(r"\b(?:chairs|dogs|people|trees|apples|books|cups|bicycles|flowers|windows|students|lanterns|pencils|sandwiches|umbrellas|leaves|mugs|chairs)\b$", text, re.IGNORECASE):
        return text
    if text.lower().endswith("y") and len(text) > 1 and text[-2].lower() not in "aeiou":
        return f"{text[:-1]}ies"
    if text.lower().endswith(("s", "x", "z", "ch", "sh")):
        return f"{text}es"
    return f"{text}s"


def _safe_reason(reason: str) -> str:
    cleaned = re.sub(r"\b(?:constraint_id|centroid|iou|object_prompt|mask|verdict)\b", "", reason, flags=re.IGNORECASE)
    cleaned = re.sub(r"[\s,;:]+", " ", cleaned).strip(" .")
    if not cleaned or re.search(r"\b(?:expected|found|outside|within|present|detected|ratio)\b", cleaned, re.IGNORECASE) is None:
        return "Did not match the rule."
    return cleaned[0].upper() + cleaned[1:] + ("." if not cleaned.endswith(".") else "")


def _build_checklist(constraints, verdicts) -> str:
    verdict_by_id = {item.get("constraint_id"): item for item in verdicts}
    rows = []
    for constraint in constraints:
        verdict = verdict_by_id.get(constraint.id, {})
        passed = bool(verdict.get("passed"))
        color = GREEN if passed else RED
        sentence = html.escape(_constraint_sentence(constraint))
        identifier = html.escape(str(constraint.id))
        row = (
            '<div style="display:flex;align-items:center;gap:12px;padding:10px 4px;'
            'border-bottom:1px solid #e5e7eb">'
            f'<img src="{_dot_data_uri(color)}" width="20" height="20" alt="">'
            f'<span style="flex:1">{sentence}</span>'
            f'<small style="color:{GREY}">{identifier}</small></div>'
        )
        if not passed:
            why = html.escape(_safe_reason(str(verdict.get("reason", ""))))
            row += (
                f'<div style="margin:-4px 0 8px 36px;color:{GREY};font-style:italic">'
                f"Why: {why}</div>"
            )
        rows.append(row)
    count = len(constraints)
    item_word = "thing" if count == 1 else "things"
    header = f"The AI broke your brief into {count} {item_word} it must check"
    return f"<h2>{header}</h2><div>{''.join(rows)}</div>"


def _round_sentence(verdicts: list[dict], constraints_by_id: dict) -> str:
    failures = [item for item in verdicts if not item.get("passed")]
    if not failures:
        count = len(verdicts)
        return f"All {count} {'rule' if count == 1 else 'rules'} matched."
    reasons = []
    for verdict in failures:
        constraint = constraints_by_id.get(verdict.get("constraint_id"))
        reason = str(verdict.get("reason", ""))
        if constraint and constraint.type == "count":
            found = re.search(r"found\s+(\d+)", reason, flags=re.IGNORECASE)
            expected = constraint.expected_count
            if found:
                noun = _pluralize(constraint.object_prompt) if int(found.group(1)) != 1 else constraint.object_prompt
                reasons.append(f"Found {found.group(1)} {noun} (expected {expected})")
                continue
        reasons.append(_safe_reason(reason))
    return "; ".join(reasons)


def _timeline_card(
    image: Image.Image,
    round_index: int,
    passed: bool,
    footer: str,
    fix_instruction: str | None = None,
) -> Image.Image:
    font = ImageFont.load_default(size=20)
    badge_text = "PASSED" if passed else "FAILED"
    badge_color = MASK_COLORS["passed" if passed else "failed"]
    footer_lines = textwrap.wrap(footer, width=max(20, image.width // 10)) or [""]
    fix_lines = (
        textwrap.wrap(f"Fix requested: {fix_instruction}", width=max(20, image.width // 10))
        if fix_instruction
        else []
    )
    footer_lines = footer_lines[:2]
    fix_lines = fix_lines[:3]
    extra_height = 56 + 26 * len(footer_lines) + (26 * len(fix_lines) if fix_lines else 0)
    canvas = Image.new("RGB", (image.width, image.height + extra_height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((14, 12), f"Round {round_index}", fill="#111827", font=font)
    badge_bounds = draw.textbbox((0, 0), badge_text, font=font)
    badge_width = badge_bounds[2] - badge_bounds[0] + 20
    badge_x = image.width - badge_width - 12
    draw.rounded_rectangle(
        (badge_x, 8, badge_x + badge_width, 38),
        radius=6,
        fill=badge_color,
    )
    draw.text((badge_x + 10, 12), badge_text, fill="white", font=font)
    canvas.paste(image.convert("RGB"), (0, 44))

    draw.multiline_text(
        (14, image.height + 52),
        "\n".join(footer_lines),
        fill="#374151",
        font=font,
        spacing=4,
    )
    if fix_lines:
        draw.multiline_text(
            (14, image.height + 54 + 26 * len(footer_lines)),
            "\n".join(fix_lines),
            fill="#374151",
            font=font,
            spacing=4,
        )
    return canvas


def _friendly_fix(round_data: dict, constraints_by_id: dict) -> str | None:
    if not round_data.get("fix_instruction"):
        return None
    for verdict in round_data.get("verdicts", []):
        if verdict.get("passed"):
            continue
        constraint = constraints_by_id.get(verdict.get("constraint_id"))
        if not constraint:
            continue
        if constraint.type == "count":
            quantity = int(constraint.expected_count or 0)
            label = constraint.object_prompt if quantity == 1 else _pluralize(constraint.object_prompt)
            return f"Adjust the image to show exactly {quantity} {label}."
        if constraint.type == "position":
            if constraint.position == "center":
                return f"Move the {constraint.object_prompt} to the center."
            return f"Move the {constraint.object_prompt} to the {constraint.position} side."
        if constraint.type == "absence":
            return f"Remove the {constraint.object_prompt}."
        if constraint.type == "presence":
            return f"Add at least one {constraint.object_prompt}."
    return "Adjust the image to meet the checklist."


def _summary_sentence(passed: bool, rounds: int) -> str:
    if passed and rounds == 1:
        return "The AI got it right on the first try."
    if passed:
        return (
            f"The AI failed the checklist on round 1, used the specific feedback "
            f"to fix the image, and passed on round {rounds}."
        )
    return f"The AI never satisfied the checklist after {rounds} rounds. The final image is shown above."


def _technical_markdown(constraints, round_data, run_dir) -> str:
    sections = ["### Parsed checklist", "```json"]
    sections.append(json.dumps(
        [constraint.model_dump() for constraint in constraints],
        indent=2,
        default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value),
    ))
    sections.append("```")
    for item in round_data:
        masks = item.get("masks_by_prompt", item.get("masks", {}))
        technical = {
            "verdicts": item.get("verdicts", []),
            "mask_centroids": {},
            "filtered_mask_counts": {},
            "mask_object_prompts": list(masks),
        }
        for prompt, entries in masks.items():
            if not isinstance(entries, list):
                entries = [entries]
            technical["filtered_mask_counts"][prompt] = len(entries)
            centers = []
            for entry in entries:
                mask = _load_mask(entry)
                if mask is None or not mask.any():
                    centers.append(None)
                    continue
                ys, xs = np.where(mask)
                centers.append([float(xs.mean() / mask.shape[1]), float(ys.mean() / mask.shape[0])])
            technical["mask_centroids"][prompt] = centers
        sections.extend([
            f"### Round {item.get('round')}",
            "```json",
            json.dumps(technical, indent=2, default=lambda value: value.tolist() if hasattr(value, "tolist") else str(value)),
            "```",
        ])
    sections.append(f"Run directory: `{run_dir or 'Not available'}`")
    return "\n\n".join(sections)


def _gallery_and_technical(constraints, round_data, run_dir):
    constraints_by_id = {constraint.id: constraint for constraint in constraints}
    gallery_items: list[tuple[Image.Image, str]] = []
    last_verdicts = []
    for item in round_data:
        round_index = int(item["round"])
        verdicts = [dict(verdict) for verdict in item.get("verdicts", [])]
        last_verdicts = verdicts
        for verdict in verdicts:
            constraint = constraints_by_id.get(verdict.get("constraint_id"))
            if constraint:
                verdict["object_prompt"] = constraint.object_prompt
        passed = bool(item.get("passed", all(v.get("passed") for v in verdicts)))
        status = "PASSED" if passed else "FAILED"
        footer = _round_sentence(verdicts, constraints_by_id)
        rendered, _ = render_round_overlay(
            item.get("image_path_or_bytes"),
            item.get("masks_by_prompt", item.get("masks", {})),
            verdicts,
            round_index,
        )
        if rendered is not None:
            caption = f"Round {round_index} — {status} — {footer}"
            fix = _friendly_fix(item, constraints_by_id)
            if fix and str(fix).strip() != "No fix needed.":
                caption += f" — Fix requested: {fix}"
            gallery_items.append(
                (_timeline_card(rendered, round_index, passed, footer, fix), caption)
            )
    return gallery_items, last_verdicts


def _four_outputs(checklist, gallery_items, summary, technical):
    return checklist, gallery_items, f'<div style="font-size:1.25em;text-align:center">{summary}</div>', technical


def replay_demo() -> tuple[str, list[tuple[Image.Image, str]], str, str]:
    demo_dir = Path("runs") / "demo"
    summary_path = demo_dir / "summary.json"
    if not summary_path.exists():
        return _four_outputs("", [], "No saved demo. Run one brief first, then copy its run folder to runs/demo.", "")
    try:
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        constraints_data = summary.get("constraints", [])
        constraints = [ConstraintAdapter(item) for item in constraints_data]
        round_data = []
        for round_index in range(1, int(summary.get("rounds", 0)) + 1):
            image_path = demo_dir / f"round_{round_index}.png"
            masks_path = demo_dir / f"round_{round_index}_masks.json"
            verdicts_path = demo_dir / f"round_{round_index}_verdicts.json"
            fix_path = demo_dir / f"round_{round_index}_fix.txt"
            masks = json.loads(masks_path.read_text(encoding="utf-8")) if masks_path.exists() else {}
            verdicts = json.loads(verdicts_path.read_text(encoding="utf-8")) if verdicts_path.exists() else []
            round_data.append({
                "round": round_index,
                "passed": all(item.get("passed") for item in verdicts),
                "image_path_or_bytes": image_path if image_path.is_file() else None,
                "masks_by_prompt": masks,
                "verdicts": verdicts,
                "fix_instruction": fix_path.read_text(encoding="utf-8").strip() if fix_path.exists() else "",
            })
        gallery_items, final_verdicts = _gallery_and_technical(
            constraints, round_data, demo_dir
        )
        checklist = _build_checklist(constraints, final_verdicts)
        summary_text = _summary_sentence(bool(summary.get("final_passed")), len(round_data))
        technical = _technical_markdown(constraints, round_data, demo_dir)
        return _four_outputs(checklist, gallery_items, summary_text, technical)
    except Exception:
        LOGGER.exception("Failed to replay demo run")
        return _four_outputs("", [], "The saved demo could not be loaded. Please try running a new brief.", "")


def run_brief(
    brief: str,
    mode: str,
) -> tuple[str, list[tuple[Image.Image, str]], str, str]:
    if not brief or not brief.strip():
        return _four_outputs("", [], "Please enter a brief first.", "")
    if mode.startswith("Demo replay"):
        return replay_demo()
    try:
        config = load_config()
        if mode.startswith("Live"):
            api_key_name = config.get("meta", {}).get("api_key_env", "")
            if not api_key_name or not os.getenv(api_key_name):
                return _four_outputs("", [], "Live mode needs the API key in your .env file.", "")
            config.setdefault("mock", {})["enabled"] = False
        elif mode.startswith("Mock"):
            config.setdefault("mock", {})["enabled"] = True
        else:
            return _four_outputs("", [], "Choose mock, live, or demo replay mode.", "")

        runs_dir = Path(config.get("runs_dir", "runs"))
        previous_runs = set(runs_dir.iterdir()) if runs_dir.exists() else set()
        spark, image, sam = get_clients(config)
        result = Agent(spark, image, sam, config).run(brief.strip())
        available_runs = [path for path in runs_dir.iterdir() if path.is_dir()] if runs_dir.exists() else []
        new_runs = [path for path in available_runs if path not in previous_runs]
        run_dir = max(new_runs or available_runs, key=lambda path: path.stat().st_mtime) if available_runs else None
        round_data = []
        for item in result["rounds"]:
            round_index = int(item["round"])
            image_path = run_dir / f"round_{round_index}.png" if run_dir else None
            round_data.append({
                **item,
                "image_path_or_bytes": image_path if image_path and image_path.is_file() else None,
                "masks_by_prompt": item.get("masks", {}),
            })
        gallery_items, final_verdicts = _gallery_and_technical(
            result["constraints"], round_data, run_dir
        )
        checklist = _build_checklist(result["constraints"], final_verdicts)
        summary_text = _summary_sentence(bool(result["final_passed"]), len(round_data))
        technical = _technical_markdown(result["constraints"], round_data, run_dir)
        return _four_outputs(checklist, gallery_items, summary_text, technical)
    except Exception:
        LOGGER.exception("Failed to run brief")
        return _four_outputs("", [], "The image run could not be completed. Please check the setup and try again.", "")


def ConstraintAdapter(data: dict):
    from src.schema import Constraint

    return Constraint.model_validate(data)


APP_CSS = f"""
    .hero-subtitle {{ color: {GREY}; font-size: 0.95em; }}
    .round-timeline {{ min-height: 360px; }}
    .round-timeline .grid-wrap {{
        display: flex !important;
        flex-wrap: nowrap !important;
        gap: 12px;
        overflow-x: auto !important;
        padding-bottom: 12px;
    }}
    .round-timeline .grid-wrap > * {{
        flex: 0 0 280px !important;
        width: 280px !important;
        max-width: 280px !important;
    }}
"""

with gr.Blocks() as app:
    gr.Markdown(
        "# Image Unit Tests\n"
        f'<div class="hero-subtitle">An AI draws an image, then checks its own work '
        f"against a checklist and fixes what's wrong.</div>"
    )
    with gr.Row():
        brief = gr.Textbox(
            label="What should the image show?",
            placeholder="e.g. exactly 3 red chairs on a beach",
            scale=5,
            lines=1,
        )
        mode = gr.Dropdown(
            list(MODES),
            label="How to run it",
            value=MODES[0],
            scale=2,
        )
        run_btn = gr.Button("Run", variant="primary")
        replay_btn = gr.Button("Replay saved demo")

    checklist_panel = gr.HTML(label="What the AI understood")
    gr.Markdown("## What happened, round by round")
    timeline_gallery = gr.Gallery(
        label="Rounds",
        columns=3,
        height=540,
        object_fit="contain",
        elem_classes=["round-timeline"],
    )
    summary_panel = gr.Markdown()
    with gr.Accordion("Technical details (for developers)", open=False):
        technical_panel = gr.Markdown()

    output_components = [checklist_panel, timeline_gallery, summary_panel, technical_panel]
    run_btn.click(fn=run_brief, inputs=[brief, mode], outputs=output_components)
    replay_btn.click(fn=replay_demo, inputs=[], outputs=output_components)

if __name__ == "__main__":
    app.launch(css=APP_CSS)
