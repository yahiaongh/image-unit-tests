from __future__ import annotations

import math
import re
from io import BytesIO

import numpy as np
from PIL import Image, ImageDraw, PngImagePlugin


_CLAUSE_BOUNDARIES = re.compile(
    r"\s+(?:on|in|at|with|and|but|that|where|beside|under|above|behind|"
    r"over|near|next to|arranged|lined)\b.*$",
    flags=re.IGNORECASE,
)


def _object_phrase(text: str) -> str:
    phrase = _CLAUSE_BOUNDARIES.sub("", text).strip(" \t\n,.;:!?")
    phrase = re.sub(r"^(?:a|an|the)\s+", "", phrase, flags=re.IGNORECASE)
    return phrase or "object"


def _brief_count_and_object(brief: str) -> tuple[int, str] | None:
    match = re.search(r"\bexactly\s+(\d+)\s+([^,.!?;]+)", brief, flags=re.IGNORECASE)
    if not match:
        return None
    return int(match.group(1)), _object_phrase(match.group(2))


class MockSparkClient:
    def generate_constraints(self, brief: str) -> list[dict]:
        constraints = []
        count_and_object = _brief_count_and_object(brief)
        if count_and_object:
            count, object_prompt = count_and_object
            constraints.append({
                "id": "count-1",
                "type": "count",
                "description": f"exactly {count} {object_prompt}",
                "object_prompt": object_prompt,
                "expected_count": count,
            })

        position_match = re.search(
            r"\b(?:on\s+the\s+)?(left|right|top|bottom|center)\b",
            brief,
            flags=re.IGNORECASE,
        )
        if position_match:
            object_prompt = count_and_object[1] if count_and_object else "object"
            constraints.append({
                "id": "position-1",
                "type": "position",
                "description": f"{object_prompt} on the {position_match.group(1).lower()}",
                "object_prompt": object_prompt,
                "position": position_match.group(1).lower(),
            })

        absence_match = re.search(r"\bno\s+([^,.!?;]+)", brief, flags=re.IGNORECASE)
        if absence_match:
            object_prompt = _object_phrase(absence_match.group(1))
            constraints.append({
                "id": "absence-1",
                "type": "absence",
                "description": f"no {object_prompt}",
                "object_prompt": object_prompt,
            })

        if constraints:
            return constraints
        return [
            {
                "id": "count-1",
                "type": "count",
                "description": "exactly 1 object",
                "object_prompt": "object",
                "expected_count": 1,
            },
            {
                "id": "position-1",
                "type": "position",
                "description": "object is centered",
                "object_prompt": "object",
                "position": "center",
            },
        ]


class MockImageClient:
    def __init__(self, config: dict | None = None):
        self.config = config or {}

    def _png_bytes(self, color: tuple[int, int, int], brief: str) -> bytes:
        image = Image.new("RGB", (512, 512), color=color)
        image.putpixel((256, 256), (50, 50, 50))
        draw = ImageDraw.Draw(image)
        max_width = image.width - 40
        words = brief.split()
        lines = []
        current_line = ""
        for word in words:
            candidate = f"{current_line} {word}".strip()
            if current_line and draw.textbbox((0, 0), candidate)[2] > max_width:
                lines.append(current_line)
                current_line = word
            else:
                current_line = candidate
        if current_line:
            lines.append(current_line)
        draw.multiline_text((20, 20), "\n".join(lines), fill=(35, 35, 35), spacing=6)

        buffer = BytesIO()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("brief", brief)
        image.save(buffer, format="PNG", pnginfo=metadata)
        return buffer.getvalue()

    def generate(self, brief: str) -> bytes:
        return self._png_bytes((200, 200, 200), brief)

    def edit(self, image_bytes: bytes, fix_instruction: str) -> bytes:
        with Image.open(BytesIO(image_bytes)) as source:
            brief = source.info.get("brief", "")
            image = source.convert("RGB")
        image.putpixel((256, 256), (120, 120, 120))
        buffer = BytesIO()
        metadata = PngImagePlugin.PngInfo()
        metadata.add_text("brief", brief)
        image.save(buffer, format="PNG", pnginfo=metadata)
        return buffer.getvalue()


class MockSamClient:
    def __init__(
        self,
        config: dict | None = None,
        mask_map: dict[str, list[dict]] | None = None,
        force_fail_first_round: bool = False,
    ):
        self.config = config or {}
        self.mask_map = mask_map or {}
        self.force_fail_first_round = force_fail_first_round
        self._forced_first_round = False

    def segment(self, image_bytes: bytes, text_prompt: str) -> list[dict]:
        if text_prompt in self.mask_map:
            return self.mask_map[text_prompt]

        with Image.open(BytesIO(image_bytes)) as image:
            brief = image.info.get("brief", "")

        count_and_object = _brief_count_and_object(brief)
        matching_chair_prompt = bool(
            count_and_object
            and "chair" in count_and_object[1].lower()
            and "chair" in text_prompt.lower()
        )
        count = 1
        if matching_chair_prompt:
            count = count_and_object[0]
        if self.force_fail_first_round and matching_chair_prompt and not self._forced_first_round:
            count += 1
            self._forced_first_round = True

        columns = min(4, max(1, math.ceil(math.sqrt(count))))
        rows = math.ceil(count / columns)
        cell_width = 64 // columns
        cell_height = 64 // rows
        masks = []
        for index in range(count):
            row, column = divmod(index, columns)
            x1 = column * cell_width + max(1, cell_width // 4)
            y1 = row * cell_height + max(1, cell_height // 4)
            x2 = min((column + 1) * cell_width - 1, x1 + max(2, cell_width // 2))
            y2 = min((row + 1) * cell_height - 1, y1 + max(2, cell_height // 2))
            mask = np.zeros((64, 64), dtype=bool)
            mask[y1:y2, x1:x2] = True
            masks.append({
                "mask": mask,
                "score": 0.95,
                "box": [x1, y1, x2, y2],
            })
        return masks
