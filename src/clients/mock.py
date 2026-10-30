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
        self._calls: dict[str, int] = {}

    def segment(self, image_bytes: bytes, text_prompt: str) -> list[dict]:
        if text_prompt in self.mask_map:
            return self.mask_map[text_prompt]

        with Image.open(BytesIO(image_bytes)) as image:
            width, height = image.size
            brief = image.info.get("brief", "")

        prompt = text_prompt.lower()
        call_number = self._calls.get(prompt, 0)
        self._calls[prompt] = call_number + 1
        count_and_object = _brief_count_and_object(brief)
        if "person" in prompt:
            count = 0
        elif "chair" in prompt:
            count = (
                count_and_object[0]
                if count_and_object and "chair" in count_and_object[1].lower()
                else 3
            )
        else:
            count = 1
        if self.force_fail_first_round and "chair" in prompt and call_number == 0:
            count += 1

        columns = min(4, max(1, math.ceil(math.sqrt(count))))
        rows = math.ceil(count / columns)
        masks = []
        for index in range(count):
            row, column = divmod(index, columns)
            if "logo" in prompt:
                x0, y0 = int(width * 0.78), int(height * 0.76)
            else:
                x0 = int(width * (0.1 + 0.2 * column))
                y0 = int(height * (0.35 + 0.3 * row))
            x1 = min(width, x0 + max(2, int(width * 0.12)))
            y1 = min(height, y0 + max(2, int(height * 0.18)))
            mask = np.zeros((height, width), dtype=bool)
            mask[y0:y1, x0:x1] = True
            masks.append({
                "mask": mask,
                "score": 0.95,
                "box": [x0, y0, x1, y1],
            })
        return masks
