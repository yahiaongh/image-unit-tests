from __future__ import annotations

import asyncio
import base64
import json
import math
from collections.abc import AsyncIterator
from io import BytesIO

import numpy as np
import requests
from meta_sam_parser import (
    ResponsesStreamSourceError,
    decode_mask_to_raster,
    image_segmentation_format,
    parse_responses_stream,
)
from PIL import Image

from src.config import get_env


class ClientError(RuntimeError):
    """Raised when a client call fails before returning usable data."""


def _image_data_url(image_bytes: bytes) -> tuple[str, tuple[int, int]]:
    with Image.open(BytesIO(image_bytes)) as image:
        image_format = image.format
        image_size = image.size
    mime_type = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }.get(image_format)
    if mime_type is None:
        raise ClientError(f"Unsupported image format for SAM: {image_format}")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}", image_size


async def _response_events(response: requests.Response) -> AsyncIterator[dict]:
    data_lines: list[str] = []
    for raw_line in response.iter_lines(decode_unicode=True):
        if raw_line is None:
            continue
        line = raw_line.decode("utf-8") if isinstance(raw_line, bytes) else raw_line
        if not line:
            if data_lines:
                data = "\n".join(data_lines)
                data_lines.clear()
                if data != "[DONE]":
                    event = json.loads(data)
                    if isinstance(event, dict):
                        yield event
            continue
        if line.startswith(":"):
            continue
        field, separator, value = line.partition(":")
        if separator and field == "data":
            data_lines.append(value.lstrip())

    if data_lines:
        data = "\n".join(data_lines)
        if data != "[DONE]":
            event = json.loads(data)
            if isinstance(event, dict):
                yield event


async def _parse_response(response: requests.Response):
    # source: https://github.com/meta-models/meta-model-cookbook/blob/main/07_sam/01_api_basics/python/sam_api.py:111-114
    # — parse_responses_stream(events, image_segmentation_format()).
    parsed = parse_responses_stream(
        _response_events(response), image_segmentation_format()
    )
    async with parsed:
        return await parsed.final_result()


def _decode_image_result(result, image_size: tuple[int, int]) -> list[dict]:
    if result.outcome.status != "completed":
        raise ClientError(f"SAM segmentation was incomplete: {result.outcome}")
    errors = [item for item in result.diagnostics if item.severity == "error"]
    if errors:
        detail = "; ".join(f"line {item.line}: {item.message}" for item in errors)
        raise ClientError(f"SAM parser rejected segmentation records: {detail}")

    width, height = image_size
    boxes = {
        record.object_id: record
        for record in result.records
        if record.kind == "box"
    }
    masks = []
    for record in result.records:
        if record.kind != "mask":
            continue
        local_mask = np.frombuffer(
            decode_mask_to_raster(record.mask), dtype=np.uint8
        ).reshape(record.mask.height, record.mask.width)
        bounds = record.bounds
        x0 = max(0, math.floor(bounds.left))
        y0 = max(0, math.floor(bounds.top))
        x1 = min(width, math.ceil(bounds.right))
        y1 = min(height, math.ceil(bounds.bottom))
        if x1 <= x0 or y1 <= y0:
            continue
        resized = Image.fromarray(local_mask * 255, mode="L").resize(
            (x1 - x0, y1 - y0), Image.Resampling.NEAREST
        )
        full_mask = np.zeros((height, width), dtype=bool)
        full_mask[y0:y1, x0:x1] = np.asarray(resized) > 0
        box_record = boxes.get(record.object_id)
        box = (
            [
                box_record.left,
                box_record.top,
                box_record.right,
                box_record.bottom,
            ]
            if box_record is not None
            else None
        )
        confidence = record.confidence
        masks.append({
            "mask": full_mask,
            "score": float(confidence) if confidence is not None else 0.0,
            "box": box,
        })
    return masks


class SamClient:
    def __init__(self, config: dict):
        self.config = config
        # source: https://github.com/meta-models/meta-model-cookbook/blob/main/07_sam/01_api_basics/python/sam_api.py:34-35
        # — 'BASE_URL = "https://api.meta.ai/v1"; MODEL = "sam-3.1"'
        self.base_url = config.get("meta", {}).get("base_url")
        self.api_key_env = config.get("meta", {}).get("api_key_env")
        self.api_key = get_env(self.api_key_env) if self.api_key_env else None
        self.model_id = config.get("models", {}).get("sam", {}).get("model_id")
        self.endpoint = config.get("models", {}).get("sam", {}).get("endpoint")

    def segment(self, image_bytes: bytes, text_prompt: str) -> list[dict]:
        if not self.endpoint or self.endpoint.startswith("TODO"):
            raise ClientError("SamClient endpoint is not configured in config.yaml")
        if not text_prompt.strip():
            raise ValueError("The concept must be a short noun phrase.")

        image_url, image_size = _image_data_url(image_bytes)
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
        }
        # source: https://github.com/meta-models/meta-model-cookbook/blob/main/07_sam/01_api_basics/python/sam_api.py:118-126
        # — stream=True and input_text/input_image content parts.
        payload = {
            "model": self.model_id,
            "stream": True,
            "input": [
                {
                    "type": "message",
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": text_prompt.strip()},
                        {"type": "input_image", "image_url": image_url},
                    ],
                }
            ],
        }

        for attempt in range(2):
            try:
                with requests.post(
                    f"{self.base_url}{self.endpoint}",
                    headers=headers,
                    json=payload,
                    stream=True,
                    timeout=120,
                ) as response:
                    if response.status_code == 429:
                        raise ClientError(
                            f"SamClient rate limit exceeded (HTTP 429): {response.text}"
                        )
                    if response.status_code >= 400:
                        raise ClientError(
                            f"SamClient request failed with status {response.status_code}: "
                            f"{response.text}"
                        )
                    result = asyncio.run(_parse_response(response))
                return _decode_image_result(result, image_size)
            except ResponsesStreamSourceError as exc:
                if attempt == 1 or not isinstance(exc.cause, requests.RequestException):
                    raise
            except requests.Timeout as exc:
                if attempt == 1:
                    raise ClientError("SamClient request timed out") from exc
            except requests.RequestException as exc:
                if attempt == 1:
                    raise ClientError(f"SamClient network error: {exc}") from exc

        raise ClientError("SamClient retry ended without a segmentation result")
