from __future__ import annotations

import base64
import time
from typing import Any

import requests

from src.config import get_env


class ClientError(RuntimeError):
    """Raised when a client call fails before returning usable data."""


def _image_data_url(image_bytes: bytes) -> str:
    import io

    from PIL import Image

    with Image.open(io.BytesIO(image_bytes)) as image:
        image_format = image.format
    mime_type = {
        "JPEG": "image/jpeg",
        "PNG": "image/png",
        "WEBP": "image/webp",
    }.get(image_format)
    if mime_type is None:
        raise ValueError(f"Unsupported input image format: {image_format}")
    encoded = base64.b64encode(image_bytes).decode("ascii")
    return f"data:{mime_type};base64,{encoded}"


def _extract_image(response_data: dict) -> bytes:
    output = response_data.get("output")
    if not isinstance(output, list):
        raise ClientError("ImageClient response did not contain an output list")
    for item in output:
        if not isinstance(item, dict) or item.get("type") != "image_generation_call":
            continue
        image_result = item.get("result")
        if not isinstance(image_result, str) or not image_result:
            continue
        try:
            return base64.b64decode(image_result, validate=True)
        except (ValueError, base64.binascii.Error) as exc:
            raise ClientError("ImageClient returned invalid base64 image data") from exc
    raise ClientError("ImageClient response did not contain a generated image")


def _post_with_timeout_retry(
    method: str,
    url: str,
    headers: dict,
    payload: dict[str, Any],
    timeout: int,
) -> requests.Response:
    for attempt in range(2):
        t0 = time.monotonic()
        try:
            return requests.post(
                url,
                headers=headers,
                json=payload,
                timeout=timeout,
            )
        except requests.Timeout:
            if attempt == 1:
                raise
        finally:
            elapsed = time.monotonic() - t0
            print(f"[ImageClient.{method}] elapsed={elapsed:.1f}s")
    raise ClientError(f"ImageClient.{method} retry ended without a response")


class ImageClient:
    def __init__(self, config: dict):
        self.config = config
        # source: https://dev.meta.ai/docs/quickstart
        # — "Base URL: https://api.meta.ai/v1"
        self.base_url = config.get("meta", {}).get("base_url")
        self.api_key_env = config.get("meta", {}).get("api_key_env")
        self.api_key = get_env(self.api_key_env) if self.api_key_env else None
        self.model_id = config.get("models", {}).get("image", {}).get("model_id")
        self.endpoint = config.get("models", {}).get("image", {}).get("endpoint")
        self.edit_endpoint = config.get("models", {}).get("image", {}).get("edit_endpoint")

    def generate(self, brief: str) -> bytes:
        """Return image bytes."""
        if not self.endpoint or self.endpoint.startswith("TODO"):
            raise ClientError("ImageClient generation endpoint is not configured in config.yaml")

        # source: https://github.com/meta-models/meta-model-cookbook/blob/main/05_muse_image/01_image_api_fundamentals/README.md:54-58
        # — the SDK client is configured with api_key=... against api.meta.ai.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # source: https://github.com/meta-models/meta-model-cookbook/blob/main/05_muse_image/01_image_api_fundamentals/README.md
        # — "model=\"muse-image-1.0\", input=(...)".
        payload: dict[str, Any] = {
            "model": self.model_id,
            "input": brief,
        }

        try:
            response = _post_with_timeout_retry(
                "generate",
                f"{self.base_url}{self.endpoint}",
                headers,
                payload,
                timeout=120,
            )
            if response.status_code == 429:
                raise ClientError(
                    f"ImageClient rate limit exceeded (HTTP 429): {response.text}"
                )
            if response.status_code >= 400:
                raise ClientError(
                    f"ImageClient request failed with status {response.status_code}: "
                    f"{response.text}"
                )
            if not response.content:
                raise ClientError("ImageClient returned an empty response")
            try:
                return _extract_image(response.json())
            except ValueError as exc:
                raise ClientError(f"ImageClient returned invalid JSON: {exc}") from exc
        except requests.Timeout:
            raise ClientError("ImageClient request timed out") from None
        except requests.RequestException as exc:
            raise ClientError(f"ImageClient network error: {exc}") from exc

    def edit(self, image_bytes: bytes, fix_instruction: str) -> bytes:
        """Return edited image bytes."""
        if not self.edit_endpoint or self.edit_endpoint.startswith("TODO"):
            raise ClientError("ImageClient edit endpoint is not configured in config.yaml")

        # source: https://github.com/meta-models/meta-model-cookbook/blob/main/05_muse_image/01_image_api_fundamentals/README.md:54-58
        # — the SDK client is configured with api_key=... against api.meta.ai.
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # source: https://github.com/meta-models/meta-model-cookbook/blob/main/05_muse_image/01_image_api_fundamentals/README.md
        # — "type": "input_image", "image_url": data_url(...).
        payload: dict[str, Any] = {
            "model": self.model_id,
            "input": [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": fix_instruction},
                        {
                            "type": "input_image",
                            "image_url": _image_data_url(image_bytes),
                        },
                    ],
                }
            ],
        }

        try:
            response = _post_with_timeout_retry(
                "edit",
                f"{self.base_url}{self.edit_endpoint}",
                headers,
                payload,
                timeout=120,
            )
            if response.status_code == 429:
                raise ClientError(
                    f"ImageClient rate limit exceeded (HTTP 429): {response.text}"
                )
            if response.status_code >= 400:
                raise ClientError(
                    f"ImageClient edit request failed with status {response.status_code}: "
                    f"{response.text}"
                )
            if not response.content:
                raise ClientError("ImageClient edit returned an empty response")
            try:
                return _extract_image(response.json())
            except ValueError as exc:
                raise ClientError(f"ImageClient edit returned invalid JSON: {exc}") from exc
        except requests.Timeout:
            raise ClientError("ImageClient edit request timed out") from None
        except requests.RequestException as exc:
            raise ClientError(f"ImageClient edit network error: {exc}") from exc
