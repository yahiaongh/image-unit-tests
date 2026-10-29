from __future__ import annotations

from typing import Any

import requests

from src.config import get_env


class ClientError(RuntimeError):
    """Raised when a client call fails before returning usable data."""


class ImageClient:
    def __init__(self, config: dict):
        self.config = config
        self.base_url = config.get("basten", {}).get("base_url") or "TODO"
        self.api_key_env = config.get("basten", {}).get("api_key_env")
        self.api_key = get_env(self.api_key_env) if self.api_key_env else None
        self.model_id = config.get("models", {}).get("image", {}).get("model_id")
        self.endpoint = config.get("models", {}).get("image", {}).get("endpoint")
        self.edit_endpoint = config.get("models", {}).get("image", {}).get("edit_endpoint")

    def generate(self, brief: str) -> bytes:
        """Return image bytes."""
        if not self.endpoint or self.endpoint == "TODO":
            raise ClientError("ImageClient generation endpoint is not configured in config.yaml")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # TODO: exact request/response contract for image generation is documented in the
        # Meta/Baseten cookbook; keep the adapter implementation narrow and placeholder-safe.
        payload: dict[str, Any] = {
            "model": self.model_id,
            "prompt": brief,
        }

        try:
            response = requests.post(
                f"{self.base_url}{self.endpoint}",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code == 429:
                raise ClientError("ImageClient rate limit exceeded (HTTP 429)")
            if response.status_code >= 400:
                raise ClientError(f"ImageClient request failed with status {response.status_code}")
            if not response.content:
                raise ClientError("ImageClient returned an empty response")
            return response.content
        except requests.Timeout:
            raise ClientError("ImageClient request timed out") from None
        except requests.RequestException as exc:
            raise ClientError(f"ImageClient network error: {exc}") from exc

    def edit(self, image_bytes: bytes, fix_instruction: str) -> bytes:
        """Return edited image bytes."""
        if not self.edit_endpoint or self.edit_endpoint == "TODO":
            raise ClientError("ImageClient edit endpoint is not configured in config.yaml")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # TODO: exact edit request body and file handling remain pending from provider docs.
        payload: dict[str, Any] = {
            "model": self.model_id,
            "image": image_bytes,
            "instruction": fix_instruction,
        }

        try:
            response = requests.post(
                f"{self.base_url}{self.edit_endpoint}",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code == 429:
                raise ClientError("ImageClient rate limit exceeded (HTTP 429)")
            if response.status_code >= 400:
                raise ClientError(f"ImageClient edit request failed with status {response.status_code}")
            if not response.content:
                raise ClientError("ImageClient edit returned an empty response")
            return response.content
        except requests.Timeout:
            raise ClientError("ImageClient edit request timed out") from None
        except requests.RequestException as exc:
            raise ClientError(f"ImageClient edit network error: {exc}") from exc
