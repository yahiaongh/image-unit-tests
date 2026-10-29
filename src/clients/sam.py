from __future__ import annotations

from typing import Any

import numpy as np
import requests

from src.config import get_env


class ClientError(RuntimeError):
    """Raised when a client call fails before returning usable data."""


class SamClient:
    def __init__(self, config: dict):
        self.config = config
        self.base_url = config.get("meta", {}).get("base_url") or config.get("basten", {}).get("base_url") or "TODO"
        self.api_key_env = config.get("meta", {}).get("api_key_env") or config.get("basten", {}).get("api_key_env")
        self.api_key = get_env(self.api_key_env) if self.api_key_env else None
        self.model_id = config.get("models", {}).get("sam", {}).get("model_id")
        self.endpoint = config.get("models", {}).get("sam", {}).get("endpoint")

    def segment(self, image_bytes: bytes, text_prompt: str) -> list[dict]:
        """
        Return a list of mask dicts:
        {"mask": np.ndarray(bool, HxW), "score": float, "box": [x1,y1,x2,y2] or None}
        """
        if not self.endpoint or self.endpoint == "TODO":
            raise ClientError("SamClient endpoint is not configured in config.yaml")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # TODO: exact SAM request/response contract is provider-specific and should be filled
        # in from the Meta cookbook when available. Keep the adapter narrow and explicit.
        payload: dict[str, Any] = {
            "model": self.model_id,
            "image": image_bytes,
            "prompt": text_prompt,
        }

        try:
            response = requests.post(
                f"{self.base_url}{self.endpoint}",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code == 429:
                raise ClientError("SamClient rate limit exceeded (HTTP 429)")
            if response.status_code >= 400:
                raise ClientError(f"SamClient request failed with status {response.status_code}")
            if not response.content:
                raise ClientError("SamClient returned an empty response")
            data = response.json()
        except requests.Timeout:
            raise ClientError("SamClient request timed out") from None
        except requests.RequestException as exc:
            raise ClientError(f"SamClient network error: {exc}") from exc

        masks: list[dict] = []
        for item in data.get("masks", []) if isinstance(data, dict) else []:
            arr = np.asarray(item.get("mask", []), dtype=bool)
            masks.append({
                "mask": arr,
                "score": float(item.get("score", 0.0)),
                "box": item.get("box"),
            })
        return masks
