from __future__ import annotations

import json
from typing import Any

import requests

from src.config import get_env


class ClientError(RuntimeError):
    """Raised when a client call fails before returning usable data."""


class SparkClient:
    def __init__(self, config: dict):
        self.config = config
        self.base_url = config.get("basten", {}).get("base_url") or "TODO"
        self.api_key_env = config.get("basten", {}).get("api_key_env")
        self.api_key = get_env(self.api_key_env) if self.api_key_env else None
        self.model_id = config.get("models", {}).get("spark", {}).get("model_id")
        self.endpoint = config.get("models", {}).get("spark", {}).get("endpoint")

    def generate_constraints(self, brief: str) -> list[dict]:
        """Return a list of constraint dicts. Raise ValueError on invalid JSON."""
        if not self.endpoint or self.endpoint == "TODO":
            raise ClientError("SparkClient endpoint is not configured in config.yaml")

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        # TODO: follow the exact Meta/Baseten chat-completions request schema from the
        # docs when available. Keep the request body shape here as a TODO placeholder.
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "user", "content": brief},
            ],
        }

        try:
            response = requests.post(
                f"{self.base_url}{self.endpoint}",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code == 429:
                raise ClientError("SparkClient rate limit exceeded (HTTP 429)")
            if response.status_code >= 400:
                raise ClientError(f"SparkClient request failed with status {response.status_code}")
            if not response.text or not response.text.strip():
                raise ClientError("SparkClient returned an empty response")
            data = response.json()
        except requests.Timeout:
            raise ClientError("SparkClient request timed out") from None
        except requests.RequestException as exc:
            raise ClientError(f"SparkClient network error: {exc}") from exc

        # TODO: parse the provider-specific JSON response into a list of constraint dicts.
        try:
            text = data["choices"][0]["message"]["content"]
            parsed = json.loads(text)
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"SparkClient returned invalid constraint JSON: {exc}") from exc

        if not isinstance(parsed, list):
            raise ValueError("SparkClient JSON did not resolve to a list of constraints")
        return parsed
