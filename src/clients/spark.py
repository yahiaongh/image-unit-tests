from __future__ import annotations

from typing import Any

import requests

from src.config import get_env
from src.json_utils import SPARK_SYSTEM_PROMPT, parse_constraints


class ClientError(RuntimeError):
    """Raised when a client call fails before returning usable data."""


class SparkTruncationError(ClientError):
    """Raised when Muse Spark exhausts its response token limit."""


class SparkClient:
    def __init__(self, config: dict):
        self.config = config
        self.base_url = config.get("basten", {}).get("base_url")
        self.api_key_env = config.get("basten", {}).get("api_key_env")
        self.api_key = get_env(self.api_key_env) if self.api_key_env else None
        self.model_id = config.get("models", {}).get("spark", {}).get("model_id")
        self.endpoint = config.get("models", {}).get("spark", {}).get("endpoint")

    def generate_constraints(self, brief: str) -> list[dict]:
        """Return parsed constraint dictionaries."""
        if not self.endpoint:
            raise ClientError("SparkClient endpoint is not configured in config.yaml")

        raw = self._raw_completion_with_retry(brief)

        def retry_fn(_system_prompt: str, repair_prompt: str) -> str:
            return self._raw_completion_with_retry(repair_prompt)

        return [
            constraint.model_dump()
            for constraint in parse_constraints(raw, llm_callable=retry_fn)
        ]

    def _raw_completion_with_retry(self, prompt: str) -> str:
        try:
            return self._raw_completion(prompt)
        except SparkTruncationError:
            return self._raw_completion(prompt)

    def _raw_completion(self, prompt: str) -> str:
        # source: /tmp/baseten-chat-completions.html:702
        # — the curl example uses an Api-Key Authorization header.
        headers = {
            "Authorization": f"Api-Key {self.api_key}",
            "Content-Type": "application/json",
        }

        # source: https://docs.baseten.co/reference/inference-api/chat-completions
        # — chat-completions request fields and JSON response mode.
        payload: dict[str, Any] = {
            "model": self.model_id,
            "messages": [
                {"role": "system", "content": SPARK_SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            "temperature": self.config.get("models", {}).get("spark", {}).get(
                "temperature", 0.0
            ),
            "response_format": {"type": "json_object"},
            "max_tokens": 8192,
        }

        try:
            response = requests.post(
                f"{self.base_url}{self.endpoint}",
                headers=headers,
                json=payload,
                timeout=30,
            )
            if response.status_code == 429:
                raise ClientError(
                    f"SparkClient rate limit exceeded (HTTP 429): {response.text}"
                )
            if response.status_code >= 400:
                raise ClientError(
                    f"SparkClient request failed with status {response.status_code}: "
                    f"{response.text}"
                )
            if not response.text or not response.text.strip():
                raise ClientError("SparkClient returned an empty response")
            data = response.json()
        except requests.Timeout:
            raise ClientError("SparkClient request timed out") from None
        except requests.RequestException as exc:
            raise ClientError(f"SparkClient network error: {exc}") from exc

        try:
            choice = data["choices"][0]
            message = choice["message"]
        except (KeyError, TypeError) as exc:
            raise ValueError(f"SparkClient returned invalid constraint JSON: {exc}") from exc

        finish = choice.get("finish_reason")
        usage = data.get("usage", {})
        if finish == "length":
            raise SparkTruncationError(
                "Muse Spark exhausted max_tokens; "
                f"finish_reason={finish!r} usage={usage!r}"
            )
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, str) or not content.strip():
            fields = list(message) if isinstance(message, dict) else []
            raise ValueError(
                "Muse Spark returned empty content; "
                f"fields seen: {fields} finish_reason={finish!r} usage={usage!r}"
            )
        return content
