"""Minimal client for TypeSafe's System One API (the API Jev is served on).

Any server that speaks the same ``/v1/systemone`` schema works: pass ``base_url``
or set ``TYPESAFE_BASE_URL``. An API key is only required for TypeSafe's own endpoint.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

DEFAULT_BASE_URL = "https://api.typesafe.ai"

# USD per 1M input tokens; output tokens are free. TypeSafe pricing as of Sept 2026.
JEV_INPUT_PRICE_PER_M = 0.042

# Jev reads ~64k tokens in total and ~32k for the state plus its longest question.
# Four characters per token is a rough English average, so this leaves headroom.
MAX_STATE_CHARS = 100_000

RETRY_STATUSES = (429, 529)


class JevError(Exception):
    pass


class JevClient:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        model: str = "jev-latest",
        timeout: float = 30.0,
        max_retries: int = 3,
    ):
        self.base_url = (base_url or os.environ.get("TYPESAFE_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.api_key = api_key or os.environ.get("TYPESAFE_API_KEY")
        if not self.api_key and self.base_url == DEFAULT_BASE_URL:
            raise JevError("Set TYPESAFE_API_KEY or pass api_key")
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def evaluate(self, state: Any, questions: dict[str, dict]) -> dict:
        """POST one request; returns the raw response ({"model", "answers", "usage"})."""
        body = json.dumps({"state": state, "model": self.model, "questions": questions}).encode()
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries + 1):
            request = urllib.request.Request(
                f"{self.base_url}/v1/systemone", data=body, headers=headers, method="POST"
            )
            try:
                with urllib.request.urlopen(request, timeout=self.timeout) as response:
                    return json.loads(response.read())
            except urllib.error.HTTPError as e:
                if e.code in RETRY_STATUSES and attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise JevError(f"System One API returned {e.code}: {e.read()[:300].decode(errors='replace')}") from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries:
                    time.sleep(min(2**attempt, 8))
                    continue
                raise JevError(f"Could not reach {self.base_url}: {e}") from e
        raise AssertionError("unreachable")
