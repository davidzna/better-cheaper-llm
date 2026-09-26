"""Yes/no checks answered by Jev, backed by the check you run today."""

from __future__ import annotations

import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from .jev import MAX_STATE_CHARS, JevClient, JevError


@dataclass
class CheckResult:
    value: bool | None  # None: Jev was unsure (or failed) and no fallback was given
    p: float | None  # Jev's probability that the statement holds; None if Jev failed
    source: str  # "jev", "fallback" or "unsure"
    error: str | None = None

    def __bool__(self) -> bool:
        return bool(self.value)


class Checker:
    """Ask Jev whether a statement holds for some state.

    mode="shadow" (default): return ``fallback()``, your existing check, unchanged,
    and log what Jev would have said. Jev runs in parallel, so this adds no latency
    unless Jev is slower than your check.

    mode="enforce": use Jev's answer when p >= threshold (yes) or p <= 1 - threshold
    (no). Otherwise, or if Jev fails, call ``fallback()``; without one the result is
    ``source="unsure"`` with ``value=None``.

    Logs hold one JSON line per check: probabilities, answers and timings, never the
    state or statements. ``better-cheaper-llm report LOG`` turns them into threshold advice.
    """

    def __init__(self, client: Any = None, *, mode: str = "shadow", log_path: str | Path | None = None):
        if mode not in ("shadow", "enforce"):
            raise ValueError("mode must be 'shadow' or 'enforce'")
        self.client = client if client is not None else JevClient()
        self.mode = mode
        self.log_path = Path(log_path) if log_path else None
        self._pool = ThreadPoolExecutor(max_workers=4)
        self._lock = threading.Lock()

    def check(
        self,
        site: str,
        statement: str | dict | list,
        state: Any,
        *,
        threshold: float = 0.9,
        fallback: Callable[[], bool] | None = None,
        criteria: dict | None = None,
    ) -> CheckResult:
        """``statement`` may be a list: the check holds only if every statement holds.

        ``criteria`` is Jev's optional {"true": ..., "false": ...} description of what
        yes and no mean, applied to every statement.
        """
        if not 0.5 <= threshold < 1:
            raise ValueError("threshold must be at least 0.5 and below 1")
        if self.mode == "shadow" and fallback is None:
            raise ValueError("shadow mode needs a fallback: the check you run today")
        statements = statement if isinstance(statement, list) else [statement]

        fallback_value = fallback_ms = None
        if self.mode == "shadow":
            pending = self._pool.submit(self._ask, statements, state, criteria)
            fallback_value, fallback_ms = _timed(fallback)
            p, error, jev_ms, tokens = pending.result()
            result = CheckResult(fallback_value, p, "fallback", error)
        else:
            p, error, jev_ms, tokens = self._ask(statements, state, criteria)
            if p is not None and (p >= threshold or p <= 1 - threshold):
                result = CheckResult(p >= threshold, p, "jev")
            elif fallback is not None:
                fallback_value, fallback_ms = _timed(fallback)
                result = CheckResult(fallback_value, p, "fallback", error)
            else:
                result = CheckResult(None, p, "unsure", error)

        self._log(
            {
                "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "site": site,
                "mode": self.mode,
                "statements": len(statements),
                "threshold": threshold,
                "p": p,
                "value": result.value,
                "source": result.source,
                "fallback_value": fallback_value,
                "jev_ms": jev_ms,
                "fallback_ms": fallback_ms,
                "jev_input_tokens": tokens,
                "error": error,
            }
        )
        return result

    def _ask(self, statements: list, state: Any, criteria: dict | None):
        """Returns (p, error, ms, input_tokens). Never raises: Jev must not break the pipeline."""
        questions = {}
        for i, s in enumerate(statements):
            questions[f"s{i}"] = {"type": "noul", "instructions": s}
            if criteria:
                questions[f"s{i}"]["criteria"] = criteria
        start = time.perf_counter()
        try:
            if not questions:
                raise ValueError("no statements to check")
            if len(json.dumps(state, default=str)) > MAX_STATE_CHARS:
                raise JevError(f"state is over {MAX_STATE_CHARS:,} characters, too large for Jev")
            response = self.client.evaluate(state, questions)
            p = min(float(response["answers"][key]["noul"]) for key in questions)
            return p, None, _ms(start), (response.get("usage") or {}).get("input_tokens")
        except Exception as e:  # noqa: BLE001
            return None, f"{type(e).__name__}: {e}", _ms(start), None

    def _log(self, record: dict) -> None:
        if not self.log_path:
            return
        with self._lock, open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record) + "\n")


def _timed(fn: Callable[[], bool]) -> tuple[bool, float]:
    start = time.perf_counter()
    value = bool(fn())
    return value, _ms(start)


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 1)
