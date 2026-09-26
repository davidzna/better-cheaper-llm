"""Pick confidence thresholds from Jev-vs-LLM agreement data.

Agreement with your current LLM is the bar, not correctness: where Jev gives the
same answer, switching changes only the bill. A threshold is recommended only if
the 95% lower bound on agreement clears the target, so small samples can't
produce confident-looking recommendations.
"""

from __future__ import annotations

import json
import math
import statistics
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

THRESHOLDS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.98, 0.99)


@dataclass
class Point:
    threshold: float  # act on Jev's answer when its confidence is at least this
    covered: int
    agreed: int
    total: int

    @property
    def coverage(self) -> float:
        return self.covered / self.total if self.total else 0.0

    @property
    def agreement(self) -> float | None:
        return self.agreed / self.covered if self.covered else None

    @property
    def agreement_lower(self) -> float | None:
        return wilson_lower(self.agreed, self.covered) if self.covered else None


def wilson_lower(successes: int, n: int, z: float = 1.645) -> float:
    """One-sided 95% lower bound of a proportion (Wilson score interval)."""
    p = successes / n
    centre = p + z * z / (2 * n)
    margin = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return (centre - margin) / (1 + z * z / n)


def sweep(results: list[tuple[float, bool]]) -> list[Point]:
    """results: (Jev confidence, whether Jev agreed with the LLM) per call."""
    return [
        Point(t, sum(c >= t for c, _ in results), sum(a for c, a in results if c >= t), len(results))
        for t in THRESHOLDS
    ]


def yes_no_results(pairs: list[tuple[float, bool]]) -> list[tuple[float, bool]]:
    """(Jev's p, reference yes/no) -> (confidence, agreed). Confidence is 0 at p=0.5, 1 at p=0 or 1."""
    return [(abs(2 * p - 1), (p >= 0.5) == reference) for p, reference in pairs]


def auroc(pairs: list[tuple[float, bool]]) -> float | None:
    """Chance a random true row scores above a random false one (ties count half)."""
    positives = sum(label for _, label in pairs)
    negatives = len(pairs) - positives
    if not positives or not negatives:
        return None
    ordered = sorted(pairs, key=lambda pair: pair[0])
    rank_sum, i = 0.0, 0
    while i < len(ordered):
        j = i
        while j < len(ordered) and ordered[j][0] == ordered[i][0]:
            j += 1
        rank_sum += (i + 1 + j) / 2 * sum(label for _, label in ordered[i:j])  # tied rows share their mean rank
        i = j
    return (rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def recommend(points: list[Point], target: float = 0.98) -> Point | None:
    """The threshold that offloads the most calls while agreement stays above target."""
    ok = [p for p in points if p.covered and p.agreement_lower >= target]
    return max(ok, key=lambda p: (p.coverage, -p.threshold)) if ok else None


def render_sweep(points: list[Point], *, as_check_threshold: bool = False) -> list[str]:
    head = "Checker threshold" if as_check_threshold else "Confidence >="
    lines = [f"| {head} | Offloaded | Agreement | 95% lower bound |", "|---:|---:|---:|---:|"]
    for p in points:
        if not p.covered:
            continue
        t = (1 + p.threshold) / 2 if as_check_threshold else p.threshold
        lines.append(f"| {t:.3g} | {p.coverage:.0%} | {p.agreement:.1%} | {p.agreement_lower:.1%} |")
    return lines


# --- Checker shadow logs -------------------------------------------------------

def load_check_log(path: str | Path) -> dict[str, dict]:
    """Per site: agreement pairs plus counts and latencies from a Checker log."""
    sites: dict[str, dict] = defaultdict(lambda: {"results": [], "calls": 0, "errors": 0, "jev_ms": [], "fallback_ms": []})
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            site = sites[rec["site"]]
            site["calls"] += 1
            site["errors"] += bool(rec.get("error"))
            if rec.get("jev_ms") is not None and not rec.get("error"):
                site["jev_ms"].append(rec["jev_ms"])
            if rec.get("fallback_ms") is not None:
                site["fallback_ms"].append(rec["fallback_ms"])
            p, incumbent = rec.get("p"), rec.get("fallback_value")
            if p is not None and incumbent is not None:
                site["results"] += yes_no_results([(p, incumbent)])
    return dict(sites)


def render_check_report(sites: dict[str, dict], *, source: str, target: float) -> str:
    out = [f"# better-cheaper-llm report: {source}", "", f"Target agreement with your current check: {target:.0%}", ""]
    for name, s in sorted(sites.items()):
        out += [f"## `{name}`", ""]
        facts = [f"{s['calls']:,} checks", f"{len(s['results']):,} with both answers"]
        if s["errors"]:
            facts.append(f"{s['errors']:,} Jev errors")
        if s["jev_ms"]:
            facts.append(f"Jev median {statistics.median(s['jev_ms']):,.0f} ms")
        if s["fallback_ms"]:
            facts.append(f"current check median {statistics.median(s['fallback_ms']):,.0f} ms")
        out += [", ".join(facts), ""]
        if not s["results"]:
            out += ["Nothing to compare yet: run the Checker in shadow mode with a fallback.", ""]
            continue
        points = sweep(s["results"])
        out += render_sweep(points, as_check_threshold=True) + [""]
        best = recommend(points, target)
        if best:
            line = (
                f"Recommended: `threshold={(1 + best.threshold) / 2:.3g}`. Jev answers {best.coverage:.0%} of checks, "
                f"agreeing {best.agreement:.1%} of the time"
            )
            out.append(line + ("; the rest go to your current check." if best.coverage < 1 else "."))
        else:
            out.append(f"No threshold reaches {target:.0%} agreement with confidence yet. Keep shadowing, or lower the target.")
        out.append("")
    return "\n".join(out)

