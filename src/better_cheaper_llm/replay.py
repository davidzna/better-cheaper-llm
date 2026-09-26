"""Re-ask logged decision calls to Jev and measure how often it agrees with the LLM."""

from __future__ import annotations

import json
import random
import statistics
import time
from dataclasses import dataclass, field
from typing import Any, Callable

from .audit import MAX_OPTIONS, Site, excerpt, label, money
from .calibrate import Point, recommend, render_sweep, sweep
from .jev import MAX_STATE_CHARS, JevError

QUESTION = "Which answer would the assistant give to the request in the state?"
MAX_DISAGREEMENTS = 5


@dataclass
class Replay:
    site: Site
    results: list[tuple[float, bool]] = field(default_factory=list)  # (confidence, agreed)
    too_large: int = 0
    errors: list[str] = field(default_factory=list)
    jev_ms: list[float] = field(default_factory=list)  # round trip per successful call
    # (LLM's answer, Jev's answer, Jev's confidence, excerpt of the request)
    disagreements: list[tuple[str, str, float, str]] = field(default_factory=list)


def replay_site(
    site: Site,
    client: Any,
    *,
    sample: int = 100,
    seed: int = 0,
    on_progress: Callable[[int, int], None] | None = None,
) -> Replay:
    """Ask Jev each sampled call as a Choice over the answers the LLM gave at this site."""
    options = [lab for lab, _ in site.labels[:MAX_OPTIONS]]
    ids = {f"o{i}": lab for i, lab in enumerate(options)}
    question = {
        "type": "choice",
        "instructions": QUESTION,
        "criteria": {key: f'The assistant answers "{lab}"' for key, lab in ids.items()},
    }
    eligible = [c for c in site.calls if label(c.output) in set(options)]
    chosen = random.Random(seed).sample(eligible, min(sample, len(eligible)))

    run = Replay(site)
    for i, call in enumerate(chosen, 1):
        state = {"instructions": call.system, "request": call.prompt}
        if len(json.dumps(state)) > MAX_STATE_CHARS:
            run.too_large += 1
            continue
        try:
            start = time.perf_counter()
            answer = client.evaluate(state, {"answer": question})["answers"]["answer"]
            run.jev_ms.append((time.perf_counter() - start) * 1000)
            confidence, jev_label, llm_label = float(answer["confidence"]), ids[answer["choice"]], label(call.output)
            run.results.append((confidence, jev_label == llm_label))
            if jev_label != llm_label and len(run.disagreements) < MAX_DISAGREEMENTS:
                run.disagreements.append((llm_label, jev_label, confidence, excerpt(call.prompt)))
        except (JevError, KeyError, TypeError, ValueError) as e:
            run.errors.append(f"{type(e).__name__}: {e}")
            if len(run.errors) >= 5 and not run.results:
                break  # every call is failing: bad key, wrong endpoint, or similar
        if on_progress:
            on_progress(i, len(chosen))
    return run


def render(runs: list[Replay], *, target: float) -> str:
    out = ["# better-cheaper-llm replay", "", f"Target agreement with your LLM: {target:.0%}", ""]
    for run in runs:
        site = run.site
        out += [f"## `{site.name}`", ""]
        facts = [f"{len(run.results):,} calls replayed"]
        if run.too_large:
            facts.append(f"{run.too_large:,} skipped (too large for Jev)")
        if run.errors:
            facts.append(f"{len(run.errors):,} errors (first: {run.errors[0]})")
        out += [", ".join(facts), ""]
        if run.jev_ms:
            out += [_speed(site, run.jev_ms), ""]
        if not run.results:
            continue
        points = sweep(run.results)
        out += render_sweep(points) + [""]
        best = recommend(points, target)
        if best:
            out.append(_projection(site, best))
        else:
            out.append(f"No confidence threshold reaches {target:.0%} agreement with confidence. Try more samples or a lower target.")
        if run.disagreements:
            out += ["", "Disagreements to review. Agreement is measured against your LLM, so some of these may be the LLM's mistakes:", ""]
            out += [f"- LLM `{llm}`, Jev `{jev}` (confidence {c:.2f}): {text}" for llm, jev, c, text in run.disagreements]
        out.append("")
    return "\n".join(out)



def _speed(site: Site, jev_ms: list[float]) -> str:
    ordered = sorted(jev_ms)
    jev = statistics.median(ordered)
    line = f"Speed: Jev took {jev:,.0f} ms per call (median; 90% within {ordered[int(0.9 * (len(ordered) - 1))]:,.0f} ms)."
    if site.latency_ms is not None:
        line += f" Your LLM's median in the log is {site.latency_ms:,.0f} ms, so about {site.latency_ms / jev:.1f}x faster per call."
    else:
        line += " Add latency_ms to your log lines to compare it with your LLM."
    return line


def _projection(site: Site, best: Point) -> str:
    line = (
        f"Recommended: act on Jev when confidence >= {best.threshold:g}. That covers {best.coverage:.0%} of calls "
        f"at {best.agreement:.1%} agreement (95% lower bound {best.agreement_lower:.1%})."
    )
    if site.cost is not None:
        # Jev sees every call; the LLM still answers the ones Jev isn't confident about.
        saving = best.coverage * site.cost - site.jev_cost
        line += f" Projected saving on this log: {money(saving)} of {money(site.cost)}."
    return line
