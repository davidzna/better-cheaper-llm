"""Score a RAG dataset with Jev: is each answer grounded, and are the passages sufficient?

Rows are JSON lines: {"id", "question", "passages": [...], "answer"} plus optional
human labels "grounded" and/or "sufficient" (true or false). With labels, the
report shows how well Jev's scores match them and which threshold to trust.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from .audit import excerpt
from .calibrate import auroc, recommend, render_sweep, sweep, yes_no_results
from .check import Checker
from .rag import grounded, sufficient

METRICS = {
    "grounded": "is every claim in the answer supported by the passages?",
    "sufficient": "do the passages contain what's needed to answer the question?",
}
MAX_LISTED = 5


@dataclass
class Row:
    id: str
    question: str | None
    passages: list[str]
    answer: str | None
    labels: dict[str, bool] = field(default_factory=dict)

    def supports(self, metric: str) -> bool:
        return bool(self.answer if metric == "grounded" else self.question)


@dataclass
class Score:
    row: Row
    metric: str
    p: float | None  # None when Jev failed
    error: str | None = None


def load_dataset(path: str | Path) -> tuple[list[Row], int]:
    """Return (rows, number of skipped lines). `contexts` is accepted as an alias for `passages`."""
    rows: list[Row] = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                passages = rec.get("passages", rec.get("contexts"))
                if isinstance(passages, str):
                    passages = [passages]
                if not isinstance(passages, list) or not passages:
                    raise ValueError("no passages")
                rows.append(
                    Row(
                        id=str(rec.get("id", f"line{n}")),
                        question=rec.get("question"),
                        passages=[str(p) for p in passages],
                        answer=rec.get("answer"),
                        labels={m: rec[m] for m in METRICS if isinstance(rec.get(m), bool)},
                    )
                )
            except (ValueError, TypeError, AttributeError):
                skipped += 1
    return rows, skipped


def score(
    rows: list[Row],
    client: Any,
    metrics: tuple[str, ...] = tuple(METRICS),
    *,
    workers: int = 4,
    on_progress: Callable[[int, int], None] | None = None,
) -> list[Score]:
    """One Jev request per row and metric, `workers` at a time."""
    checker = Checker(client, mode="enforce")
    jobs = [(row, m) for row in rows for m in metrics if row.supports(m)]

    def run(job: tuple[Row, str]) -> Score:
        row, metric = job
        if metric == "grounded":
            result = grounded(checker, row.answer, row.passages)
        else:
            result = sufficient(checker, row.question, row.passages)
        return Score(row, metric, result.p, result.error)

    scores = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for i, s in enumerate(pool.map(run, jobs), 1):
            scores.append(s)
            if on_progress:
                on_progress(i, len(jobs))
    return scores


def write_scores(scores: list[Score], path: str | Path) -> None:
    """One JSON line per row: {"id", "scores": {metric: p}, "errors"?: {metric: message}}."""
    by_row: dict[str, dict] = {}
    for s in scores:
        rec = by_row.setdefault(s.row.id, {"id": s.row.id, "scores": {}})
        rec["scores"][s.metric] = s.p
        if s.error:
            rec.setdefault("errors", {})[s.metric] = s.error
    Path(path).write_text("".join(json.dumps(r) + "\n" for r in by_row.values()))


def render(scores: list[Score], *, source: str, skipped: int, threshold: float, target: float) -> str:
    out = [f"# better-cheaper-llm eval: {source}", ""]
    facts = [f"{len({s.row.id for s in scores}):,} rows"]
    if skipped:
        facts.append(f"{skipped:,} lines skipped (no passages or not JSON)")
    out += [", ".join(facts), ""]

    for metric, meaning in METRICS.items():
        scored = [s for s in scores if s.metric == metric]
        if not scored:
            continue
        ok = [s for s in scored if s.p is not None]
        out += [f"## {metric}: {meaning}", ""]
        if len(ok) < len(scored):
            first = next(s.error for s in scored if s.p is None)
            out += [f"{len(scored) - len(ok):,} rows failed (first error: {first})", ""]
        if not ok:
            continue
        yes = sum(s.p >= threshold for s in ok)
        no = sum(s.p <= 1 - threshold for s in ok)
        out += [
            "| Jev's verdict | Rows |",
            "|---|---:|",
            f"| Yes (p >= {threshold:g}) | {yes:,} ({yes / len(ok):.0%}) |",
            f"| Unsure | {len(ok) - yes - no:,} ({(len(ok) - yes - no) / len(ok):.0%}) |",
            f"| No (p <= {1 - threshold:.2g}) | {no:,} ({no / len(ok):.0%}) |",
            "",
            "Lowest scores:",
            "",
        ]
        for s in sorted(ok, key=lambda s: s.p)[:MAX_LISTED]:
            text = s.row.answer if metric == "grounded" else s.row.question
            out.append(f"- `{s.row.id}` p={s.p:.2f}: {excerpt(text or '')}")
        labelled = [s for s in ok if metric in s.row.labels]
        if labelled:
            out += _against_labels(labelled, metric, target)
        out.append("")
    return "\n".join(out)


def _against_labels(labelled: list[Score], metric: str, target: float) -> list[str]:
    pairs = [(s.p, s.row.labels[metric]) for s in labelled]
    out = ["", f"### Against your labels ({len(pairs):,} rows)", ""]
    area = auroc(pairs)
    if area is None:
        out.append("AUROC needs both true and false labels.")
    else:
        out.append(
            f"AUROC {area:.3f}: the chance Jev scores a random true row above a random false one "
            "(1.0 is perfect, 0.5 is chance)."
        )
    points = sweep(yes_no_results(pairs))
    out += [""] + render_sweep(points, as_check_threshold=True) + [""]
    best = recommend(points, target)
    if best:
        line = (
            f"Recommended: `threshold={(1 + best.threshold) / 2:.3g}`. Jev decides {best.coverage:.0%} of rows "
            f"at {best.agreement:.1%} agreement with your labels"
        )
        out.append(line + ("; send the rest to a person or an LLM judge." if best.coverage < 1 else "."))
    else:
        out.append(f"No threshold reaches {target:.0%} agreement with your labels with confidence. Add labelled rows or lower the target.")
    wrong = [s for s in labelled if (s.p >= 0.5) != s.row.labels[metric]]
    if wrong:
        out += ["", "Most confident disagreements with your labels (the label may be the mistake):", ""]
        for s in sorted(wrong, key=lambda s: -abs(2 * s.p - 1))[:MAX_LISTED]:
            out.append(f"- `{s.row.id}` labelled {str(s.row.labels[metric]).lower()}, p={s.p:.2f}")
    return out
