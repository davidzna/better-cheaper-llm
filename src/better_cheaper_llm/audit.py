"""Where a pipeline's tokens go, and which of its LLM calls are really decisions."""

from __future__ import annotations

import json
import re
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .jev import JEV_INPUT_PRICE_PER_M
from .passages import extract_passages, used_labels
from .traces import Call

# A decision's answer is short. Judge by the visible text, not output_tokens, so a
# reasoning model that thinks for 800 tokens before saying "yes" still counts.
MAX_DECISION_CHARS = 60
MAX_OPTIONS = 255  # Jev's limit for one Choice question


@dataclass
class Price:
    """USD per 1M tokens."""

    input: float
    output: float
    cache_read: float
    cache_write: float


def load_prices(path: str | Path) -> dict[str, Price]:
    """JSON of {model: {input, output, cache_read?, cache_write?}}; keys starting with _ are comments.

    Missing cache prices default to Anthropic's multipliers (0.1x read, 1.25x write).
    """
    raw = json.loads(Path(path).read_text())
    return {
        model: Price(
            input=p["input"],
            output=p["output"],
            cache_read=p.get("cache_read", p["input"] * 0.1),
            cache_write=p.get("cache_write", p["input"] * 1.25),
        )
        for model, p in raw.items()
        if not model.startswith("_")
    }


def price_for(model: str, prices: dict[str, Price]) -> Price | None:
    """Exact match, else the longest configured name the model starts with (handles dated IDs)."""
    if model in prices:
        return prices[model]
    matches = [name for name in prices if model.startswith(name)]
    return prices[max(matches, key=len)] if matches else None


def call_cost(call: Call, price: Price) -> float:
    return (
        call.input_tokens * price.input
        + call.cache_read_tokens * price.cache_read
        + call.cache_write_tokens * price.cache_write
        + call.output_tokens * price.output
    ) / 1e6


def label(output: str) -> str:
    """Normalise a short answer so 'Yes.', ' yes' and '{"relevant": "yes"}' are one label."""
    text = output.strip()
    try:
        value = json.loads(text)
    except ValueError:
        value = None
    if isinstance(value, dict) and len(value) == 1:
        value = next(iter(value.values()))
    if isinstance(value, (str, int, float, bool)):
        text = str(value)
    return re.sub(r"\s+", " ", text).strip().strip("`\"'*.!").strip().lower()[:80]


@dataclass
class Site:
    name: str
    model: str
    calls: list[Call]
    kind: str = "generation"  # "decision", "generation" or "few calls"
    labels: list[tuple[str, int]] = field(default_factory=list)
    cost: float | None = None
    rag_calls: int = 0
    passages: int = 0
    passage_tokens: int = 0
    unused_passage_tokens: int = 0
    unused_passage_cost: float | None = None

    @property
    def input_tokens(self) -> int:
        return sum(c.all_input_tokens for c in self.calls)

    @property
    def output_tokens(self) -> int:
        return sum(c.output_tokens for c in self.calls)

    @property
    def latency_ms(self) -> float | None:
        """Median latency of the calls that have one."""
        timed = [c.latency_ms for c in self.calls if c.latency_ms is not None]
        return statistics.median(timed) if timed else None

    @property
    def jev_cost(self) -> float:
        """What Jev would charge to read every call's input once (output is free)."""
        return self.input_tokens * JEV_INPUT_PRICE_PER_M / 1e6

    @property
    def decision_saving(self) -> float | None:
        if self.kind != "decision" or self.cost is None:
            return None
        return self.cost - self.jev_cost

    @property
    def passage_saving(self) -> float | None:
        if not self.rag_calls or self.unused_passage_cost is None:
            return None
        # Jev has to read every passage to decide which ones to drop.
        return self.unused_passage_cost - self.passage_tokens * JEV_INPUT_PRICE_PER_M / 1e6


def audit(calls: list[Call], prices: dict[str, Price] | None = None, min_calls: int = 5) -> list[Site]:
    """Group calls into sites and analyse each one. Sorted by cost (or tokens without prices)."""
    by_site: dict[str, list[Call]] = defaultdict(list)
    for call in calls:
        by_site[call.site].append(call)

    sites = []
    for name, site_calls in by_site.items():
        site = Site(name, Counter(c.model for c in site_calls).most_common(1)[0][0], site_calls)
        _classify(site, min_calls)
        price = price_for(site.model, prices) if prices else None
        if price and all(price_for(c.model, prices) for c in site_calls):
            site.cost = sum(call_cost(c, price_for(c.model, prices)) for c in site_calls)
        if site.kind != "decision":
            _measure_passages(site, price)
        sites.append(site)

    sites.sort(key=lambda s: (s.cost or 0, s.input_tokens + s.output_tokens), reverse=True)
    return sites


def _classify(site: Site, min_calls: int) -> None:
    if len(site.calls) < min_calls:
        site.kind = "few calls"
        return
    lengths = sorted(len(c.output.strip()) for c in site.calls)
    p90 = lengths[int(0.9 * (len(lengths) - 1))]
    labels = Counter(label(c.output) for c in site.calls)
    if p90 <= MAX_DECISION_CHARS and 2 <= len(labels) <= min(MAX_OPTIONS, len(site.calls) // 2):
        site.kind = "decision"
        site.labels = labels.most_common()


def _measure_passages(site: Site, price: Price | None) -> None:
    for call in site.calls:
        text = f"{call.system}\n\n{call.prompt}"
        passages = extract_passages(text)
        if len(passages) < 2 or not call.output.strip():
            continue
        used = used_labels(passages, call.output)
        tokens_per_char = call.all_input_tokens / max(len(text), 1)
        site.rag_calls += 1
        site.passages += len(passages)
        site.passage_tokens += round(sum(len(p.text) for p in passages) * tokens_per_char)
        site.unused_passage_tokens += round(sum(len(p.text) for p in passages if p.label not in used) * tokens_per_char)
    if site.rag_calls and price:
        site.unused_passage_cost = site.unused_passage_tokens * price.input / 1e6


def render(sites: list[Site], *, source: str, skipped: int, priced: bool) -> str:
    calls = [c for s in sites for c in s.calls]
    total_cost = sum(s.cost or 0 for s in sites)
    total_tokens = sum(s.input_tokens + s.output_tokens for s in sites)

    def share(site: Site) -> str:
        if priced and total_cost:
            return f"{(site.cost or 0) / total_cost:.0%}"
        return f"{(site.input_tokens + site.output_tokens) / max(total_tokens, 1):.0%}"

    out = [f"# better-cheaper-llm audit: {source}", ""]
    notes = [
        f"{len(calls):,} calls",
        f"{len(sites)} call sites",
        f"{sum(c.site_tagged for c in calls):,} calls tagged with a site name",
    ]
    if skipped:
        notes.append(f"{skipped:,} lines skipped (unrecognised)")
    if estimated := sum(c.estimated for c in calls):
        notes.append(f"{estimated:,} calls without usage data (tokens estimated)")
    timed = sum(c.latency_ms is not None for c in calls)
    if timed:
        notes.append(f"{timed:,} calls with latency")
    out += [", ".join(notes), ""]
    if priced:
        out += [f"**Total cost in this log: {money(total_cost)}**", ""]
    else:
        out += ["No prices given; pass `--prices prices.json` for dollar figures.", ""]

    latency_head, latency_rule = (" Median latency |", "---:|") if timed else ("", "")
    out += ["## Call sites", "",
            f"| Site | Model | Calls | Input tokens | Output tokens | Cost | Share |{latency_head} Kind |",
            f"|---|---|---:|---:|---:|---:|---:|{latency_rule}---|"]
    for s in sites:
        latency = f" {_ms(s.latency_ms)} |" if timed else ""
        out.append(
            f"| `{s.name}` | {s.model} | {len(s.calls):,} | {s.input_tokens:,} | {s.output_tokens:,} "
            f"| {money(s.cost)} | {share(s)} |{latency} {s.kind} |"
        )

    decisions = [s for s in sites if s.kind == "decision"]
    out += ["", "## Decisions an LLM is making today", ""]
    if decisions:
        out += [
            "These sites answer from a small, repeated set. Jev can answer them: yes/no with "
            "`better_cheaper_llm.Checker`, larger sets with a Choice question.",
            "",
            "| Site | Calls | Answers seen | Cost now | Jev cost | Upper-bound saving |",
            "|---|---:|---|---:|---:|---:|",
        ]
        for s in decisions:
            seen = ", ".join(f"{lab or '(empty)'} {n:,}" for lab, n in s.labels[:4])
            if len(s.labels) > 4:
                seen += f", +{len(s.labels) - 4} more"
            out.append(
                f"| `{s.name}` | {len(s.calls):,} | {seen} | {money(s.cost)} | {money(s.jev_cost)} "
                f"| {money(s.decision_saving)} |"
            )
    else:
        out.append("None found.")

    rag = [s for s in sites if s.rag_calls]
    out += ["", "## Retrieved passages the answer may not use", ""]
    if rag:
        out += [
            "A passage counts as used if the answer cites it or shares wording with it. This is an "
            "estimate: an answer that paraphrases a passage without citing it makes that passage look unused.",
            "",
            "| Site | RAG calls | Passages per call | Passage tokens | Likely unused | Upper-bound saving |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for s in rag:
            out.append(
                f"| `{s.name}` | {s.rag_calls:,} | {s.passages / s.rag_calls:.1f} | {s.passage_tokens:,} "
                f"| {s.unused_passage_tokens:,} ({s.unused_passage_tokens / max(s.passage_tokens, 1):.0%}) "
                f"| {money(s.passage_saving)} |"
            )
    else:
        out.append("No calls with two or more recognisable passages (<document> tags or [1]-style numbering).")

    out += ["", "## Bottom line", ""]
    if priced and total_cost:
        saving = sum(max(s.decision_saving or 0, 0) + max(s.passage_saving or 0, 0) for s in sites)
        out.append(
            f"Upper-bound saving: **{money(saving)} of {money(total_cost)} ({saving / total_cost:.0%})**, "
            "if every decision moved to Jev and every unused passage were filtered out."
        )
    if decisions:
        out.append(
            "Real savings depend on how often Jev agrees with your LLM. Measure that with "
            "`better-cheaper-llm replay` (needs `TYPESAFE_API_KEY`)."
        )
    return "\n".join(out) + "\n"


def excerpt(text: str, width: int = 90) -> str:
    text = " ".join(text.split())
    return text if len(text) <= 2 * width else f"{text[:width]} ... {text[-width:]}"


def _ms(x: float | None) -> str:
    return "n/a" if x is None else f"{x:,.0f} ms"


def money(x: float | None) -> str:
    if x is None:
        return "n/a"
    if abs(x) < 0.01:
        return f"${x:.4f}"
    return f"${x:,.2f}"
