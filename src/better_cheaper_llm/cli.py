"""better-cheaper-llm audit | replay | report | eval"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .audit import audit, load_prices, render
from .calibrate import load_check_log, render_check_report
from .evaluate import METRICS, load_dataset, score, write_scores
from .evaluate import render as render_eval
from .jev import JevClient, JevError
from .replay import render as render_replay
from .replay import replay_site
from .traces import load_calls


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="better-cheaper-llm", description="Find the LLM calls in a RAG pipeline that Jev can take over, then measure it."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    a = sub.add_parser("audit", help="where the tokens go, and which calls are really decisions (offline)")
    _trace_args(a)

    r = sub.add_parser("replay", help="re-ask decision calls to Jev and measure agreement (needs TYPESAFE_API_KEY)")
    _trace_args(r)
    r.add_argument("--sample", type=int, default=100, help="calls to replay per site (default 100)")
    r.add_argument("--site", action="append", help="only replay this site (repeatable)")
    r.add_argument("--target", type=float, default=0.98, help="agreement to aim for (default 0.98)")

    c = sub.add_parser("report", help="recommend Checker thresholds from a shadow-mode log")
    c.add_argument("log", help="JSONL written by Checker(log_path=...)")
    c.add_argument("--target", type=float, default=0.98, help="agreement to aim for (default 0.98)")

    e = sub.add_parser("eval", help="score a RAG dataset for groundedness and sufficiency (needs TYPESAFE_API_KEY)")
    e.add_argument("dataset", help="JSONL rows of question, passages, answer, and optional true/false labels")
    e.add_argument("--metrics", default=",".join(METRICS), help=f"comma-separated, from: {', '.join(METRICS)}")
    e.add_argument("--threshold", type=float, default=0.9, help="p at or above this is yes, at or below 1 - this is no (default 0.9)")
    e.add_argument("--target", type=float, default=0.98, help="agreement with your labels to aim for (default 0.98)")
    e.add_argument("--out", help="write per-row scores to this JSONL file")
    e.add_argument("--workers", type=int, default=4, help="requests in parallel (default 4)")

    args = parser.parse_args(argv)
    if args.command == "report":
        print(render_check_report(load_check_log(args.log), source=Path(args.log).name, target=args.target))
        return 0
    if args.command == "eval":
        metrics = tuple(m.strip() for m in args.metrics.split(",") if m.strip())
        if not metrics or set(metrics) - set(METRICS):
            parser.error(f"--metrics must be from: {', '.join(METRICS)}")
        return _eval(args, metrics)

    calls, skipped = load_calls(args.traces)
    if not calls:
        print(f"No recognisable LLM calls in {args.traces} ({skipped} lines skipped).", file=sys.stderr)
        return 1
    prices = load_prices(args.prices) if args.prices else None
    sites = audit(calls, prices, args.min_calls)

    if args.command == "audit":
        print(render(sites, source=Path(args.traces).name, skipped=skipped, priced=prices is not None))
        return 0

    targets = [s for s in sites if s.kind == "decision" and (not args.site or s.name in args.site)]
    if not targets:
        print("No decision sites to replay. Run `better-cheaper-llm audit` to see what was found.", file=sys.stderr)
        return 1
    client = _client()
    if client is None:
        return 1
    runs = []
    for site in targets:
        runs.append(replay_site(site, client, sample=args.sample, on_progress=_progress(site.name)))
        print(file=sys.stderr)
    print(render_replay(runs, target=args.target))
    return 0


def _eval(args: argparse.Namespace, metrics: tuple[str, ...]) -> int:
    rows, skipped = load_dataset(args.dataset)
    if not rows:
        print(f"No usable rows in {args.dataset} ({skipped} lines skipped).", file=sys.stderr)
        return 1
    client = _client()
    if client is None:
        return 1
    scores = score(rows, client, metrics, workers=args.workers, on_progress=_progress("eval"))
    print(file=sys.stderr)
    if args.out:
        write_scores(scores, args.out)
    print(render_eval(scores, source=Path(args.dataset).name, skipped=skipped, threshold=args.threshold, target=args.target))
    return 0


def _client() -> JevClient | None:
    try:
        return JevClient()
    except JevError as e:
        print(e, file=sys.stderr)
        return None


def _trace_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("traces", help="JSONL of logged LLM calls, one per line")
    p.add_argument("--prices", help="JSON of USD per 1M tokens per model (see examples/prices.json)")
    p.add_argument("--min-calls", type=int, default=5, help="calls a site needs before it's classified (default 5)")


def _progress(name: str):
    def show(done: int, total: int) -> None:
        print(f"\r{name}: {done}/{total}", end="", file=sys.stderr, flush=True)

    return show


if __name__ == "__main__":
    sys.exit(main())
