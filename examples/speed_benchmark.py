"""Time the same RAG decisions on Jev and on OpenAI models.

Two tasks from the sample pipeline:

* grade: is this passage relevant to the question? (short prompt)
* hallucination: is this answer supported by the five passages? Half the answers
  are correct and half have one fact altered, so accuracy is measured too.

Calls run one at a time from your machine, so every system pays the same network
cost. Needs TYPESAFE_API_KEY, OPENAI_API_KEY and the sample files:

    uv run python examples/make_samples.py
    uv run python examples/speed_benchmark.py --models gpt-5.4-nano gpt-6-astra --n 20

Add --reasoning-effort low|medium|high to measure models that reason before answering.
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import time
import urllib.request
from pathlib import Path

from make_samples import CHECK_SYSTEM

from better_cheaper_llm import Checker, JevClient, grounded
from better_cheaper_llm.traces import load_calls

HERE = Path(__file__).parent


def ask_openai(model: str, system: str, user: str, effort: str | None) -> tuple[bool, float, int]:
    """(answered yes, milliseconds, completion tokens including hidden reasoning)."""
    body = {"model": model, "max_completion_tokens": 4000,
            "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
    if effort:
        body["reasoning_effort"] = effort
    request = urllib.request.Request(
        "https://api.openai.com/v1/chat/completions", data=json.dumps(body).encode(),
        headers={"Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}", "Content-Type": "application/json"},
    )
    start = time.perf_counter()
    response = json.loads(urllib.request.urlopen(request, timeout=300).read())
    ms = (time.perf_counter() - start) * 1000
    text = (response["choices"][0]["message"]["content"] or "").strip().lower()
    return text.startswith("yes"), ms, response["usage"]["completion_tokens"]


def ask_jev(fn) -> tuple[bool, float]:
    start = time.perf_counter()
    result = fn()
    return result.p is not None and result.p >= 0.5, (time.perf_counter() - start) * 1000


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["gpt-5.4-nano", "gpt-6-astra"])
    parser.add_argument("--n", type=int, default=20, help="items per task")
    parser.add_argument("--reasoning-effort", help="passed to the OpenAI models as reasoning_effort")
    args = parser.parse_args()

    checker = Checker(JevClient(), mode="enforce")
    systems = ["Jev", *args.models]
    stats = {(task, s): {"ms": [], "correct": 0, "tokens": []} for task in ("grade", "hallucination") for s in systems}

    calls, _ = load_calls(HERE / "rag_traces.jsonl")
    graded = [c for c in calls if c.system.startswith("You are a grader")][: args.n]
    for i, call in enumerate(graded, 1):
        _, ms = ask_jev(lambda: checker.check("grade", "The document is relevant to the question.", call.prompt))
        stats["grade", "Jev"]["ms"].append(ms)
        for model in args.models:
            _, ms, tokens = ask_openai(model, call.system, call.prompt, args.reasoning_effort)
            stats["grade", model]["ms"].append(ms)
            stats["grade", model]["tokens"].append(tokens)
        print(f"\rgrade {i}/{len(graded)}", end="", flush=True)
    print()

    rows = [json.loads(line) for line in (HERE / "rag_eval.jsonl").read_text().splitlines()]
    original = [r for r in rows if r["id"].endswith("-original")]
    altered = [r for r in rows if r["id"].endswith("-altered")]
    items = [r for pair in zip(original, altered) for r in pair][: args.n]
    for i, row in enumerate(items, 1):
        documents = "\n".join(f'<document index="{j + 1}">\n{p}\n</document>' for j, p in enumerate(row["passages"]))
        user = f"<documents>\n{documents}\n</documents>\n\nAnswer: {row['answer']}"
        verdict, ms = ask_jev(lambda: grounded(checker, row["answer"], row["passages"]))
        stats["hallucination", "Jev"]["ms"].append(ms)
        stats["hallucination", "Jev"]["correct"] += verdict == row["grounded"]
        for model in args.models:
            verdict, ms, tokens = ask_openai(model, CHECK_SYSTEM, user, args.reasoning_effort)
            s = stats["hallucination", model]
            s["ms"].append(ms)
            s["tokens"].append(tokens)
            s["correct"] += verdict == row["grounded"]
        print(f"\rhallucination {i}/{len(items)}", end="", flush=True)
    print("\n")

    print("| Task | System | Median | 90% within | Output tokens (median) | Correct |")
    print("|---|---|---:|---:|---:|---:|")
    for (task, system), s in stats.items():
        ordered = sorted(s["ms"])
        tokens = f"{statistics.median(s['tokens']):,.0f}" if s["tokens"] else "0"
        correct = f"{s['correct']}/{len(ordered)}" if task == "hallucination" else ""
        print(f"| {task} | {system} | {statistics.median(ordered):,.0f} ms "
              f"| {ordered[int(0.9 * (len(ordered) - 1))]:,.0f} ms | {tokens} | {correct} |")


if __name__ == "__main__":
    main()
