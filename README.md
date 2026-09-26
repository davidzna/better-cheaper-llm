# better-cheaper-llm

Make RAG pipelines and LLM apps faster and cheaper by moving the yes/no decisions an LLM makes (relevance grading, query routing, hallucination checks, LLM-as-a-judge evals) to [Jev](https://typesafe.ai/blog/introducing-system-one-models-and-jev), TypeSafe's System One decision model. Measure first, then switch.

A lot of the LLM calls in a RAG pipeline aren't asking the model to write anything. Is this passage relevant? Is the answer backed by the sources? Should this question go to the vector store or the web? The model reads a whole prompt, thinks about it, and says "yes".

better-cheaper-llm finds those calls and moves them to Jev. Jev doesn't generate text. It reads the same prompt and returns a typed answer with a probability, in about 200 ms, for $0.042 per million input tokens (output is free). better-cheaper-llm measures before it switches anything, so you only hand over the decisions Jev actually gets right on your traffic.

```
$ better-cheaper-llm audit logs.jsonl --prices prices.json
...
| Site                | Calls | Answers seen   | Cost now | Jev cost | Upper-bound saving |
| check_hallucination |   215 | yes 204, no 11 |    $1.42 |  $0.0038 |              $1.42 |
```

better-cheaper-llm is an independent project and isn't affiliated with TypeSafe. It's early, and the API will change.

## What's in the box

- `better-cheaper-llm audit` reads the LLM logs you already have and shows where the money goes, which calls are really decisions, and how many retrieved passages the answer never used. It runs offline and needs no API key.
- `better-cheaper-llm replay` asks Jev the decisions from your log and reports how often it agrees with your LLM, and how much faster it answered.
- `Checker` sits in your pipeline next to the check you already run. In shadow mode it changes nothing and logs both answers. Once `better-cheaper-llm report` says a threshold is safe, enforce mode lets Jev answer and falls back to your LLM when it's unsure.
- `better-cheaper-llm eval` scores a RAG dataset for groundedness and retrieval sufficiency. If you have human labels, it tells you how far to trust the scores.

It only uses the Python standard library and needs Python 3.10 or newer.

## Why this exists

It started with a question: if decision models like Jev end up inside every agent loop, how much would token spend actually drop? The answer turned out to be "it depends a lot on your pipeline". For a coding agent that mostly writes code, very little. For a RAG pipeline that grades passages, routes queries and checks its own answers, a big share of the bill. There's no way to know which one you have without measuring, so better-cheaper-llm starts by measuring.

## Should you use it?

Probably, if:

- your pipeline asks an LLM to grade, route, classify or fact-check, especially with a reasoning model;
- you log your LLM calls, or can start;
- you'd rather see numbers from your own traffic before changing anything.

Probably not, if:

- most of your spend is generation: long answers, code, summaries. Jev doesn't write text, so there's nothing to move.
- your decisions hinge on arithmetic, counting, or exact dates and figures. TypeSafe lists these as Jev's weak spots, and in our tests a changed digit was the hardest error to catch.
- you can't send prompts to a third-party API. You can point better-cheaper-llm at any server that implements the same API (see [Other servers](#other-servers)).
- your decision prompts run past about 64k tokens, which is Jev's limit.

## Getting started

better-cheaper-llm isn't on PyPI yet, so install it from GitHub:

```bash
git clone https://github.com/davidzna/better-cheaper-llm
cd better-cheaper-llm
uv sync
uv run python examples/make_samples.py     # a synthetic RAG pipeline log and a labelled eval set
uv run better-cheaper-llm audit examples/rag_traces.jsonl --prices examples/prices.json
```

The sample pipeline routes each question, grades five retrieved passages, writes an answer and then checks it for hallucinations. It's built to show every kind of finding, so don't read its totals as typical.

`replay`, `eval` and the `Checker` need a TypeSafe key:

```bash
export TYPESAFE_API_KEY=...
```

## Typical use cases

### Find out where the bill goes

You run a RAG app and the LLM invoice keeps growing, but you don't know which calls are responsible. Export a week of logs and audit them:

```bash
better-cheaper-llm audit last_week.jsonl --prices prices.json
```

The usual surprise is a reasoning model doing a yes/no job, like a "did the answer stay on topic?" check that thinks for a few hundred tokens before saying "yes". Those show up at the top of the decisions table. Nothing in your app changes until you decide it should.

### Replace an LLM relevance grader

Your pipeline retrieves ten chunks from a product knowledge base, then asks an LLM to grade each one before building the prompt (the corrective-RAG pattern). That's ten LLM calls per question just to throw chunks away. Keep your grader as the fallback and let Jev take the confident cases:

```python
from better_cheaper_llm import Checker

checker = Checker(mode="shadow", log_path="grading.jsonl")

relevant = [
    chunk for chunk in chunks
    if checker.check("grade_chunk", "The document is relevant to the question.",
                     {"question": question, "document": chunk},
                     fallback=lambda: llm_grades_relevant(question, chunk))
]
```

Run `better-cheaper-llm report grading.jsonl` after a day of traffic, then switch to `mode="enforce"` with the threshold it suggests. Your LLM only sees the chunks Jev wasn't sure about.

### Say "I don't know" before generating

An internal wiki assistant is at its worst when search comes back with nothing useful: the model writes a confident answer anyway. Check first, and skip the expensive call entirely when the passages can't answer the question:

```python
from better_cheaper_llm import Checker, sufficient

checker = Checker(mode="enforce")

pages = wiki_search(question)
if not sufficient(checker, question, pages, fallback=lambda: llm_says_pages_cover_it(question, pages)):
    return "I couldn't find that in the wiki. Try the #eng-help channel."
return generate_answer(question, pages)
```

### Fact-check a support bot before it replies

A help-center bot drafts replies from support articles. Before a draft goes out, check that every sentence is backed by the articles it retrieved, and hand the ticket to a person when it isn't or when Jev can't tell:

```python
from better_cheaper_llm import Checker, grounded

checker = Checker(mode="enforce", log_path="support_checks.jsonl")

result = grounded(checker, draft, articles, threshold=0.95)
if result.value is True:
    send_reply(ticket, draft)
else:  # False means unsupported, None means Jev wasn't confident either way
    assign_to_agent(ticket, note="Draft may not match the help center")
```

### Groundedness checks in CI

Keep a golden set of questions, retrieved passages and your app's answers in the repo, and score it on every pull request. At Jev's prices you can afford to score every row, every time:

```bash
better-cheaper-llm eval tests/golden_set.jsonl --out scores.jsonl
jq -s '[.[] | select(.scores.grounded < 0.9)] | length' scores.jsonl
```

That second line counts answers Jev didn't confidently call grounded (failed rows count too). Fail the build when it goes up. There's no built-in pass/fail gate yet, so that's your call to make. The same scores work for production monitoring: run `eval` on a sample of yesterday's answers, or all of them.

## Auditing your logs

`better-cheaper-llm audit` takes JSONL with one LLM call per line. It understands Anthropic Messages and OpenAI (Chat Completions and Responses) request/response pairs, plus a plain format for everything else:

```json
{"site": "grade_document", "model": "...", "system": "...", "input": "...", "output": "yes",
 "usage": {"input_tokens": 812, "output_tokens": 1}, "latency_ms": 640}
```

`site` (the place in your code that made the call) and `latency_ms` are optional but worth adding. Without `site`, calls are grouped by model and system prompt, which merges calls that happen to share a generic system prompt.

The report has three parts:

- every call site with its tokens, cost, share of the bill and median latency;
- the sites that are really decisions: short answers from a small repeated set, like yes/no or a route name;
- RAG prompts where retrieved passages were probably never used. That's a lexical guess: a passage counts as used if the answer cites it or shares wording with it.

Pass `--prices` (see [examples/prices.json](examples/prices.json)) to get dollars instead of just tokens. The savings it shows are upper bounds. `replay` tells you how much of that is real.

## Replaying decisions

```bash
uv run better-cheaper-llm replay logs.jsonl --prices prices.json --sample 100
```

For each decision site, replay sends a sample of the logged calls to Jev and compares answers. You get a table of confidence thresholds, how many calls Jev would take at each one, and how often it agreed with your LLM. It recommends the threshold that hands Jev the most calls while agreement stays above your target (98% by default). It uses the low end of a 95% confidence interval, so a lucky small sample can't talk you into anything.

Agreement is measured against your LLM, and your LLM isn't always right. Replay prints the calls where they disagreed so you can look for yourself. On the sample pipeline, all six disagreements in a batch of 80 turned out to be mistakes in the log, not in Jev.

## Using it in your pipeline

Start in shadow mode. Your existing check keeps making the call. Jev runs alongside it and only gets logged.

```python
from better_cheaper_llm import Checker, grounded, sufficient

checker = Checker(mode="shadow", log_path="decisions.jsonl")

ok = grounded(checker, answer, passages,
              fallback=lambda: my_llm_says_grounded(answer, passages))

if not sufficient(checker, question, passages,
                  fallback=lambda: my_llm_says_sufficient(question, passages)):
    passages = retrieve_again(question)
```

After a few hundred requests:

```bash
uv run better-cheaper-llm report decisions.jsonl
```

That gives you a threshold for each check, with Jev's median latency next to your current check's. Then switch:

```python
checker = Checker(mode="enforce", log_path="decisions.jsonl")
ok = grounded(checker, answer, passages, threshold=0.95, fallback=...)
```

In enforce mode Jev answers when it's confident either way (p >= threshold for yes, p <= 1 - threshold for no), and everything in between goes to your fallback. `grounded` asks one question per sentence of the answer, all in one request, and the answer only passes if every sentence does. For any other yes/no decision, use `checker.check(site, statement, state, fallback=...)`.

If Jev is down, times out, or the prompt is too big, the call goes to your fallback and the error goes in the log. The log keeps probabilities, answers and timings, never your prompts.

## Evaluating a dataset

The same checks work as eval metrics, and they're cheap enough to run on every answer you serve instead of a sample.

```bash
uv run better-cheaper-llm eval dataset.jsonl --out scores.jsonl
```

Each row needs `question`, `passages` (or `contexts`) and `answer`. You get two scores per row: `grounded` (is every claim supported by the passages?) and `sufficient` (do the passages contain what's needed to answer?), plus a count of yes, no and unsure verdicts and the lowest-scoring rows. Add human labels like `"grounded": false` to some rows and the report adds AUROC, a threshold table, and the rows where Jev most confidently disagrees with your labels.

Compare Jev against people, not against another LLM judge. That's the lesson of the replay result above.

On the sample eval set (120 rows: each answer as written, with one fact changed, and with its sources removed), Jev matched every label on both metrics in about 12 seconds. The sample answers copy their sources word for word, so treat that as a smoke test. In a separate run with one digit changed per answer, 27 of 30 were caught outright and 3 landed in the unsure band. None were passed.

## How fast is it?

[examples/speed_benchmark.py](examples/speed_benchmark.py) sends identical decisions to Jev and to OpenAI models, one call at a time from the same laptop:

| | Jev | GPT-5.4 nano | GPT-6 Astra | o4-mini |
|---|---:|---:|---:|---:|
| Grade a passage (median) | 185 ms | 572 ms | 1,161 ms | 1,646 ms |
| Check an answer for hallucination (median) | 185 ms | 594 ms | 1,142 ms | 1,922 ms |
| Hallucination checks right | 20/20 | 19/20 | 20/20 | 10/10 |

That was September 2026, with 20 items per task (10 for o4-mini). Jev's time barely moved between a 36-token and a 418-token prompt. These are easy synthetic decisions: the OpenAI models answered in about 4 tokens, and o4-mini reasoned for 83 to 147. One benchmark never tells the whole story, so measure your own with shadow mode or replay.

## Other servers

better-cheaper-llm talks to TypeSafe's `/v1/systemone` API. To use another server that speaks it, set `TYPESAFE_BASE_URL`. A key is only required for TypeSafe's own endpoint.

## Related projects

Jev picked up a lot of tooling quickly, and [awesome-jev](https://github.com/cobanov/awesome-jev) keeps a list. A few that overlap with better-cheaper-llm:

- [Janus](https://github.com/FirasSX914/Janus) picks Jev-versus-LLM routing thresholds from labelled data. Its finding that thresholds don't carry over between datasets is why better-cheaper-llm calibrates per call site.
- [jev-router](https://github.com/prismhq/jev-router) uses Jev to choose which model serves each request, on top of LiteLLM.
- [typesafe_agent_gates](https://github.com/ThiagaoBR/typesafe_agent_gates) gates shell commands and triages issues for coding agents built on LangChain.
- [llm-typesafe](https://simonwillison.net/2026/Sep/22/llm-typesafe/) brings Jev to Simon Willison's `llm` command-line tool.

## Roadmap

- Filtering retrieved passages with Jev before they reach the LLM, measured the same way
- Choice and score checks in the pipeline, not just in replay
- Reading OpenTelemetry, Langfuse and LangSmith exports
- Adapters for LangChain, LlamaIndex and Haystack, and a TypeScript port

## Development

```bash
uv sync
uv run pytest
```

## License

Apache-2.0. See [LICENSE](LICENSE).
