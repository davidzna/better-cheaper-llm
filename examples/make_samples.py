"""Write two synthetic samples of a CRAG-style RAG pipeline:

* examples/rag_traces.jsonl, a log of its LLM calls, for `better-cheaper-llm audit` and `better-cheaper-llm replay`
* examples/rag_eval.jsonl, labelled question/passages/answer rows, for `better-cheaper-llm eval`

Per question the pipeline makes these calls:

  route                small-model       Anthropic format, untagged  -> vectorstore | web_search | direct
  grade (x5)           small-model       Anthropic format, untagged  -> yes | no, one per retrieved passage
  generate_answer      large-model       Anthropic format, tagged    -> cited answer using 1-2 of 5 passages
  check_hallucination  large-model       generic format, tagged      -> yes | no after hidden reasoning

Only questions routed to the vectorstore go on to retrieval. All companies and
facts are fictional. Token counts are characters / 4, so the audit's passage
estimates can be checked against the ground truth that generate() returns.

    uv run python examples/make_samples.py
"""

from __future__ import annotations

import json
import random
import re
from pathlib import Path

COMPANIES = [
    "Acme Corp", "Globex", "Initech", "Umbrella Group", "Stark Industries", "Wayne Enterprises",
    "Hooli", "Soylent Co", "Tyrell Systems", "Cyberdyne", "Vandelay Imports", "Massive Dynamic",
]
PEOPLE = ["Dana Whitfield", "Omar Castell", "Priya Raman", "Lena Brandt", "Tomas Okafor", "Mei Lin"]
MONTHS = ["March", "June", "September", "November"]

# (fact template, question template). Templates are worded differently on purpose.
FACTS = [
    ("{co} reported quarterly revenue of ${a}.{b} billion, driven by strong demand in its cloud segment.",
     "What revenue did {co} report last quarter?"),
    ("Headcount at {co} reached {a}{b},{c}00 employees after the company opened two new engineering offices.",
     "How many employees does {co} have?"),
    ("{co} settled a patent dispute for ${a}{b} million without admitting wrongdoing.",
     "Did {co} settle its patent dispute?"),
    ("Analysts expect {co} to launch its next flagship device in {month}, according to supply chain checks.",
     "When will {co} launch its next flagship device?"),
    ("{co} raised its dividend to ${a}.{b}{c} per share and announced a share buyback program.",
     "What dividend does {co} pay?"),
    ("The board of {co} appointed {person} as chief financial officer effective immediately.",
     "Who is the CFO of {co}?"),
    ("{co} cut its full-year margin outlook to {a}{b} percent citing higher component costs.",
     "What margin does {co} expect this year?"),
]
FILLER = [
    "The disclosure appeared in a filing published on the investor relations site.",
    "Shares moved modestly in after-hours trading following the announcement.",
    "A spokesperson declined to comment beyond the published statement.",
    "The figures had not been independently audited at the time of publication.",
    "Regional press coverage described the update as broadly in line with guidance.",
    "Management reiterated its long-term strategy during the accompanying call.",
]

ROUTE_SYSTEM = "Route the user's question to one of: vectorstore, web_search, direct. Reply with the route name only."
GRADE_SYSTEM = (
    "You are a grader assessing whether a retrieved document is relevant to a user question. "
    "Reply yes or no."
)
RULES = [
    "Answer only from the documents provided.",
    "Cite every fact with the index of its document, like [2].",
    "If the documents do not contain the answer, say so plainly.",
    "Prefer the most recent figure when documents disagree.",
    "Quote numbers exactly as they appear, including units.",
    "Keep answers under three sentences unless asked for detail.",
    "Do not speculate about future performance.",
    "Name the company in the first sentence.",
]
ANSWER_SYSTEM = "You are a financial research assistant.\n" + "\n".join(
    f"{i}. {RULES[i % len(RULES)]}" for i in range(1, 49)
)
CHECK_SYSTEM = (
    "Decide whether the answer is fully supported by the documents. "
    "Think it through, then reply with only yes or no."
)


def generate(questions: int = 300, seed: int = 7) -> tuple[list[dict], dict]:
    rng = random.Random(seed)
    records: list[dict] = []
    truth = {"passage_chars": 0, "unused_passage_chars": 0, "answer_calls": 0, "qa": []}
    counter = iter(range(1, 10**9))

    def anthropic(model: str, system: str, user: str, output: str, trace: str, site: str | None = None,
                  cached: bool = False, output_tokens: int | None = None) -> None:
        n = next(counter)
        system_tokens, user_tokens = len(system) // 4, len(user) // 4
        record = {
            "id": f"msg_{n:06d}",
            "trace_id": trace,
            "request": {"model": model, "max_tokens": 1024, "system": system,
                        "messages": [{"role": "user", "content": user}]},
            "response": {
                "id": f"msg_{n:06d}", "type": "message", "role": "assistant", "model": model,
                "content": [{"type": "text", "text": output}], "stop_reason": "end_turn",
                "usage": {
                    "input_tokens": user_tokens if cached else system_tokens + user_tokens,
                    "cache_read_input_tokens": system_tokens if cached else 0,
                    "cache_creation_input_tokens": 0,
                    "output_tokens": output_tokens or max(len(output) // 4, 1),
                },
            },
        }
        if site:
            record["site"] = site
        records.append(record)

    for q in range(questions):
        trace = f"q{q:04d}"
        company = rng.choice(COMPANIES)
        target = rng.randrange(len(FACTS))
        question = FACTS[target][1].format(co=company)

        route = rng.choices(["vectorstore", "web_search", "direct"], weights=[70, 20, 10])[0]
        anthropic("small-model", ROUTE_SYSTEM, question, route, trace)
        if route != "vectorstore":
            continue

        # Five passages with distinct templates; the target fact is always among them,
        # and 30% of the time a second one is relevant too.
        templates = [target] + rng.sample([i for i in range(len(FACTS)) if i != target], 4)
        rng.shuffle(templates)
        relevant = {templates.index(target)}
        if rng.random() < 0.3:
            relevant.add(rng.choice([i for i in range(5) if i not in relevant]))
        passages, key_sentences = [], []
        for i, t in enumerate(templates):
            co = company if i in relevant else rng.choice([c for c in COMPANIES if c != company])
            sentence = FACTS[t][0].format(
                co=co, a=rng.randint(1, 9), b=rng.randint(0, 9), c=rng.randint(0, 9),
                month=rng.choice(MONTHS), person=rng.choice(PEOPLE),
            )
            key_sentences.append(sentence)
            passages.append(" ".join([sentence, *rng.sample(FILLER, 2)]))

        for i, passage in enumerate(passages):
            verdict = "yes" if i in relevant else "no"
            if rng.random() < 0.05:
                verdict = "no" if verdict == "yes" else "yes"
            anthropic("small-model", GRADE_SYSTEM, f"Document:\n{passage}\n\nQuestion: {question}", verdict, trace)

        documents = "<documents>\n" + "\n".join(
            f'<document index="{i + 1}">\n{p}\n</document>' for i, p in enumerate(passages)
        ) + "\n</documents>"
        cited = sorted(relevant)
        answer = f"According to [{cited[0] + 1}], {key_sentences[cited[0]]}"
        if len(cited) > 1:
            answer += f" [{cited[1] + 1}] adds that {key_sentences[cited[1]]}"
        anthropic("large-model", ANSWER_SYSTEM, f"{documents}\n\nQuestion: {question}", answer, trace,
                  site="generate_answer", cached=truth["answer_calls"] > 0)
        truth["answer_calls"] += 1
        truth["qa"].append({"question": question, "passages": passages, "answer": answer, "relevant": sorted(relevant)})
        truth["passage_chars"] += sum(len(p) for p in passages)
        truth["unused_passage_chars"] += sum(len(p) for i, p in enumerate(passages) if i not in relevant)

        user = f"{documents}\n\nAnswer: {answer}"
        verdict = "yes" if rng.random() < 0.94 else "no"
        records.append({
            "site": "check_hallucination",
            "trace_id": trace,
            "model": "large-model",
            "system": CHECK_SYSTEM,
            "input": [{"role": "user", "content": user}],
            "output": verdict,
            # Hidden reasoning is billed as output even though the visible answer is one word.
            "usage": {"input_tokens": (len(CHECK_SYSTEM) + len(user)) // 4, "output_tokens": rng.randint(300, 900)},
        })

    return records, truth


_CITATION = re.compile(r"\[\d+\]")


def eval_rows(qa: list[dict], n: int = 40, seed: int = 11) -> list[dict]:
    """Three labelled rows per answer: as written, with one fact altered, and without its sources."""
    rng = random.Random(seed)
    rows = []
    for i, item in enumerate(qa[:n]):
        question, passages, answer = item["question"], item["passages"], item["answer"]
        without_sources = [p for j, p in enumerate(passages) if j not in item["relevant"]]
        rows += [
            {"id": f"{i:03d}-original", "question": question, "passages": passages, "answer": answer,
             "grounded": True, "sufficient": True},
            {"id": f"{i:03d}-altered", "question": question, "passages": passages,
             "answer": _alter(answer, passages, rng), "grounded": False, "sufficient": True},
            {"id": f"{i:03d}-no-sources", "question": question, "passages": without_sources, "answer": answer,
             "grounded": False, "sufficient": False},
        ]
    return rows


def _alter(answer: str, passages: list[str], rng: random.Random) -> str:
    """Change one digit outside the citations, or swap the company when there is no figure."""
    citations = [m.span() for m in _CITATION.finditer(answer)]
    digits = [m.start() for m in re.finditer(r"\d", answer) if not any(a <= m.start() < b for a, b in citations)]
    if digits:
        pos = digits[0]
        return answer[:pos] + str((int(answer[pos]) + rng.randint(1, 9)) % 10) + answer[pos + 1 :]
    company = next(co for co in COMPANIES if co in answer)
    other = rng.choice([co for co in COMPANIES if co != company and all(co not in p for p in passages)])
    return answer.replace(company, other)


if __name__ == "__main__":
    records, truth = generate()
    here = Path(__file__).parent
    (here / "rag_traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    rows = eval_rows(truth["qa"])
    (here / "rag_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(f"wrote {len(records):,} calls to {here / 'rag_traces.jsonl'}")
    print(f"wrote {len(rows):,} labelled rows to {here / 'rag_eval.jsonl'}")
    print(
        f"ground truth: {truth['unused_passage_chars'] / truth['passage_chars']:.1%} of passage text "
        f"is unused (~{truth['unused_passage_chars'] // 4:,} tokens)"
    )
