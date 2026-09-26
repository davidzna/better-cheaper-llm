import pytest

from better_cheaper_llm.audit import Price, audit, call_cost, label, load_prices, price_for, render
from better_cheaper_llm.traces import Call, load_calls


def make_call(output, site="s", model="m", tokens=(100, 10)):
    return Call(id="x", site=site, model=model, system="", prompt="p", output=output,
                input_tokens=tokens[0], output_tokens=tokens[1])


@pytest.mark.parametrize("raw, expected", [
    ("Yes.", "yes"), (" yes\n", "yes"), ('{"relevant": "Yes"}', "yes"), ("**No**", "no"),
    ("true", "true"), ("7", "7"), ('"web_search"', "web_search"),
])
def test_label_normalisation(raw, expected):
    assert label(raw) == expected


def test_classifies_decisions_generations_and_small_sites():
    calls = [make_call(o, "grade") for o in ["yes", "no", "Yes.", "no", "yes", "no"]]
    calls += [make_call(f"A long, different answer number {i} about something.", "gen") for i in range(6)]
    calls += [make_call("yes", "tiny")]
    kinds = {s.name: s.kind for s in audit(calls)}
    assert kinds == {"grade": "decision", "gen": "generation", "tiny": "few calls"}


def test_an_answer_that_is_always_the_same_is_not_a_decision():
    assert audit([make_call("yes") for _ in range(10)])[0].kind == "generation"


def test_prices_match_dated_model_ids_and_price_cache():
    prices = {"claude-sonnet-5": Price(2.0, 10.0, 0.2, 2.5), "claude": Price(9, 9, 9, 9)}
    assert price_for("claude-sonnet-5-20260101", prices) is prices["claude-sonnet-5"]
    assert price_for("gpt-x", prices) is None
    call = Call(id="x", site="s", model="m", system="", prompt="", output="", input_tokens=1_000_000,
                output_tokens=100_000, cache_read_tokens=1_000_000, cache_write_tokens=1_000_000)
    assert call_cost(call, prices["claude-sonnet-5"]) == pytest.approx(2.0 + 0.2 + 2.5 + 1.0)


def test_load_prices_defaults_cache_multipliers(tmp_path):
    path = tmp_path / "prices.json"
    path.write_text('{"_note": "comment", "m": {"input": 2.0, "output": 8.0}}')
    assert load_prices(path) == {"m": Price(2.0, 8.0, pytest.approx(0.2), pytest.approx(2.5))}


def test_sample_pipeline_findings_match_ground_truth(sample_traces, prices_path):
    path, truth = sample_traces
    calls, skipped = load_calls(path)
    sites = {s.name: s for s in audit(calls, load_prices(prices_path))}
    assert skipped == 0

    kinds = {name.split("#")[0]: s.kind for name, s in sites.items()}
    assert kinds == {
        "check_hallucination": "decision",
        "generate_answer": "generation",
        "you-are-a-grader-assessing": "decision",
        "route-the-user-s-question": "decision",
    }
    answer = sites["generate_answer"]
    assert answer.rag_calls == truth["answer_calls"]
    assert answer.unused_passage_tokens == pytest.approx(truth["unused_passage_chars"] / 4, rel=0.01)
    # The hidden reasoning in check_hallucination makes it the most expensive site.
    assert next(iter(sites)) == "check_hallucination"


def test_render_without_prices(sample_traces):
    calls, _ = load_calls(sample_traces[0])
    text = render(audit(calls), source="log", skipped=0, priced=False)
    assert "pass `--prices" in text and "Upper-bound saving:" not in text
    assert "## Decisions an LLM is making today" in text


def test_latency_column_only_when_logs_have_it():
    timed = [make_call(o) for o in ["yes", "no"] * 3]
    for i, c in enumerate(timed):
        c.latency_ms = 700.0 + i * 100
    text = render(audit(timed), source="log", skipped=0, priced=False)
    assert "| Median latency |" in text and "| 950 ms |" in text
    assert "Median latency" not in render(audit([make_call(o) for o in ["yes", "no"] * 3]), source="log", skipped=0, priced=False)
