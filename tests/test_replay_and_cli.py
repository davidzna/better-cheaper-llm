from better_cheaper_llm.audit import audit, label, load_prices
from better_cheaper_llm.cli import main
from better_cheaper_llm.jev import JevError
from better_cheaper_llm.replay import render, replay_site
from better_cheaper_llm.traces import load_calls


def grader_site(sample_traces, prices_path):
    calls, _ = load_calls(sample_traces[0])
    return next(s for s in audit(calls, load_prices(prices_path)) if s.name.startswith("you-are-a-grader"))


def test_replay_measures_agreement(sample_traces, prices_path, fake_jev):
    site = grader_site(sample_traces, prices_path)
    expected = {c.prompt: label(c.output) for c in site.calls}

    def answer(state, question):  # agrees with the LLM, except on requests mentioning Acme
        key = next(k for k, v in question["criteria"].items() if v.endswith(f'"{expected[state["request"]]}"'))
        confidence = 0.4 if "Acme" in state["request"] else 0.95
        if "Acme" in state["request"]:
            key = next(k for k in question["criteria"] if k != key)
        return {"type": "choice", "choice": key, "confidence": confidence, "probabilities": {}}

    run = replay_site(site, fake_jev(choice=answer), sample=200)
    assert len(run.results) == 200 and not run.errors
    text = render([run], target=0.95)
    assert "Recommended: act on Jev when confidence >= 0.5" in text
    assert "Projected saving on this log" in text
    assert 0 < len(run.disagreements) <= 5
    assert "Disagreements to review" in text


def test_replay_compares_jev_speed_with_the_logged_llm(fake_jev):
    from better_cheaper_llm.traces import Call

    calls = [Call(id=str(i), site="grade", model="m", system="Grade it.", prompt=f"doc {i}", output=o,
                  input_tokens=100, output_tokens=1, latency_ms=900.0) for i, o in enumerate(["yes", "no"] * 5)]
    site = audit(calls)[0]
    agree = lambda state, q: {"choice": next(k for k, v in q["criteria"].items() if v.endswith('"yes"')), "confidence": 0.9}
    run = replay_site(site, fake_jev(choice=agree), sample=10)
    assert len(run.jev_ms) == 10
    assert "Your LLM's median in the log is 900 ms, so about" in render([run], target=0.95)


def test_replay_gives_up_when_every_call_fails(sample_traces, prices_path, fake_jev):
    run = replay_site(grader_site(sample_traces, prices_path), fake_jev(error=JevError("401")), sample=50)
    assert len(run.errors) == 5 and not run.results


def test_cli_audit(sample_traces, prices_path, capsys):
    assert main(["audit", str(sample_traces[0]), "--prices", str(prices_path)]) == 0
    out = capsys.readouterr().out
    assert "# better-cheaper-llm audit" in out and "Upper-bound saving:" in out


def test_cli_replay_needs_a_key(sample_traces, monkeypatch, capsys):
    monkeypatch.delenv("TYPESAFE_API_KEY", raising=False)
    monkeypatch.delenv("TYPESAFE_BASE_URL", raising=False)
    assert main(["replay", str(sample_traces[0])]) == 1
    assert "TYPESAFE_API_KEY" in capsys.readouterr().err
