import importlib.util
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


class FakeJev:
    """Stands in for JevClient. `p` is one probability for every question, a list per question,
    or a function of (state, question)."""

    def __init__(self, p=0.5, error=None, choice=None):
        self.p, self.error, self.choice = p, error, choice
        self.requests = []

    def evaluate(self, state, questions):
        self.requests.append((state, questions))
        if self.error:
            raise self.error
        answers = {}
        for i, (key, q) in enumerate(questions.items()):
            if q["type"] == "choice":
                answers[key] = self.choice(state, q)
            else:
                p = self.p(state, q) if callable(self.p) else self.p[i] if isinstance(self.p, list) else self.p
                answers[key] = {"type": "noul", "noul": p}
        return {"model": "jev-test", "answers": answers, "usage": {"input_tokens": 42, "output_tokens": 0}}


@pytest.fixture
def fake_jev():
    return FakeJev


@pytest.fixture(scope="session")
def samples(tmp_path_factory):
    """Generated sample files: {"traces": path, "eval": path, "truth": ground truth}."""
    spec = importlib.util.spec_from_file_location("make_samples", ROOT / "examples" / "make_samples.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    records, truth = module.generate()
    folder = tmp_path_factory.mktemp("samples")
    (folder / "rag_traces.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
    (folder / "rag_eval.jsonl").write_text("".join(json.dumps(r) + "\n" for r in module.eval_rows(truth["qa"])))
    return {"traces": folder / "rag_traces.jsonl", "eval": folder / "rag_eval.jsonl", "truth": truth}


@pytest.fixture(scope="session")
def sample_traces(samples):
    """(path to a generated sample log, its ground truth)."""
    return samples["traces"], samples["truth"]


@pytest.fixture
def prices_path():
    return ROOT / "examples" / "prices.json"
