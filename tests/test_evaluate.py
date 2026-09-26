import json
import re

import pytest

from better_cheaper_llm.calibrate import auroc
from better_cheaper_llm.cli import main
from better_cheaper_llm.evaluate import load_dataset, render, score, write_scores


COMPANIES = ["Acme Corp", "Globex", "Initech", "Umbrella Group", "Stark Industries", "Wayne Enterprises",
             "Hooli", "Soylent Co", "Tyrell Systems", "Cyberdyne", "Vandelay Imports", "Massive Dynamic"]


def verbatim_jev(state, question):
    """Supported only if the claim, minus its citation, appears word for word in a passage."""
    instructions = question["instructions"]
    if isinstance(instructions, dict):  # grounded(): one claim per question
        claim = re.sub(r"^According to \[\d+\], |\[\d+\] adds that ", "", instructions["claim"])
        return 0.95 if any(claim in p for p in state["passages"]) else 0.03
    # sufficient(): the sample's passages suffice exactly when one is about the company asked about.
    company = next(co for co in COMPANIES if co in state["question"])
    return 0.95 if any(company in p for p in state["passages"]) else 0.05


@pytest.mark.parametrize("pairs, expected", [
    ([(0.9, True), (0.8, True), (0.2, False)], 1.0),
    ([(0.1, True), (0.9, False)], 0.0),
    ([(0.5, True), (0.5, False)], 0.5),
    ([(0.9, True), (0.2, True)], None),
])
def test_auroc(pairs, expected):
    assert auroc(pairs) == expected


def test_load_dataset(tmp_path):
    path = tmp_path / "data.jsonl"
    path.write_text("\n".join([
        json.dumps({"id": "a", "question": "Q?", "passages": ["P1", "P2"], "answer": "A.", "grounded": True}),
        json.dumps({"question": "Q?", "contexts": "one passage", "sufficient": "yes"}),  # alias; non-bool label ignored
        json.dumps({"question": "Q?", "passages": []}),
        "not json",
    ]))
    rows, skipped = load_dataset(path)
    assert skipped == 2
    assert (rows[0].id, rows[0].passages, rows[0].labels) == ("a", ["P1", "P2"], {"grounded": True})
    assert (rows[1].id, rows[1].passages, rows[1].labels, rows[1].answer) == ("line2", ["one passage"], {}, None)
    assert rows[1].supports("sufficient") and not rows[1].supports("grounded")


def test_scores_the_labelled_sample(samples, fake_jev, tmp_path):
    rows, _ = load_dataset(samples["eval"])
    scores = score(rows, fake_jev(p=verbatim_jev), workers=4)
    assert len(scores) == 2 * len(rows) and all(s.p is not None for s in scores)

    text = render(scores, source="rag_eval.jsonl", skipped=0, threshold=0.9, target=0.95)
    assert text.count("AUROC 1.000") == 2
    assert "## grounded" in text and "## sufficient" in text and "Recommended: `threshold=" in text

    out = tmp_path / "scores.jsonl"
    write_scores(scores, out)
    first = json.loads(out.read_text().splitlines()[0])
    assert first["id"] == "000-original" and set(first["scores"]) == {"grounded", "sufficient"}


def test_failures_are_reported_not_raised(samples, fake_jev):
    rows, _ = load_dataset(samples["eval"])
    scores = score(rows[:3], fake_jev(error=RuntimeError("boom")), ["grounded"])
    assert all(s.p is None and "boom" in s.error for s in scores)
    assert "3 rows failed (first error: RuntimeError: boom)" in render(scores, source="x", skipped=0, threshold=0.9, target=0.95)


def test_disagreements_with_labels_are_listed(samples, fake_jev):
    rows, _ = load_dataset(samples["eval"])
    text = render(score(rows, fake_jev(p=0.97), ["grounded"]), source="x", skipped=0, threshold=0.9, target=0.95)
    assert "Most confident disagreements with your labels" in text and "labelled false, p=0.97" in text


def test_cli_eval(samples, fake_jev, monkeypatch, tmp_path, capsys):
    monkeypatch.setattr("better_cheaper_llm.cli.JevClient", lambda: fake_jev(p=verbatim_jev))
    out = tmp_path / "scores.jsonl"
    assert main(["eval", str(samples["eval"]), "--metrics", "grounded", "--out", str(out)]) == 0
    assert "# better-cheaper-llm eval" in capsys.readouterr().out
    assert len(out.read_text().splitlines()) == 120
    with pytest.raises(SystemExit):
        main(["eval", str(samples["eval"]), "--metrics", "vibes"])
