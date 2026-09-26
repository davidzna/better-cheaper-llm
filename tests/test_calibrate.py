import pytest

from better_cheaper_llm.calibrate import load_check_log, recommend, render_check_report, sweep, wilson_lower
from better_cheaper_llm.check import Checker


def test_wilson_lower_bound_penalises_small_samples():
    assert wilson_lower(20, 20) == pytest.approx(0.88, abs=0.01)
    assert wilson_lower(1000, 1000) > 0.99


def test_recommend_offloads_the_most_calls_that_still_meet_target():
    # 400 confident calls that always agree, 100 unsure ones that agree half the time.
    results = [(0.95, True)] * 400 + [(0.2, i % 2 == 0) for i in range(100)]
    best = recommend(sweep(results), target=0.98)
    assert best.threshold == 0.3 and best.coverage == pytest.approx(0.8)
    assert recommend(sweep(results[400:]), target=0.98) is None


def test_report_from_a_shadow_log(fake_jev, tmp_path):
    log = tmp_path / "decisions.jsonl"
    confident, unsure = Checker(fake_jev(p=0.99), log_path=log), Checker(fake_jev(p=0.62), log_path=log)
    for _ in range(300):
        confident.check("rag.grounded", "It holds.", "state", fallback=lambda: True)
    for i in range(50):
        unsure.check("rag.grounded", "It holds.", "state", fallback=lambda i=i: i % 2 == 0)

    sites = load_check_log(log)
    assert sites["rag.grounded"]["calls"] == 350 and len(sites["rag.grounded"]["results"]) == 350
    text = render_check_report(sites, source="decisions.jsonl", target=0.98)
    assert "Recommended: `threshold=0.65`" in text  # the lowest threshold that excludes the p=0.62 checks
    assert "Jev answers 86% of checks" in text
