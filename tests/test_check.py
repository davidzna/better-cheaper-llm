import json
import time

import pytest

from better_cheaper_llm.check import Checker
from better_cheaper_llm.jev import MAX_STATE_CHARS, JevError


def test_enforce_uses_confident_jev_answers(fake_jev):
    checker = Checker(fake_jev(p=0.97), mode="enforce")
    result = checker.check("s", "It holds.", {"x": 1}, fallback=lambda: pytest.fail("fallback ran"))
    assert (result.value, result.source, result.p) == (True, "jev", 0.97)

    checker = Checker(fake_jev(p=0.02), mode="enforce")
    assert checker.check("s", "It holds.", "state").value is False


def test_enforce_falls_back_when_unsure_or_failing(fake_jev):
    unsure = Checker(fake_jev(p=0.6), mode="enforce")
    assert unsure.check("s", "It holds.", "state", fallback=lambda: True).source == "fallback"
    no_fallback = unsure.check("s", "It holds.", "state")
    assert (no_fallback.value, no_fallback.source, bool(no_fallback)) == (None, "unsure", False)

    failing = Checker(fake_jev(error=JevError("down")), mode="enforce")
    result = failing.check("s", "It holds.", "state", fallback=lambda: False)
    assert (result.value, result.source, result.p) == (False, "fallback", None)
    assert "down" in result.error


def test_shadow_returns_the_current_check_and_logs_jev(fake_jev, tmp_path):
    log = tmp_path / "decisions.jsonl"
    jev = fake_jev(p=0.99)
    checker = Checker(jev, log_path=log)
    result = checker.check("rag.grounded", "Secret statement.", {"doc": "classified-xyz"}, fallback=lambda: False)
    assert (result.value, result.source, result.p) == (False, "fallback", 0.99)
    assert len(jev.requests) == 1

    record = json.loads(log.read_text())
    assert record["site"] == "rag.grounded" and record["mode"] == "shadow"
    assert (record["p"], record["fallback_value"], record["jev_input_tokens"]) == (0.99, False, 42)
    assert "Secret" not in log.read_text() and "classified-xyz" not in log.read_text()


def test_shadow_runs_jev_alongside_the_fallback(fake_jev):
    class SlowJev(fake_jev):
        def evaluate(self, state, questions):
            time.sleep(0.2)
            return super().evaluate(state, questions)

    def slow_fallback():
        time.sleep(0.2)
        return True

    start = time.perf_counter()
    Checker(SlowJev(p=0.9)).check("s", "It holds.", "state", fallback=slow_fallback)
    assert time.perf_counter() - start < 0.35


def test_shadow_requires_a_fallback(fake_jev):
    with pytest.raises(ValueError, match="fallback"):
        Checker(fake_jev()).check("s", "It holds.", "state")


def test_a_list_holds_only_if_every_statement_holds(fake_jev):
    jev = fake_jev(p=[0.99, 0.05, 0.97])
    result = Checker(jev, mode="enforce").check("s", ["a", "b", "c"], "state")
    assert (result.value, result.p) == (False, 0.05)
    assert list(jev.requests[0][1]) == ["s0", "s1", "s2"]


def test_criteria_are_sent_with_each_statement(fake_jev):
    jev = fake_jev(p=0.99)
    criteria = {"true": "explicitly stated", "false": "absent or contradicted"}
    Checker(jev, mode="enforce").check("s", ["a", "b"], "state", criteria=criteria)
    assert all(q["criteria"] == criteria for q in jev.requests[0][1].values())


def test_oversized_state_never_reaches_jev(fake_jev):
    jev = fake_jev(p=0.99)
    result = Checker(jev, mode="enforce").check("s", "It holds.", "x" * (MAX_STATE_CHARS + 1), fallback=lambda: True)
    assert result.source == "fallback" and "too large" in result.error and not jev.requests


def test_threshold_is_validated(fake_jev):
    with pytest.raises(ValueError):
        Checker(fake_jev(), mode="enforce").check("s", "It holds.", "state", threshold=0.4)
