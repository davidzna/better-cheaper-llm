from better_cheaper_llm.check import Checker
from better_cheaper_llm.rag import grounded, split_claims, sufficient


def test_split_claims_keeps_every_sentence():
    text = "Acme grew revenue 12% last year. It did. Globex hired 200 people! Is that all? Yes."
    assert split_claims(text) == ["Acme grew revenue 12% last year. It did.", "Globex hired 200 people!",
                                  "Is that all? Yes."]
    many = " ".join(f"Sentence number {i} is here." for i in range(10))
    grouped = split_claims(many, max_claims=4)
    assert len(grouped) <= 4 and " ".join(grouped) == many


def test_grounded_asks_one_question_per_claim_in_one_request(fake_jev):
    jev = fake_jev(p=[0.98, 0.3])
    result = grounded(Checker(jev, mode="enforce"), "Acme grew revenue last year. Globex hired staff too.",
                      ["Acme grew revenue."], fallback=lambda: False)
    state, questions = jev.requests[0]
    assert state == {"passages": ["Acme grew revenue."]}
    assert [q["instructions"]["claim"] for q in questions.values()] == [
        "Acme grew revenue last year.", "Globex hired staff too."]
    assert (result.p, result.source, result.value) == (0.3, "fallback", False)  # weakest claim decides


def test_sufficient_sends_question_and_passages(fake_jev):
    jev = fake_jev(p=0.04)
    result = sufficient(Checker(jev, mode="enforce"), "Who is the CFO?", iter(["Revenue rose."]))
    assert result.value is False and result.source == "jev"
    assert jev.requests[0][0] == {"question": "Who is the CFO?", "passages": ["Revenue rose."]}
