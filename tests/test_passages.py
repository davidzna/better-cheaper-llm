from better_cheaper_llm.passages import Passage, extract_passages, used_labels


def test_tagged_passages_use_index_attribute():
    prompt = (
        '<documents>\n<document index="3">Alpha text.</document>\n'
        '<document index="7">Beta text.</document>\n</documents>\nQuestion: ?'
    )
    assert extract_passages(prompt) == [Passage("3", "Alpha text."), Passage("7", "Beta text.")]


def test_numbered_passages_stop_before_the_question():
    prompt = "Context:\n[1] Acme grew revenue.\n[2] Globex hired staff.\n\nQuestion: what did Acme do?"
    assert extract_passages(prompt) == [Passage("1", "Acme grew revenue."), Passage("2", "Globex hired staff.")]


def test_a_single_passage_is_not_rag_context():
    assert extract_passages("<document>only one</document>") == []


def test_used_by_citation_or_shared_wording():
    passages = [
        Passage("1", "Acme Corp reported quarterly revenue of $4.2 billion in its cloud segment."),
        Passage("2", "Globex appointed a new chief financial officer."),
        Passage("3", "Initech cut its margin outlook."),
    ]
    answer = "Acme Corp reported quarterly revenue of $4.2 billion. See also [3]."
    assert used_labels(passages, answer) == {"1", "3"}


def test_common_words_alone_do_not_count_as_use():
    passages = [Passage("1", "The company said that it was in the market."), Passage("2", "Other text here.")]
    assert used_labels(passages, "The company said that it was fine.") == set()
