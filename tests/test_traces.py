import json

from better_cheaper_llm.traces import load_calls, parse_record


def test_anthropic_with_cache_tool_use_and_documents():
    call = parse_record({
        "site": "answer",
        "request": {
            "model": "claude-sonnet-5",
            "system": [{"type": "text", "text": "Be brief."}],
            "messages": [{"role": "user", "content": [
                {"type": "document", "source": {"type": "text", "media_type": "text/plain", "data": "Doc one."}},
                {"type": "text", "text": "Question?"},
            ]}],
        },
        "response": {
            "id": "msg_1", "model": "claude-sonnet-5",
            "content": [{"type": "thinking", "thinking": ""}, {"type": "text", "text": "Checking."},
                        {"type": "tool_use", "id": "t1", "name": "search", "input": {"q": "x"}}],
            "usage": {"input_tokens": 10, "cache_read_input_tokens": 90, "cache_creation_input_tokens": 5,
                      "output_tokens": 7},
        },
    })
    assert call.site == "answer" and call.site_tagged
    assert call.system == "Be brief."
    assert "<document>Doc one.</document>" in call.prompt
    assert call.output == 'Checking.\ntool:search({"q": "x"})'
    assert (call.input_tokens, call.cache_read_tokens, call.cache_write_tokens, call.output_tokens) == (10, 90, 5, 7)
    assert call.all_input_tokens == 105
    assert not call.estimated


def test_openai_chat_counts_cached_tokens_once():
    call = parse_record({
        "request": {"model": "gpt-x", "messages": [
            {"role": "system", "content": "Route it."}, {"role": "user", "content": "hello"}]},
        "response": {
            "model": "gpt-x",
            "choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "lookup", "arguments": "{\"id\": 3}"}}]}}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 4, "prompt_tokens_details": {"cached_tokens": 80}},
        },
    })
    assert call.system == "Route it." and call.prompt == "hello"
    assert call.output == 'tool:lookup({"id": 3})'
    assert (call.input_tokens, call.cache_read_tokens, call.output_tokens) == (20, 80, 4)


def test_openai_responses():
    call = parse_record({
        "request": {"model": "gpt-y", "instructions": "Grade it.", "input": "Is this relevant?"},
        "response": {
            "model": "gpt-y",
            "output": [{"type": "reasoning", "summary": []},
                       {"type": "message", "content": [{"type": "output_text", "text": "yes"}]}],
            "usage": {"input_tokens": 50, "output_tokens": 300, "input_tokens_details": {"cached_tokens": 0}},
        },
    })
    assert (call.system, call.prompt, call.output) == ("Grade it.", "Is this relevant?", "yes")
    assert (call.input_tokens, call.output_tokens) == (50, 300)


def test_generic_without_usage_is_estimated():
    call = parse_record({"model": "m", "input": "x" * 40, "output": "yes"})
    assert call.estimated
    assert call.input_tokens == 10 and call.output_tokens == 1


def test_untagged_calls_group_by_system_prompt():
    a = parse_record({"model": "m", "system": "Grade 1 doc.", "input": "first question", "output": "yes"})
    b = parse_record({"model": "m", "system": "Grade 2 doc.", "input": "another one", "output": "no"})
    c = parse_record({"model": "m", "system": "Something else.", "input": "first question", "output": "no"})
    assert a.site == b.site != c.site  # digits don't split a template
    assert a.site.startswith("grade-doc#") and not a.site_tagged


def test_load_calls_skips_bad_lines(tmp_path):
    path = tmp_path / "log.jsonl"
    good = {"model": "m", "input": "hi", "output": "yes", "usage": {"input_tokens": 1, "output_tokens": 1}}
    path.write_text(json.dumps(good) + "\nnot json\n\n" + json.dumps({"unrelated": True}) + "\n")
    calls, skipped = load_calls(path)
    assert len(calls) == 1 and skipped == 2


def test_latency_from_the_line_or_its_metadata():
    base = {"model": "m", "input": "hi", "output": "yes"}
    assert parse_record({**base, "latency_ms": 812}).latency_ms == 812.0
    assert parse_record({**base, "metadata": {"duration_ms": 95.5}}).latency_ms == 95.5
    assert parse_record({**base, "latency_ms": "fast"}).latency_ms is None
    assert parse_record(base).latency_ms is None
