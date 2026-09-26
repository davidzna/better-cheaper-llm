"""Load logged LLM calls from JSONL, one call per line.

Recognised line shapes:

* ``{"request": <Anthropic Messages request>, "response": <Anthropic message>}``
* ``{"request": <OpenAI Chat Completions request>, "response": <its response>}``
* ``{"request": <OpenAI Responses request>, "response": <its response>}``
* ``{"model": ..., "system": ..., "input": str | [messages], "output": str, "usage": {...}}``

Any line may also carry ``"site"`` (a name for the place in your code that made the
call; strongly recommended), ``"latency_ms"`` or ``"duration_ms"``, ``"trace_id"``
and ``"id"``, at the top level or under ``"metadata"``. Lines that match none of
the shapes are skipped and counted.
"""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class Call:
    id: str
    site: str
    model: str
    system: str
    prompt: str  # every non-system input message, flattened to text
    output: str  # response text; tool calls rendered as tool:name(args)
    input_tokens: int  # uncached input only
    output_tokens: int  # includes hidden reasoning tokens
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    estimated: bool = False  # no usage block; token counts guessed from characters
    site_tagged: bool = False  # site came from the log rather than a fingerprint
    latency_ms: float | None = None
    trace_id: str | None = None

    @property
    def all_input_tokens(self) -> int:
        return self.input_tokens + self.cache_read_tokens + self.cache_write_tokens


def load_calls(path: str | Path) -> tuple[list[Call], int]:
    """Return (calls, number of skipped lines)."""
    calls: list[Call] = []
    skipped = 0
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            try:
                call = parse_record(json.loads(line), default_id=f"line{n}")
            except (ValueError, TypeError, KeyError, AttributeError, IndexError):
                call = None
            if call is None:
                skipped += 1
            else:
                calls.append(call)
    return calls, skipped


def parse_record(rec: dict, default_id: str = "") -> Call | None:
    req, resp = rec.get("request"), rec.get("response")
    if isinstance(req, dict) and isinstance(resp, dict):
        if "choices" in resp:
            model, system, prompt, output, tokens = _openai_chat(req, resp)
        elif isinstance(resp.get("output"), list):
            model, system, prompt, output, tokens = _openai_responses(req, resp)
        elif isinstance(resp.get("content"), list):
            model, system, prompt, output, tokens = _anthropic(req, resp)
        else:
            return None
        call_id = rec.get("id") or resp.get("id") or default_id
    elif "output" in rec and ("input" in rec or "prompt" in rec):
        model, system, prompt, output, tokens = _generic(rec)
        call_id = rec.get("id") or default_id
    else:
        return None

    estimated = tokens is None
    if tokens is None:
        tokens = {
            "input_tokens": _estimate(system) + _estimate(prompt),
            "output_tokens": _estimate(output),
            "cache_read_tokens": 0,
            "cache_write_tokens": 0,
        }
    meta = rec.get("metadata") or {}
    tag = rec.get("site") or rec.get("name") or meta.get("site")
    timings = (rec.get("latency_ms"), rec.get("duration_ms"), meta.get("latency_ms"), meta.get("duration_ms"))
    latency = next((float(t) for t in timings if isinstance(t, (int, float)) and not isinstance(t, bool)), None)
    return Call(
        id=str(call_id),
        site=str(tag) if tag else fingerprint(model, system, prompt),
        model=model,
        system=system,
        prompt=prompt,
        output=output,
        estimated=estimated,
        site_tagged=bool(tag),
        latency_ms=latency,
        trace_id=rec.get("trace_id"),
        **{k: int(v or 0) for k, v in tokens.items()},
    )


def fingerprint(model: str, system: str, prompt: str) -> str:
    """Name an untagged call site: same model + same system prompt => same site.

    Without a system prompt, the first characters of the prompt stand in for the
    template. Calls that share a generic system prompt will merge; tag them instead.
    """
    template = _normalize(system) if system.strip() else _normalize(prompt)[:24]
    digest = hashlib.sha1(f"{model}|{template}".encode()).hexdigest()[:6]
    words = re.findall(r"[a-z]+", (system or prompt).lower())[:5]
    return f"{'-'.join(words) or 'call'}#{digest}"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"\d", "0", text.lower())).strip()


def _estimate(text: str) -> int:
    return (len(text) + 3) // 4


def _tool(name: Any, args: Any) -> str:
    if not isinstance(args, str):
        args = json.dumps(args, sort_keys=True)
    return f"tool:{name}({args})"


def _text(content: Any) -> str:
    """Flatten message content (a string or a list of blocks) to text.

    Document and search-result blocks are wrapped in tags so passage detection can
    still tell them apart after flattening.
    """
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        content = [content]
    parts = []
    for block in content:
        if isinstance(block, str):
            parts.append(block)
        elif not isinstance(block, dict) or block.get("type") in ("thinking", "redacted_thinking", "reasoning"):
            continue
        elif block.get("type") in ("document", "search_result"):
            source = block.get("source") or {}
            inner = source.get("data") if isinstance(source.get("data"), str) else _text(
                block.get("content") or source.get("content")
            )
            parts.append(f"<{block['type']}>{inner}</{block['type']}>")
        elif isinstance(block.get("text"), str):
            parts.append(block["text"])
        elif block.get("type") in ("tool_use", "function_call"):
            parts.append(_tool(block.get("name"), block.get("input", block.get("arguments"))))
        elif "content" in block:  # tool_result, output messages and similar wrappers
            parts.append(_text(block["content"]))
        elif "output" in block:  # function_call_output
            parts.append(_text(block["output"]))
    return "\n".join(p for p in parts if p)


def _split_messages(messages: list) -> tuple[str, str]:
    system, prompt = [], []
    for m in messages or []:
        if not isinstance(m, dict):
            prompt.append(str(m))
        elif m.get("role") in ("system", "developer"):
            system.append(_text(m.get("content")))
        else:
            prompt.append(_text(m.get("content", m.get("output"))))
    return "\n\n".join(p for p in system if p), "\n\n".join(p for p in prompt if p)


def _anthropic(req: dict, resp: dict):
    prompt = "\n\n".join(_text(m.get("content")) for m in req.get("messages", []))
    u = resp.get("usage")
    tokens = None
    if isinstance(u, dict):
        tokens = {
            "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"),
            "cache_read_tokens": u.get("cache_read_input_tokens"),
            "cache_write_tokens": u.get("cache_creation_input_tokens"),
        }
    model = resp.get("model") or req.get("model") or "unknown"
    return model, _text(req.get("system")), prompt, _text(resp["content"]), tokens


def _openai_chat(req: dict, resp: dict):
    system, prompt = _split_messages(req.get("messages", []))
    message = (resp.get("choices") or [{}])[0].get("message") or {}
    calls = [_tool(tc["function"]["name"], tc["function"].get("arguments")) for tc in message.get("tool_calls") or []]
    output = "\n".join(p for p in [_text(message.get("content")), *calls] if p)
    u = resp.get("usage")
    tokens = None
    if isinstance(u, dict):
        cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        tokens = {
            "input_tokens": (u.get("prompt_tokens") or 0) - cached,
            "output_tokens": u.get("completion_tokens"),
            "cache_read_tokens": cached,
            "cache_write_tokens": 0,
        }
    return resp.get("model") or req.get("model") or "unknown", system, prompt, output, tokens


def _openai_responses(req: dict, resp: dict):
    inp = req.get("input")
    if isinstance(inp, str):
        system, prompt = "", inp
    else:
        system, prompt = _split_messages(inp)
    system = "\n\n".join(p for p in [_text(req.get("instructions")), system] if p)
    u = resp.get("usage")
    tokens = None
    if isinstance(u, dict):
        cached = (u.get("input_tokens_details") or {}).get("cached_tokens") or 0
        tokens = {
            "input_tokens": (u.get("input_tokens") or 0) - cached,
            "output_tokens": u.get("output_tokens"),
            "cache_read_tokens": cached,
            "cache_write_tokens": 0,
        }
    return resp.get("model") or req.get("model") or "unknown", system, prompt, _text(resp["output"]), tokens


def _generic(rec: dict):
    inp = rec.get("input", rec.get("prompt"))
    system, prompt = _split_messages(inp) if isinstance(inp, list) else ("", _text(inp))
    system = "\n\n".join(p for p in [_text(rec.get("system")), system] if p)
    u = rec.get("usage")
    tokens = None
    if isinstance(u, dict):
        tokens = {
            "input_tokens": u.get("input_tokens"),
            "output_tokens": u.get("output_tokens"),
            "cache_read_tokens": u.get("cache_read_tokens"),
            "cache_write_tokens": u.get("cache_write_tokens"),
        }
    return rec.get("model") or "unknown", system, prompt, _text(rec.get("output")), tokens
