"""Find retrieved passages in a RAG prompt and guess which ones the answer used.

The guess is lexical: a passage counts as used if the answer cites it or shares
two content-word trigrams with it. Unused-token totals are an estimate of what
passage filtering could remove, not a measurement.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_TAGGED = re.compile(
    r"<(document|doc|passage|source|chunk|context_item|search_result)\b([^>]*)>(.*?)</\1\s*>",
    re.S | re.I,
)
_TAG_INDEX = re.compile(r"""\b(?:index|id|n)\s*=\s*["']?([\w-]+)""", re.I)
_NUMBERED = re.compile(
    r"^[ \t]*(?:\[(\d{1,3})\]|(?:document|doc|passage|source|chunk)[ \t]*#?(\d{1,3})[ \t]*[:.)\]-])",
    re.I | re.M,
)
_CITATION = re.compile(r"\[(\d{1,3}(?:\s*,\s*\d{1,3})*)\]")
_WORD = re.compile(r"[a-z0-9]+(?:[.,][0-9]+)*")
_STOPWORDS = frozenset(
    "a an and are as at be been but by for from has have in is it its of on or that the "
    "their this to was were which will with".split()
)
MIN_SHARED_TRIGRAMS = 2


@dataclass
class Passage:
    label: str
    text: str


def extract_passages(prompt: str) -> list[Passage]:
    """Tagged blocks (<document>, <passage>, ...) first, then [1]-style numbering."""
    tagged = _TAGGED.findall(prompt)
    if len(tagged) >= 2:
        passages = []
        for i, (_, attrs, body) in enumerate(tagged, 1):
            index = _TAG_INDEX.search(attrs)
            passages.append(Passage(index.group(1) if index else str(i), body.strip()))
        return passages

    markers = list(_NUMBERED.finditer(prompt))
    if len(markers) < 2:
        return []
    passages = []
    for i, m in enumerate(markers):
        if i + 1 < len(markers):
            end = markers[i + 1].start()
        else:  # the last passage ends at the next blank line (usually before the question)
            blank = prompt.find("\n\n", m.end())
            end = blank if blank != -1 else len(prompt)
        passages.append(Passage(m.group(1) or m.group(2), prompt[m.end() : end].strip()))
    return passages


def used_labels(passages: list[Passage], answer: str) -> set[str]:
    cited = {n.strip() for group in _CITATION.findall(answer) for n in group.split(",")}
    answer_grams = _trigrams(answer)
    return {
        p.label
        for p in passages
        if p.label in cited or len(answer_grams & _trigrams(p.text)) >= MIN_SHARED_TRIGRAMS
    }


def _trigrams(text: str) -> set[tuple[str, ...]]:
    words = [w for w in _WORD.findall(text.lower()) if w not in _STOPWORDS]
    return {tuple(words[i : i + 3]) for i in range(len(words) - 2)}
