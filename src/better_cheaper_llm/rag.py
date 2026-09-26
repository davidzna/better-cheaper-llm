"""Checks a RAG pipeline commonly asks an LLM to make."""

from __future__ import annotations

import re
from typing import Callable, Iterable

from .check import Checker, CheckResult

_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'(\[])")
MIN_CLAIM_CHARS = 12


def sufficient(
    checker: Checker,
    question: str,
    passages: Iterable[str],
    *,
    site: str = "rag.sufficient",
    threshold: float = 0.9,
    fallback: Callable[[], bool] | None = None,
) -> CheckResult:
    """Do the retrieved passages contain what's needed to answer the question?

    A confident no lets you skip generation: retrieve again, or say you don't know.
    """
    return checker.check(
        site,
        "The passages contain the information needed to answer the question.",
        {"question": question, "passages": list(passages)},
        threshold=threshold,
        fallback=fallback,
    )


def grounded(
    checker: Checker,
    answer: str,
    passages: Iterable[str],
    *,
    site: str = "rag.grounded",
    threshold: float = 0.9,
    fallback: Callable[[], bool] | None = None,
    max_claims: int = 32,
) -> CheckResult:
    """Is every claim in the answer supported by the passages?

    Each sentence becomes its own Jev question, all sent in one request; the answer
    is grounded only if its weakest claim is.
    """
    statements = [
        {"question": "Do the passages in the state support this claim?", "claim": claim}
        for claim in split_claims(answer, max_claims)
    ]
    return checker.check(site, statements, {"passages": list(passages)}, threshold=threshold, fallback=fallback)


def split_claims(text: str, max_claims: int = 32) -> list[str]:
    """Split into sentences, folding short fragments into the previous one.

    Past ``max_claims``, neighbouring sentences are grouped so every sentence is
    still checked.
    """
    claims: list[str] = []
    for sentence in _SENTENCE_END.split(text.strip()):
        sentence = sentence.strip()
        if not sentence:
            continue
        if claims and len(sentence) < MIN_CLAIM_CHARS:
            claims[-1] += " " + sentence
        else:
            claims.append(sentence)
    if len(claims) > max_claims:
        size = -(-len(claims) // max_claims)
        claims = [" ".join(claims[i : i + size]) for i in range(0, len(claims), size)]
    return claims
