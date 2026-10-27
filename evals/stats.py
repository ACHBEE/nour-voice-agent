"""Small stats helpers shared by the eval and latency scripts (no numpy needed)."""

from __future__ import annotations

from collections.abc import Iterable, Sequence


def percentile(values: Sequence[float], p: float) -> float | None:
    """Linear-interpolated percentile (p in 0..100), like numpy's default. None if empty."""
    xs = sorted(v for v in values if v is not None)
    if not xs:
        return None
    pos = (len(xs) - 1) * p / 100
    lo = int(pos)
    hi = min(lo + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def rate(flags: Iterable[bool]) -> float | None:
    xs = list(flags)
    return sum(xs) / len(xs) if xs else None


def first_relevant_rank(ranked_doc_ids: Sequence[str], relevant: set[str]) -> int | None:
    """1-based position of the first chunk from a relevant document, or None."""
    for i, doc_id in enumerate(ranked_doc_ids, 1):
        if doc_id in relevant:
            return i
    return None


def ranking_metrics(ranks: Sequence[int | None], ks: Sequence[int] = (1, 3, 5)) -> dict[str, float]:
    """hit@k and MRR over queries, given each query's first relevant rank (None = miss)."""
    n = len(ranks)
    if n == 0:
        return {}
    out = {f"hit@{k}": sum(1 for r in ranks if r is not None and r <= k) / n for k in ks}
    out["mrr"] = sum(1 / r for r in ranks if r is not None) / n
    return out
