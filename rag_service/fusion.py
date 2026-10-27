"""Reciprocal rank fusion (Cormack et al., 2009).

score(d) = sum over ranked lists of 1 / (k + rank(d)), rank starting at 1.
Only ranks are used, never raw scores, so cosine similarities and ts_rank values (which live on
incomparable scales) can be fused without any normalization or tuning.
"""

from __future__ import annotations

from dataclasses import dataclass, field

RRF_K = 60  # the paper's default; dampens the advantage of being #1 vs #2 in a single list


@dataclass
class Fused:
    id: int
    score: float
    ranks: dict[str, int] = field(default_factory=dict)  # list name -> 1-based rank


def rrf(
    ranked: dict[str, list[int]], k: int = RRF_K, weights: dict[str, float] | None = None
) -> list[Fused]:
    """Fuse named ranked lists of ids. Ties are broken by best single rank, then by id.

    `weights` scales each list's contribution (weighted RRF); missing lists weigh 1.
    """
    weights = weights or {}
    fused: dict[int, Fused] = {}
    for name, ids in ranked.items():
        w = weights.get(name, 1.0)
        for rank, doc_id in enumerate(ids, start=1):
            f = fused.setdefault(doc_id, Fused(doc_id, 0.0))
            f.score += w / (k + rank)
            f.ranks[name] = rank
    return sorted(fused.values(), key=lambda f: (-f.score, min(f.ranks.values()), f.id))
