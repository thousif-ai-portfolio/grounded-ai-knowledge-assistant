"""Small, dependency-free metric helpers."""

import math


def recall_at(ranks: list[int | None], k: int) -> float:
    """Share of questions whose first relevant chunk is in the top k."""
    if not ranks:
        return 0.0
    return sum(1 for r in ranks if r is not None and r <= k) / len(ranks)


def mrr_at(ranks: list[int | None], k: int) -> float:
    """Mean reciprocal rank, counting only hits within the top k."""
    if not ranks:
        return 0.0
    return sum(1 / r for r in ranks if r is not None and r <= k) / len(ranks)


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile, e.g. pct=95 for p95."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(pct / 100 * len(ordered)) - 1)
    return ordered[index]


def rate(flags: list[bool]) -> float:
    return sum(flags) / len(flags) if flags else 0.0
