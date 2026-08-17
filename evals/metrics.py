"""Retrieval metrics.

Deliberately dependency-free and pure so they are unit-testable without Milvus,
an LLM, or a network. If the metrics themselves are wrong, every conclusion
drawn from the eval harness is wrong too.
"""

import math
from collections.abc import Sequence


def recall_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of relevant documents appearing in the top k.

    Returns 0.0 when nothing is relevant — a question with no known answer
    cannot be scored, and 1.0 would flatter the system.
    """
    if not relevant:
        return 0.0
    top_k = set(retrieved[:k])
    hits = sum(1 for doc_id in set(relevant) if doc_id in top_k)
    return hits / len(set(relevant))


def precision_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Fraction of the top k that is relevant."""
    if k <= 0 or not retrieved:
        return 0.0
    top_k = retrieved[:k]
    relevant_set = set(relevant)
    return sum(1 for doc_id in top_k if doc_id in relevant_set) / len(top_k)


def reciprocal_rank(retrieved: Sequence[str], relevant: Sequence[str]) -> float:
    """1/rank of the first relevant hit; 0.0 if none. Rewards ranking it first."""
    relevant_set = set(relevant)
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / index
    return 0.0


def dcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """Discounted cumulative gain with binary relevance."""
    relevant_set = set(relevant)
    return sum(
        1.0 / math.log2(index + 1)
        for index, doc_id in enumerate(retrieved[:k], start=1)
        if doc_id in relevant_set
    )


def ndcg_at_k(retrieved: Sequence[str], relevant: Sequence[str], k: int) -> float:
    """DCG normalised by the best achievable ordering.

    Unlike recall, this rewards putting relevant documents *near the top* —
    which is what actually matters when only the first few chunks fit in the
    context window.
    """
    if not relevant:
        return 0.0
    ideal = dcg_at_k(list(dict.fromkeys(relevant)), relevant, k)
    if ideal == 0:
        return 0.0
    return dcg_at_k(retrieved, relevant, k) / ideal


def evaluate_one(
    retrieved: Sequence[str],
    relevant: Sequence[str],
    ks: Sequence[int] = (1, 3, 5, 10),
) -> dict[str, float]:
    """All metrics for a single query."""
    results: dict[str, float] = {}
    for k in ks:
        results[f"recall@{k}"] = recall_at_k(retrieved, relevant, k)
        results[f"precision@{k}"] = precision_at_k(retrieved, relevant, k)
        results[f"ndcg@{k}"] = ndcg_at_k(retrieved, relevant, k)
    results["mrr"] = reciprocal_rank(retrieved, relevant)
    # Tracked separately because it is the failure people actually feel: the
    # answer simply was not in the corpus slice the agent saw.
    results["miss"] = 0.0 if any(d in set(relevant) for d in retrieved) else 1.0
    return results


def aggregate(per_query: list[dict[str, float]]) -> dict[str, float]:
    """Mean of each metric across queries."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {key: sum(q[key] for q in per_query) / len(per_query) for key in keys}


def compare(current: dict[str, float], baseline: dict[str, float]) -> dict[str, dict]:
    """Per-metric delta against a baseline.

    `regressed` accounts for direction: a rise in `miss` is bad, a rise in
    everything else is good.
    """
    out: dict[str, dict] = {}
    for key in sorted(set(current) | set(baseline)):
        now = current.get(key)
        was = baseline.get(key)
        if now is None or was is None:
            out[key] = {"current": now, "baseline": was, "delta": None, "regressed": False}
            continue
        delta = now - was
        lower_is_better = key == "miss"
        regressed = (delta > 1e-9) if lower_is_better else (delta < -1e-9)
        out[key] = {
            "current": now,
            "baseline": was,
            "delta": delta,
            "regressed": regressed,
        }
    return out
