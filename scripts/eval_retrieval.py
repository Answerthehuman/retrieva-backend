#!/usr/bin/env python
"""Retrieval evaluation harness.

Runs a golden set against live Milvus and reports Recall/Precision/nDCG/MRR,
so retrieval changes can be judged on measured deltas instead of intuition.

**Retrieval only — no LLM calls.** It exercises the embed → search → rank path
directly, which keeps it free to run and makes results attributable to
retrieval rather than to generation.

    python scripts/eval_retrieval.py                                # run + print
    python scripts/eval_retrieval.py --save evals/results/baseline.json
    python scripts/eval_retrieval.py --compare evals/results/baseline.json

Results are tagged with the active profile: lite and full are not comparable,
since one has reranking off.
"""

import argparse
import asyncio
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.config.settings import get_settings  # noqa: E402
from core.providers import get_reranker, get_retriever  # noqa: E402
from evals.metrics import aggregate, compare, evaluate_one  # noqa: E402

EVALS_DIR = Path(__file__).resolve().parents[1] / "evals"
DEFAULT_GOLDEN_SET = EVALS_DIR / "golden_set.json"


def load_golden_set(path: Path) -> list[dict]:
    if not path.exists():
        example = EVALS_DIR / "golden_set.example.json"
        raise SystemExit(
            f"No golden set at {path}.\n"
            f"Copy the example and fill it in with questions over your own documents:\n"
            f"    cp {example} {path}\n"
            f"See the _readme block inside it for guidance."
        )
    data = json.loads(path.read_text(encoding="utf-8"))
    queries = data.get("queries", [])
    if not queries:
        raise SystemExit(f"{path} contains no queries.")

    unscoreable = [q["id"] for q in queries if not q.get("relevant_doc_ids")]
    if unscoreable:
        print(
            f"note: {len(unscoreable)} query(ies) have no relevant_doc_ids and will "
            f"score 0 by definition: {', '.join(unscoreable[:5])}"
            + (" …" if len(unscoreable) > 5 else ""),
            file=sys.stderr,
        )
    return queries


async def run_eval(queries: list[dict], top_k: int) -> dict:
    settings = get_settings()
    retriever = get_retriever(settings)
    reranker = get_reranker(settings)

    from core.utils.bm25 import build_bm25_loader

    bm25_loader = build_bm25_loader(settings)
    collection = settings.milvus_default_collection

    per_query, rows = [], []
    for entry in queries:
        question = entry["question"]
        relevant = entry.get("relevant_doc_ids", [])

        docs = await retriever.search_parallel(
            [{"query": question, "collection": collection}],
            output_fields=["content", "source", "page", "id"],
            bm25_loader=bm25_loader,
        )
        if reranker and docs:
            docs = await reranker.rerank(question, docs, top_k=top_k)

        retrieved_ids = [d.get("id") for d in docs[:top_k] if d.get("id")]
        metrics = evaluate_one(retrieved_ids, relevant)
        per_query.append(metrics)
        rows.append(
            {
                "id": entry.get("id"),
                "question": question,
                "retrieved": retrieved_ids[:10],
                "relevant": relevant,
                "metrics": metrics,
                "tags": entry.get("tags", []),
            }
        )

    return {
        "generated_at": datetime.now(UTC).isoformat(),
        # Results are only comparable within the same profile and embedding
        # model — record both so a stale comparison is obvious.
        "profile": settings.retrieva_profile,
        "embedding_model": settings.embedding_model,
        "embedding_dim": settings.embedding_dim,
        "rerank_enabled": settings.rerank_enabled,
        "hybrid_search_enabled": settings.hybrid_search_enabled,
        "top_k": top_k,
        "query_count": len(queries),
        "aggregate": aggregate(per_query),
        "per_query": rows,
    }


def print_report(result: dict) -> None:
    print()
    print("=" * 68)
    print("RETRIEVAL EVALUATION")
    print("=" * 68)
    print(f"  profile          : {result['profile']}")
    print(f"  embedding        : {result['embedding_model']} ({result['embedding_dim']}d)")
    print(f"  reranking        : {'on' if result['rerank_enabled'] else 'off'}")
    print(f"  hybrid search    : {'on' if result['hybrid_search_enabled'] else 'off'}")
    print(f"  queries          : {result['query_count']}")
    print()
    print(f"  {'METRIC':<16}{'VALUE':>10}")
    print(f"  {'-' * 26}")
    for key, value in sorted(result["aggregate"].items()):
        print(f"  {key:<16}{value:>10.4f}")

    misses = [r for r in result["per_query"] if r["metrics"]["miss"] == 1.0]
    if misses:
        print()
        print(f"  COMPLETE MISSES ({len(misses)}) — nothing relevant retrieved:")
        for row in misses[:10]:
            print(f"    [{row['id']}] {row['question'][:60]}")
        if len(misses) > 10:
            print(f"    … and {len(misses) - 10} more")
    print()


def print_comparison(current: dict, baseline: dict) -> bool:
    """Print deltas. Returns True if anything regressed."""
    print("=" * 68)
    print("COMPARISON VS BASELINE")
    print("=" * 68)

    if baseline.get("profile") != current.get("profile"):
        print(
            f"  ⚠️  PROFILE MISMATCH: baseline={baseline.get('profile')} "
            f"current={current.get('profile')} — these are not comparable."
        )
    if baseline.get("embedding_model") != current.get("embedding_model"):
        print(
            f"  ⚠️  EMBEDDING MODEL CHANGED: {baseline.get('embedding_model')} → "
            f"{current.get('embedding_model')} — expect large, expected shifts."
        )

    deltas = compare(current["aggregate"], baseline.get("aggregate", {}))
    print()
    print(f"  {'METRIC':<16}{'BASELINE':>10}{'CURRENT':>10}{'DELTA':>10}")
    print(f"  {'-' * 46}")
    regressed = False
    for key, info in deltas.items():
        if info["delta"] is None:
            print(f"  {key:<16}{'—':>10}{info['current']:>10.4f}{'new':>10}")
            continue
        marker = "  ⚠️" if info["regressed"] else ""
        regressed = regressed or info["regressed"]
        print(
            f"  {key:<16}{info['baseline']:>10.4f}{info['current']:>10.4f}"
            f"{info['delta']:>+10.4f}{marker}"
        )
    print()
    print("  ⚠️  one or more metrics regressed" if regressed else "  ✓ no regressions")
    print()
    return regressed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden-set", type=Path, default=DEFAULT_GOLDEN_SET)
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--save", type=Path, help="write results JSON here")
    parser.add_argument("--compare", type=Path, help="compare against a saved results file")
    parser.add_argument(
        "--fail-on-regression",
        action="store_true",
        help="exit non-zero if any metric regressed (for CI)",
    )
    args = parser.parse_args()

    queries = load_golden_set(args.golden_set)
    result = asyncio.run(run_eval(queries, args.top_k))
    print_report(result)

    regressed = False
    if args.compare:
        if not args.compare.exists():
            print(f"note: no baseline at {args.compare}; skipping comparison", file=sys.stderr)
        else:
            baseline = json.loads(args.compare.read_text(encoding="utf-8"))
            regressed = print_comparison(result, baseline)

    if args.save:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"saved → {args.save}")

    return 1 if (regressed and args.fail_on_regression) else 0


if __name__ == "__main__":
    raise SystemExit(main())
