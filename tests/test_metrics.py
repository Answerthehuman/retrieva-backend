"""Tests for the retrieval metrics.

If these are wrong, every conclusion drawn from the eval harness is wrong —
and a bad metric is worse than no metric, because it looks authoritative.
"""

import pytest

from evals.metrics import (
    aggregate,
    compare,
    evaluate_one,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)


class TestRecall:
    def test_perfect(self):
        assert recall_at_k(["a", "b", "c"], ["a", "b"], 3) == 1.0

    def test_partial(self):
        assert recall_at_k(["a", "x", "y"], ["a", "b"], 3) == 0.5

    def test_none_found(self):
        assert recall_at_k(["x", "y"], ["a"], 2) == 0.0

    def test_respects_k_cutoff(self):
        # "b" sits at rank 3, outside k=2.
        assert recall_at_k(["a", "x", "b"], ["a", "b"], 2) == 0.5

    def test_no_relevant_scores_zero(self):
        """An unanswerable question must not score 1.0 by vacuous truth."""
        assert recall_at_k(["a"], [], 5) == 0.0

    def test_duplicates_do_not_inflate(self):
        assert recall_at_k(["a", "a", "a"], ["a", "b"], 3) == 0.5


class TestPrecision:
    def test_all_relevant(self):
        assert precision_at_k(["a", "b"], ["a", "b"], 2) == 1.0

    def test_half_relevant(self):
        assert precision_at_k(["a", "x"], ["a"], 2) == 0.5

    def test_empty_retrieval(self):
        assert precision_at_k([], ["a"], 5) == 0.0

    def test_zero_k(self):
        assert precision_at_k(["a"], ["a"], 0) == 0.0


class TestReciprocalRank:
    def test_first_position(self):
        assert reciprocal_rank(["a", "x"], ["a"]) == 1.0

    def test_second_position(self):
        assert reciprocal_rank(["x", "a"], ["a"]) == 0.5

    def test_third_position(self):
        assert reciprocal_rank(["x", "y", "a"], ["a"]) == pytest.approx(1 / 3)

    def test_absent(self):
        assert reciprocal_rank(["x", "y"], ["a"]) == 0.0


class TestNDCG:
    def test_ideal_ordering_is_one(self):
        assert ndcg_at_k(["a", "b"], ["a", "b"], 2) == pytest.approx(1.0)

    def test_rewards_ranking_relevant_higher(self):
        """The whole point of nDCG over recall: position matters."""
        good = ndcg_at_k(["a", "x", "y"], ["a"], 3)
        bad = ndcg_at_k(["x", "y", "a"], ["a"], 3)
        assert good > bad

    def test_no_relevant(self):
        assert ndcg_at_k(["x"], [], 3) == 0.0

    def test_nothing_retrieved(self):
        assert ndcg_at_k([], ["a"], 3) == 0.0


class TestEvaluateOne:
    def test_emits_all_metrics(self):
        result = evaluate_one(["a", "b"], ["a"], ks=(1, 3))
        for key in ("recall@1", "recall@3", "precision@1", "ndcg@1", "ndcg@3", "mrr", "miss"):
            assert key in result

    def test_miss_flag_set_when_nothing_relevant_found(self):
        assert evaluate_one(["x", "y"], ["a"])["miss"] == 1.0

    def test_miss_flag_clear_on_a_hit(self):
        assert evaluate_one(["a"], ["a"])["miss"] == 0.0


class TestAggregate:
    def test_means_across_queries(self):
        agg = aggregate([{"recall@1": 1.0}, {"recall@1": 0.0}])
        assert agg["recall@1"] == 0.5

    def test_empty_input(self):
        assert aggregate([]) == {}


class TestCompare:
    def test_improvement_not_flagged(self):
        result = compare({"recall@10": 0.8}, {"recall@10": 0.6})
        assert result["recall@10"]["delta"] == pytest.approx(0.2)
        assert result["recall@10"]["regressed"] is False

    def test_regression_flagged(self):
        result = compare({"recall@10": 0.5}, {"recall@10": 0.7})
        assert result["recall@10"]["regressed"] is True

    def test_miss_direction_inverted(self):
        """`miss` is the one metric where lower is better."""
        worse = compare({"miss": 0.3}, {"miss": 0.1})
        assert worse["miss"]["regressed"] is True

        better = compare({"miss": 0.1}, {"miss": 0.3})
        assert better["miss"]["regressed"] is False

    def test_identical_is_not_a_regression(self):
        assert compare({"mrr": 0.5}, {"mrr": 0.5})["mrr"]["regressed"] is False

    def test_missing_metric_handled(self):
        """A newly-added metric must not crash a comparison to an old baseline."""
        result = compare({"new_metric": 0.5}, {})
        assert result["new_metric"]["delta"] is None
        assert result["new_metric"]["regressed"] is False
