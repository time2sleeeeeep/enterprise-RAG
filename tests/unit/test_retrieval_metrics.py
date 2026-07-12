"""测试 retrieval_metrics.py 全部 6 个纯函数 + RETRIEVAL_METRIC_SPECS。"""

import pytest
from src.evaluation.retrieval_metrics import (
    precision_at_k,
    recall_at_k,
    rrf_score,
    ndcg_at_k,
    hit_rate_at_k,
    average_precision,
    RETRIEVAL_METRIC_SPECS,
)


class TestPrecisionAtK:
    def test_basic(self):
        assert precision_at_k({"a", "b"}, ["a", "c", "b"], k=3) == 2 / 3

    def test_k_smaller_than_hits(self):
        assert precision_at_k({"a", "b", "c"}, ["a", "x", "b"], k=2) == 1 / 2

    def test_empty_retrieved(self):
        assert precision_at_k({"a"}, [], k=5) == 0.0

    def test_no_relevant(self):
        assert precision_at_k({"x"}, ["a", "b"], k=5) == 0.0


class TestRecallAtK:
    def test_basic(self):
        assert recall_at_k({"a", "b"}, ["a", "c", "b"], k=3) == 1.0

    def test_partial(self):
        assert recall_at_k({"a", "b", "c"}, ["a", "x"], k=2) == 1 / 3

    def test_no_relevant_docs(self):
        assert recall_at_k(set(), ["a", "b"], k=5) == 0.0

    def test_empty_retrieved(self):
        assert recall_at_k({"a", "b"}, [], k=5) == 0.0


class TestMRR:
    def test_first_at_rank_1(self):
        assert rrf_score({"a"}, ["a", "b"]) == 1.0

    def test_first_at_rank_3(self):
        assert rrf_score({"c"}, ["a", "b", "c"]) == 1.0 / 3

    def test_no_relevant(self):
        assert rrf_score({"x"}, ["a", "b"]) == 0.0


class TestNDCG:
    def test_perfect_ranking(self):
        rel = {"a": 2, "b": 1}
        assert ndcg_at_k(rel, ["a", "b"], k=2) == pytest.approx(1.0)

    def test_worse_ranking(self):
        rel = {"a": 2, "b": 1}
        assert ndcg_at_k(rel, ["b", "a"], k=2) < 1.0

    def test_only_irrelevant(self):
        assert ndcg_at_k({"x": 0}, ["a", "b"], k=2) == 0.0


class TestHitRate:
    def test_hit(self):
        assert hit_rate_at_k({"a"}, ["x", "a"], k=2) == 1.0

    def test_miss(self):
        assert hit_rate_at_k({"a"}, ["x", "y"], k=2) == 0.0


class TestAveragePrecision:
    def test_perfect(self):
        assert average_precision({"a", "b"}, ["a", "b"]) == 1.0

    def test_imperfect(self):
        # 两个相关: 位置 2 和 3, precision@2=0.5, precision@3=2/3
        ap = average_precision({"b", "c"}, ["a", "b", "c"])
        expected = (0.5 + 2 / 3) / 2
        assert ap == pytest.approx(expected)

    def test_no_relevant(self):
        assert average_precision(set(), ["a", "b"]) == 0.0


def test_metric_specs_consistency():
    """RETRIEVAL_METRIC_SPECS 应该包含预期键且每个 spec 有 fn/kwargs。"""
    assert "precision_at_5" in RETRIEVAL_METRIC_SPECS
    assert "recall_at_10" in RETRIEVAL_METRIC_SPECS
    assert "mrr" in RETRIEVAL_METRIC_SPECS
    assert "ndcg_at_10" in RETRIEVAL_METRIC_SPECS
    assert "hit_rate_at_5" in RETRIEVAL_METRIC_SPECS
    assert "map_score" in RETRIEVAL_METRIC_SPECS
    for name, spec in RETRIEVAL_METRIC_SPECS.items():
        assert "fn" in spec, name
        assert "kwargs" in spec, name
