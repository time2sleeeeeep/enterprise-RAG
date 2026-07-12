"""测试 retriever.py 纯函数：reciprocal_rank_fusion、_hit_to_doc、fuse_and_select。"""

from src.core.retriever import reciprocal_rank_fusion, _hit_to_doc, fuse_and_select
from tests.conftest import FakeHit


class TestHitToDoc:
    def test_full_fields(self):
        h = FakeHit("id1", 0.9, {"content": "hello", "source": "a.md", "page_num": 3, "doc_id": "d1"})
        d = _hit_to_doc(h)
        assert d == {"id": "id1", "score": 0.9, "content": "hello", "source": "a.md", "page_num": 3, "doc_id": "d1"}

    def test_missing_fields_fallback(self):
        h = FakeHit("id2", 0.5, {})
        d = _hit_to_doc(h)
        assert d["id"] == "id2"
        assert d["content"] == ""
        assert d["page_num"] == 0


class TestReciprocalRankFusion:
    def test_single_ranking(self):
        r = reciprocal_rank_fusion([[{"id": "A"}, {"id": "B"}]], k=60)
        assert r[0][0] == "A"

    def test_multi_ranking_overlap(self):
        # B 在两路都出现 → 累积分最高
        dense = [{"id": "A"}, {"id": "B"}]
        sparse = [{"id": "B"}, {"id": "C"}]
        r = reciprocal_rank_fusion([dense, sparse], k=60)
        assert r[0][0] == "B"

    def test_empty_ranking(self):
        r = reciprocal_rank_fusion([[]], k=60)
        assert r == []


class TestFuseAndSelect:
    def test_basic(self):
        dense = [
            {"id": "A", "score": 0.9, "content": "ca", "source": "s", "page_num": 1, "doc_id": "d"},
        ]
        sparse = [
            {"id": "A", "score": 5.0, "content": "ca2", "source": "s", "page_num": 1, "doc_id": "d"},
        ]
        results = fuse_and_select([dense, sparse], top_k=1, rrf_k=60)
        assert len(results) == 1
        assert results[0]["content"] == "ca"  # setdefault 保留 dense 先出现的版本

    def test_multiple_rankings_no_overlap(self):
        r1 = [{"id": "X", "content": "cx", "source": "sx", "page_num": 1, "doc_id": "dx"}]
        r2 = [{"id": "Y", "content": "cy", "source": "sy", "page_num": 2, "doc_id": "dy"}]
        results = fuse_and_select([r1, r2], top_k=2)
        assert len(results) == 2
