"""测试 LatencyTracker + ComponentTiming。"""

import time
from src.evaluation.latency_tracker import LatencyTracker, ComponentTiming


class TestLatencyTracker:
    def test_record_and_summary(self):
        tracker = LatencyTracker()
        tracker.start()
        time.sleep(0.01)
        tracker.record("embedding_ms")

        tracker.start()
        time.sleep(0.01)
        tracker.record("generation_ms")

        s = tracker.summary()
        assert s["embedding_ms"]["count"] == 1
        assert s["generation_ms"]["count"] == 1
        assert s["embedding_ms"]["total"] > 0

    def test_to_component_timing(self):
        tracker = LatencyTracker()
        tracker.start()
        time.sleep(0.005)
        tracker.record("scoring_ms")
        ct = tracker.to_component_timing()
        assert ct.scoring_ms > 0
        assert ct.embedding_ms == 0.0

    def test_reset(self):
        tracker = LatencyTracker()
        tracker.start()
        time.sleep(0.01)
        tracker.record("embedding_ms")
        tracker.reset()
        s = tracker.summary()
        assert s["embedding_ms"]["count"] == 0


class TestComponentTiming:
    def test_as_dict_keys(self):
        ct = ComponentTiming(embedding_ms=1.0, generation_ms=2.0)
        d = ct.as_dict()
        for k in ("embedding_ms", "dense_search_ms", "sparse_search_ms", "reranking_ms", "generation_ms", "scoring_ms"):
            assert k in d

    def test_total_ms(self):
        ct = ComponentTiming(embedding_ms=1.5, reranking_ms=3.0)
        assert ct.total_ms == 4.5
