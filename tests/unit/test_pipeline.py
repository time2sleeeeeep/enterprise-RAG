"""测试 generate_chunk_id（确定性、唯一性）。"""

from src.ingestion.pipeline import generate_chunk_id


class TestGenerateChunkId:
    def test_deterministic(self):
        a = generate_chunk_id("doc1", 0)
        b = generate_chunk_id("doc1", 0)
        assert a == b

    def test_unique_across_indices(self):
        ids = {generate_chunk_id("d", i) for i in range(10)}
        assert len(ids) == 10

    def test_hex_format(self):
        cid = generate_chunk_id("d", 5)
        assert len(cid) == 32
        assert all(c in "0123456789abcdef" for c in cid)
