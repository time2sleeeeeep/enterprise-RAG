"""mocked 集成测试：缓存 SCAN + 精确/语义命中/未命中路径（H2）。"""

import asyncio
import fnmatch
import json

import numpy as np
import pytest


class FakeRedis:
    def __init__(self, store):
        self.store = store
        self.emb_get_count = 0

    async def get(self, key):
        if "cache_emb:" in key:
            self.emb_get_count += 1
        return self.store.get(key)

    async def set(self, key, value, ex=None):
        self.store[key] = value

    async def scan_iter(self, match=None, count=None):
        for k in list(self.store.keys()):
            if fnmatch.fnmatch(k, match):
                yield k


class FakeEmbedder:
    def __init__(self, arr):
        self.emb = np.array(arr, dtype=float)

    def encode_query(self, q):
        return {"dense": self.emb}


@pytest.fixture
def store():
    return {}


@pytest.fixture
def patch_cache(monkeypatch, store):
    fake_redis = FakeRedis(store)

    async def fake_get_redis():
        return fake_redis

    monkeypatch.setattr("src.core.cache.get_redis", fake_get_redis)


def _embedder(arr):
    return lambda: FakeEmbedder(arr)


async def test_exact_hit(store, patch_cache, monkeypatch):
    monkeypatch.setattr("src.core.cache.get_embedder", _embedder([1.0, 0.0]))
    from src.core.cache import get_cached_answer, set_cached_answer

    await set_cached_answer("hello", {"answer": "A", "sources": []})
    r = await get_cached_answer("hello")
    assert r["answer"] == "A"


async def test_semantic_hit(store, patch_cache, monkeypatch):
    monkeypatch.setattr("src.core.cache.get_embedder", _embedder([1.0, 0.0]))
    from src.core.cache import get_cached_answer, set_cached_answer

    await set_cached_answer("q1", {"answer": "X", "sources": []})
    # 不同 query 文本，但相同 embedding → 语义命中
    r = await get_cached_answer("q2")
    assert r is not None and r["answer"] == "X"


async def test_miss(store, patch_cache, monkeypatch):
    monkeypatch.setattr("src.core.cache.get_embedder", _embedder([0.0, 1.0]))
    from src.core.cache import get_cached_answer, set_cached_answer

    await set_cached_answer("q1", {"answer": "Y", "sources": []})
    # 正交 embedding → miss
    monkeypatch.setattr("src.core.cache.get_embedder", _embedder([1.0, 0.0]))
    r = await get_cached_answer("q2")
    assert r is None


async def test_scan_limit_enforced(store, patch_cache, monkeypatch):
    # 5 条 emb 缓存，limit=2 → 只比对 2 条
    monkeypatch.setattr("src.core.cache.get_embedder", _embedder([0.0, 1.0]))
    from src.core.cache import get_cached_answer, set_cached_answer

    for i in range(5):
        await set_cached_answer(f"q{i}", {"answer": str(i), "sources": []})

    monkeypatch.setattr("src.core.cache.get_embedder", _embedder([1.0, 0.0]))
    import src.core.cache as c

    saved = c.settings.semantic_cache_scan_limit
    c.settings.semantic_cache_scan_limit = 2
    r = await get_cached_answer("qmiss")
    c.settings.semantic_cache_scan_limit = saved
    assert r is None
    # 精确 miss (1 get) + 语义比对 (limit=2 → 2 gets on emb keys) = 3 total
    # emb_get_count 应 ≤ 2
    from src.core.cache import get_redis

    fake = await get_redis()
    assert fake.emb_get_count == 2, f"expected 2, got {fake.emb_get_count}"


async def test_set_writes_both_keys(store, patch_cache, monkeypatch):
    monkeypatch.setattr("src.core.cache.get_embedder", _embedder([0.5, 0.5]))
    from src.core.cache import set_cached_answer

    await set_cached_answer("q", {"answer": "Z", "sources": []})
    assert any(k.startswith("rag:cache:") for k in store)
    assert any(k.startswith("rag:cache_emb:") for k in store)
