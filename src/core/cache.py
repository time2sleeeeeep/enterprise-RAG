# 答案缓存模块：基于 Redis 的双层缓存，先做精确哈希匹配，未命中时再做语义相似度匹配。
# 相似度阈值 0.95，缓存 TTL 3600 秒。

import hashlib
import json

import numpy as np
from loguru import logger

from src.core.embeddings import get_embedder
from src.db.redis_client import get_redis


CACHE_TTL = 3600
SIMILARITY_THRESHOLD = 0.95


def _hash_query(query: str) -> str:
    """对查询字符串做 MD5 哈希，用作 Redis key 的唯一标识。"""
    return hashlib.md5(query.encode()).hexdigest()


async def get_cached_answer(query: str) -> dict | None:
    """查询缓存：先精确哈希匹配，未命中再遍历语义向量做余弦相似度匹配（阈值 0.95）。"""
    redis = await get_redis()
    if redis is None:
        return None

    exact_key = f"rag:cache:{_hash_query(query)}"
    cached = await redis.get(exact_key)
    if cached:
        logger.debug(f"Cache hit (exact): {query[:50]}")
        return json.loads(cached)

    embedder = get_embedder()
    query_emb = embedder.encode_query(query)["dense"]

    keys = await redis.keys("rag:cache_emb:*")
    if not keys:
        return None

    for key in keys[:100]:
        raw = await redis.get(key)
        if raw is None:
            continue
        entry = json.loads(raw)
        stored_emb = np.array(entry["embedding"])
        sim = float(np.dot(query_emb, stored_emb) / (
            np.linalg.norm(query_emb) * np.linalg.norm(stored_emb) + 1e-8
        ))
        if sim >= SIMILARITY_THRESHOLD:
            logger.debug(f"Cache hit (semantic, sim={sim:.4f}): {query[:50]}")
            return entry["result"]

    return None


async def set_cached_answer(query: str, result: dict) -> None:
    """写入缓存：同时存储精确哈希键和嵌入向量键，TTL 均为 3600 秒。"""
    redis = await get_redis()
    if redis is None:
        return

    exact_key = f"rag:cache:{_hash_query(query)}"
    await redis.set(exact_key, json.dumps(result, ensure_ascii=False), ex=CACHE_TTL)

    embedder = get_embedder()
    query_emb = embedder.encode_query(query)["dense"]

    emb_key = f"rag:cache_emb:{_hash_query(query)}"
    entry = {"embedding": query_emb.tolist(), "result": result}
    await redis.set(emb_key, json.dumps(entry, ensure_ascii=False), ex=CACHE_TTL)

    logger.debug(f"Cached answer for: {query[:50]}")
