# 混合检索模块：结合稠密向量检索（COSINE）和稀疏向量检索（BM25-like），
# 通过 Reciprocal Rank Fusion 融合两路结果，从 Milvus 获取最终 Top-K 文档。

import numpy as np
from loguru import logger
from pymilvus import Collection

from src.config import settings
from src.core.embeddings import get_embedder
from src.db.milvus_client import create_collection


def fuse_and_select(
    rankings: list[list[dict]],
    top_k: int,
    rrf_k: int = 60,
) -> list[dict]:
    """多路排名列表 RRF 融合 + 按融合分值选取 top_k 完整文档。

    各路列表的元素 dict 应带 id/content/source/page_num/doc_id 字段；
    返回的每个 dict 含上述全部字段，score 为 RRF 融合分值。
    """
    fused = reciprocal_rank_fusion(rankings, k=rrf_k)

    # 按 id 建索引，优先保留先出现的文档（setdefault）
    id_to_doc: dict[str, dict] = {}
    for ranking in rankings:
        for doc in ranking:
            id_to_doc.setdefault(doc["id"], doc)

    results = []
    for doc_id, score in fused[:top_k]:
        doc = id_to_doc.get(doc_id)
        if doc is not None:
            results.append({
                "id": doc["id"],
                "content": doc["content"],
                "source": doc["source"],
                "page_num": doc["page_num"],
                "doc_id": doc["doc_id"],
                "score": score,
            })
    return results


def reciprocal_rank_fusion(
    rankings: list[list[dict]],
    k: int = 60,
) -> list[tuple[str, float]]:
    """对多路检索结果做 RRF 融合，返回按融合分值降序排列的 (id, score) 列表。"""
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc in enumerate(ranking):
            doc_id = doc["id"]
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


def _hit_to_doc(hit) -> dict:
    """将 Milvus 搜索命中对象转为带完整字段的文档字典（content 等随检索一次取回）。"""
    return {
        "id": hit.id,
        "score": hit.score,
        "content": hit.get("content") or "",
        "source": hit.get("source") or "",
        "page_num": hit.get("page_num") or 0,
        "doc_id": hit.get("doc_id") or "",
    }


class HybridRetriever:
    """混合检索器，结合稠密和稀疏向量搜索，通过 RRF 融合后从 Milvus 拉取完整文档内容。"""

    def __init__(self):
        self.embedder = get_embedder()
        self.collection: Collection | None = None

    def _get_collection(self) -> Collection:
        """懒加载并返回已加载的 Milvus Collection。

        注意：空集合调用 load() 会触发 Milvus QueryCoord 的无限等待循环（等待从未
        出现的 segment），约 5 分钟后 gRPC 超时。因此先检查 num_entities，为 0 时
        跳过 load() —— 空集合搜索会立即返回空结果，无需加载。[MILVUS_EMPTY_LOAD]
        """
        if self.collection is None:
            self.collection = create_collection(settings.milvus_collection)
            if self.collection.num_entities > 0:
                self.collection.load()
            else:
                logger.warning(
                    f"Collection '{settings.milvus_collection}' has 0 entities, "
                    "skipping load() to avoid Milvus empty-collection hang. "
                    "Search will return empty results until data is ingested."
                )
        return self.collection

    def dense_search(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[dict]:
        """执行稠密向量 COSINE 相似度搜索，返回带完整字段的文档列表。"""
        collection = self._get_collection()
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k,
            output_fields=["id", "content", "source", "page_num", "doc_id"],
        )
        return [_hit_to_doc(hit) for hit in results[0]]

    def sparse_search(
        self, query_sparse: dict, top_k: int = 20
    ) -> list[dict]:
        """执行稀疏向量内积搜索（BM25-like），返回带完整字段的文档列表。"""
        collection = self._get_collection()
        results = collection.search(
            data=[query_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=top_k,
            output_fields=["id", "content", "source", "page_num", "doc_id"],
        )
        return [_hit_to_doc(hit) for hit in results[0]]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[dict]:
        """稠密 + 稀疏检索 → 一路 RRF 融合，返回 top_k 完整文档。"""
        query_emb = self.embedder.encode_query(query)
        dense_results = self.dense_search(query_emb["dense"], top_k=top_k * 3)
        sparse_results = self.sparse_search(query_emb["sparse"], top_k=top_k * 3)

        logger.debug(
            f"Dense hits: {len(dense_results)}, Sparse hits: {len(sparse_results)}"
        )
        return fuse_and_select([dense_results, sparse_results], top_k, rrf_k)

    def multi_query_search(
        self,
        queries: list[str],
        top_k: int = 10,
        rrf_k: int = 60,
    ) -> list[dict]:
        """多查询扁平 RRF 检索：每个子查询分别 dense+sparse 检索，全部 2×len(queries) 路
        结果一次性喂给 RRF 融合，无中间截断（避免两级 RRF 丢信息）。"""
        all_rankings: list[list[dict]] = []
        for q in queries:
            q_emb = self.embedder.encode_query(q)
            all_rankings.append(self.dense_search(q_emb["dense"], top_k=top_k * 3))
            all_rankings.append(self.sparse_search(q_emb["sparse"], top_k=top_k * 3))

        return fuse_and_select(all_rankings, top_k, rrf_k)


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    """获取 HybridRetriever 全局单例。"""
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
