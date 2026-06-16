import numpy as np
from loguru import logger
from pymilvus import Collection

from src.config import settings
from src.core.embeddings import get_embedder
from src.db.milvus_client import create_collection


def reciprocal_rank_fusion(
    rankings: list[list[tuple[str, float]]],
    k: int = 60,
) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, (doc_id, _) in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return fused


class HybridRetriever:
    def __init__(self):
        self.embedder = get_embedder()
        self.collection: Collection | None = None

    def _get_collection(self) -> Collection:
        if self.collection is None:
            self.collection = create_collection(settings.milvus_collection)
            self.collection.load()
        return self.collection

    def dense_search(
        self, query_embedding: np.ndarray, top_k: int = 20
    ) -> list[tuple[str, float]]:
        collection = self._get_collection()
        results = collection.search(
            data=[query_embedding.tolist()],
            anns_field="dense_vector",
            param={"metric_type": "COSINE", "params": {"ef": 128}},
            limit=top_k,
            output_fields=["id", "content", "source", "page_num", "doc_id"],
        )
        hits = []
        for hit in results[0]:
            hits.append((hit.id, hit.score))
        return hits

    def sparse_search(
        self, query_sparse: dict, top_k: int = 20
    ) -> list[tuple[str, float]]:
        collection = self._get_collection()
        results = collection.search(
            data=[query_sparse],
            anns_field="sparse_vector",
            param={"metric_type": "IP"},
            limit=top_k,
            output_fields=["id", "content", "source", "page_num", "doc_id"],
        )
        hits = []
        for hit in results[0]:
            hits.append((hit.id, hit.score))
        return hits

    def hybrid_search(
        self,
        query: str,
        top_k: int = 10,
        dense_weight: float = 0.7,
        sparse_weight: float = 0.3,
        rrf_k: int = 60,
    ) -> list[dict]:
        query_emb = self.embedder.encode_query(query)
        dense_vector = query_emb["dense"]
        sparse_vector = query_emb["sparse"]

        dense_results = self.dense_search(dense_vector, top_k=top_k * 3)
        sparse_results = self.sparse_search(sparse_vector, top_k=top_k * 3)

        logger.debug(f"Dense hits: {len(dense_results)}, Sparse hits: {len(sparse_results)}")

        fused = reciprocal_rank_fusion([dense_results, sparse_results], k=rrf_k)
        top_ids = [doc_id for doc_id, _ in fused[:top_k]]

        collection = self._get_collection()
        docs = collection.query(
            expr=f'id in {top_ids}',
            output_fields=["id", "content", "source", "page_num", "doc_id"],
        )

        id_to_doc = {d["id"]: d for d in docs}
        results = []
        for doc_id, score in fused[:top_k]:
            if doc_id in id_to_doc:
                doc = id_to_doc[doc_id]
                results.append({
                    "id": doc["id"],
                    "content": doc["content"],
                    "source": doc["source"],
                    "page_num": doc["page_num"],
                    "doc_id": doc["doc_id"],
                    "score": score,
                })
        return results


_retriever: HybridRetriever | None = None


def get_retriever() -> HybridRetriever:
    global _retriever
    if _retriever is None:
        _retriever = HybridRetriever()
    return _retriever
