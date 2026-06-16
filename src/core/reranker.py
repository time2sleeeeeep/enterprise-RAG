import numpy as np
from loguru import logger
from FlagEmbedding import FlagReranker

from src.config import settings


class BGEReranker:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info(f"Loading reranker model on {settings.reranker_device}...")
        self.model = FlagReranker(
            settings.reranker_model_name,
            use_fp16=(settings.reranker_device == "cuda"),
            device=settings.reranker_device,
        )
        self._initialized = True
        logger.info("Reranker model loaded successfully")

    def rerank(
        self,
        query: str,
        documents: list[dict],
        top_k: int = 5,
    ) -> list[dict]:
        if not documents:
            return []

        pairs = [[query, doc["content"]] for doc in documents]
        scores = self.model.compute_score(pairs, normalize=True)

        if isinstance(scores, float):
            scores = [scores]

        for i, doc in enumerate(documents):
            doc["rerank_score"] = float(scores[i])

        reranked = sorted(documents, key=lambda x: x["rerank_score"], reverse=True)
        return reranked[:top_k]


_reranker: BGEReranker | None = None


def get_reranker() -> BGEReranker:
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker
