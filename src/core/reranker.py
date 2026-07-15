# 重排序模块：封装 BGE Reranker 模型（单例），对检索召回的文档按查询相关性重新打分排序。

import threading

import numpy as np
from loguru import logger
from FlagEmbedding import FlagReranker

from src.config import settings


class BGEReranker:
    """BGE Reranker 模型单例封装，对 (query, document) 对计算相关性分数。"""

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        """确保全局只初始化一次重排序模型（单例模式）。"""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        """加载 BGE Reranker 模型到指定设备，已初始化则直接返回。"""
        if self._initialized:
            return
        with BGEReranker._lock:
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
        """对文档列表按 query 相关性重新打分，返回分值最高的 top_k 个文档。"""
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
    """获取 BGEReranker 全局单例。"""
    global _reranker
    if _reranker is None:
        _reranker = BGEReranker()
    return _reranker
