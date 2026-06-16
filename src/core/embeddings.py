import torch
import numpy as np
from FlagEmbedding import BGEM3FlagModel
from loguru import logger

from src.config import settings


class BGEm3Embedder:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        logger.info(f"Loading bge-m3 model on {settings.embedding_device}...")
        self.model = BGEM3FlagModel(
            settings.embedding_model_name,
            use_fp16=(settings.embedding_device == "cuda"),
            device=settings.embedding_device,
        )
        self._initialized = True
        logger.info("bge-m3 model loaded successfully")

    def encode(
        self,
        texts: list[str],
        batch_size: int | None = None,
        return_sparse: bool = True,
    ) -> dict[str, np.ndarray | list[dict]]:
        batch_size = batch_size or settings.embedding_batch_size
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            return_dense=True,
            return_sparse=return_sparse,
            return_colbert_vecs=False,
        )
        result = {"dense": np.array(output["dense_vecs"])}
        if return_sparse and "lexical_weights" in output:
            result["sparse"] = output["lexical_weights"]
        return result

    def encode_query(self, query: str) -> dict[str, np.ndarray | dict]:
        output = self.model.encode(
            [query],
            batch_size=1,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense": np.array(output["dense_vecs"][0]),
            "sparse": output["lexical_weights"][0] if "lexical_weights" in output else {},
        }


def get_embedder() -> BGEm3Embedder:
    return BGEm3Embedder()
