# 文本分块模块：提供递归字符切分（RecursiveChunker）和语义切分（semantic_chunk）两种策略。
# 递归切分按标点/换行符层级拆分；语义切分根据相邻句子嵌入余弦相似度决定是否合并。

import re
from dataclasses import dataclass
from enum import Enum

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingStrategy(Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class RecursiveChunker:
    """基于 LangChain RecursiveCharacterTextSplitter 的递归分块器，支持中英文标点。"""

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " "],
            length_function=len,
        )

    def split_text(self, text: str) -> list[str]:
        """将输入文本按配置切分，返回文本片段列表。"""
        docs = self.splitter.create_documents([text])
        return [doc.page_content for doc in docs]


class SemanticChunker:
    """语义分块器：正则切句 → 批量编码句子嵌入 → 按相邻句余弦相似度合并。

    与 RecursiveChunker 拥有统一的 split_text(self, text) -> list[str] 接口，
    通过 get_chunker(ChunkingStrategy.SEMANTIC) 创建。
    """

    def __init__(
        self,
        chunk_size: int = 512,
        chunk_overlap: int = 64,
        threshold: float = 0.75,
    ):
        # chunk_size / chunk_overlap 保留以统一接口，但语义分块不受严格 size 约束
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.threshold = threshold
        self._embedder = None  # 懒加载，与 pipeline 共用 get_embedder 单例

    def _get_embedder(self):
        """懒加载 BGEm3Embedder 单例，避免重复加载模型。"""
        if self._embedder is None:
            from src.core.embeddings import get_embedder

            self._embedder = get_embedder()
        return self._embedder

    def split_text(self, text: str) -> list[str]:
        """对文本做语义分块：切句 → 批量编码 → 相邻句相似度合并 → 返回文本块列表。"""
        if not text.strip():
            return []

        # 1. 正则切句（中英文标点，与 RecursiveChunker 分隔符一致，不引入 nltk）
        sentence_pat = re.compile(r"(?<=[。！？；.!?;])\s*")
        raw_sentences = [s.strip() for s in sentence_pat.split(text) if s.strip()]
        if len(raw_sentences) <= 1:
            return [text]

        # 2. 批量编码句子嵌入（GPU，复用 pipeline 已加载的 embedder）
        embedder = self._get_embedder()
        emb_result = embedder.encode(raw_sentences, return_sparse=False)
        embeddings = emb_result["dense"]

        # 3. 按相邻句余弦相似度合并（复用既有 semantic_chunk 函数）
        chunks = semantic_chunk(
            text, embeddings, raw_sentences, threshold=self.threshold
        )

        return [c.content for c in chunks]


def get_chunker(
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> "RecursiveChunker | SemanticChunker":
    """根据策略类型返回对应的分块器实例。SEMANTIC 返回 SemanticChunker，否则返回 RecursiveChunker。"""
    if strategy == ChunkingStrategy.SEMANTIC:
        return SemanticChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    return RecursiveChunker(chunk_size=chunk_size, chunk_overlap=chunk_overlap)


@dataclass
class TextChunk:
    content: str
    chunk_id: int
    page_num: int
    start_char: int
    end_char: int


def recursive_chunk(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    page_num: int = 0,
) -> list[TextChunk]:
    """对单段文本做递归字符分块，返回带字符偏移量的 TextChunk 列表。"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " "],
        length_function=len,
    )
    docs = splitter.create_documents([text])
    chunks = []
    offset = 0
    for i, doc in enumerate(docs):
        content = doc.page_content
        start = text.find(content, offset)
        if start == -1:
            start = offset
        end = start + len(content)
        offset = start + 1
        chunks.append(TextChunk(
            content=content,
            chunk_id=i,
            page_num=page_num,
            start_char=start,
            end_char=end,
        ))
    return chunks


def semantic_chunk(
    text: str,
    embeddings: np.ndarray,
    sentences: list[str],
    threshold: float = 0.75,
    page_num: int = 0,
) -> list[TextChunk]:
    """Group consecutive sentences by cosine similarity above threshold."""
    if len(sentences) == 0:
        return []

    chunks = []
    current_sentences = [sentences[0]]
    chunk_id = 0

    for i in range(1, len(sentences)):
        sim = float(np.dot(embeddings[i - 1], embeddings[i]) / (
            np.linalg.norm(embeddings[i - 1]) * np.linalg.norm(embeddings[i]) + 1e-8
        ))
        if sim >= threshold:
            current_sentences.append(sentences[i])
        else:
            content = "\n".join(current_sentences)
            start = text.find(current_sentences[0])
            chunks.append(TextChunk(
                content=content,
                chunk_id=chunk_id,
                page_num=page_num,
                start_char=max(0, start),
                end_char=start + len(content) if start >= 0 else len(content),
            ))
            chunk_id += 1
            current_sentences = [sentences[i]]

    if current_sentences:
        content = "\n".join(current_sentences)
        start = text.find(current_sentences[0])
        chunks.append(TextChunk(
            content=content,
            chunk_id=chunk_id,
            page_num=page_num,
            start_char=max(0, start),
            end_char=start + len(content) if start >= 0 else len(content),
        ))

    return chunks
