from dataclasses import dataclass
from enum import Enum

import numpy as np
from langchain_text_splitters import RecursiveCharacterTextSplitter


class ChunkingStrategy(Enum):
    RECURSIVE = "recursive"
    SEMANTIC = "semantic"


class RecursiveChunker:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", "。", ".", "！", "!", "？", "?", "；", ";", " "],
            length_function=len,
        )

    def split_text(self, text: str) -> list[str]:
        docs = self.splitter.create_documents([text])
        return [doc.page_content for doc in docs]


def get_chunker(
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> RecursiveChunker:
    if strategy == ChunkingStrategy.SEMANTIC:
        pass
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
