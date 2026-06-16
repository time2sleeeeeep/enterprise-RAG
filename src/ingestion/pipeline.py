import hashlib
from pathlib import Path

from loguru import logger

from src.config import settings
from src.core.chunking import ChunkingStrategy, get_chunker
from src.core.embeddings import get_embedder
from src.db.milvus_client import connect_milvus, create_collection
from src.db.mysql_client import SessionLocal, Document
from src.ingestion.loader import load_document
from src.ingestion.parser import clean_text, extract_page_markers


def generate_chunk_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}_{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


def ingest_document(
    file_path: str,
    strategy: ChunkingStrategy = ChunkingStrategy.RECURSIVE,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict:
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")

    logger.info(f"Ingesting document: {path.name}")

    loaded = load_document(path)
    if not loaded.content.strip():
        raise ValueError(f"No content extracted from {file_path}")

    pages = extract_page_markers(loaded.content)
    logger.info(f"Loaded {len(pages)} page(s) from {path.name}")

    chunker = get_chunker(strategy, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []
    for page_num, page_content in pages:
        cleaned = clean_text(page_content)
        if not cleaned.strip():
            continue
        texts = chunker.split_text(cleaned)
        for text in texts:
            all_chunks.append({
                "content": text,
                "source": path.name,
                "page_num": page_num,
            })

    if not all_chunks:
        raise ValueError(f"No chunks generated from {file_path}")
    logger.info(f"Generated {len(all_chunks)} chunks")

    doc_id = hashlib.md5(path.name.encode()).hexdigest()[:16]

    embedder = get_embedder()
    texts = [c["content"] for c in all_chunks]
    embeddings = embedder.encode(texts, return_sparse=True)

    connect_milvus()
    collection = create_collection(settings.milvus_collection)

    entities = []
    for i, chunk in enumerate(all_chunks):
        entities.append({
            "id": generate_chunk_id(doc_id, i),
            "doc_id": doc_id,
            "chunk_id": i,
            "content": chunk["content"][:8192],
            "source": chunk["source"],
            "page_num": chunk["page_num"],
            "dense_vector": embeddings["dense"][i].tolist(),
            "sparse_vector": embeddings["sparse"][i],
        })

    batch_size = 100
    total_inserted = 0
    for start in range(0, len(entities), batch_size):
        end = min(start + batch_size, len(entities))
        batch = entities[start:end]
        collection.insert(batch)
        total_inserted += end - start
        logger.info(f"Inserted batch {start}-{end} ({total_inserted}/{len(entities)})")

    collection.flush()
    logger.info(f"Flushed {total_inserted} vectors to Milvus")

    db = SessionLocal()
    try:
        doc_record = db.query(Document).filter(Document.id == doc_id).first()
        if doc_record:
            doc_record.chunk_count = len(all_chunks)
        else:
            doc_record = Document(
                id=doc_id,
                filename=path.name,
                file_type=path.suffix.lstrip("."),
                chunk_count=len(all_chunks),
            )
            db.add(doc_record)
        db.commit()
    finally:
        db.close()

    logger.info(f"Document '{path.name}' ingested: {len(all_chunks)} chunks stored")
    return {
        "doc_id": doc_id,
        "filename": path.name,
        "chunk_count": len(all_chunks),
        "strategy": strategy.value,
    }
