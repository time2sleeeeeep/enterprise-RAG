import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, UploadFile, HTTPException
from pymilvus import Collection, utility
from pydantic import BaseModel
from sqlalchemy.orm import Session
from loguru import logger

from src.config import settings
from src.db.mysql_client import get_db, Document
from src.ingestion.pipeline import ingest_document

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".md", ".markdown", ".docx"}


class DocumentInfo(BaseModel):
    id: str
    filename: str
    file_type: str
    chunk_count: int


class IngestResponse(BaseModel):
    doc_id: str
    filename: str
    chunk_count: int
    message: str


@router.post("/upload", response_model=IngestResponse)
async def upload_document(file: UploadFile = File(...)):
    suffix = Path(file.filename).suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {list(ALLOWED_EXTENSIONS)}",
        )

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        with open(tmp_path, "wb") as f:
            content = await file.read()
            f.write(content)

        result = ingest_document(str(tmp_path))

        return IngestResponse(
            doc_id=result["doc_id"],
            filename=result["filename"],
            chunk_count=result["chunk_count"],
            message=f"Document '{file.filename}' ingested successfully",
        )
    except Exception as e:
        logger.error(f"Failed to ingest {file.filename}: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get("/", response_model=list[DocumentInfo])
def list_documents(db: Session = Depends(get_db)):
    docs = db.query(Document).order_by(Document.created_at.desc()).all()
    return [
        DocumentInfo(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            chunk_count=doc.chunk_count,
        )
        for doc in docs
    ]


@router.delete("/{doc_id}")
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail="Document not found")

    collection_name = settings.milvus_collection
    if utility.has_collection(collection_name):
        collection = Collection(collection_name)
        collection.load()
        collection.delete(expr=f'doc_id == "{doc_id}"')
        collection.flush()

    db.delete(doc)
    db.commit()

    return {"message": f"Document '{doc.filename}' deleted", "doc_id": doc_id}
