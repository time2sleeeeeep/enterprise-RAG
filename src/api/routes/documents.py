# 文档管理路由：提供文档上传（POST /upload）、批量导入（POST /bulk-import）、
# 列表查询（GET /）和删除（DELETE /{doc_id}）接口。
# 上传时将文件写入临时目录后触发摄取流水线，删除时同步清理 Milvus 中的向量数据。

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from pymilvus import Collection, utility
from pymilvus.exceptions import MilvusException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from loguru import logger

from src.api.responses import BulkImportItemResult, BulkImportResponse, DeleteResponse, ErrorResponse
from src.config import settings
from src.db.mysql_client import get_db, Document
from src.ingestion.bulk import (
    ALLOWED_EXTENSIONS,
    ALLOWED_ARCHIVE_EXTENSIONS,
    bulk_import,
    collect_files,
    extract_zip,
)
from src.ingestion.pipeline import ingest_document

router = APIRouter(prefix="/documents")

# 本模块各端点公共的错误响应文档
_COMMON_ERRORS: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "请求参数不合法"},
    500: {"model": ErrorResponse, "description": "服务器内部错误"},
}


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


@router.post(
    "/upload",
    response_model=IngestResponse,
    responses={
        **_COMMON_ERRORS,
        413: {"model": ErrorResponse, "description": "上传文件过大"},
    },
)
async def upload_document(file: UploadFile = File(...)):
    """接收上传文件，写入临时目录后触发摄取流水线，完成后删除临时文件。"""
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type: {suffix}. Allowed: {list(ALLOWED_EXTENSIONS)}",
        )

    tmp_dir = Path(tempfile.mkdtemp())
    tmp_path = tmp_dir / file.filename
    try:
        content = await file.read()
        with open(tmp_path, "wb") as f:
            f.write(content)

        result = ingest_document(str(tmp_path))

        return IngestResponse(
            doc_id=result["doc_id"],
            filename=result["filename"],
            chunk_count=result["chunk_count"],
            message=f"Document '{file.filename}' ingested successfully",
        )
    except HTTPException:
        raise
    except FileNotFoundError as e:
        logger.error(f"Temporary file missing for {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="File processing failed")
    except Exception as e:
        logger.error(f"Failed to ingest {file.filename}: {e}")
        raise HTTPException(status_code=500, detail="Document ingestion failed")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.post(
    "/bulk-import",
    response_model=BulkImportResponse,
    responses={
        **_COMMON_ERRORS,
        413: {"model": ErrorResponse, "description": "上传文件过大"},
    },
)
async def bulk_import_documents(
    files: list[UploadFile] = File(..., description="支持多文件上传或单个 zip 压缩包"),
    max_concurrency: int = Form(4, ge=1, le=8, description="并发处理上限（1-8）"),
):
    """批量导入文档。

    上传方式：
    - **多文件上传** — 直接上传多个受支持的文件（.pdf / .md / .docx）
    - **zip 压缩包** — 上传一个 .zip 文件，服务端解压后递归提取其中的支持文件

    每个文件根据扩展名自动匹配处理配置（chunk 大小、重叠量等）。处理的
    文件写入临时目录，完成后清理。
    """
    tmp_dir = Path(tempfile.mkdtemp())
    collected: list[Path] = []
    skipped: list[BulkImportItemResult] = []

    try:
        # 1. 保存所有上传文件到临时目录，区分 zip 与普通文件
        for upload in files:
            filename = upload.filename or "untitled"
            suffix = Path(filename).suffix.lower()

            # zip 压缩包 → 解压后收集
            if suffix in ALLOWED_ARCHIVE_EXTENSIONS:
                content = await upload.read()
                zip_tmp = tmp_dir / filename
                zip_tmp.write_bytes(content)
                extract_dir = tmp_dir / f"_extracted_{Path(filename).stem}"
                extract_dir.mkdir(exist_ok=True)
                extract_zip(zip_tmp, extract_dir)
                for fp in collect_files(extract_dir):
                    collected.append(fp)
                # TODO: 未来可在此处依据 zip 内文件结构进一步分类路由
                continue

            # 普通文件 → 直接保存
            if suffix not in ALLOWED_EXTENSIONS:
                skipped.append(BulkImportItemResult(
                    filename=filename,
                    status="skipped",
                    doc_id=None,
                    chunk_count=None,
                    error=f"Unsupported file type: {suffix}",
                ))
                continue

            content = await upload.read()
            tmp_path = tmp_dir / filename
            tmp_path.write_bytes(content)
            collected.append(tmp_path)

        # 2. 并发调用摄取流水线
        if collected:
            processed = await bulk_import(collected, max_concurrency=max_concurrency)
        else:
            processed = []

        # 3. 汇总结果
        all_results = skipped + processed
        success_count = sum(1 for r in all_results if r["status"] == "success")
        failed_count = sum(1 for r in all_results if r["status"] == "failed")
        skipped_count = sum(1 for r in all_results if r["status"] == "skipped")

        return BulkImportResponse(
            total=len(all_results),
            success_count=success_count,
            failed_count=failed_count,
            skipped_count=skipped_count,
            results=[BulkImportItemResult(**r) for r in all_results],
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Bulk import failed: {e}")
        raise HTTPException(status_code=500, detail="Bulk import failed")
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


@router.get(
    "/",
    response_model=list[DocumentInfo],
    responses={
        500: {"model": ErrorResponse, "description": "数据库查询失败"},
    },
)
def list_documents(db: Session = Depends(get_db)):
    """按创建时间倒序返回所有已摄取文档的元信息列表。"""
    try:
        docs = db.query(Document).order_by(Document.created_at.desc()).all()
    except Exception as e:
        logger.error(f"Failed to query documents: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve documents")

    return [
        DocumentInfo(
            id=doc.id,
            filename=doc.filename,
            file_type=doc.file_type,
            chunk_count=doc.chunk_count,
        )
        for doc in docs
    ]


@router.delete(
    "/{doc_id}",
    response_model=DeleteResponse,
    responses={
        **_COMMON_ERRORS,
        404: {"model": ErrorResponse, "description": "文档不存在"},
    },
)
def delete_document(doc_id: str, db: Session = Depends(get_db)):
    """删除指定文档：同步从 Milvus 删除向量数据，再从 MySQL 删除元数据记录。"""
    doc = db.query(Document).filter(Document.id == doc_id).first()
    if not doc:
        raise HTTPException(status_code=404, detail=f"Document not found: {doc_id}")

    collection_name = settings.milvus_collection
    if utility.has_collection(collection_name):
        try:
            collection = Collection(collection_name)
            collection.load()
            collection.delete(expr=f'doc_id == "{doc_id}"')
            collection.flush()
        except MilvusException as e:
            logger.error(f"Milvus delete failed for doc {doc_id}: {e}")
            raise HTTPException(status_code=500, detail="Failed to delete vector data")

    try:
        db.delete(doc)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"MySQL delete failed for doc {doc_id}: {e}")
        raise HTTPException(status_code=500, detail="Failed to delete document metadata")

    return DeleteResponse(
        message=f"Document '{doc.filename}' deleted",
        doc_id=doc_id,
    )
