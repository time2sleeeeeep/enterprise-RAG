# 文档管理路由：提供文档上传（POST /upload）、批量导入（POST /bulk-import）、
# 列表查询（GET /）和删除（DELETE /{doc_id}）接口。
# 上传时将文件写入临时目录后触发摄取流水线，删除时同步清理 Milvus 中的向量数据。

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Form, UploadFile, HTTPException
from pymilvus import Collection, utility
from pymilvus.exceptions import MilvusException
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session
from loguru import logger

from src.api.responses import BulkDeleteResponse, BulkImportItemResult, BulkImportResponse, DeleteResponse, ErrorResponse
from src.config import settings
from src.db.milvus_client import delete_doc_vectors
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


class BulkDeleteRequest(BaseModel):
    doc_ids: list[str] = Field(..., min_length=1, description="要删除的文档 ID 列表")


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
    files: list[UploadFile] = File(..., description="多文件 / 文件夹 / zip 压缩包"),
    max_concurrency: int = Form(4, ge=1, le=8, description="并发处理上限（1-8）"),
):
    """批量导入文档，**支持直接上传整个文件夹**。

    上传方式：
    - **文件夹上传** — 浏览器 `<input webkitdirectory>` 或客户端将文件夹内
      所有文件打包为 multipart 请求，`filename` 携带相对路径（如
      `docs/chapter1/report.pdf`），服务端自动重建目录结构。
    - **多文件上传** — 同时选择多个受支持的文件（.pdf / .md / .docx）。
    - **zip 压缩包** — 上传一个 .zip 文件，服务端解压后递归提取支持文件。

    每个文件根据扩展名自动匹配处理配置（chunk 大小、重叠量等）。所有
    文件写入临时目录，完成后清理。
    """
    tmp_dir = Path(tempfile.mkdtemp())
    collected: list[Path] = []
    skipped: list[BulkImportItemResult] = []

    def _safe_relative(raw: str) -> Path:
        """将上传的 filename 转为安全的相对路径，防止路径穿越攻击。"""
        # 标准化路径分隔符，剔除绝对路径前缀和 `..` 穿越
        normalized = raw.replace("\\", "/").lstrip("/")
        parts = [p for p in normalized.split("/") if p and p != ".."]
        if not parts:
            return Path("untitled")
        return Path(*parts)

    try:
        # 1. 保存所有上传文件到临时目录，区分 zip 与普通文件
        for upload in files:
            raw_filename = upload.filename or "untitled"
            suffix = Path(raw_filename).suffix.lower()

            # zip 压缩包 → 解压后收集
            if suffix in ALLOWED_ARCHIVE_EXTENSIONS:
                content = await upload.read()
                zip_tmp = tmp_dir / Path(raw_filename).name
                zip_tmp.write_bytes(content)
                extract_dir = tmp_dir / f"_extracted_{Path(raw_filename).stem}"
                extract_dir.mkdir(exist_ok=True)
                extract_zip(zip_tmp, extract_dir)
                for fp in collect_files(extract_dir):
                    collected.append(fp)
                continue

            # 不支持的类型 → 跳过
            if suffix not in ALLOWED_EXTENSIONS:
                skipped.append(BulkImportItemResult(
                    filename=raw_filename,
                    status="skipped",
                    doc_id=None,
                    chunk_count=None,
                    error=f"Unsupported file type: {suffix}",
                ))
                continue

            # 普通文件 → 写入临时目录（保留相对路径结构）
            content = await upload.read()
            rel_path = _safe_relative(raw_filename)
            tmp_path = tmp_dir / rel_path
            tmp_path.parent.mkdir(parents=True, exist_ok=True)
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

    try:
        delete_doc_vectors(doc_id)
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


# Milvus `in` 表达式单次最大 ID 数量（避免表达式过长导致解析失败）
_MILVUS_IN_BATCH_SIZE = 200


@router.delete(
    "/",
    response_model=BulkDeleteResponse,
    responses={
        **_COMMON_ERRORS,
        404: {"model": ErrorResponse, "description": "部分或全部文档不存在"},
    },
)
def bulk_delete_documents(body: BulkDeleteRequest, db: Session = Depends(get_db)):
    """批量删除文档：一次性清理 Milvus 向量和 MySQL 元数据，显著减少网络往返和磁盘刷新次数。"""
    requested = body.doc_ids

    # 1. 查询 MySQL 中实际存在的文档
    existing_docs = db.query(Document).filter(Document.id.in_(requested)).all()
    existing_ids = {doc.id for doc in existing_docs}
    not_found = [doc_id for doc_id in requested if doc_id not in existing_ids]

    if not existing_ids:
        return BulkDeleteResponse(
            total_requested=len(requested),
            deleted_count=0,
            not_found=not_found,
            message="No requested documents found in database",
        )

    # 2. 从 Milvus 批量删除向量数据（一次 load + 分批 or 表达式 + 一次 flush）
    collection_name = settings.milvus_collection
    milvus_deleted = 0
    milvus_error = None
    if utility.has_collection(collection_name):
        try:
            collection = Collection(collection_name)
            collection.load()
            ids_list = list(existing_ids)
            for i in range(0, len(ids_list), _MILVUS_IN_BATCH_SIZE):
                batch = ids_list[i : i + _MILVUS_IN_BATCH_SIZE]
                # 使用 or 连接多个 == 条件，Milvus 原生支持且兼容性最好
                or_clause = " or ".join(f'doc_id == "{doc_id}"' for doc_id in batch)
                result = collection.delete(expr=or_clause)
                milvus_deleted += getattr(result, "delete_count", 0)
            collection.flush()
            logger.info(f"Bulk deleted {milvus_deleted} Milvus entries for {len(existing_ids)} docs")
        except MilvusException as e:
            logger.error(f"Milvus bulk delete failed: {e}")
            milvus_error = str(e)

    # 3. 从 MySQL 批量删除元数据（单事务）
    try:
        db.query(Document).filter(Document.id.in_(list(existing_ids))).delete(
            synchronize_session=False
        )
        db.commit()
    except Exception as e:
        db.rollback()
        logger.error(f"MySQL bulk delete failed: {e}")
        raise HTTPException(
            status_code=500,
            detail="Failed to delete document metadata",
        )

    # 4. 构造响应（Milvus 异常不阻断 MySQL 删除，尽力而为）
    message = f"Deleted {len(existing_ids)} documents"
    if not_found:
        message += f"; {len(not_found)} IDs not found"
    if milvus_error:
        message += f" (Milvus cleanup incomplete: {milvus_error[:120]})"

    return BulkDeleteResponse(
        total_requested=len(requested),
        deleted_count=len(existing_ids),
        not_found=not_found,
        message=message,
    )
