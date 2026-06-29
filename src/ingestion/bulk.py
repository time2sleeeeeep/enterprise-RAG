# 批量导入模块：支持 zip 压缩包解压、多文件目录扫描、按扩展名分类路由到不同处理配置，
# 并通过线程池并发调用 ingest_document 流水线，汇总每个文件的处理结果。

import asyncio
import zipfile
from dataclasses import dataclass
from pathlib import Path

from loguru import logger

from src.ingestion.pipeline import ingest_document
from src.ingestion.loader import LOADERS

# ---------------------------------------------------------------------------
# 处理配置（可按扩展名定制 chunk 策略 / 大小 / 重叠量，后续可扩展模型选择等）
# ---------------------------------------------------------------------------

@dataclass
class ProcessingProfile:
    """某类文件的摄取参数配置。"""
    strategy: str = "recursive"
    chunk_size: int = 512
    chunk_overlap: int = 64

# 按扩展名映射处理配置，未匹配到的后缀使用 _DEFAULT
PROCESSING_PROFILES: dict[str, ProcessingProfile] = {
    ".pdf": ProcessingProfile(strategy="recursive", chunk_size=512, chunk_overlap=64),
    ".md": ProcessingProfile(strategy="recursive", chunk_size=512, chunk_overlap=64),
    ".markdown": ProcessingProfile(strategy="recursive", chunk_size=512, chunk_overlap=64),
    ".docx": ProcessingProfile(strategy="recursive", chunk_size=512, chunk_overlap=64),
}
_DEFAULT_PROFILE = ProcessingProfile()

# 支持的文件后缀（与 LOADERS 保持一致，额外允许 .zip 作为压缩包入口）
ALLOWED_EXTENSIONS = set(LOADERS.keys())
ALLOWED_ARCHIVE_EXTENSIONS = {".zip"}

# 忽略的隐藏文件/目录前缀
_IGNORED_PREFIXES = (".", "__MACOSX")

# 并发摄取上限
_DEFAULT_MAX_CONCURRENCY = 4


# ---------------------------------------------------------------------------
# 工具函数
# ---------------------------------------------------------------------------

def get_profile(extension: str) -> ProcessingProfile:
    """根据小写扩展名返回对应的处理配置，未知类型返回默认配置。"""
    return PROCESSING_PROFILES.get(extension, _DEFAULT_PROFILE)


def is_supported(file_path: Path) -> bool:
    """检查文件是否可被摄取（扩展名在 LOADERS 注册表中）。"""
    return file_path.suffix.lower() in ALLOWED_EXTENSIONS


def _should_skip(path: Path) -> bool:
    """跳过隐藏文件/目录和 macOS 资源叉目录。"""
    return any(part.startswith(_IGNORED_PREFIXES) for part in path.parts)


# ---------------------------------------------------------------------------
# Zip 解压
# ---------------------------------------------------------------------------

def extract_zip(zip_path: Path, target_dir: Path) -> list[Path]:
    """解压 zip 文件到目标目录，返回所有解压出的 *文件* 路径列表（已过滤目录和隐藏项）。"""
    extracted: list[Path] = []
    with zipfile.ZipFile(str(zip_path), "r") as zf:
        for member in zf.infolist():
            if member.is_dir():
                continue
            member_path = target_dir / member.filename
            if _should_skip(member_path):
                continue
            member_path.parent.mkdir(parents=True, exist_ok=True)
            zf.extract(member, str(target_dir))
            extracted.append(member_path)
    logger.info(f"Extracted {len(extracted)} file(s) from {zip_path.name}")
    return extracted


def collect_files(root_dir: Path) -> list[Path]:
    """递归扫描目录，收集所有受支持的文件（跳过隐藏目录/文件），返回绝对路径列表。"""
    collected: list[Path] = []
    for entry in root_dir.rglob("*"):
        if entry.is_file() and not _should_skip(entry) and is_supported(entry):
            collected.append(entry.resolve())
    logger.info(f"Collected {len(collected)} supported file(s) from {root_dir}")
    return collected


# ---------------------------------------------------------------------------
# 单文件处理（同步，在线程池中执行）
# ---------------------------------------------------------------------------

def _process_one(file_path: Path, profile: ProcessingProfile) -> dict:
    """对单个文件调用摄取流水线，始终返回 dict（成功/失败均不抛异常）。"""
    try:
        result = ingest_document(
            str(file_path),
            strategy=profile.strategy,       # type: ignore[arg-type]
            chunk_size=profile.chunk_size,
            chunk_overlap=profile.chunk_overlap,
        )
        return {
            "filename": file_path.name,
            "status": "success",
            "doc_id": result["doc_id"],
            "chunk_count": result["chunk_count"],
            "error": None,
        }
    except Exception as exc:
        logger.error(f"Bulk ingest failed for {file_path.name}: {exc}")
        return {
            "filename": file_path.name,
            "status": "failed",
            "doc_id": None,
            "chunk_count": None,
            "error": str(exc),
        }


# ---------------------------------------------------------------------------
# 批量导入编排（异步）
# ---------------------------------------------------------------------------

async def bulk_import(
    file_paths: list[Path],
    max_concurrency: int = _DEFAULT_MAX_CONCURRENCY,
) -> list[dict]:
    """并发处理文件列表，每文件通过线程池执行同步摄取流水线。

    返回值按原始 file_paths 顺序排列，每个元素为包含 filename/status/doc_id/chunk_count/error 的 dict。
    """
    semaphore = asyncio.Semaphore(max_concurrency)

    async def run_one(path: Path) -> dict:
        profile = get_profile(path.suffix.lower())
        async with semaphore:
            return await asyncio.to_thread(_process_one, path, profile)

    tasks = [run_one(p) for p in file_paths]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    # 兜底：将 gather 产生的异常对象转换为统一 dict
    safe_results: list[dict] = []
    for path, res in zip(file_paths, results):
        if isinstance(res, Exception):
            logger.error(f"Bulk ingest task crashed for {path.name}: {res}")
            safe_results.append({
                "filename": path.name,
                "status": "failed",
                "doc_id": None,
                "chunk_count": None,
                "error": str(res),
            })
        else:
            safe_results.append(res)

    return safe_results
