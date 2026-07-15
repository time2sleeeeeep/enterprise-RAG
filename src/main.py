# FastAPI 应用入口：初始化服务、注册路由、配置 CORS 和全局异常处理。
# 启动时连接 MySQL/Milvus，关闭时断开连接并清理 Redis 连接池。

import asyncio
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger
from starlette.exceptions import HTTPException as StarletteHTTPException

from src.api.health import check_health
from src.api.responses import ErrorResponse, HealthResponse
from src.api.routes import chat, documents, eval
from src.config import settings
from src.db.mysql_client import init_db
from src.db.milvus_client import connect_milvus, disconnect_milvus
from src.db.redis_client import close_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期管理：启动时初始化 DB、连接 Milvus、预热加载 collection，关闭时释放资源。

    Collection 在启动时加载（而非首次请求延迟加载）可避免冷启动请求遭遇
    Milvus 加载延迟。加载操作通过 asyncio.to_thread 卸载到线程池，不阻塞事件循环。[MILVUS_EMPTY_LOAD]
    """
    logger.info("Starting Enterprise RAG service...")
    init_db()
    connect_milvus()

    # 预热：加载 Milvus collection，避免首次请求的冷启动延迟
    try:
        from src.db.milvus_client import create_collection
        from src.config import settings

        def _warmup():
            col = create_collection(settings.milvus_collection)
            if col.num_entities > 0:
                col.load()
                logger.info(
                    f"Collection '{settings.milvus_collection}' loaded "
                    f"({col.num_entities} entities)"
                )
            else:
                logger.warning(
                    f"Collection '{settings.milvus_collection}' is empty — "
                    "search will return empty results until documents are ingested"
                )

        await asyncio.to_thread(_warmup)
    except Exception as e:
        logger.error(f"Collection warmup failed (service will load lazily): {e}")

    yield
    disconnect_milvus()
    await close_redis()
    logger.info("Shutting down Enterprise RAG service...")


app = FastAPI(
    title="Enterprise RAG Knowledge Base",
    description="Production-grade RAG pipeline with hybrid retrieval and reranking",
    version="0.1.0",
    lifespan=lifespan,
)

_cors_origins = settings.cors_origins
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=("*" not in _cors_origins),  # 通配源时关 credentials，符合 CORS 规范
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# 全局异常处理器
# ---------------------------------------------------------------------------


@app.exception_handler(StarletteHTTPException)
async def http_exception_handler(request: Request, exc: StarletteHTTPException):
    """统一格式化 HTTPException 响应，使 OpenAPI 可正确生成 ErrorResponse schema。"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail} — {request.method} {request.url.path}")
    return JSONResponse(
        status_code=exc.status_code,
        content=ErrorResponse(
            detail=str(exc.detail),
            message=None,
        ).model_dump(),
    )


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    """统一格式化请求校验错误，返回 422 和友好的错误详情。"""
    logger.warning(f"Validation error on {request.method} {request.url.path}: {exc.errors()}")
    return JSONResponse(
        status_code=422,
        content={"detail": exc.errors()},
    )


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """捕获所有未处理的内部异常，返回 500 且不泄露内部错误细节。"""
    logger.error(f"Unhandled exception on {request.method} {request.url.path}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content=ErrorResponse(
            detail="Internal server error",
            message=None,
        ).model_dump(),
    )


# ---------------------------------------------------------------------------
# 路由注册
# ---------------------------------------------------------------------------


@app.get(
    "/health",
    response_model=HealthResponse,
    responses={
        500: {"model": ErrorResponse, "description": "服务不可用"},
        503: {"model": HealthResponse, "description": "依赖服务不可用"},
    },
)
async def health_check():
    """健康检查：探活 Milvus/MySQL/Redis，任一不可用返回 503 degraded。"""
    result = await check_health()
    status = "healthy" if result["all_up"] else "degraded"
    body = HealthResponse(
        status=status, service="enterprise-rag", dependencies=result["dependencies"]
    )
    if result["all_up"]:
        return body
    return JSONResponse(status_code=503, content=body.model_dump())


app.include_router(chat.router, prefix="/api/v1", tags=["Chat"])
app.include_router(documents.router, prefix="/api/v1", tags=["Documents"])
app.include_router(eval.router, prefix="/api/v1", tags=["Evaluation"])


if __name__ == "__main__":
    import uvicorn
    from src.config import settings

    uvicorn.run(
        "src.main:app",
        host=settings.server_host,
        port=settings.server_port,
        reload=True,
    )
