from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from loguru import logger

from src.api.routes import chat, documents, eval
from src.db.mysql_client import init_db
from src.db.milvus_client import connect_milvus, disconnect_milvus
from src.db.redis_client import close_redis


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Starting Enterprise RAG service...")
    init_db()
    connect_milvus()
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

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error", "message": str(exc)},
    )


@app.get("/health")
async def health_check():
    return {"status": "healthy", "service": "enterprise-rag"}


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
