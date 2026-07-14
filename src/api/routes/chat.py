# 对话路由：提供 POST /chat 接口，实现完整的 RAG 问答流程。
# 依次执行：查询缓存 -> 查询改写 -> 混合检索 -> 重排序 -> LLM 生成 -> 结果缓存 -> 写入聊天历史。

import asyncio
import json
import uuid

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session
from loguru import logger

from src.api.responses import ErrorResponse
from src.config import settings
from src.core.cache import get_cached_answer, set_cached_answer
from src.core.generator import generate_answer
from src.core.query_rewriter import expand_query, rewrite_query
from src.core.reranker import get_reranker
from src.core.retriever import get_retriever
from src.db.mysql_client import ChatHistory, get_db

router = APIRouter(prefix="/chat")


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    top_k: int = 5
    use_reranker: bool = True
    use_cache: bool = True
    use_query_rewrite: bool = False
    use_multi_query: bool = False


class SourceDoc(BaseModel):
    content: str
    source: str
    page_num: int
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceDoc]
    session_id: str
    cached: bool = False


@router.post(
    "/",
    response_model=ChatResponse,
    responses={
        400: {"model": ErrorResponse, "description": "请求参数不合法"},
        500: {"model": ErrorResponse, "description": "检索或生成失败"},
        503: {"model": ErrorResponse, "description": "依赖服务不可用（Milvus / LLM）"},
    },
)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """处理问答请求：查缓存 -> 改写查询 -> 混合检索 -> 重排序 -> LLM 生成 -> 缓存结果 -> 写历史。"""
    session_id = request.session_id or str(uuid.uuid4())[:16]

    # 1. 查询缓存
    if request.use_cache:
        try:
            cached = await get_cached_answer(request.question)
            if cached:
                return ChatResponse(
                    answer=cached["answer"],
                    sources=[SourceDoc(**s) for s in cached["sources"]],
                    session_id=session_id,
                    cached=True,
                )
        except Exception as e:
            logger.warning(f"Cache read failed, continuing without cache: {e}")

    # 2. 查询改写（同步 LLM 调用，丢入线程池避免阻塞事件循环）
    query = request.question
    if request.use_query_rewrite:
        try:
            query = await asyncio.to_thread(rewrite_query, request.question)
        except Exception as e:
            logger.warning(f"Query rewrite failed, using original query: {e}")

    # 3. 混合检索（pymilvus 同步 + torch 编码，丢入线程池避免阻塞事件循环；
    #    use_multi_query 时先扩展子查询再扁平 RRF 融合全部结果）
    try:
        retriever = get_retriever()
        if request.use_multi_query:
            sub_queries = await asyncio.to_thread(expand_query, query)
            documents = await asyncio.to_thread(
                retriever.multi_query_search, sub_queries, top_k=request.top_k * 3
            )
        else:
            documents = await asyncio.to_thread(
                retriever.hybrid_search, query, top_k=request.top_k * 3
            )
    except Exception as e:
        logger.error(f"Hybrid search failed: {e}")
        raise HTTPException(status_code=503, detail="Search service unavailable")

    # 4. 重排序（torch 推理，丢入线程池避免阻塞事件循环）
    if request.use_reranker and documents:
        try:
            reranker = get_reranker()
            documents = await asyncio.to_thread(
                reranker.rerank,
                query=request.question,
                documents=documents,
                top_k=request.top_k,
            )
        except Exception as e:
            logger.warning(f"Reranker failed, continuing without reranking: {e}")
            documents = documents[: request.top_k]
    else:
        documents = documents[: request.top_k]

    # 5. 加载历史对话
    history_messages: list[dict] = []
    if request.session_id and settings.chat_history_max_turns > 0:
        try:
            records = (
                db.query(ChatHistory)
                .filter(ChatHistory.session_id == request.session_id)
                .order_by(ChatHistory.created_at.desc())
                .limit(settings.chat_history_max_turns)
                .all()
            )
            for r in reversed(records):
                history_messages.append({"role": "user", "content": r.question})
                history_messages.append({"role": "assistant", "content": r.answer})
        except Exception as e:
            logger.warning(f"Failed to load chat history: {e}")

    # 6. LLM 生成（同步 LLM 调用，丢入线程池避免阻塞事件循环）
    try:
        result = await asyncio.to_thread(
            generate_answer,
            request.question,
            documents,
            history=history_messages or None,
        )
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise HTTPException(status_code=503, detail="LLM service unavailable")

    sources = [
        SourceDoc(
            content=doc["content"][:200],
            source=doc["source"],
            page_num=doc.get("page_num", 0),
            score=doc.get("rerank_score", doc.get("score", 0.0)),
        )
        for doc in documents
    ]

    # 7. 写入缓存（失败不影响主流程）
    if request.use_cache:
        try:
            cache_data = {
                "answer": result["answer"],
                "sources": [s.model_dump() for s in sources],
            }
            await set_cached_answer(request.question, cache_data)
        except Exception as e:
            logger.warning(f"Cache write failed: {e}")

    # 8. 写入历史（失败不影响主流程）
    try:
        history = ChatHistory(
            session_id=session_id,
            question=request.question,
            answer=result["answer"],
            sources=json.dumps([s.model_dump() for s in sources], ensure_ascii=False),
        )
        db.add(history)
        db.commit()
    except Exception as e:
        db.rollback()
        logger.warning(f"Failed to save chat history: {e}")

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        session_id=session_id,
        cached=False,
    )
