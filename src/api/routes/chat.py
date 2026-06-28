# 对话路由：提供 POST /chat 接口，实现完整的 RAG 问答流程。
# 依次执行：查询缓存 -> 查询改写 -> 混合检索 -> 重排序 -> LLM 生成 -> 结果缓存 -> 写入聊天历史。

import uuid
import json

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session
from loguru import logger

from src.db.mysql_client import get_db, ChatHistory
from src.config import settings
from src.core.retriever import get_retriever
from src.core.reranker import get_reranker
from src.core.generator import generate_answer
from src.core.query_rewriter import rewrite_query
from src.core.cache import get_cached_answer, set_cached_answer

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    question: str
    session_id: str | None = None
    top_k: int = 5
    use_reranker: bool = True
    use_cache: bool = True
    use_query_rewrite: bool = True


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


@router.post("/", response_model=ChatResponse)
async def chat(request: ChatRequest, db: Session = Depends(get_db)):
    """处理问答请求：查缓存 -> 改写查询 -> 混合检索 -> 重排序 -> LLM 生成 -> 缓存结果 -> 写历史。"""
    session_id = request.session_id or str(uuid.uuid4())[:16]

    if request.use_cache:
        cached = await get_cached_answer(request.question)
        if cached:
            return ChatResponse(
                answer=cached["answer"],
                sources=[SourceDoc(**s) for s in cached["sources"]],
                session_id=session_id,
                cached=True,
            )

    query = request.question
    if request.use_query_rewrite:
        query = rewrite_query(request.question)

    retriever = get_retriever()
    documents = retriever.hybrid_search(query, top_k=request.top_k * 3)

    if request.use_reranker and documents:
        reranker = get_reranker()
        documents = reranker.rerank(
            query=request.question,
            documents=documents,
            top_k=request.top_k,
        )
    else:
        documents = documents[: request.top_k]

    # 查询该 session 的历史对话，注入 LLM 上下文
    history_messages: list[dict] = []
    if request.session_id and settings.chat_history_max_turns > 0:
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

    result = generate_answer(
        request.question,
        documents,
        history=history_messages or None,
    )

    sources = [
        SourceDoc(
            content=doc["content"][:200],
            source=doc["source"],
            page_num=doc.get("page_num", 0),
            score=doc.get("score", 0.0) if not request.use_reranker else doc.get("rerank_score", doc.get("score", 0.0)),
        )
        for doc in documents
    ]

    if request.use_cache:
        cache_data = {
            "answer": result["answer"],
            "sources": [s.model_dump() for s in sources],
        }
        await set_cached_answer(request.question, cache_data)

    history = ChatHistory(
        session_id=session_id,
        question=request.question,
        answer=result["answer"],
        sources=json.dumps([s.model_dump() for s in sources], ensure_ascii=False),
    )
    db.add(history)
    db.commit()

    return ChatResponse(
        answer=result["answer"],
        sources=sources,
        session_id=session_id,
        cached=False,
    )
