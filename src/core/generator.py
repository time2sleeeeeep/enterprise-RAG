from openai import OpenAI
from loguru import logger

from src.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI(
            api_key=settings.deepseek_api_key,
            base_url=settings.deepseek_base_url,
        )
    return _client


SYSTEM_PROMPT = """你是一个专业的知识库问答助手。请根据提供的参考资料回答用户问题。

要求：
1. 只基于参考资料中的信息回答，不要编造内容
2. 如果参考资料不足以回答问题，明确告知用户
3. 回答要准确、简洁、有条理
4. 在回答末尾标注引用来源"""

CONTEXT_TEMPLATE = """参考资料：
{context}

用户问题：{question}

请基于以上参考资料回答问题，并在末尾标注引用来源（格式：[来源: 文件名, 第X页]）。"""


def format_context(documents: list[dict]) -> str:
    parts = []
    for i, doc in enumerate(documents, 1):
        source_info = f"[{doc['source']}"
        if doc.get("page_num", 0) > 0:
            source_info += f", 第{doc['page_num']}页"
        source_info += "]"
        parts.append(f"【参考{i}】{source_info}\n{doc['content']}")
    return "\n\n".join(parts)


def generate_answer(
    question: str,
    documents: list[dict],
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> dict:
    if not documents:
        return {
            "answer": "抱歉，未找到相关参考资料来回答您的问题。",
            "sources": [],
        }

    context = format_context(documents)
    user_message = CONTEXT_TEMPLATE.format(context=context, question=question)

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_message},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        answer = response.choices[0].message.content.strip()
        sources = [
            {"source": doc["source"], "page_num": doc.get("page_num", 0)}
            for doc in documents
        ]
        return {"answer": answer, "sources": sources}
    except Exception as e:
        logger.error(f"LLM generation failed: {e}")
        raise
