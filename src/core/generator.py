# 答案生成模块：调用 DeepSeek LLM，将检索到的文档片段格式化为上下文后生成回答。
# 要求模型仅基于参考资料作答并标注来源。

from openai import OpenAI
from loguru import logger

from src.config import settings

_client: OpenAI | None = None


def _get_client() -> OpenAI:
    """获取 DeepSeek OpenAI 兼容客户端单例。"""
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
4. 在回答末尾标注引用来源
5. 如果对话历史中存在上下文，结合历史理解用户意图（如指代消解、追问等），但仍以参考资料为主要依据"""

CONTEXT_TEMPLATE = """参考资料：
{context}

用户问题：{question}

请基于以上参考资料回答问题，并在末尾标注引用来源（格式：[来源: 文件名, 第X页]）。"""


def format_context(documents: list[dict]) -> str:
    """将文档列表格式化为带编号和来源标注的参考资料文本块。"""
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
    history: list[dict] | None = None,
    temperature: float = 0.3,
    max_tokens: int = 1024,
) -> dict:
    """调用 LLM 生成回答，无文档时返回固定提示；有文档时拼接上下文后调用 DeepSeek。

    history 为历史对话消息列表，格式为 [{"role":"user","content":"..."}, {"role":"assistant","content":"..."}, ...]，
    按时间正序排列，会被注入到 system prompt 与当前用户消息之间。
    """
    if not documents:
        return {
            "answer": "抱歉，未找到相关参考资料来回答您的问题。",
            "sources": [],
        }

    context = format_context(documents)
    user_message = CONTEXT_TEMPLATE.format(context=context, question=question)

    # 构建消息列表：system prompt → 历史对话 → 当前用户消息
    messages: list[dict] = [{"role": "system", "content": SYSTEM_PROMPT}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_message})

    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=messages,
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
