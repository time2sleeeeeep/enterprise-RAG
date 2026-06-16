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


REWRITE_PROMPT = """你是一个查询改写专家。请将用户的原始问题改写为更适合检索的形式。
要求：
1. 保留原始问题的核心语义
2. 补充可能的同义词或相关术语
3. 如果问题过于简短，适当扩展
4. 输出改写后的查询，不要输出其他内容

原始问题：{query}
改写后的查询："""

MULTI_QUERY_PROMPT = """你是一个查询扩展专家。请为用户的问题生成3个不同角度的子查询，用于提高检索召回率。
要求：
1. 每个子查询覆盖原始问题的不同方面
2. 保持语义相关性
3. 每行输出一个子查询，不要编号

原始问题：{query}
子查询："""


def rewrite_query(query: str) -> str:
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": REWRITE_PROMPT.format(query=query)}],
            temperature=0.1,
            max_tokens=256,
        )
        rewritten = response.choices[0].message.content.strip()
        logger.debug(f"Query rewritten: '{query}' -> '{rewritten}'")
        return rewritten
    except Exception as e:
        logger.warning(f"Query rewrite failed: {e}, using original")
        return query


def expand_query(query: str) -> list[str]:
    client = _get_client()
    try:
        response = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": MULTI_QUERY_PROMPT.format(query=query)}],
            temperature=0.3,
            max_tokens=512,
        )
        content = response.choices[0].message.content.strip()
        sub_queries = [q.strip() for q in content.split("\n") if q.strip()]
        logger.debug(f"Query expanded into {len(sub_queries)} sub-queries")
        return sub_queries[:3]
    except Exception as e:
        logger.warning(f"Query expansion failed: {e}")
        return [query]
