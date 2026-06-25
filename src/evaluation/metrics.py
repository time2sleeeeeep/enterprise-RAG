# RAG 评估指标模块：通过 LLM 裁判计算五种指标（忠实度、答案相关性、上下文精确率、上下文召回率、正确性）。
# 每个指标让 LLM 输出 0.0~1.0 的评分。

import re
from openai import OpenAI
from loguru import logger

from src.config import settings


def _get_llm_client() -> OpenAI:
    """创建指向 DeepSeek 的 OpenAI 兼容客户端。"""
    return OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)


def _llm_judge(prompt: str) -> str:
    """向 LLM 发送评分提示，返回模型输出的原始文本（temperature=0 保证确定性）。"""
    client = _get_llm_client()
    resp = client.chat.completions.create(
        model=settings.deepseek_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0,
        max_tokens=256,
    )
    return resp.choices[0].message.content.strip()


def faithfulness_score(answer: str, contexts: list[str]) -> float:
    """评估答案是否忠实于检索上下文，返回 0.0~1.0 的忠实度分值。"""
    if not answer or not contexts:
        return 0.0
    context_text = "\n---\n".join(contexts[:5])
    prompt = f"""Judge whether the answer is faithful to the given context.
Score from 0.0 to 1.0 where 1.0 means fully supported by context.

Context:
{context_text}

Answer:
{answer}

Reply with ONLY a number between 0.0 and 1.0."""
    try:
        score = float(_llm_judge(prompt))
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.0


def answer_relevancy_score(question: str, answer: str) -> float:
    """评估答案与问题的相关程度，返回 0.0~1.0 的相关性分值。"""
    if not answer:
        return 0.0
    prompt = f"""Judge how relevant the answer is to the question.
Score from 0.0 to 1.0 where 1.0 means perfectly relevant.

Question: {question}
Answer: {answer}

Reply with ONLY a number between 0.0 and 1.0."""
    try:
        score = float(_llm_judge(prompt))
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.0


def context_precision_score(question: str, contexts: list[str], ground_truth: str) -> float:
    """计算上下文精确率：相关上下文在已检索上下文中的加权精确率（MAP 风格）。"""
    if not contexts:
        return 0.0
    relevant_count = 0
    precision_sum = 0.0
    for i, ctx in enumerate(contexts[:10]):
        prompt = f"""Is this context relevant to answering the question correctly?
Question: {question}
Expected answer: {ground_truth}
Context: {ctx}

Reply with ONLY "yes" or "no"."""
        result = _llm_judge(prompt).lower()
        if "yes" in result:
            relevant_count += 1
            precision_sum += relevant_count / (i + 1)
    if relevant_count == 0:
        return 0.0
    return precision_sum / relevant_count


def context_recall_score(ground_truth: str, contexts: list[str]) -> float:
    """评估检索上下文对标准答案的覆盖程度，返回 0.0~1.0 的召回分值。"""
    if not ground_truth or not contexts:
        return 0.0
    context_text = "\n---\n".join(contexts[:5])
    prompt = f"""What fraction of the ground truth answer can be attributed to the given contexts?
Score from 0.0 to 1.0 where 1.0 means the contexts fully cover the ground truth.

Ground truth: {ground_truth}
Contexts:
{context_text}

Reply with ONLY a number between 0.0 and 1.0."""
    try:
        score = float(_llm_judge(prompt))
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.0


def correctness_score(answer: str, ground_truth: str) -> float:
    """对比答案与标准答案的事实准确性，返回 0.0~1.0 的正确性分值。"""
    if not answer or not ground_truth:
        return 0.0
    prompt = f"""Judge the correctness of the answer compared to ground truth.
Score from 0.0 to 1.0 where 1.0 means factually equivalent.

Ground truth: {ground_truth}
Answer: {answer}

Reply with ONLY a number between 0.0 and 1.0."""
    try:
        score = float(_llm_judge(prompt))
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.0


METRIC_FUNCTIONS = {
    "faithfulness": faithfulness_score,
    "answer_relevancy": answer_relevancy_score,
    "context_precision": context_precision_score,
    "context_recall": context_recall_score,
    "correctness": correctness_score,
}
