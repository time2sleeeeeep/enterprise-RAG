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
    """向 LLM 发送评分提示，返回模型输出的原始文本。

    temperature=0 保证确定性，但 DeepSeek 某些情况下会返回空字符串。
    出现空输出时自动用 temperature=0.1 重试一次。
    """
    client = _get_llm_client()
    for attempt, temp in enumerate([0, 0.1]):
        resp = client.chat.completions.create(
            model=settings.deepseek_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        if raw:
            return raw
        logger.debug(f"LLM judge returned empty on attempt {attempt + 1} (temp={temp}), retrying...")
    return ""


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


import json


# ---------------------------------------------------------------------------
# 声明级忠实度评估（Claim-Level Faithfulness）
# 比朴素 LLM 打分更细粒度：先拆解 atomic claims，再逐条验证
# ---------------------------------------------------------------------------

CLAIM_DECOMPOSE_PROMPT = """Break down the following answer into atomic factual claims.
Each claim must be a single, self-contained, verifiable statement.
Do NOT include opinions, hedges, or meta-commentary.
Output each claim on its own line, prefixed with a dash and space.

Answer:
{answer}

Output format:
- claim 1
- claim 2
- claim 3"""


CLAIM_VERIFY_BATCH_PROMPT = """For each claim below, determine whether it is factually supported by the given context.
Reply with ONLY "yes" or "no" on each line, one per claim in order.

Context:
{context}

Claims:
{claims}"""


def _decompose_claims(answer: str) -> list[str]:
    """使用 LLM 将答案分解为原子声明列表。

    使用简单的逐行格式而非 JSON，兼容 DeepSeek 等对 JSON 输出不稳定的模型。
    首次调用失败时自动重试（切换 temperature）。
    """
    if not answer:
        return []
    prompt = CLAIM_DECOMPOSE_PROMPT.format(answer=answer)

    for attempt in range(2):
        raw = _llm_judge(prompt)
        if not raw:
            # DeepSeek temperature=0 偶发空输出，重试一次
            logger.debug(f"Claim decomposition empty on attempt {attempt + 1}, retrying...")
            continue
        # 按 dash 前缀解析
        lines = [l.strip().lstrip("- ").strip() for l in raw.split("\n") if l.strip()]
        claims = [l for l in lines if len(l) > 5]
        if claims:
            return claims
        # 如果逐行解析也失败，尝试 JSON 作为降级
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, list) and all(isinstance(c, str) for c in parsed):
                return parsed
        except (json.JSONDecodeError, Exception):
            pass

    return []


def _verify_claims_batch(
    claims: list[str], contexts: list[str]
) -> list[bool]:
    """批量验证所有声明是否被上下文支持，一次 LLM 调用完成。"""
    if not claims or not contexts:
        return [False] * len(claims)

    claims_text = "\n".join(
        f"{i}. {c}" for i, c in enumerate(claims)
    )
    context_text = "\n---\n".join(contexts[:5])
    prompt = CLAIM_VERIFY_BATCH_PROMPT.format(
        context=context_text, claims=claims_text,
    )
    raw = _llm_judge(prompt)
    if not raw:
        # 空输出降级为逐条验证
        return [_verify_single_claim(c, contexts) for c in claims]

    # 解析逐行 yes/no
    supported = []
    for line in raw.split("\n"):
        line = line.strip().lower()
        supported.append("yes" in line)

    # 补齐长度
    while len(supported) < len(claims):
        supported.append(False)

    return supported[:len(claims)]


def _verify_single_claim(claim: str, contexts: list[str]) -> bool:
    """逐条验证单个声明（批处理失败时的降级方案）。"""
    context_text = "\n---\n".join(contexts[:5])
    prompt = f"""Determine whether the following claim is factually supported by the given context.
Reply with ONLY "yes" or "no".

Context:
{context_text}

Claim: {claim}"""
    result = _llm_judge(prompt).lower()
    return "yes" in result


def claim_faithfulness_score(answer: str, contexts: list[str]) -> float:
    """声明级忠实度：将答案拆为原子声明，逐条验证后返回支持率。

    返回 0.0~1.0：被支持的声明数 / 总声明数。
    同时将详细结果记录到日志供调试。

    Args:
        answer: LLM 生成的答案
        contexts: 检索到的参考上下文列表
    Returns:
        0.0~1.0 的忠实度分值；无声明或无上下文时返回 0.0
    """
    if not answer or not contexts:
        return 0.0

    claims = _decompose_claims(answer)
    if not claims:
        logger.warning("Claim decomposition returned no claims, falling back to naive faithfulness")
        return faithfulness_score(answer, contexts)

    supported = _verify_claims_batch(claims, contexts)
    num_supported = sum(1 for s in supported if s)
    score = num_supported / len(claims)

    logger.debug(
        f"Claim faithfulness: {num_supported}/{len(claims)} claims supported = {score:.3f}"
    )
    return max(0.0, min(1.0, score))


METRIC_FUNCTIONS = {
    # 核心指标
    "faithfulness": faithfulness_score,
    "faithfulness_claim": claim_faithfulness_score,  # 声明级忠实度（更严格）
    "answer_relevancy": answer_relevancy_score,
    "context_precision": context_precision_score,
    "context_recall": context_recall_score,
    "correctness": correctness_score,
}
