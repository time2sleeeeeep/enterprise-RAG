# RAG 评估指标模块：通过 LLM 裁判计算五种指标（忠实度、答案相关性、上下文精确率、上下文召回率、正确性）。
# 每个指标让 LLM 输出 0.0~1.0 的评分。
# 裁判模型由 eval_judge_model 配置，缺省回退到 deepseek_model，可换用独立模型缓解自评偏置。

import re
from openai import OpenAI
from loguru import logger

from src.config import settings


def _get_llm_client() -> OpenAI:
    """创建指向 DeepSeek 的 OpenAI 兼容客户端。"""
    return OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)


def _llm_judge(prompt: str) -> str:
    """向 LLM 发送评分提示，返回模型输出的原始文本。

    temperature=0.5 作为默认值，避免 DeepSeek 低温下复杂 prompt 频繁返回空输出的问题。
    若仍出现空输出，降低温度至 0.3 重试一次，兼顾稳定性与输出质量。
    """
    client = _get_llm_client()
    judge_model = settings.eval_judge_model or settings.deepseek_model
    for attempt, temp in enumerate([0.5, 0.3]):
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=temp,
            max_tokens=1024,
        )
        raw = resp.choices[0].message.content
        if raw is not None:
            raw = raw.strip()
            if raw:
                return raw
        logger.debug(
            f"LLM judge returned empty on attempt {attempt + 1} (temp={temp}), retrying..."
        )
    logger.warning("LLM judge exhausted all retries, returning empty string")
    return ""


def faithfulness_score(answer: str, contexts: list[str]) -> float:
    """评估答案是否忠实于检索上下文，返回 0.0~1.0 的忠实度分值。"""
    if not answer or not contexts:
        return 0.0
    context_text = "\n---\n".join(contexts[:5])
    prompt = f"""请判断以下答案是否忠实于给定的上下文。
请给出 0.0 到 1.0 的评分，1.0 表示答案完全由上下文支持。

上下文：
{context_text}

答案：
{answer}

请只回复一个 0.0 到 1.0 之间的数字。"""
    try:
        score = float(_llm_judge(prompt))
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.0


def answer_relevancy_score(question: str, answer: str) -> float:
    """评估答案与问题的相关程度，返回 0.0~1.0 的相关性分值。"""
    if not answer:
        return 0.0
    prompt = f"""请判断以下答案与问题的相关程度。
请给出 0.0 到 1.0 的评分，1.0 表示完全相关。

问题：{question}
答案：{answer}

请只回复一个 0.0 到 1.0 之间的数字。"""
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
        prompt = f"""以下上下文是否与正确回答该问题相关？
问题：{question}
期望答案：{ground_truth}
上下文：{ctx}

请只回复"是"或"否"。"""
        result = _llm_judge(prompt).lower()
        if "是" in result or "yes" in result:
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
    prompt = f"""标准答案中有多少比例的内容可以由给定上下文支持？
请给出 0.0 到 1.0 的评分，1.0 表示上下文完全覆盖了标准答案。

标准答案：{ground_truth}
上下文：
{context_text}

请只回复一个 0.0 到 1.0 之间的数字。"""
    try:
        score = float(_llm_judge(prompt))
        return max(0.0, min(1.0, score))
    except (ValueError, TypeError):
        return 0.0


def correctness_score(answer: str, ground_truth: str) -> float:
    """对比答案与标准答案的事实准确性，返回 0.0~1.0 的正确性分值。"""
    if not answer or not ground_truth:
        return 0.0
    prompt = f"""请判断答案相对于标准答案的正确性。
请给出 0.0 到 1.0 的评分，1.0 表示与标准答案在事实上等价。

标准答案：{ground_truth}
答案：{answer}

请只回复一个 0.0 到 1.0 之间的数字。"""
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

CLAIM_DECOMPOSE_PROMPT = """请将以下答案拆解为若干原子事实声明。
每条声明必须是一个独立、完整、可验证的陈述。
不要包含观点、模棱两可的表述或元评论。
每条声明单独占一行，以短横线和空格开头。

答案：
{answer}

输出格式：
- 声明 1
- 声明 2
- 声明 3"""


CLAIM_VERIFY_BATCH_PROMPT = """请判断以下每条声明是否被给定上下文在事实上支持。
每行只回复"是"或"否"，按声明顺序逐行对应。

上下文：
{context}

声明：
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

    # 解析逐行 是/否
    supported = []
    for line in raw.split("\n"):
        line = line.strip().lower()
        supported.append("是" in line or "yes" in line)

    # 补齐长度
    while len(supported) < len(claims):
        supported.append(False)

    return supported[:len(claims)]


def _verify_single_claim(claim: str, contexts: list[str]) -> bool:
    """逐条验证单个声明（批处理失败时的降级方案）。"""
    context_text = "\n---\n".join(contexts[:5])
    prompt = f"""请判断以下声明是否被给定上下文在事实上支持。
请只回复"是"或"否"。

上下文：
{context_text}

声明：{claim}"""
    result = _llm_judge(prompt).lower()
    return "是" in result or "yes" in result


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
