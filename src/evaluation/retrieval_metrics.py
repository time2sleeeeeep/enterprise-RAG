# 标准检索指标模块：纯公式计算，无需 LLM 调用，依赖 EvalSample.relevance_labels。
# 包括 Precision@K、Recall@K、MRR、NDCG@K，执行速度在毫秒级。

import math
from collections import OrderedDict


def precision_at_k(
    relevant_ids: set[str], retrieved_ids: list[str], k: int = 5
) -> float:
    """前 K 个检索结果中相关文档的占比。

    Args:
        relevant_ids: 相关文档 ID 集合（relevance >= 1）
        retrieved_ids: 检索返回的文档 ID 列表（按排名正序）
        k: 截断值
    Returns:
        0.0~1.0 的精确率
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0
    return len(set(top_k) & relevant_ids) / min(k, len(top_k))


def recall_at_k(
    relevant_ids: set[str], retrieved_ids: list[str], k: int = 10
) -> float:
    """所有相关文档中，被前 K 个检索结果覆盖的比例。

    Args:
        relevant_ids: 相关文档 ID 集合
        retrieved_ids: 检索返回的文档 ID 列表（按排名正序）
        k: 截断值
    Returns:
        0.0~1.0 的召回率；如果无相关文档则返回 0.0
    """
    if not relevant_ids:
        return 0.0
    top_k = retrieved_ids[:k]
    return len(set(top_k) & relevant_ids) / len(relevant_ids)


def rrf_score(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    """倒排排名：第一个相关文档排名的倒数，即 1 / rank_of_first_relevant。

    单条查询的 Reciprocal Rank；跨多条查询求均值即为 MRR。

    Args:
        relevant_ids: 相关文档 ID 集合
        retrieved_ids: 检索返回的文档 ID 列表（按排名正序）
    Returns:
        0.0~1.0 — 第一个相关文档的排名倒数；无相关文档时返回 0.0
    """
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            return 1.0 / (i + 1)
    return 0.0


def ndcg_at_k(
    relevance_scores: dict[str, int],
    retrieved_ids: list[str],
    k: int = 10,
) -> float:
    """归一化折损累计增益 NDCG@K。

    使用多级相关性分数：0=不相关, 1=相关, 2=高度相关。
    DCG = Σ (2^rel_i - 1) / log2(i+2)，IDCG 来自 ideal 排序。

    Args:
        relevance_scores: {doc_id: relevance_level (0/1/2), ...}
        retrieved_ids: 检索返回的文档 ID 列表（按排名正序）
        k: 截断值
    Returns:
        0.0~1.0 的 NDCG；如果 IDCG 为 0（无相关文档）则返回 0.0
    """
    top_k = retrieved_ids[:k]
    if not top_k:
        return 0.0

    # DCG
    dcg = 0.0
    for i, doc_id in enumerate(top_k):
        rel = relevance_scores.get(doc_id, 0)
        gain = (2 ** rel) - 1
        dcg += gain / math.log2(i + 2)

    # IDCG
    all_rels = sorted(relevance_scores.values(), reverse=True)[:k]
    idcg = 0.0
    for i, rel in enumerate(all_rels):
        gain = (2 ** rel) - 1
        idcg += gain / math.log2(i + 2)

    if idcg == 0:
        return 0.0
    return dcg / idcg


def hit_rate_at_k(
    relevant_ids: set[str], retrieved_ids: list[str], k: int = 5
) -> float:
    """前 K 个结果中是否至少有一个相关文档（命中率）。

    用于批量评估时求平均；单样本返回 0.0 或 1.0。

    Args:
        relevant_ids: 相关文档 ID 集合
        retrieved_ids: 检索返回的文档 ID 列表（按排名正序）
        k: 截断值
    Returns:
        1.0 如果前 K 个中至少有一个相关文档，否则 0.0
    """
    top_k = set(retrieved_ids[:k])
    return 1.0 if (top_k & relevant_ids) else 0.0


def average_precision(relevant_ids: set[str], retrieved_ids: list[str]) -> float:
    """平均精度 AP：每个相关文档位置上的 precision 之和 / 总相关数。

    Args:
        relevant_ids: 相关文档 ID 集合
        retrieved_ids: 检索返回的文档 ID 列表（按排名正序）
    Returns:
        0.0~1.0 的 AP 值；无相关文档时返回 0.0
    """
    if not relevant_ids:
        return 0.0
    hits = 0
    precision_sum = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_ids:
            hits += 1
            precision_sum += hits / (i + 1)
    return precision_sum / len(relevant_ids)


# 指标注册表（签名均为 (relevant_ids, retrieved_ids, **kwargs) 的便捷封装）
# 注意：这些封装函数签名与 METRIC_FUNCTIONS 需求不同，由 run_eval 的指标分发逻辑统一处理。
RETRIEVAL_METRIC_SPECS = OrderedDict({
    "precision_at_5": {"fn": precision_at_k, "kwargs": {"k": 5}},
    "precision_at_10": {"fn": precision_at_k, "kwargs": {"k": 10}},
    "recall_at_5": {"fn": recall_at_k, "kwargs": {"k": 5}},
    "recall_at_10": {"fn": recall_at_k, "kwargs": {"k": 10}},
    "mrr": {"fn": rrf_score, "kwargs": {}},
    "ndcg_at_10": {"fn": ndcg_at_k, "kwargs": {"k": 10}},
    "hit_rate_at_5": {"fn": hit_rate_at_k, "kwargs": {"k": 5}},
    "map_score": {"fn": average_precision, "kwargs": {}},
})
