# 评估运行器：支持单次评估（run_single_eval）和多配置消融实验（run_ablation_study）。
# 预置六种消融配置（有无重排序、有无混合检索、有无查询改写、不同 top_k），结果写入 JSON 文件。

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from loguru import logger

from src.core.retriever import get_retriever
from src.core.reranker import get_reranker
from src.core.generator import generate_answer
from src.core.query_rewriter import expand_query, rewrite_query
from src.evaluation.dataset import EvalSample, load_eval_dataset
from src.evaluation.metrics import METRIC_FUNCTIONS
from src.evaluation.latency_tracker import LatencyTracker, ComponentTiming
from src.evaluation.retrieval_metrics import RETRIEVAL_METRIC_SPECS


@dataclass
class AblationConfig:
    name: str = "baseline"
    use_reranker: bool = True
    use_query_rewrite: bool = False
    use_multi_query: bool = False
    use_hybrid: bool = True
    dense_weight: float = 0.7
    sparse_weight: float = 0.3
    top_k: int = 5
    chunk_size: int = 512


@dataclass
class EvalResult:
    config_name: str
    metrics: dict[str, float] = field(default_factory=dict)
    per_sample: list[dict] = field(default_factory=list)
    total_time: float = 0.0
    # 业务延迟：用户实际感知的 RAG 管线耗时（检索 + 生成）
    avg_service_latency: float = 0.0
    # 评估开销：LLM 打分耗时（不属于业务延迟）
    avg_scoring_latency: float = 0.0
    # 向后兼容别名
    @property
    def avg_latency(self) -> float:
        return self.avg_service_latency
    @avg_latency.setter
    def avg_latency(self, val: float):
        self.avg_service_latency = val
    # 分组件延迟
    per_component_timing: ComponentTiming = field(default_factory=ComponentTiming)
    # 指标统计（多次运行时填充）
    metric_stats: dict[str, dict[str, float]] = field(default_factory=dict)
    # 延迟百分位（业务延迟）
    latency_percentiles: dict[str, float] = field(default_factory=dict)


ABLATION_CONFIGS = [
    AblationConfig(name="baseline", use_reranker=True, use_hybrid=True),
    AblationConfig(name="no_reranker", use_reranker=False, use_hybrid=True),
    AblationConfig(name="dense_only", use_reranker=True, use_hybrid=False),
    AblationConfig(name="with_rewrite", use_reranker=True, use_hybrid=True, use_query_rewrite=True),
    AblationConfig(name="top3", use_reranker=True, use_hybrid=True, top_k=3),
    AblationConfig(name="top10", use_reranker=True, use_hybrid=True, top_k=10),
    AblationConfig(
        name="with_multi_query",
        use_reranker=True,
        use_hybrid=True,
        use_multi_query=True,
    ),
]


def _retrieve_for_sample(
    query: str, config: AblationConfig, tracker: LatencyTracker | None = None
) -> list[dict]:
    """根据消融配置执行检索：可选查询改写、混合/纯稠密检索、重排序。"""
    retriever = get_retriever()

    if config.use_query_rewrite:
        query = rewrite_query(query)

    if config.use_multi_query:
        sub_queries = expand_query(query)
        top_k_for_search = config.top_k * 2 if config.use_reranker else config.top_k
        docs = retriever.multi_query_search(sub_queries, top_k=top_k_for_search)
    elif config.use_hybrid:
        if tracker:
            tracker.start()
        docs = retriever.hybrid_search(
            query=query,
            top_k=config.top_k * 2 if config.use_reranker else config.top_k,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
        )
        if tracker:
            tracker.record("dense_search_ms")
            tracker.record("sparse_search_ms")
    else:
        from src.core.embeddings import get_embedder
        embedder = get_embedder()
        if tracker:
            tracker.start()
        query_result = embedder.encode_query(query)
        if tracker:
            tracker.record("embedding_ms")
            tracker.start()
        # dense_search 已通过 output_fields 带回完整字段（含 content），无需再补查询
        docs = retriever.dense_search(query_result["dense"], top_k=config.top_k * 2)
        if tracker:
            tracker.record("dense_search_ms")

    if config.use_reranker and docs:
        if tracker:
            tracker.start()
        reranker = get_reranker()
        docs = reranker.rerank(query, docs, top_k=config.top_k)
        if tracker:
            tracker.record("reranking_ms")
    else:
        docs = docs[:config.top_k]

    return docs


def run_single_eval(
    samples: list[EvalSample],
    config: AblationConfig,
    seed: int = 42,
    enable_claim_faithfulness: bool = False,
) -> EvalResult:
    """对给定样本集运行一次完整评估，逐条检索、生成、打分，汇总各指标均值和延迟。

    Args:
        samples: 评估样本列表
        config: 消融配置
        seed: 随机种子（预留，用于未来扩展）
        enable_claim_faithfulness: 是否启用声明级忠实度（更严格但多 2 次 LLM 调用）
    """
    import statistics as stats_lib

    result = EvalResult(config_name=config.name)
    tracker = LatencyTracker()

    # 扩展指标列表：LLM 指标 + 检索指标
    llm_metric_names = list(METRIC_FUNCTIONS.keys())
    retrieval_metric_names = list(RETRIEVAL_METRIC_SPECS.keys())
    all_scores: dict[str, list[float]] = {
        name: [] for name in llm_metric_names + retrieval_metric_names
    }
    all_service_latencies: list[float] = []
    all_scoring_latencies: list[float] = []

    start = time.time()

    for i, sample in enumerate(samples):
        sample_start = time.time()
        logger.info(
            f"[{config.name}] Evaluating sample {i+1}/{len(samples)}: {sample.question[:50]}"
        )

        # --- 检索阶段（带计时）---
        docs = _retrieve_for_sample(sample.question, config, tracker)
        contexts = [d["content"] for d in docs if d.get("content")]
        retrieved_ids = [d.get("id", "") for d in docs if d.get("id")]

        # --- 生成阶段（带计时）---
        tracker.start()
        gen_result = generate_answer(sample.question, docs)
        answer = gen_result["answer"]
        tracker.record("generation_ms")

        # 业务延迟到此为止（检索 + 生成 = 用户实际等待时间）
        service_end = time.time()
        service_latency = service_end - sample_start

        # --- 打分阶段（评估开销，不算业务延迟）---
        tracker.start()
        sample_scores = {}

        # LLM 指标
        for metric_name in llm_metric_names:
            metric_fn = METRIC_FUNCTIONS[metric_name]
            try:
                if metric_name == "faithfulness":
                    score = metric_fn(answer, contexts)
                elif metric_name == "faithfulness_claim":
                    if enable_claim_faithfulness:
                        score = metric_fn(answer, contexts)
                    else:
                        continue  # 跳过声明级忠实度
                elif metric_name == "answer_relevancy":
                    score = metric_fn(sample.question, answer)
                elif metric_name == "context_precision":
                    score = metric_fn(sample.question, contexts, sample.ground_truth)
                elif metric_name == "context_recall":
                    score = metric_fn(sample.ground_truth, contexts)
                elif metric_name == "correctness":
                    score = metric_fn(answer, sample.ground_truth)
                else:
                    score = 0.0
                sample_scores[metric_name] = score
                all_scores[metric_name].append(score)
            except Exception as e:
                logger.warning(f"Metric {metric_name} failed: {e}")
                sample_scores[metric_name] = None

        # 检索指标（无需 LLM，极快）
        if sample.relevance_labels:
            relevant_ids = {
                cid for cid, rel in sample.relevance_labels.items() if rel >= 1
            }
            for metric_name in retrieval_metric_names:
                try:
                    spec = RETRIEVAL_METRIC_SPECS[metric_name]
                    fn = spec["fn"]
                    kwargs = spec["kwargs"]
                    if metric_name == "ndcg_at_10":
                        score = fn(sample.relevance_labels, retrieved_ids, **kwargs)
                    else:
                        score = fn(relevant_ids, retrieved_ids, **kwargs)
                    sample_scores[metric_name] = score
                    all_scores[metric_name].append(score)
                except Exception as e:
                    logger.warning(f"Retrieval metric {metric_name} failed: {e}")
                    sample_scores[metric_name] = None

        tracker.record("scoring_ms")

        scoring_latency = time.time() - service_end
        all_service_latencies.append(service_latency)
        all_scoring_latencies.append(scoring_latency)

        result.per_sample.append({
            "question": sample.question,
            "answer": answer,
            "ground_truth": sample.ground_truth,
            "contexts_count": len(contexts),
            "retrieved_ids": retrieved_ids,
            "scores": sample_scores,
            "service_latency": service_latency,    # 业务延迟
            "scoring_latency": scoring_latency,    # 评估开销
        })

    result.total_time = time.time() - start
    result.avg_service_latency = sum(all_service_latencies) / max(len(all_service_latencies), 1)
    result.avg_scoring_latency = sum(all_scoring_latencies) / max(len(all_scoring_latencies), 1)
    result.per_component_timing = tracker.to_component_timing()

    # 计算各指标均值
    result.metrics = {}
    result.metric_stats = {}
    for name, scores in all_scores.items():
        valid_scores = [s for s in scores if s is not None]
        if valid_scores:
            result.metrics[name] = sum(valid_scores) / len(valid_scores)
            result.metric_stats[name] = {
                "mean": result.metrics[name],
                "std": stats_lib.stdev(valid_scores) if len(valid_scores) > 1 else 0.0,
                "min": min(valid_scores),
                "max": max(valid_scores),
                "count": len(valid_scores),
            }

    # 延迟百分位（业务延迟）
    if all_service_latencies:
        sorted_lat = sorted(all_service_latencies)
        n = len(sorted_lat)
        result.latency_percentiles = {
            "p50": sorted_lat[int(n * 0.50)],
            "p95": sorted_lat[int(n * 0.95)] if n > 1 else sorted_lat[0],
            "p99": sorted_lat[int(n * 0.99)] if n > 1 else sorted_lat[0],
        }

    return result


def run_ablation_study(
    dataset_path: str,
    output_dir: str = "eval_results",
    configs: list[AblationConfig] | None = None,
) -> dict[str, EvalResult]:
    """遍历所有消融配置运行评估，将汇总和详细结果写入 JSON 文件后返回结果字典。"""
    samples = load_eval_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} evaluation samples")

    if configs is None:
        configs = ABLATION_CONFIGS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, EvalResult] = {}
    for config in configs:
        logger.info(f"Running ablation: {config.name}")
        result = run_single_eval(samples, config)
        results[config.name] = result
        logger.info(
            f"[{config.name}] Metrics: "
            + ", ".join(f"{k}={v:.4f}" for k, v in result.metrics.items())
        )

    summary = {}
    for name, result in results.items():
        summary[name] = {
            "metrics": result.metrics,
            "avg_latency": result.avg_latency,
            "total_time": result.total_time,
        }

    summary_path = output_path / "ablation_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Ablation summary saved to {summary_path}")

    for name, result in results.items():
        detail_path = output_path / f"{name}_detail.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump(asdict(result), f, ensure_ascii=False, indent=2)

    return results


def print_comparison_table(results: dict[str, EvalResult]) -> str:
    """将各消融配置的指标和平均延迟格式化为对齐的对比表格字符串并打印日志。"""
    metrics = list(next(iter(results.values())).metrics.keys())
    header = f"{'Config':<15}" + "".join(f"{m:<18}" for m in metrics) + f"{'Latency':<10}"
    lines = [header, "-" * len(header)]
    for name, result in results.items():
        row = f"{name:<15}"
        row += "".join(f"{result.metrics.get(m, 0):<18.4f}" for m in metrics)
        row += f"{result.avg_latency:<10.2f}s"
        lines.append(row)
    table = "\n".join(lines)
    logger.info(f"\n{table}")
    return table
