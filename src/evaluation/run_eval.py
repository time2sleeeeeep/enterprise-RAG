import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

from loguru import logger

from src.core.retriever import get_retriever
from src.core.reranker import get_reranker
from src.core.generator import generate_answer
from src.core.query_rewriter import rewrite_query
from src.evaluation.dataset import EvalSample, load_eval_dataset
from src.evaluation.metrics import METRIC_FUNCTIONS


@dataclass
class AblationConfig:
    name: str = "baseline"
    use_reranker: bool = True
    use_query_rewrite: bool = False
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
    avg_latency: float = 0.0


ABLATION_CONFIGS = [
    AblationConfig(name="baseline", use_reranker=True, use_hybrid=True),
    AblationConfig(name="no_reranker", use_reranker=False, use_hybrid=True),
    AblationConfig(name="dense_only", use_reranker=True, use_hybrid=False),
    AblationConfig(name="with_rewrite", use_reranker=True, use_hybrid=True, use_query_rewrite=True),
    AblationConfig(name="top3", use_reranker=True, use_hybrid=True, top_k=3),
    AblationConfig(name="top10", use_reranker=True, use_hybrid=True, top_k=10),
]


def _retrieve_for_sample(query: str, config: AblationConfig) -> list[dict]:
    retriever = get_retriever()

    if config.use_query_rewrite:
        query = rewrite_query(query)

    if config.use_hybrid:
        docs = retriever.hybrid_search(
            query=query,
            top_k=config.top_k * 2 if config.use_reranker else config.top_k,
            dense_weight=config.dense_weight,
            sparse_weight=config.sparse_weight,
        )
    else:
        from src.core.embeddings import get_embedder
        embedder = get_embedder()
        query_result = embedder.encode_query(query)
        results = retriever.dense_search(query_result["dense"], top_k=config.top_k * 2)
        docs = []
        for chunk_id, score in results:
            docs.append({"id": chunk_id, "score": score, "content": "", "source": "", "page_num": 0})

    if config.use_reranker and docs:
        reranker = get_reranker()
        docs = reranker.rerank(query, docs, top_k=config.top_k)
    else:
        docs = docs[:config.top_k]

    return docs


def run_single_eval(samples: list[EvalSample], config: AblationConfig) -> EvalResult:
    result = EvalResult(config_name=config.name)
    all_scores: dict[str, list[float]] = {name: [] for name in METRIC_FUNCTIONS}
    start = time.time()

    for i, sample in enumerate(samples):
        sample_start = time.time()
        logger.info(f"[{config.name}] Evaluating sample {i+1}/{len(samples)}: {sample.question[:50]}")

        docs = _retrieve_for_sample(sample.question, config)
        contexts = [d["content"] for d in docs if d.get("content")]

        gen_result = generate_answer(sample.question, docs)
        answer = gen_result["answer"]

        sample_scores = {}
        for metric_name, metric_fn in METRIC_FUNCTIONS.items():
            if metric_name == "faithfulness":
                score = metric_fn(answer, contexts)
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

        result.per_sample.append({
            "question": sample.question,
            "answer": answer,
            "ground_truth": sample.ground_truth,
            "contexts_count": len(contexts),
            "scores": sample_scores,
            "latency": time.time() - sample_start,
        })

    result.total_time = time.time() - start
    result.avg_latency = result.total_time / max(len(samples), 1)
    result.metrics = {
        name: sum(scores) / max(len(scores), 1) for name, scores in all_scores.items()
    }
    return result


def run_ablation_study(
    dataset_path: str,
    output_dir: str = "eval_results",
    configs: list[AblationConfig] | None = None,
) -> dict[str, EvalResult]:
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
