# 统计评估模块：多次运行评估计算均值/标准差/置信区间。
# 支持统计显著性检验和多轮种子控制。

import random
import statistics as stats_lib
from dataclasses import dataclass, field

import numpy as np
from loguru import logger

from src.evaluation.dataset import EvalSample, load_eval_dataset
from src.evaluation.run_eval import AblationConfig, EvalResult, run_single_eval


@dataclass
class StatisticalResult:
    """多轮评估的统计结果。"""
    config_name: str
    num_runs: int
    num_samples: int
    metrics: dict[str, float] = field(default_factory=dict)
    metric_stats: dict[str, dict] = field(default_factory=dict)
    #  {metric_name: {mean, std, ci_lower, ci_upper, values: [...]}}
    latency_stats: dict[str, dict] = field(default_factory=dict)
    #  {component: {mean_ms, std_ms, p50_ms, p95_ms}}
    per_runs: list[EvalResult] = field(default_factory=list)


def _bootstrap_ci(
    values: list[float],
    n_bootstrap: int = 1000,
    alpha: float = 0.05,
) -> tuple[float, float]:
    """Bootstrap 95% 置信区间。"""
    if len(values) < 3:
        return values[0], values[0]
    rng = np.random.RandomState(42)
    means = []
    arr = np.array(values)
    for _ in range(n_bootstrap):
        sample = rng.choice(arr, size=len(arr), replace=True)
        means.append(float(np.mean(sample)))
    means.sort()
    lower_idx = int(n_bootstrap * alpha / 2)
    upper_idx = int(n_bootstrap * (1 - alpha / 2))
    return means[lower_idx], means[upper_idx - 1]


def run_eval_with_stats(
    samples: list[EvalSample],
    config: AblationConfig,
    n_runs: int = 3,
    base_seed: int = 42,
    enable_claim_faithfulness: bool = False,
) -> StatisticalResult:
    """对给定样本集运行 N 次评估，汇总统计量。

    Args:
        samples: 评估样本列表
        config: 消融配置
        n_runs: 运行次数（>=1）
        base_seed: 基础随机种子，每轮递增
        enable_claim_faithfulness: 是否启用声明级忠实度

    Returns:
        StatisticalResult 包含所有统计信息
    """
    logger.info(
        f"[{config.name}] Running {n_runs} eval rounds on {len(samples)} samples"
    )

    all_runs: list[EvalResult] = []
    for run_idx in range(n_runs):
        seed = base_seed + run_idx
        logger.info(f"[{config.name}] Round {run_idx + 1}/{n_runs} (seed={seed})")
        result = run_single_eval(
            samples, config, seed=seed,
            enable_claim_faithfulness=enable_claim_faithfulness,
        )
        all_runs.append(result)

    if n_runs == 1:
        result = all_runs[0]
        return StatisticalResult(
            config_name=config.name,
            num_runs=1,
            num_samples=len(samples),
            metrics=result.metrics,
            metric_stats=result.metric_stats,
            per_runs=all_runs,
        )

    # 汇总多轮统计
    all_metric_names = set()
    for r in all_runs:
        all_metric_names.update(r.metrics.keys())

    aggregated_metrics: dict[str, float] = {}
    aggregated_stats: dict[str, dict] = {}

    for metric_name in sorted(all_metric_names):
        values = [
            r.metrics.get(metric_name, 0.0) for r in all_runs
        ]
        valid = [v for v in values if v is not None]
        if not valid:
            continue
        mean_val = float(np.mean(valid))
        std_val = float(np.std(valid)) if len(valid) > 1 else 0.0
        ci_lower, ci_upper = _bootstrap_ci(valid)
        aggregated_metrics[metric_name] = mean_val
        aggregated_stats[metric_name] = {
            "mean": mean_val,
            "std": std_val,
            "ci_lower": ci_lower,
            "ci_upper": ci_upper,
            "values": valid,
            "num_valid_runs": len(valid),
        }

    # 延迟统计
    timing_keys = [
        "embedding_ms", "dense_search_ms", "sparse_search_ms",
        "reranking_ms", "generation_ms", "scoring_ms",
    ]
    latency_stats: dict[str, dict] = {}
    for key in timing_keys:
        values = [
            getattr(r.per_component_timing, key, 0.0)
            for r in all_runs
            if hasattr(r, "per_component_timing")
        ]
        valid = [v for v in values if v is not None]
        if not valid:
            continue
        latency_stats[key] = {
            "mean_ms": float(np.mean(valid)),
            "std_ms": float(np.std(valid)) if len(valid) > 1 else 0.0,
        }

    return StatisticalResult(
        config_name=config.name,
        num_runs=n_runs,
        num_samples=len(samples),
        metrics=aggregated_metrics,
        metric_stats=aggregated_stats,
        latency_stats=latency_stats,
        per_runs=all_runs,
    )


def run_ablation_with_stats(
    dataset_path: str,
    output_dir: str = "eval_results",
    configs: list[AblationConfig] | None = None,
    n_runs: int = 3,
) -> dict[str, StatisticalResult]:
    """多轮消融实验：遍历所有配置运行统计评估，结果写入 JSON。

    Args:
        dataset_path: 评估数据集 JSON 路径
        output_dir: 输出目录
        configs: 消融配置列表
        n_runs: 每个配置运行的次数

    Returns:
        {config_name: StatisticalResult}
    """
    from pathlib import Path
    import json
    from dataclasses import asdict

    from src.evaluation.run_eval import ABLATION_CONFIGS

    samples = load_eval_dataset(dataset_path)
    logger.info(f"Loaded {len(samples)} evaluation samples from {dataset_path}")

    if configs is None:
        configs = ABLATION_CONFIGS

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    results: dict[str, StatisticalResult] = {}
    for config in configs:
        logger.info(f"Running statistical ablation: {config.name}")
        result = run_eval_with_stats(samples, config, n_runs=n_runs)
        results[config.name] = result

        # 日志输出
        metric_lines = []
        for name, stats in sorted(result.metric_stats.items()):
            if "ci_lower" in stats and "ci_upper" in stats:
                ci = f"[{stats['ci_lower']:.3f}, {stats['ci_upper']:.3f}]"
                metric_lines.append(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f} 95%CI {ci}")
            else:
                metric_lines.append(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        logger.info(f"[{config.name}] Stats:\n" + "\n".join(metric_lines))

    # 保存汇总 JSON
    summary = {}
    for name, sr in results.items():
        summary[name] = {
            "metrics": sr.metrics,
            "metric_stats": {
                k: {kk: vv for kk, vv in v.items() if kk != "values"}
                for k, v in sr.metric_stats.items()
            },
            "latency_stats": sr.latency_stats,
            "num_runs": sr.num_runs,
            "num_samples": sr.num_samples,
        }

    summary_path = output_path / "ablation_statistical_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info(f"Statistical summary saved to {summary_path}")

    for name, sr in results.items():
        detail_path = output_path / f"{name}_stats_detail.json"
        with open(detail_path, "w", encoding="utf-8") as f:
            json.dump({
                "config_name": sr.config_name,
                "metrics": sr.metrics,
                "metric_stats": sr.metric_stats,
                "latency_stats": sr.latency_stats,
            }, f, ensure_ascii=False, indent=2)

    return results
