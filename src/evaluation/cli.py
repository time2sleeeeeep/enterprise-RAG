# 评估 CLI 入口：提供命令行接口用于数据集生成、评估运行、消融实验和报告生成。
#
# 用法：
#   python -m src.evaluation.cli generate-dataset --num-samples 50
#   python -m src.evaluation.cli run --dataset eval_data/milvus_qa_dataset.json
#   python -m src.evaluation.cli ablation --dataset eval_data/milvus_qa.json
#   python -m src.evaluation.cli report --results-dir eval_results --format html

import argparse
import sys
from pathlib import Path

from loguru import logger


def cmd_generate_dataset(args):
    """生成合成 QA 评估数据集。"""
    from src.evaluation.synthetic_data import SyntheticQAConfig, generate_synthetic_dataset
    from src.db.milvus_client import connect_milvus, disconnect_milvus

    connect_milvus()
    try:
        config = SyntheticQAConfig(
            num_questions=args.num_samples,
            sampling_strategy=args.sampling,
            seed=args.seed,
            validate_sample_size=args.validate_size,
        )
        samples = generate_synthetic_dataset(config, args.output)
        print(f"\n✓ Generated {len(samples)} QA pairs → {args.output}")
    finally:
        disconnect_milvus()


def cmd_run_eval(args):
    """运行单次评估。"""
    from src.evaluation.dataset import load_eval_dataset
    from src.evaluation.run_eval import AblationConfig, run_single_eval, ABLATION_CONFIGS
    from src.evaluation.statistical import run_eval_with_stats
    from src.db.milvus_client import connect_milvus, disconnect_milvus

    samples = load_eval_dataset(args.dataset)
    print(f"Loaded {len(samples)} samples from {args.dataset}")

    # 查找配置
    config = None
    for c in ABLATION_CONFIGS:
        if c.name == args.config:
            config = c
            break
    if config is None:
        print(f"Unknown config '{args.config}'. Available: {[c.name for c in ABLATION_CONFIGS]}")
        sys.exit(1)

    connect_milvus()
    try:
        if args.runs > 1:
            result = run_eval_with_stats(samples, config, n_runs=args.runs, base_seed=args.seed)
            print(f"\n✓ Statistical evaluation complete ({args.runs} runs)")
            print(f"  Config: {result.config_name}")
            for name, stats in result.metric_stats.items():
                print(f"  {name}: {stats['mean']:.4f} ± {stats['std']:.4f}")
        else:
            result = run_single_eval(samples, config, seed=args.seed)
            print(f"\n✓ Evaluation complete")
            print(f"  Config: {result.config_name}")
            for name, score in result.metrics.items():
                print(f"  {name}: {score:.4f}")
            print(f"  Avg service latency: {result.avg_service_latency:.2f}s")
            print(f"  Avg scoring latency: {result.avg_scoring_latency:.2f}s (LLM judge overhead)")
    finally:
        disconnect_milvus()

    # 保存结果
    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    import json
    from dataclasses import asdict
    detail_path = output_path / f"{config.name}_detail.json"
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(asdict(result), f, ensure_ascii=False, indent=2)
    print(f"  Results saved to {detail_path}")


def cmd_run_ablation(args):
    """运行消融实验。"""
    from src.evaluation.statistical import run_ablation_with_stats
    from src.db.milvus_client import connect_milvus, disconnect_milvus

    connect_milvus()
    try:
        results = run_ablation_with_stats(
            dataset_path=args.dataset,
            output_dir=args.output_dir,
            n_runs=args.runs,
        )
    finally:
        disconnect_milvus()

    print(f"\n✓ Ablation study complete ({len(results)} configs × {args.runs} runs)")
    print(f"\n{'Config':<16} {'Faithfulness':<14} {'Answer Rel':<12} {'Correctness':<12} {'Latency':<10}")
    print("-" * 64)
    for name, sr in results.items():
        m = sr.metrics
        print(
            f"{name:<16} {m.get('faithfulness', 0):<14.4f} "
            f"{m.get('answer_relevancy', 0):<12.4f} "
            f"{m.get('correctness', 0):<12.4f} "
            f"{sr.per_runs[0].avg_latency if sr.per_runs else 0:<10.2f}"
        )


def cmd_report(args):
    """从评估结果生成报告。"""
    from src.evaluation.report import generate_reports_from_dir

    outputs = generate_reports_from_dir(
        results_dir=args.results_dir,
        output_dir=args.output_dir or args.results_dir,
        format=args.format,
    )
    for fmt, path in outputs.items():
        print(f"✓ {fmt.upper()} report → {path}")


def cmd_list_datasets(args):
    """列出可用的评估数据集。"""
    import json
    eval_dir = Path("eval_data")
    if not eval_dir.exists():
        print("No eval_data/ directory found.")
        return

    for fpath in sorted(eval_dir.glob("*.json")):
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                has_labels = any(
                    isinstance(item, dict) and item.get("relevance_labels")
                    for item in data
                )
                print(f"  {fpath.name:<40} {len(data):>4} samples  relevance_labels={'✓' if has_labels else '✗'}")
        except Exception:
            print(f"  {fpath.name:<40} (unreadable)")


def main():
    parser = argparse.ArgumentParser(
        description="Enterprise RAG Evaluation Framework CLI"
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate-dataset
    gen_parser = subparsers.add_parser("generate-dataset", help="Generate synthetic QA dataset")
    gen_parser.add_argument("--num-samples", type=int, default=50)
    gen_parser.add_argument("--output", type=str, default="eval_data/milvus_qa_dataset.json")
    gen_parser.add_argument("--sampling", type=str, default="stratified", choices=["random", "stratified"])
    gen_parser.add_argument("--seed", type=int, default=42)
    gen_parser.add_argument("--validate-size", type=int, default=10)

    # run
    run_parser = subparsers.add_parser("run", help="Run single evaluation")
    run_parser.add_argument("--dataset", type=str, required=True)
    run_parser.add_argument("--config", type=str, default="baseline")
    run_parser.add_argument("--output-dir", type=str, default="eval_results")
    run_parser.add_argument("--runs", type=int, default=1)
    run_parser.add_argument("--seed", type=int, default=42)

    # ablation
    abl_parser = subparsers.add_parser("ablation", help="Run ablation study")
    abl_parser.add_argument("--dataset", type=str, required=True)
    abl_parser.add_argument("--output-dir", type=str, default="eval_results")
    abl_parser.add_argument("--runs", type=int, default=1)

    # report
    rep_parser = subparsers.add_parser("report", help="Generate evaluation report")
    rep_parser.add_argument("--results-dir", type=str, required=True)
    rep_parser.add_argument("--output-dir", type=str, default=None)
    rep_parser.add_argument("--format", type=str, default="html", choices=["html", "markdown", "both"])

    # list-datasets
    subparsers.add_parser("list-datasets", help="List available eval datasets")

    args = parser.parse_args()

    if args.command == "generate-dataset":
        cmd_generate_dataset(args)
    elif args.command == "run":
        cmd_run_eval(args)
    elif args.command == "ablation":
        cmd_run_ablation(args)
    elif args.command == "report":
        cmd_report(args)
    elif args.command == "list-datasets":
        cmd_list_datasets(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
