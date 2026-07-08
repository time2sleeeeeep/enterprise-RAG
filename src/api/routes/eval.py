# 评估路由：提供 POST /eval/run（运行评估）和 GET /eval/datasets（列出数据集）两个接口。
# 数据集生成、消融实验、报告生成等离线任务请使用 CLI：
#   python -m src.evaluation.cli --help

import json
from pathlib import Path

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from src.api.responses import ErrorResponse
from src.evaluation.dataset import load_eval_dataset
from src.evaluation.run_eval import ABLATION_CONFIGS

router = APIRouter(prefix="/eval")

_COMMON_ERRORS: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "请求参数不合法"},
    404: {"model": ErrorResponse, "description": "评估数据集或资源不存在"},
    500: {"model": ErrorResponse, "description": "评估运行失败"},
}


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------


class EvalRequest(BaseModel):
    dataset_path: str
    config_name: str = "baseline"                 # baseline | no_reranker | dense_only | ...
    n_runs: int = 1                                # >=2 时启用统计模式（均值±标准差）
    enable_claim_faithfulness: bool = False         # 声明级忠实度（更严格，但多 2 次 LLM 调用）
    use_reranker: bool = True
    use_query_rewrite: bool = False
    top_k: int = 5


class MetricStatsModel(BaseModel):
    mean: float
    std: float
    min: float
    max: float
    count: int


class EvalResponse(BaseModel):
    config_name: str
    num_samples: int
    num_runs: int
    metrics: dict[str, float]
    avg_service_latency: float            # 业务延迟（检索+生成，用户实际等待时间）
    avg_scoring_latency: float | None = None  # 评估开销（LLM 打分耗时）
    # 以下字段仅在 n_runs > 1 时填充
    metric_stats: dict[str, MetricStatsModel] | None = None
    latency_breakdown: dict[str, float] | None = None
    latency_percentiles: dict[str, float] | None = None


class DatasetInfoModel(BaseModel):
    name: str
    num_samples: int
    has_relevance_labels: bool


class ListDatasetsResponse(BaseModel):
    datasets: list[DatasetInfoModel]


# ---------------------------------------------------------------------------
# POST /eval/run
# ---------------------------------------------------------------------------


@router.post(
    "/run",
    response_model=EvalResponse,
    responses=_COMMON_ERRORS,
)
async def run_evaluation(request: EvalRequest):
    """运行一次评估，返回指标均值和延迟。

    支持两种模式：
    - 快速模式（n_runs=1, enable_claim_faithfulness=False）：最快，适合日常检查
    - 详细模式（n_runs>=2, enable_claim_faithfulness=True）：含统计信息和声明级忠实度

    可指定的 config_name：
        baseline, no_reranker, dense_only, with_rewrite, top3, top10
    """
    from src.evaluation.run_eval import AblationConfig, run_single_eval
    from src.evaluation.statistical import run_eval_with_stats
    from src.db.milvus_client import connect_milvus, disconnect_milvus

    # 加载数据集
    try:
        samples = load_eval_dataset(request.dataset_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")
    except Exception as e:
        logger.error(f"Failed to load dataset {request.dataset_path}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {e}")

    if not samples:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    # 查找配置
    config = None
    for c in ABLATION_CONFIGS:
        if c.name == request.config_name:
            config = c
            break
    if config is None:
        if request.config_name == "api_eval":
            config = AblationConfig(
                name="api_eval",
                use_reranker=request.use_reranker,
                use_query_rewrite=request.use_query_rewrite,
                top_k=request.top_k,
            )
        else:
            raise HTTPException(
                status_code=400,
                detail=f"Unknown config '{request.config_name}'. "
                       f"Options: {[c.name for c in ABLATION_CONFIGS]}",
            )
    else:
        # 允许覆盖已有配置的部分字段
        if not request.use_reranker:
            config.use_reranker = False
        if request.use_query_rewrite:
            config.use_query_rewrite = True
        if request.top_k != 5:
            config.top_k = request.top_k

    logger.info(
        f"Running eval: {len(samples)} samples, config={config.name}, "
        f"runs={request.n_runs}, claim_faithfulness={request.enable_claim_faithfulness}"
    )

    connect_milvus()
    try:
        if request.n_runs > 1:
            stat_result = run_eval_with_stats(
                samples, config,
                n_runs=request.n_runs,
                enable_claim_faithfulness=request.enable_claim_faithfulness,
            )
            first = stat_result.per_runs[0] if stat_result.per_runs else None

            metric_stats_response = {}
            for name, stats in stat_result.metric_stats.items():
                metric_stats_response[name] = MetricStatsModel(
                    mean=stats.get("mean", 0),
                    std=stats.get("std", 0),
                    min=stats.get("min", stats.get("mean", 0)),
                    max=stats.get("max", stats.get("mean", 0)),
                    count=stats.get("num_valid_runs", stats.get("count", 0)),
                )

            return EvalResponse(
                config_name=stat_result.config_name,
                num_samples=stat_result.num_samples,
                num_runs=stat_result.num_runs,
                metrics=stat_result.metrics,
                avg_service_latency=first.avg_service_latency if first else 0,
                avg_scoring_latency=first.avg_scoring_latency if first else 0,
                metric_stats=metric_stats_response,
                latency_breakdown=first.per_component_timing.as_dict() if first else None,
                latency_percentiles=first.latency_percentiles if first else None,
            )
        else:
            result = run_single_eval(
                samples, config,
                enable_claim_faithfulness=request.enable_claim_faithfulness,
            )
            return EvalResponse(
                config_name=result.config_name,
                num_samples=len(samples),
                num_runs=1,
                metrics=result.metrics,
                avg_service_latency=result.avg_service_latency,
                avg_scoring_latency=result.avg_scoring_latency,
                latency_breakdown=result.per_component_timing.as_dict(),
                latency_percentiles=result.latency_percentiles,
            )
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        raise HTTPException(status_code=500, detail=f"Evaluation failed: {e}")
    finally:
        disconnect_milvus()


# ---------------------------------------------------------------------------
# GET /eval/datasets
# ---------------------------------------------------------------------------


@router.get(
    "/datasets",
    response_model=ListDatasetsResponse,
    responses=_COMMON_ERRORS,
)
async def list_datasets():
    """列出 eval_data/ 目录中所有可用的评估数据集及其基本信息。"""
    eval_dir = Path("eval_data")
    datasets: list[DatasetInfoModel] = []

    if eval_dir.exists():
        for fpath in sorted(eval_dir.glob("*.json")):
            try:
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if isinstance(data, list):
                    has_labels = any(
                        isinstance(item, dict) and item.get("relevance_labels")
                        for item in data
                    )
                    datasets.append(DatasetInfoModel(
                        name=fpath.name,
                        num_samples=len(data),
                        has_relevance_labels=has_labels,
                    ))
            except Exception:
                continue

    return ListDatasetsResponse(datasets=datasets)
