# 评估路由：提供 POST /eval/run（单次评估）和 POST /eval/ablation（消融实验）两个接口。
# 允许通过 API 触发 RAG 管道的评估并返回各项指标结果。

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from src.evaluation.dataset import load_eval_dataset
from src.evaluation.run_eval import run_single_eval, run_ablation_study, AblationConfig

router = APIRouter(prefix="/eval", tags=["evaluation"])


class EvalRequest(BaseModel):
    dataset_path: str
    metrics: list[str] = ["faithfulness", "answer_relevancy", "context_precision"]
    use_reranker: bool = True
    use_query_rewrite: bool = False
    top_k: int = 5


class AblationRequest(BaseModel):
    dataset_path: str
    output_dir: str = "eval_results"


class EvalResultResponse(BaseModel):
    metrics: dict[str, float]
    num_samples: int
    avg_latency: float


class AblationResultResponse(BaseModel):
    configs: dict[str, dict[str, float]]
    output_dir: str


@router.post("/run", response_model=EvalResultResponse)
async def run_evaluation(request: EvalRequest):
    """加载评估数据集，按请求参数构造评估配置并运行，返回指定指标的平均分。"""
    try:
        samples = load_eval_dataset(request.dataset_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

    config = AblationConfig(
        name="api_eval",
        use_reranker=request.use_reranker,
        use_query_rewrite=request.use_query_rewrite,
        top_k=request.top_k,
    )

    logger.info(f"Running evaluation: {len(samples)} samples, config={config.name}")
    result = run_single_eval(samples, config)

    filtered_metrics = {
        k: v for k, v in result.metrics.items() if k in request.metrics
    }

    return EvalResultResponse(
        metrics=filtered_metrics,
        num_samples=len(samples),
        avg_latency=result.avg_latency,
    )


@router.post("/ablation", response_model=AblationResultResponse)
async def run_ablation(request: AblationRequest):
    """触发消融实验，遍历所有预置配置，结果写入 output_dir 并返回各配置的指标汇总。"""
    try:
        results = run_ablation_study(request.dataset_path, request.output_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")

    configs = {
        name: result.metrics for name, result in results.items()
    }

    return AblationResultResponse(configs=configs, output_dir=request.output_dir)
