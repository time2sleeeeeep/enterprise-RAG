# 评估路由：提供 POST /eval/run（单次评估）和 POST /eval/ablation（消融实验）两个接口。
# 允许通过 API 触发 RAG 管道的评估并返回各项指标结果。

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from loguru import logger

from src.api.responses import ErrorResponse
from src.evaluation.dataset import load_eval_dataset
from src.evaluation.run_eval import AblationConfig, run_ablation_study, run_single_eval

router = APIRouter(prefix="/eval")

# 本模块各端点公共的错误响应文档
_COMMON_ERRORS: dict[int | str, dict] = {
    400: {"model": ErrorResponse, "description": "请求参数不合法"},
    404: {"model": ErrorResponse, "description": "评估数据集或资源不存在"},
    500: {"model": ErrorResponse, "description": "评估运行失败"},
}


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


@router.post(
    "/run",
    response_model=EvalResultResponse,
    responses=_COMMON_ERRORS,
)
async def run_evaluation(request: EvalRequest):
    """加载评估数据集，按请求参数构造评估配置并运行，返回指定指标的平均分。"""
    try:
        samples = load_eval_dataset(request.dataset_path)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")
    except Exception as e:
        logger.error(f"Failed to load dataset {request.dataset_path}: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to load dataset: {e}")

    if not samples:
        raise HTTPException(status_code=400, detail="Dataset is empty")

    config = AblationConfig(
        name="api_eval",
        use_reranker=request.use_reranker,
        use_query_rewrite=request.use_query_rewrite,
        top_k=request.top_k,
    )

    logger.info(f"Running evaluation: {len(samples)} samples, config={config.name}")
    try:
        result = run_single_eval(samples, config)
    except Exception as e:
        logger.error(f"Evaluation run failed: {e}")
        raise HTTPException(status_code=500, detail="Evaluation run failed")

    filtered_metrics = {
        k: v for k, v in result.metrics.items() if k in request.metrics
    }

    return EvalResultResponse(
        metrics=filtered_metrics,
        num_samples=len(samples),
        avg_latency=result.avg_latency,
    )


@router.post(
    "/ablation",
    response_model=AblationResultResponse,
    responses=_COMMON_ERRORS,
)
async def run_ablation(request: AblationRequest):
    """触发消融实验，遍历所有预置配置，结果写入 output_dir 并返回各配置的指标汇总。"""
    try:
        results = run_ablation_study(request.dataset_path, request.output_dir)
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Dataset not found: {request.dataset_path}")
    except Exception as e:
        logger.error(f"Ablation study failed: {e}")
        raise HTTPException(status_code=500, detail="Ablation study failed")

    configs = {
        name: result.metrics for name, result in results.items()
    }

    return AblationResultResponse(configs=configs, output_dir=request.output_dir)
