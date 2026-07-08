# 评估数据集模块：定义 EvalSample 数据类，提供 JSON 格式评估集的加载和保存工具函数。

import json
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class EvalSample:
    question: str
    ground_truth: str
    contexts: list[str] = field(default_factory=list)
    answer: str = ""
    source_documents: list[dict] = field(default_factory=list)
    # 新增字段（全部可选，保持向后兼容）
    relevance_labels: dict[str, int] = field(default_factory=dict)
    #  ^ chunk_id -> relevance: 0=不相关, 1=相关, 2=高度相关（合成数据的种子 chunk）
    expected_sources: list[str] = field(default_factory=list)
    #  ^ 期望回答引用的来源文件名列表
    seed_chunk_id: str = ""
    #  ^ 合成数据集中生成该 QA 的源 chunk id
    difficulty: str = ""
    #  ^ simple / medium / complex


def load_eval_dataset(path: str) -> list[EvalSample]:
    """从 JSON 文件加载评估样本列表，文件不存在时抛出 FileNotFoundError。"""
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")

    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    samples = []
    for item in data:
        samples.append(EvalSample(
            question=item["question"],
            ground_truth=item["ground_truth"],
            contexts=item.get("contexts", []),
            relevance_labels=item.get("relevance_labels", {}),
            expected_sources=item.get("expected_sources", []),
            seed_chunk_id=item.get("seed_chunk_id", ""),
            difficulty=item.get("difficulty", ""),
        ))
    return samples


def save_eval_dataset(samples: list[EvalSample], path: str) -> None:
    """将评估样本列表序列化为 JSON 文件，编码 Unicode 字符。"""
    data = []
    for s in samples:
        data.append({
            "question": s.question,
            "ground_truth": s.ground_truth,
            "contexts": s.contexts,
            "answer": s.answer,
            "relevance_labels": s.relevance_labels,
            "expected_sources": s.expected_sources,
            "seed_chunk_id": s.seed_chunk_id,
            "difficulty": s.difficulty,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
