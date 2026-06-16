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


def load_eval_dataset(path: str) -> list[EvalSample]:
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
        ))
    return samples


def save_eval_dataset(samples: list[EvalSample], path: str) -> None:
    data = []
    for s in samples:
        data.append({
            "question": s.question,
            "ground_truth": s.ground_truth,
            "contexts": s.contexts,
            "answer": s.answer,
        })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
