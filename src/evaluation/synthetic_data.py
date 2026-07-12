# 合成 QA 数据集生成器：从已摄入的 Milvus 文档中采样 chunk，
# 使用 LLM 自动生成领域特定的问答对，并自动标注相关文档标签。
#
# 用法：
#   python -m src.evaluation.synthetic_data --num-samples 50 --output eval_data/milvus_qa.json

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from loguru import logger
from openai import OpenAI

from src.config import settings
from src.core.embeddings import get_embedder
from src.db.milvus_client import create_collection
from src.db.mysql_client import SessionLocal, Document
from src.evaluation.dataset import EvalSample, save_eval_dataset


# ---------------------------------------------------------------------------
# 配置
# ---------------------------------------------------------------------------

@dataclass
class SyntheticQAConfig:
    """合成数据集生成配置。"""
    num_questions: int = 50                # 目标 QA 对数量
    sampling_strategy: str = "stratified"   # random | stratified | diverse
    questions_per_chunk: int = 1            # 每个 chunk 最多生成几条
    difficulty_mix: dict = field(default_factory=lambda: {
        "simple": 0.4, "medium": 0.4, "complex": 0.2,
    })
    seed: int = 42
    min_chunk_length: int = 200             # 跳过内容太短的 chunk
    dedup_threshold: float = 0.90           # 问题去重的余弦相似度阈值
    validate_sample_size: int = 5           # 验证阶段抽查数量


# ---------------------------------------------------------------------------
# LLM 提示词
# ---------------------------------------------------------------------------

QA_GENERATION_SYSTEM = """你是一位用于评估 RAG（检索增强生成）系统的数据集构建专家。
你的任务是从文档片段中生成高质量的问答对。"""

QA_GENERATION_PROMPT = """请严格根据提供的上下文片段，生成一道{difficulty}难度的问答对。

上下文：
来源文档：{source}
{page_info}
正文：
{content}

要求：
1. 问题必须只能通过提供的上下文来回答
2. 问题应当具体、明确，并以真实用户的提问方式表述
3. {difficulty_instruction}
4. 答案必须详尽，且只能使用上下文中的事实
5. 答案中应尽可能包含具体细节（名称、数字、命令、参数等）

输出格式：
问题：<一行问题>
答案：<基于上下文得出的详尽答案>"""

DIFFICULTY_INSTRUCTIONS = {
    "simple": "问题应当简单直接，可由上下文中的单句话直接回答",
    "medium": "问题需要综合上下文中多个部分的信息才能回答",
    "complex": "问题需要超越表面阅读的推理或推断才能回答",
}

# 难度级别到中文标签的映射（仅用于拼装提示词，存储时仍使用英文键）
DIFFICULTY_LABELS = {
    "simple": "简单",
    "medium": "中等",
    "complex": "复杂",
}

QA_VALIDATION_PROMPT = """请验证该问题是否可以仅使用提供的上下文完整回答。
请只回复一个 JSON 对象：{{"valid": true/false, "reason": "<简要说明>"}}

上下文：{content}

问题：{question}
标准答案：{answer}

该答案是否完全由上下文支持？"""


# ---------------------------------------------------------------------------
# 核心逻辑
# ---------------------------------------------------------------------------

def _get_llm_client() -> OpenAI:
    """创建指向 DeepSeek 的 OpenAI 兼容客户端。"""
    model = settings.eval_qa_generation_model or settings.deepseek_model
    return OpenAI(api_key=settings.deepseek_api_key, base_url=settings.deepseek_base_url)


def _get_document_sources() -> list[dict]:
    """从 MySQL 查询所有已摄入文档的元信息（id, filename, chunk_count）。"""
    db = SessionLocal()
    try:
        docs = db.query(Document).all()
        return [
            {"id": d.id, "filename": d.filename, "chunk_count": d.chunk_count}
            for d in docs
        ]
    finally:
        db.close()


def _sample_chunks(
    config: SyntheticQAConfig,
) -> list[dict]:
    """从 Milvus 中按采样策略选取 chunk。

    当前使用 stratified 策略：按文档来源分层，每层按比例分配配额，
    从 Milvus 查询该文档的所有 chunk，随机选取。

    Returns:
        [{id, content, source, page_num, doc_id}, ...]
    """
    random.seed(config.seed)

    sources = _get_document_sources()
    if not sources:
        raise RuntimeError("No documents found in MySQL. Ingest documents first.")

    total_chunks = sum(s["chunk_count"] for s in sources)
    logger.info(
        f"Found {len(sources)} documents, {total_chunks} total chunks in MySQL"
    )

    collection = create_collection(settings.milvus_collection)
    collection.load()

    # 要采样的 chunk 数量（略多于目标，去重后会减少）
    target_samples = min(
        config.num_questions * 2, total_chunks
    )

    if config.sampling_strategy == "stratified":
        # 按文档 chunk_count 比例分配采样配额
        chunks_per_doc = {}
        remaining = target_samples
        for s in sorted(sources, key=lambda x: x["chunk_count"], reverse=True):
            if remaining <= 0:
                break
            quota = max(1, int(target_samples * s["chunk_count"] / total_chunks))
            quota = min(quota, remaining, s["chunk_count"])
            chunks_per_doc[s["id"]] = quota
            remaining -= quota

        logger.info(f"Stratified sampling across {len(chunks_per_doc)} documents")
    else:
        # 纯随机：每个文档均匀分配
        quota = max(1, target_samples // max(len(sources), 1))
        chunks_per_doc = {s["id"]: min(quota, s["chunk_count"]) for s in sources}

    sampled_chunks = []
    for doc_id, quota in chunks_per_doc.items():
        try:
            results = collection.query(
                expr=f'doc_id == "{doc_id}"',
                output_fields=["id", "content", "source", "page_num", "doc_id"],
                limit=10000,
            )
            if not results:
                logger.warning(f"No chunks found for doc_id={doc_id} in Milvus")
                continue

            # 过滤太短的 chunk
            eligible = [
                r for r in results
                if len(r.get("content", "")) >= config.min_chunk_length
            ]
            if not eligible:
                continue

            selected = random.sample(eligible, min(quota, len(eligible)))
            sampled_chunks.extend(selected)
            logger.debug(f"Sampled {len(selected)} chunks from doc {doc_id}")
        except Exception as e:
            logger.warning(f"Failed to query chunks for doc_id={doc_id}: {e}")
            continue

    logger.info(f"Sampled {len(sampled_chunks)} chunks total")
    return sampled_chunks


def _pick_difficulty(config: SyntheticQAConfig) -> str:
    """按配置的概率分布选择难度级别。"""
    r = random.random()
    cumulative = 0.0
    for level, prob in config.difficulty_mix.items():
        cumulative += prob
        if r <= cumulative:
            return level
    return "medium"


def _generate_single_qa(
    chunk: dict, difficulty: str, llm_client: OpenAI
) -> tuple[str, str] | None:
    """从单个 chunk 生成一条 QA 对。返回 (question, ground_truth) 或 None。"""
    content = chunk.get("content", "")
    source = chunk.get("source", "unknown")
    page_num = chunk.get("page_num", 0)
    page_info = f"页码：{page_num}" if page_num > 0 else ""

    difficulty_instruction = DIFFICULTY_INSTRUCTIONS.get(
        difficulty, DIFFICULTY_INSTRUCTIONS["medium"]
    )
    difficulty_label = DIFFICULTY_LABELS.get(difficulty, difficulty)

    prompt = QA_GENERATION_PROMPT.format(
        difficulty=difficulty_label,
        source=source,
        page_info=page_info,
        content=content[:4000],  # DeepSeek context limit safety
        difficulty_instruction=difficulty_instruction,
    )

    model = settings.eval_qa_generation_model or settings.deepseek_model
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": QA_GENERATION_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        raw = response.choices[0].message.content.strip()
    except Exception as e:
        logger.error(f"LLM generation failed for chunk {chunk.get('id')}: {e}")
        return None

    # 解析 问题:/答案: 格式（兼容全角/半角冒号及英文标记）
    question = ""
    answer = ""
    normalized = raw.replace("：", ":")
    if "问题:" in normalized and "答案:" in normalized:
        parts = normalized.split("答案:", 1)
        question = parts[0].replace("问题:", "").strip()
        answer = parts[1].strip()
    elif "QUESTION:" in normalized and "ANSWER:" in normalized:
        parts = normalized.split("ANSWER:", 1)
        question = parts[0].replace("QUESTION:", "").strip()
        answer = parts[1].strip()
    elif "\n" in raw:
        lines = raw.split("\n")
        question = lines[0].strip()
        answer = "\n".join(lines[1:]).strip()

    if not question or not answer:
        logger.warning(f"Failed to parse QA from LLM output: {raw[:100]}")
        return None
    if len(question) < 10 or len(answer) < 30:
        logger.warning(f"QA too short: Q={len(question)}chars, A={len(answer)}chars")
        return None

    return question, answer


def _validate_qa(
    chunk: dict, question: str, answer: str, llm_client: OpenAI
) -> bool:
    """LLM 抽查验证 QA 对是否忠实于源 chunk。"""
    prompt = QA_VALIDATION_PROMPT.format(
        content=chunk.get("content", "")[:4000],
        question=question,
        answer=answer,
    )
    model = settings.eval_qa_generation_model or settings.deepseek_model
    try:
        response = llm_client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=128,
        )
        result = json.loads(response.choices[0].message.content.strip())
        return result.get("valid", False)
    except Exception:
        return True  # 解析失败时宽松处理


def _deduplicate_questions(
    samples: list[EvalSample], threshold: float = 0.90
) -> list[EvalSample]:
    """基于嵌入余弦相似度去除语义重复的问题。"""
    if len(samples) <= 1:
        return samples

    embedder = get_embedder()
    questions = [s.question for s in samples]
    embeddings = embedder.encode(questions, return_sparse=False)["dense"]

    # 归一化
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-8
    normalized = embeddings / norms
    sim_matrix = np.dot(normalized, normalized.T)

    keep = [True] * len(samples)
    for i in range(len(samples)):
        if not keep[i]:
            continue
        for j in range(i + 1, len(samples)):
            if sim_matrix[i, j] >= threshold:
                keep[j] = False

    kept = [s for i, s in enumerate(samples) if keep[i]]
    removed = len(samples) - len(kept)
    if removed > 0:
        logger.info(f"Deduplication removed {removed} near-duplicate questions")
    return kept


def _auto_label_relevance(sample: EvalSample, seed_chunk: dict) -> EvalSample:
    """为合成样本自动标注相关文档标签。

    种子 chunk 标为 2（高度相关）。
    后续可以通过语义搜索找到其他相关 chunk 标记为 1，当前版本仅标记种子 chunk。
    """
    seed_id = seed_chunk.get("id", "")
    if seed_id:
        sample.relevance_labels[seed_id] = 2
        sample.seed_chunk_id = seed_id
    source = seed_chunk.get("source", "")
    if source:
        sample.expected_sources = [source]
    return sample


def generate_synthetic_dataset(
    config: SyntheticQAConfig | None = None,
    output_path: str = "eval_data/milvus_qa_dataset.json",
) -> list[EvalSample]:
    """主入口：生成合成 QA 评估数据集。

    Args:
        config: 生成配置，None 则使用默认值
        output_path: 输出 JSON 文件路径

    Returns:
        生成的 EvalSample 列表
    """
    if config is None:
        config = SyntheticQAConfig()

    random.seed(config.seed)
    llm_client = _get_llm_client()

    # 1. 采样 chunk
    logger.info(f"Step 1/5: Sampling chunks (strategy={config.sampling_strategy})...")
    chunks = _sample_chunks(config)
    if not chunks:
        raise RuntimeError("No chunks sampled. Check Milvus connection and document ingestion.")

    # 2. 生成 QA 对
    logger.info(f"Step 2/5: Generating QA pairs for {len(chunks)} chunks...")
    samples = []
    difficulties = [config.difficulty_mix.copy() for _ in range(len(chunks))]
    for i, chunk in enumerate(chunks):
        difficulty = _pick_difficulty(config)
        qa = _generate_single_qa(chunk, difficulty, llm_client)
        if qa is None:
            continue
        question, answer = qa
        sample = EvalSample(question=question, ground_truth=answer, difficulty=difficulty)
        sample = _auto_label_relevance(sample, chunk)
        samples.append(sample)

        if (i + 1) % 10 == 0:
            logger.info(f"  Generated {len(samples)} valid QA pairs from {i + 1} chunks")

    logger.info(f"Generated {len(samples)} raw QA pairs")

    # 3. 去重
    logger.info("Step 3/5: Deduplicating questions...")
    samples = _deduplicate_questions(samples, config.dedup_threshold)

    # 4. 截取到目标数量
    if len(samples) > config.num_questions:
        random.shuffle(samples)
        samples = samples[:config.num_questions]
        logger.info(f"Truncated to {config.num_questions} samples")

    # 5. 验证（抽查）
    logger.info(f"Step 4/5: Validating {config.validate_sample_size} random samples...")
    validate_samples = random.sample(
        samples, min(config.validate_sample_size, len(samples))
    )
    valid_count = 0
    for vs in validate_samples:
        # 找到对应的 seed chunk 进行验证
        if vs.seed_chunk_id:
            seed = next((c for c in chunks if c.get("id") == vs.seed_chunk_id), None)
            if seed and _validate_qa(seed, vs.question, vs.ground_truth, llm_client):
                valid_count += 1
            elif seed:
                logger.warning(f"Validation failed for Q: {vs.question[:60]}...")
    logger.info(f"Validation: {valid_count}/{len(validate_samples)} passed")

    # 6. 保存
    logger.info(f"Step 5/5: Saving {len(samples)} samples to {output_path}...")
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    save_eval_dataset(samples, output_path)
    logger.info(f"Dataset saved to {output_path}")

    # 打印统计
    difficulties_seen = {}
    sources_seen = set()
    for s in samples:
        d = s.difficulty or "unknown"
        difficulties_seen[d] = difficulties_seen.get(d, 0) + 1
        for src in s.expected_sources:
            sources_seen.add(src)

    logger.info(
        f"Dataset summary: {len(samples)} questions, "
        f"{len(sources_seen)} unique sources, "
        f"difficulties: {difficulties_seen}"
    )

    return samples


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic QA evaluation dataset from ingested Milvus documents"
    )
    parser.add_argument(
        "--num-samples", type=int, default=50,
        help="Target number of QA pairs (default: 50)"
    )
    parser.add_argument(
        "--output", type=str, default="eval_data/milvus_qa_dataset.json",
        help="Output JSON path"
    )
    parser.add_argument(
        "--sampling", type=str, default="stratified",
        choices=["random", "stratified"],
        help="Chunk sampling strategy (default: stratified)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed"
    )
    parser.add_argument(
        "--validate-size", type=int, default=10,
        help="Number of samples to validate (default: 10)"
    )
    args = parser.parse_args()

    config = SyntheticQAConfig(
        num_questions=args.num_samples,
        sampling_strategy=args.sampling,
        seed=args.seed,
        validate_sample_size=args.validate_size,
    )

    try:
        samples = generate_synthetic_dataset(config, args.output)
        print(f"\n✓ Generated {len(samples)} QA pairs → {args.output}")
        # 打印前 3 条样例
        for i, s in enumerate(samples[:3]):
            print(f"\n--- Sample {i+1} ---")
            print(f"Q: {s.question}")
            print(f"A: {s.ground_truth[:200]}...")
            print(f"Difficulty: {s.difficulty}, Source: {s.expected_sources}")
    except Exception as e:
        logger.error(f"Dataset generation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
