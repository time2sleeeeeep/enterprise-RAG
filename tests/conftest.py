"""pytest 共享 fixtures 与工具类。"""

from dataclasses import dataclass

import pytest


@dataclass
class FakeHit:
    """模拟 pymilvus Hit 对象，供 _hit_to_doc / 检索测试使用。"""

    id: str
    score: float
    fields: dict

    @property
    def distance(self):
        return self.score

    def get(self, name):
        return self.fields.get(name)


@pytest.fixture
def sample_documents():
    """一组带完整字段的文档字典。"""
    return [
        {
            "id": "a1",
            "content": "RAG 是检索增强生成。",
            "source": "intro.md",
            "page_num": 1,
            "doc_id": "d1",
            "score": 0.9,
        },
        {
            "id": "b2",
            "content": "Milvus 是向量数据库。",
            "source": "milvus.md",
            "page_num": 3,
            "doc_id": "d2",
            "score": 0.8,
        },
    ]


@pytest.fixture
def sample_eval_samples():
    """三条简化的 EvalSample，用于 retrieval_metrics / dataset 测试。"""
    from src.evaluation.dataset import EvalSample

    return [
        EvalSample(
            question="什么是 RAG？",
            ground_truth="检索增强生成",
            relevance_labels={"a1": 2, "b2": 1},
            difficulty="simple",
        ),
        EvalSample(
            question="Milvus 是什么？",
            ground_truth="向量数据库",
            relevance_labels={"b2": 2},
            difficulty="simple",
        ),
    ]
