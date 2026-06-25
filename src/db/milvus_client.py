# Milvus 向量数据库客户端：管理连接/断开，以及集合的创建（含稠密/稀疏向量索引）。
# 若集合已存在则直接返回，避免重复创建。

from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType, utility
from src.config import settings


def connect_milvus():
    """建立与 Milvus 服务的默认连接。"""
    connections.connect(
        alias="default",
        host=settings.milvus_host,
        port=settings.milvus_port,
    )


def disconnect_milvus():
    """断开默认 Milvus 连接。"""
    connections.disconnect(alias="default")


def create_collection(collection_name: str = "documents") -> Collection:
    """获取或创建指定名称的集合，包含稠密（HNSW）和稀疏（SPARSE_INVERTED_INDEX）两个向量索引。"""
    if utility.has_collection(collection_name):
        return Collection(collection_name)

    fields = [
        FieldSchema(name="id", dtype=DataType.VARCHAR, is_primary=True, max_length=64),
        FieldSchema(name="doc_id", dtype=DataType.VARCHAR, max_length=64),
        FieldSchema(name="chunk_id", dtype=DataType.INT64),
        FieldSchema(name="content", dtype=DataType.VARCHAR, max_length=8192),
        FieldSchema(name="source", dtype=DataType.VARCHAR, max_length=512),
        FieldSchema(name="page_num", dtype=DataType.INT64),
        FieldSchema(name="dense_vector", dtype=DataType.FLOAT_VECTOR, dim=1024),
        FieldSchema(name="sparse_vector", dtype=DataType.SPARSE_FLOAT_VECTOR),
    ]

    schema = CollectionSchema(fields=fields, description="RAG document chunks")
    collection = Collection(name=collection_name, schema=schema)

    collection.create_index(
        field_name="dense_vector",
        index_params={
            "metric_type": "COSINE",
            "index_type": "HNSW",
            "params": {"M": 16, "efConstruction": 256},
        },
    )
    collection.create_index(
        field_name="sparse_vector",
        index_params={
            "metric_type": "IP",
            "index_type": "SPARSE_INVERTED_INDEX",
        },
    )

    return collection
