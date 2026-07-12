# 全局配置模块：通过 pydantic-settings 从 .env 文件加载所有服务配置，
# 包括 DeepSeek API、Milvus、MySQL、Redis、嵌入模型、重排序模型及服务器参数。

import json
import os
from typing import Annotated

from pydantic_settings import BaseSettings, NoDecode
from pydantic import Field, field_validator


class Settings(BaseSettings):
    """项目全局配置，字段值优先从环境变量/.env 文件读取。"""
    # DeepSeek API
    deepseek_api_key: str = Field(default="")
    deepseek_base_url: str = Field(default="https://api.deepseek.com")
    deepseek_model: str = Field(default="deepseek-chat")

    # Milvus
    milvus_host: str = Field(default="localhost")
    milvus_port: int = Field(default=19530)
    milvus_collection: str = Field(default="enterprise_rag_chunks")

    # MySQL
    mysql_host: str = Field(default="localhost")
    mysql_port: int = Field(default=3306)
    mysql_user: str = Field(default="rag_user")
    mysql_password: str = Field(default="rag_password_123")
    mysql_database: str = Field(default="enterprise_rag")

    # Redis
    redis_host: str = Field(default="localhost")
    redis_port: int = Field(default=6379)
    redis_db: int = Field(default=0)
    redis_password: str = Field(default="")
    redis_max_memory: str = Field(default="512mb")
    redis_max_memory_policy: str = Field(default="allkeys-lru")
    # 语义缓存 SCAN 比对上限（避免缓存增长后单次 miss 线性扫描过久）
    semantic_cache_scan_limit: int = Field(default=500)

    # Embedding
    embedding_model_name: str = Field(default="BAAI/bge-m3")
    embedding_device: str = Field(default="cuda")
    embedding_batch_size: int = Field(default=32)

    # Reranker
    reranker_model_name: str = Field(default="BAAI/bge-reranker-v2-m3")
    reranker_device: str = Field(default="cuda")

    # Chat
    chat_history_max_turns: int = Field(default=5)

    # Retrieval
    dense_top_k: int = Field(default=20)
    sparse_top_k: int = Field(default=20)
    rerank_top_k: int = Field(default=5)
    rrf_k: int = Field(default=60)

    # Evaluation
    eval_qa_generation_model: str = Field(default="")
    eval_judge_model: str = Field(default="")
    eval_default_runs: int = Field(default=3)
    eval_output_dir: str = Field(default="eval_results")
    eval_claim_batch_verify: bool = Field(default=True)

    # Server
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8000)
    # CORS：env 可传逗号分隔字符串或 JSON 数组，如 CORS_ORIGINS=http://a.com,http://b.com
    cors_origins: Annotated[list[str], NoDecode] = Field(default_factory=lambda: ["*"])

    @field_validator("cors_origins", mode="before")
    @classmethod
    def _parse_cors_origins(cls, v):
        """兼容逗号分隔字符串与 JSON 数组两种 env 写法。"""
        if isinstance(v, str):
            v = v.strip()
            if v.startswith("["):
                try:
                    return json.loads(v)
                except json.JSONDecodeError:
                    pass
            return [s.strip() for s in v.split(",") if s.strip()]
        return v

    @property
    def mysql_url(self) -> str:
        """拼接 SQLAlchemy 格式的 MySQL 连接 URL。"""
        return (
            f"mysql+pymysql://{self.mysql_user}:{self.mysql_password}"
            f"@{self.mysql_host}:{self.mysql_port}/{self.mysql_database}"
        )

    @property
    def redis_url(self) -> str:
        """拼接 Redis 连接 URL，有密码时自动携带认证信息。"""
        if self.redis_password:
            return f"redis://:{self.redis_password}@{self.redis_host}:{self.redis_port}/{self.redis_db}"
        return f"redis://{self.redis_host}:{self.redis_port}/{self.redis_db}"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8", "extra": "ignore"}


settings = Settings()
