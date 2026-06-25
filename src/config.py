# 全局配置模块：通过 pydantic-settings 从 .env 文件加载所有服务配置，
# 包括 DeepSeek API、Milvus、MySQL、Redis、嵌入模型、重排序模型及服务器参数。

from pydantic_settings import BaseSettings
from pydantic import Field


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

    # Embedding
    embedding_model_name: str = Field(default="BAAI/bge-m3")
    embedding_device: str = Field(default="cuda")
    embedding_batch_size: int = Field(default=32)

    # Reranker
    reranker_model_name: str = Field(default="BAAI/bge-reranker-v2-m3")
    reranker_device: str = Field(default="cuda")

    # Retrieval
    dense_top_k: int = Field(default=20)
    sparse_top_k: int = Field(default=20)
    rerank_top_k: int = Field(default=5)
    rrf_k: int = Field(default=60)

    # Server
    server_host: str = Field(default="0.0.0.0")
    server_port: int = Field(default=8000)

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

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
