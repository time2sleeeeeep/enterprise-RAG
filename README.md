# 企业级 RAG 知识库

生产级检索增强生成系统，支持混合检索、语义重排序和完整的评估框架。

## 架构

```
┌─────────────────────────────────────────────────────────────────┐
│                        FastAPI 服务                               │
├─────────────┬──────────────────┬──────────────────┬─────────────┤
│  /chat      │  /documents      │  /eval           │  /health    │
└──────┬──────┴────────┬─────────┴────────┬─────────┴─────────────┘
       │               │                  │
       ▼               ▼                  ▼
┌─────────────────────────────────────────────────────────────────┐
│                      核心管线                                      │
│  查询改写 → 混合检索 → 重排序 → 生成                                │
└──────┬───────────────┬─────────────────┬────────────────────────┘
       │               │                 │
       ▼               ▼                 ▼
┌────────────┐  ┌─────────────┐  ┌─────────────────┐
│   Redis    │  │   Milvus    │  │     MySQL        │
│  (缓存)    │  │  (向量库)    │  │  (元数据)        │
└────────────┘  └─────────────┘  └─────────────────┘
```

## 核心特性

- **混合检索**：稠密向量（BGE-M3）+ 稀疏向量（BM25），RRF 融合（k=60）
- **语义重排序**：BGE-Reranker-v2-M3 精确相关性打分
- **两级缓存**：Redis 语义缓存（精确 MD5 + 向量相似度阈值 0.95）
- **查询改写**：基于 LLM 的查询扩展，提升召回率
- **多格式文档摄入**：支持 PDF、Markdown、DOCX，智能分块
- **消融实验评估**：RAGAS 风格指标（忠实度、相关性、精确率、召回率、正确性）
- **GPU 加速**：本地 BGE-M3 和重排序模型基于 CUDA 推理

## 性能指标

| 配置            | 忠实度       | 相关性      | 精确率      | 延迟        |
|:----------------|:-------------|:-----------|:-----------|:-----------|
| 基线配置        | 0.91         | 0.89       | 0.85       | 2.1s       |
| 无重排序        | 0.82         | 0.80       | 0.72       | 1.4s       |
| 仅稠密检索      | 0.85         | 0.83       | 0.76       | 1.8s       |
| 含查询改写      | 0.92         | 0.91       | 0.88       | 3.2s       |

## 技术栈

| 组件            | 技术方案                       |
|:----------------|:------------------------------|
| 框架            | FastAPI + Uvicorn             |
| 向量模型        | BAAI/bge-m3（1024维，本地部署）|
| 重排序模型      | BAAI/bge-reranker-v2-m3      |
| 向量数据库      | Milvus 2.4（HNSW, M=16）     |
| 缓存            | Redis 7                       |
| 元数据库        | MySQL 8.0                     |
| 大语言模型      | DeepSeek（兼容 OpenAI 接口）  |
| 分块策略        | RecursiveCharacterTextSplitter（512/64）|

## 快速开始

### 前置条件

- Docker & Docker Compose
- Python 3.10+
- NVIDIA GPU，支持 CUDA（建议 8GB+ 显存）
- DeepSeek API 密钥

### 1. 启动基础设施

```bash
docker compose up -d
```

启动以下服务：Milvus（standalone + etcd + minio）、MySQL、Redis、Attu（向量数据库管理界面）。

### 2. 安装依赖

```bash
cd enterprise-rag
pip install -e .
```

### 3. 配置环境变量

```bash
cp .env.example .env
# 编辑 .env 文件，填入你的 DeepSeek API 密钥
```

### 4. 启动服务

```bash
python -m src.main
# 或者
uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## API 接口

### 对话
```
POST /api/v1/chat/
Body: {"question": "什么是RAG？", "top_k": 5, "use_reranker": true, "use_cache": true}
```

### 文档管理
```
POST   /api/v1/documents/upload    # 上传并摄入文档（PDF/MD/DOCX）
GET    /api/v1/documents/          # 列出所有文档
DELETE /api/v1/documents/{doc_id}  # 删除文档及其向量数据
```

### 评估
```
POST /api/v1/eval/run       # 在数据集上运行评估
POST /api/v1/eval/ablation  # 运行消融实验（6 种配置）
```

### 健康检查
```
GET /health
```

## 项目结构

```
enterprise-rag/
├── src/
│   ├── main.py                 # FastAPI 应用 + 生命周期管理
│   ├── config.py               # Pydantic Settings（.env 配置）
│   ├── api/routes/
│   │   ├── chat.py             # RAG 管线接口
│   │   ├── documents.py        # 文档 CRUD + 摄入
│   │   └── eval.py             # 评估与消融实验
│   ├── core/
│   │   ├── embeddings.py       # BGE-M3 单例（稠密+稀疏）
│   │   ├── retriever.py        # 混合检索 + RRF 融合
│   │   ├── reranker.py         # BGE-Reranker-v2-M3
│   │   ├── generator.py        # DeepSeek LLM 生成
│   │   ├── query_rewriter.py   # LLM 查询扩展
│   │   ├── cache.py            # Redis 两级语义缓存
│   │   └── chunking.py         # 分块策略
│   ├── db/
│   │   ├── milvus_client.py    # Milvus 连接与集合管理
│   │   ├── mysql_client.py     # SQLAlchemy ORM 模型
│   │   └── redis_client.py     # 异步 Redis 客户端
│   ├── ingestion/
│   │   ├── loader.py           # 多格式文档加载
│   │   ├── parser.py           # PDF/MD/DOCX 解析器
│   │   └── pipeline.py         # 端到端摄入管线
│   └── evaluation/
│       ├── dataset.py          # 评估数据集加载
│       ├── metrics.py          # LLM-as-judge 指标
│       └── run_eval.py         # 消融实验运行器
├── eval_data/
│   └── sample_dataset.json     # 示例评估数据集
├── tests/
├── docker-compose.yml          # 基础设施服务
├── Dockerfile                  # 应用容器
├── pyproject.toml              # 依赖与配置
└── .env.example                # 环境变量模板
```

## 评估

运行消融实验，对比不同管线配置的效果：

```bash
# 通过 API 调用
curl -X POST http://localhost:8000/api/v1/eval/ablation \
  -H "Content-Type: application/json" \
  -d '{"dataset_path": "eval_data/sample_dataset.json"}'
```

消融实验测试 6 种配置：baseline（基线）、no_reranker（无重排序）、dense_only（仅稠密检索）、with_rewrite（含查询改写）、top3、top10。结果保存至 `eval_results/ablation_summary.json`。

## 设计决策

1. **混合检索优于单一稠密检索**：RRF 融合稠密+稀疏向量在精确率上稳定提升 8-12%
2. **两阶段检索**：先进行宽泛检索（top_k × 3），再通过重排序模型精确筛选 top-k，兼顾召回率与精确率
3. **语义缓存阈值 0.95**：高阈值避免缓存误命中，同时仍能捕获同义改写查询
4. **单例模型加载**：BGE-M3 和重排序模型仅加载一次并在请求间共享，避免 GPU 显存抖动
5. **LLM-as-judge 评估指标**：使用 DeepSeek 评估忠实度/相关性，提供比词汇重叠指标更细粒度的评分
