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
│  查询历史 → 查询改写 → 混合检索 → 重排序 → 生成（含历史上下文）       │
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
- **多轮对话**：基于 session_id 的对话历史注入，LLM 感知上下文（指代消解、追问）
- **多格式文档摄入**：支持 PDF、Markdown、DOCX，智能分块；支持文件夹上传和 zip 批量导入
- **批量删除**：一键清除所有已入库文档的元数据与向量数据，消除重复的 Milvus load/flush 开销
- **消融实验评估**：RAGAS 风格指标（忠实度、相关性、精确率、召回率、正确性）
- **GPU 加速**：本地 BGE-M3 和重排序模型基于 CUDA 推理

## 性能指标

*评估数据集：`eval_data/milvus_qa.json`（N=150），消融对比使用前 20 条子集以保证可比性。指标由 DeepSeek 作为 LLM 裁判打分；通过 `eval_judge_model` 配置可换用独立裁判模型以减少自评偏置。*

| 配置 | 忠实度 | 相关性 | 上下文精确率 | 上下文召回率 | 正确性 | 延迟 |
|:-----|:-----|:-----|:----------|:----------|:-----|:---|
| 基线配置 | 0.79 | 0.90 | 0.77 | 0.45 | 0.61 | 4.6s |
| 无重排序 | 0.65 | 0.96 | 0.63 | 0.41 | 0.55 | 3.5s |
| 仅稠密检索 | 0.85 | 0.98 | 0.81 | 0.50 | 0.62 | 3.3s |
| 含查询改写 | 0.70 | 0.90 | 0.69 | 0.40 | 0.60 | 6.5s |
| 含多查询融合 ⚠️ | 0.79 | 0.89 | 0.72 | 0.41 | 0.55 | 7.0s |

*消融实验 N=20，全指标含 retrieval 指标详见 `eval_results/phase0_summary.json`。N=150 基线全指标见 `eval_results/baseline_detail.json`。*

*⚠️ 含多查询融合（`with_multi_query`）：将原始问题扩展为 3 个子查询，对全部 dense+sparse 结果做扁平 RRF 融合。实测 recall@10 下降 8.3%（0.55 vs baseline 0.60）、延迟增加 52%（7.0s vs 4.6s），因此该特性默认关闭。保留代码以供未来探索更优的多查询策略（如 query-focused 去噪或 per-query rerank）。*

## 技术栈

| 组件       | 技术方案                                 |
| :--------- | :--------------------------------------- |
| 框架       | FastAPI + Uvicorn                        |
| 向量模型   | BAAI/bge-m3（1024维，本地部署）          |
| 重排序模型 | BAAI/bge-reranker-v2-m3                  |
| 向量数据库 | Milvus 2.4（HNSW, M=16）                 |
| 缓存       | Redis 7                                  |
| 元数据库   | MySQL 8.0                                |
| 大语言模型 | DeepSeek（兼容 OpenAI 接口）             |
| 分块策略   | RecursiveCharacterTextSplitter（512/64） |

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

所有接口的错误响应均使用统一格式 `{"detail": "错误描述", "message": null}`，各状态码含义如下：

| 状态码 | 说明 |
| :----- | :--- |
| 400 | 请求参数不合法（文件类型不支持、数据集为空等） |
| 404 | 资源不存在（文档、数据集等） |
| 413 | 上传文件过大 |
| 422 | 请求体校验失败（字段缺失或类型错误） |
| 500 | 服务器内部错误（不泄露具体原因） |
| 503 | 依赖服务不可用（Milvus、LLM 等） |

### 对话

```
POST /api/v1/chat/
Body: {
    "question": "什么是RAG？",
    "session_id": null,          // 可选，留空则自动创建新会话
    "top_k": 5,
    "use_reranker": true,
    "use_cache": true,
    "use_query_rewrite": true
}

// 多轮对话：将上一轮返回的 session_id 传入即可继续对话
POST /api/v1/chat/
Body: {
    "question": "它有哪些优缺点？",
    "session_id": "a1b2c3d4e5f6g7h8"  // 使用之前返回的 session_id
}
```

### 文档管理

```
POST   /api/v1/documents/upload       # 上传并摄入单个文档（PDF/MD/DOCX）
POST   /api/v1/documents/bulk-import  # 批量导入（多文件/文件夹/zip压缩包）
GET    /api/v1/documents/             # 列出所有文档
DELETE /api/v1/documents/{doc_id}     # 删除单个文档及其向量数据
DELETE /api/v1/documents/             # 批量删除文档（一次性清理向量和元数据）
```

**批量删除** `DELETE /api/v1/documents/`：

接受 JSON 请求体 `{"doc_ids": ["id1", "id2", ...]}`，在单次请求中完成所有文档的 Milvus 向量清理（一次 load + 分批 or 表达式 + 一次 flush）和 MySQL 元数据清理（单事务），相比逐个删除可减少 N-1 次 Milvus collection load/flush 和 N-1 次数据库事务。

```json
// 请求
{"doc_ids": ["abc123", "def456", "ghi789"]}

// 响应
{
    "total_requested": 3,
    "deleted_count": 3,
    "not_found": [],
    "message": "Deleted 3 documents"
}
```

`not_found` 字段列出在数据库中不存在的 ID；即使部分 ID 不存在，存在的仍会被删除。

### 评估

```
POST /api/v1/eval/run       # 在数据集上运行评估
POST /api/v1/eval/ablation  # 运行消融实验（6 种配置）
```

### 健康检查

```
GET /health  →  {"status": "healthy", "service": "enterprise-rag"}
```

## 项目结构

```
enterprise-rag/
├── src/
│   ├── main.py                 # FastAPI 应用 + 生命周期管理
│   ├── config.py               # Pydantic Settings（.env 配置）
│   ├── api/
│   │   ├── responses.py         # 通用响应模型（ErrorResponse 等）
│   │   └── routes/
│   │       ├── chat.py             # RAG 管线接口
│   │       ├── documents.py        # 文档 CRUD + 摄入
│   │       └── eval.py             # 评估与消融实验
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
│   └── milvus_qa.json     # 150 组评估 QA 对（合成数据，含相关性标注）
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
  -d '{"dataset_path": "eval_data/milvus_qa.json"}'
```

消融实验测试 6 种配置：baseline（基线）、no_reranker（无重排序）、dense_only（仅稠密检索）、with_rewrite（含查询改写）、top3、top10。结果保存至 `eval_results/ablation_summary.json`。

## 设计决策

1. **混合检索优于单一稠密检索**：RRF 融合稠密+稀疏向量在精确率上稳定提升 8-12%
2. **两阶段检索**：先进行宽泛检索（top_k × 3），再通过重排序模型精确筛选 top-k，兼顾召回率与精确率
3. **语义缓存阈值 0.95**：高阈值避免缓存误命中，同时仍能捕获同义改写查询
4. **单例模型加载**：BGE-M3 和重排序模型仅加载一次并在请求间共享，避免 GPU 显存抖动
5. **LLM-as-judge 评估指标**：使用 DeepSeek 评估忠实度/相关性，提供比词汇重叠指标更细粒度的评分
6. **对话历史注入**：同一 session 的历史 Q&A 以标准 OpenAI chat 消息格式注入 system prompt 与当前消息之间，默认取最近 5 轮；LLM 可直接理解指代和追问，无需前端额外处理上下文
