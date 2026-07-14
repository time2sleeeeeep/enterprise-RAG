# 企业级 RAG 知识库

生产级检索增强生成系统，支持混合检索、语义重排序和完整的评估框架。

## 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                         FastAPI 服务                               │
├──────────────┬──────────────────┬──────────────────┬─────────────┤
│  /chat       │  /documents      │  /eval           │  /health    │
└──────┬───────┴────────┬─────────┴────────┬─────────┴─────────────┘
       │                │                  │
       ▼                ▼                  ▼
┌──────────────────────────────────────────────────────────────────┐
│                       核心管线（/chat）                             │
│  缓存检查 → (查询改写) → 混合检索 → (重排序) → 历史注入 → LLM 生成   │
└──────┬────────────────┬──────────────────┬───────────────────────┘
       │                │                  │
       ▼                ▼                  ▼
┌────────────┐  ┌──────────────┐  ┌──────────────────┐
│   Redis    │  │   Milvus     │  │     MySQL         │
│  (缓存)    │  │  (向量库)     │  │  (元数据/历史)     │
└────────────┘  └──────────────┘  └──────────────────┘
```

## 核心特性

- **混合检索**：稠密向量（BGE-M3）+ 稀疏向量（BM25），RRF 融合（k=60）
- **语义重排序**：BGE-Reranker-v2-M3 精确相关性打分
- **两级缓存**：Redis 语义缓存（精确 MD5 + 向量相似度阈值 0.95）
- **查询改写**：基于 LLM 的查询扩展（默认关闭，实验性功能）
- **多查询融合**：LLM 扩展 3 个子查询做扁平 RRF 融合（默认关闭，实验性功能）
- **多轮对话**：基于 session_id 的对话历史注入，LLM 感知上下文（指代消解、追问），默认注入最近 5 轮
- **多格式文档摄入**：支持 PDF、Markdown、DOCX，递归分块 + 语义分块双策略；支持文件夹上传和 zip 批量导入
- **批量删除**：一键清除所有已入库文档的元数据与向量数据，消除重复的 Milvus load/flush 开销
- **消融实验评估**：RAGAS 风格指标（忠实度、相关性、精确率、召回率、正确性）+ 检索指标（Prec@K、Rec@K、MRR、NDCG@K），含统计评估（均值、标准差、置信区间）
- **依赖探活**：`/health` 端点并发探测 Milvus/MySQL/Redis 状态，任一不可用返回 503 degraded
- **GPU 加速**：本地 BGE-M3 和重排序模型基于 CUDA 推理

## 性能指标

*评估数据集 N=70，每配置 3 次独立运行取均值（Bootstrap 95% 置信区间）。指标由 DeepSeek 作为 LLM 裁判打分（忠实度采用声明级评估）；通过 `EVAL_JUDGE_MODEL` 环境变量可指定独立裁判模型以减少自评偏置。延迟为分组件计时之和（检索 + 重排序 + 生成），不含查询改写/多查询扩展的额外 LLM 调用及 Python 框架开销，实际端到端耗时略高。*

### 生成与答案质量（LLM-as-Judge）

| 配置 | 忠实度 | 答案相关性 | 上下文精确率 | 上下文召回率 | 正确性 | 延迟 |
|:-----|:-----|:-----|:----------|:----------|:-----|:---|
| 基线配置 | 0.9445 | 0.9240 | 0.7734 | 0.7819 | 0.6783 | 3.62s |
| 无重排序 | 0.9352 | 0.9386 | 0.6055 | 0.7213 | 0.6593 | 3.94s |
| 仅稠密检索 | 0.9067 | 0.9405 | 0.8135 | 0.7907 | 0.7195 | 3.45s |
| 含查询改写 | 0.9181 | 0.8867 | 0.6933 | 0.7312 | 0.6410 | 3.61s |
| top3 | 0.9414 | 0.9205 | 0.7460 | 0.7200 | 0.6626 | 2.94s |
| top10 | 0.8300 | 0.9107 | 0.7169 | 0.7682 | 0.7283 | 3.78s |
| 含多查询融合 ⚠️ | 0.9138 | 0.9076 | 0.7706 | 0.7632 | 0.7024 | 3.31s |

### 检索指标

| 配置 | Prec@5 | Rec@5 | MRR | NDCG@10 | Hit@5 |
|:-----|:-----|:-----|:-----|:-----|:-----|
| 基线配置 | 0.1371 | 0.6857 | 0.5736 | 0.6017 | 0.6857 |
| 无重排序 | 0.1314 | 0.6571 | 0.4267 | 0.4841 | 0.6571 |
| 仅稠密检索 | 0.1429 | 0.7143 | 0.6052 | 0.6328 | 0.7143 |
| 含查询改写 | 0.1295 | 0.6476 | 0.5012 | 0.5380 | 0.6476 |
| top3 | 0.2048 | 0.6143 | 0.5452 | 0.5631 | 0.6143 |
| top10 | 0.1343 | 0.6714 | 0.5663 | 0.5993 | 0.6714 |
| 含多查询融合 ⚠️ | 0.1343 | 0.6714 | 0.5752 | 0.5996 | 0.6714 |

*延迟为分组件计时之和（dense_search + sparse_search + reranking + generation），来自 3 次运行 × 70 样本的均值。`with_rewrite` 不含查询改写 LLM 调用耗时；`with_multi_query` 的 dense/sparse 检索耗时未通过 LatencyTracker 记录，实际端到端延迟显著高于表中数值。延迟百分位（p50/p95/p99）及各指标 95% 置信区间详见评估输出 JSON。*

*⚠️ 含多查询融合（`with_multi_query`）：将原始问题扩展为 3 个子查询，对全部 dense+sparse 结果做扁平 RRF 融合。检索与生成指标均无显著提升，且额外 LLM 调用增加端到端延迟，因此该特性默认关闭。保留代码以供未来探索更优的多查询策略（如 query-focused 去噪或 per-query rerank）。*

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
| 分块策略   | RecursiveCharacterTextSplitter（512/64）+ SemanticChunker |

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
# 一键启动（基础设施 + 应用）
./services.sh start

# 或者手动启动应用
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
    "top_k": 5,                  // 最终返回文档数，内部宽召回使用 top_k × 3
    "use_reranker": true,
    "use_cache": true,
    "use_query_rewrite": false,  // 实验性，默认关闭
    "use_multi_query": false     // 实验性，默认关闭
}

// 响应
{
    "answer": "RAG（检索增强生成）是一种...",
    "sources": [
        {
            "content": "RAG 是检索增强生成技术的简称...",   // 截断至 200 字符
            "source": "rag_intro.pdf",
            "page_num": 3,
            "score": 0.9521
        }
    ],
    "session_id": "a1b2c3d4e5f6g7h8",   // 多轮对话时传入此值
    "cached": false                       // 是否命中语义缓存
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

**上传单个文档** `POST /api/v1/documents/upload`：

```bash
curl -X POST http://localhost:8000/api/v1/documents/upload \
  -F "file=@report.pdf"
```

```json
// 响应
{"doc_id": "abc123", "filename": "report.pdf", "chunk_count": 15, "message": "Ingested successfully"}
```

重传同名文件时，旧向量和元数据会先被清除再重新摄入，不会产生孤儿向量或主键冲突。

**列出文档** `GET /api/v1/documents/`：

```json
// 响应
[
    {"id": "abc123", "filename": "report.pdf", "file_type": "pdf", "chunk_count": 15},
    {"id": "def456", "filename": "intro.md",   "file_type": "md",  "chunk_count": 8}
]
```

**删除单个文档** `DELETE /api/v1/documents/{doc_id}`：

```json
// 响应
{"message": "Document deleted successfully", "doc_id": "abc123"}
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

**批量导入** `POST /api/v1/documents/bulk-import`：

支持三种上传模式，单次请求可混合携带多个文件：

```bash
# 多文件上传
curl -X POST http://localhost:8000/api/v1/documents/bulk-import \
  -F "files=@doc1.pdf" \
  -F "files=@doc2.md" \
  -F "max_concurrency=4"

# 文件夹上传（webkitdirectory，客户端发送带相对路径的文件）
curl -X POST http://localhost:8000/api/v1/documents/bulk-import \
  -F "files=@docs/chapter1.md" \
  -F "files=@docs/chapter2.md" \
  -F "files=@docs/sub/basics.pdf"

# zip 压缩包上传
curl -X POST http://localhost:8000/api/v1/documents/bulk-import \
  -F "files=@archive.zip"
```

参数说明：
- `files`：一个或多个文件（.pdf / .md / .markdown / .docx / .zip），必需
- `max_concurrency`：并发摄入上限（默认 4，最大 8），通过 `asyncio.Semaphore` 控制
- zip 包内的隐藏文件（`.` 开头）和 `__MACOSX` 目录自动跳过
- 文件名重复时，旧文档的向量和元数据会先被清除再重新摄入

```json
// 响应
{
    "total": 5,
    "success_count": 4,
    "failed_count": 1,
    "skipped_count": 0,
    "results": [
        {"filename": "doc1.pdf",  "status": "success", "doc_id": "abc123", "chunk_count": 12, "error": null},
        {"filename": "doc2.md",   "status": "success", "doc_id": "def456", "chunk_count": 8,  "error": null},
        {"filename": "bad.docx",  "status": "failed",  "doc_id": null,     "chunk_count": null, "error": "Unsupported format"}
    ]
}

### 评估

```
POST /api/v1/eval/run        # 运行评估（支持 7 种消融配置 + 自定义配置）
GET  /api/v1/eval/datasets   # 列出可用的评估数据集
```

消融实验、报告生成、数据集生成等离线任务请使用 CLI（详见下方[评估](#评估)章节）。

### 健康检查

```
GET /health

// 所有依赖正常 → 200
{
    "status": "healthy",
    "service": "enterprise-rag",
    "dependencies": {"milvus": "up", "mysql": "up", "redis": "up"}
}

// 任一依赖不可用 → 503
{
    "status": "degraded",
    "service": "enterprise-rag",
    "dependencies": {"milvus": "up", "mysql": "down: OperationalError", "redis": "up"}
}
```

健康检查并发探测 Milvus（`list_collections`）、MySQL（`SELECT 1`）、Redis（`PING`），各依赖独立 2s 超时，sync 探活丢入独立线程池避免耗尽主事件循环。

## 项目结构

```
enterprise-rag/
├── src/
│   ├── main.py                 # FastAPI 应用 + 生命周期管理
│   ├── config.py               # Pydantic Settings（.env 配置）
│   ├── api/
│   │   ├── health.py            # 依赖探活（Milvus/MySQL/Redis）
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
│   │   └── chunking.py         # 递归分块 + 语义分块
│   ├── db/
│   │   ├── milvus_client.py    # Milvus 连接与集合管理
│   │   ├── mysql_client.py     # SQLAlchemy ORM 模型
│   │   └── redis_client.py     # 异步 Redis 客户端
│   ├── ingestion/
│   │   ├── bulk.py             # 批量导入（多文件/文件夹/zip）
│   │   ├── loader.py           # 多格式文档加载
│   │   ├── parser.py           # PDF/MD/DOCX 解析器
│   │   └── pipeline.py         # 端到端摄入管线
│   └── evaluation/
│       ├── dataset.py          # 评估数据集加载
│       ├── metrics.py          # LLM-as-judge 生成指标
│       ├── retrieval_metrics.py # 检索指标（Prec@K、Rec@K、MRR、NDCG@K）
│       ├── statistical.py      # 统计评估（均值、标准差、置信区间）
│       ├── synthetic_data.py   # 合成数据生成（QA 对生成 + 相关性标注）
│       ├── latency_tracker.py  # 延迟分解追踪
│       ├── report.py           # 评估报告生成
│       ├── run_eval.py         # 消融实验运行器
│       └── cli.py              # 评估 CLI
├── eval_data/
│   ├── milvus_qa.json      # 150 组合成 QA 对（含相关性标注）
│   ├── milvus_qa_70.json   # 70 组（消融实验用）
│   ├── milvus_qa_20.json   # 20 组（快速测试）
│   └── milvus_qa_smoke5.json # 5 组（冒烟测试）
├── tests/
├── docker-compose.yml          # 基础设施服务
├── Dockerfile                  # 应用容器
├── pyproject.toml              # 依赖与配置
└── .env.example                # 环境变量模板
```

## 评估

系统提供完整的评估框架，支持 **CLI 命令行**和 **REST API** 两种使用方式。评估指标分为 LLM-as-Judge 生成指标和纯计算检索指标两类，支持单次评估、多轮统计评估和消融实验。

### 评估指标

**生成与答案质量（LLM-as-Judge）** — 由 LLM 裁判打分，0.0~1.0：

| 指标 | 说明 |
|:-----|:-----|
| `faithfulness` | 答案是否忠实于检索上下文（朴素整体打分） |
| `faithfulness_claim` | 声明级忠实度：拆解原子声明 → 逐条验证 → 返回支持率（更严格，需额外 LLM 调用） |
| `answer_relevancy` | 答案与问题的相关程度 |
| `context_precision` | 检索上下文对问题的精确率（MAP 风格，逐条判定相关/不相关） |
| `context_recall` | 检索上下文对标准答案的覆盖程度 |
| `correctness` | 答案与标准答案的事实一致性 |

**检索指标** — 纯公式计算，毫秒级，无需 LLM 调用（依赖数据集中的 `relevance_labels` 标注）：

| 指标 | 说明 |
|:-----|:-----|
| `precision_at_k` | 前 K 个检索结果中相关文档的占比（Prec@K） |
| `recall_at_k` | 所有相关文档中被前 K 个结果覆盖的比例（Rec@K） |
| `mrr` | 第一个相关文档排名的倒数（Mean Reciprocal Rank） |
| `ndcg_at_k` | 归一化折损累计增益，支持多级相关性（0/1/2） |
| `hit_rate_at_k` | 前 K 个结果中是否至少命中一个相关文档 |
| `map_score` | 平均精度（Mean Average Precision） |

### 消融配置

预置 7 种管线配置，覆盖核心组件的消融对比：

| 配置名 | 检索方式 | 重排序 | 查询改写 | 多查询融合 | top_k | 说明 |
|:-----|:-----|:-----|:-----|:-----|:-----|:-----|
| `baseline` | 混合 | ✅ | ❌ | ❌ | 5 | 默认生产配置 |
| `no_reranker` | 混合 | ❌ | ❌ | ❌ | 5 | 消融重排序的影响 |
| `dense_only` | 仅稠密 | ✅ | ❌ | ❌ | 5 | 消融稀疏检索的影响 |
| `with_rewrite` | 混合 | ✅ | ✅ | ❌ | 5 | 测试查询改写效果 |
| `top3` | 混合 | ✅ | ❌ | ❌ | 3 | 减少上下文窗口 |
| `top10` | 混合 | ✅ | ❌ | ❌ | 10 | 扩大上下文窗口 |
| `with_multi_query` | 混合 | ✅ | ❌ | ✅ | 5 | 多查询扁平 RRF 融合（实验性） |

### CLI 命令行

```bash
python -m src.evaluation.cli <command> [options]
```

**列出可用数据集：**

```bash
python -m src.evaluation.cli list-datasets
# 输出示例：
#   milvus_qa.json         150 samples  relevance_labels=✓
#   milvus_qa_70.json       70 samples  relevance_labels=✓
#   milvus_qa_20.json       20 samples  relevance_labels=✓
#   milvus_qa_smoke5.json    5 samples  relevance_labels=✓
```

**运行单次评估：**

```bash
# 使用预置配置 baseline，对 70 条样本运行 1 次评估
python -m src.evaluation.cli run \
  --dataset eval_data/milvus_qa_70.json \
  --config baseline \
  --output-dir eval_results

# 多轮统计模式（3 次运行，输出均值 ± 标准差 + 95% 置信区间）
python -m src.evaluation.cli run \
  --dataset eval_data/milvus_qa_70.json \
  --config dense_only \
  --runs 3
```

**运行消融实验（全部 7 种配置 × 多轮统计）：**

```bash
# 每种配置运行 3 次，汇总统计结果
python -m src.evaluation.cli ablation \
  --dataset eval_data/milvus_qa_70.json \
  --runs 3 \
  --output-dir eval_results
```

输出文件：
- `eval_results/ablation_statistical_summary.json` — 所有配置的汇总统计
- `eval_results/<config_name>_stats_detail.json` — 单配置的详细统计
- `eval_results/<config_name>_detail.json` — 单次运行的逐样本结果

**生成评估报告：**

```bash
# 从 eval_results/ 中的 JSON 结果生成可视化报告
python -m src.evaluation.cli report \
  --results-dir eval_results \
  --format both        # html | markdown | both

# 输出：eval_results/report.html、eval_results/report.md
```

HTML 报告为单文件自包含（内联 CSS + SVG 图表），无需外部依赖，可直接在浏览器中打开。

**生成合成评估数据集：**

```bash
# 从已摄入的 Milvus 文档中采样 chunk，LLM 自动生成 QA 对并标注相关性
python -m src.evaluation.cli generate-dataset \
  --num-samples 50 \
  --sampling stratified \
  --output eval_data/milvus_qa_custom.json \
  --seed 42
```

参数说明：
- `--num-samples`：目标 QA 对数量（默认 50）
- `--sampling`：采样策略 — `random`（随机）| `stratified`（分层，按文档分布）
- `--output`：输出 JSON 文件路径
- `--seed`：随机种子（默认 42）
- `--validate-size`：验证阶段抽查数量（默认 10）

生成的数据集每条包含 `question`、`ground_truth`、`relevance_labels`（chunk_id → 0/1/2）、`difficulty`（simple/medium/complex）等字段，可直接用于评估。

### REST API

**运行评估 `POST /api/v1/eval/run`：**

```json
// 快速模式（n_runs=1）
{
    "dataset_path": "eval_data/milvus_qa_70.json",
    "config_name": "baseline",
    "n_runs": 1,
    "enable_claim_faithfulness": false
}

// 统计模式（n_runs>=2，含均值 ± 标准差 + 95%CI）+ 声明级忠实度
{
    "dataset_path": "eval_data/milvus_qa_70.json",
    "config_name": "baseline",
    "n_runs": 3,
    "enable_claim_faithfulness": true
}

// 自定义配置（config_name="api_eval" 时启用自定义参数）
{
    "dataset_path": "eval_data/milvus_qa_70.json",
    "config_name": "api_eval",
    "top_k": 8,
    "use_reranker": true,
    "use_query_rewrite": true,
    "use_multi_query": false,
    "n_runs": 1
}
```

响应包含 `metrics`（各指标均值）、`avg_service_latency`（业务延迟）、`avg_scoring_latency`（评估开销）、`latency_breakdown`（分组件耗时）、`latency_percentiles`（p50/p95/p99）。多轮模式下额外包含 `metric_stats`（均值/标准差/置信区间）。

**列出数据集 `GET /api/v1/eval/datasets`：**

```json
// 响应
{
    "datasets": [
        {"name": "milvus_qa.json", "num_samples": 150, "has_relevance_labels": true},
        {"name": "milvus_qa_70.json", "num_samples": 70, "has_relevance_labels": true}
    ]
}
```

### 延迟分解

评估管线对每个样本记录 6 个组件的耗时（毫秒），可用于定位性能瓶颈：

| 组件 | 说明 |
|:-----|:-----|
| `embedding_ms` | 查询向量编码耗时 |
| `dense_search_ms` | Milvus 稠密向量检索耗时 |
| `sparse_search_ms` | Milvus 稀疏向量检索耗时 |
| `reranking_ms` | BGE-Reranker 重排序耗时 |
| `generation_ms` | DeepSeek LLM 生成耗时（用户感知的业务延迟终点） |
| `scoring_ms` | LLM 裁判打分耗时（不属业务延迟，仅为评估开销） |

业务延迟（`avg_service_latency`）= 检索 + 重排序 + 生成，即用户实际等待时间。LLM 裁判打分（`scoring_ms`）在业务延迟之后发生，不计入用户体验。

### 配置参考

```bash
# .env 中评估相关配置项及默认值
EVAL_JUDGE_MODEL=              # 裁判模型名，为空则回退到 DEEPSEEK_MODEL
EVAL_DEFAULT_RUNS=3            # 统计评估默认运行次数
EVAL_OUTPUT_DIR=eval_results   # 评估结果输出目录
EVAL_CLAIM_BATCH_VERIFY=true   # 声明级忠实度：批量验证（一次 LLM 调用）vs 逐条验证
CHAT_HISTORY_MAX_TURNS=5       # 注入 LLM 上下文的历史对话轮数
```

## 设计决策

1. **混合检索 + 重排序是核心质量保障**：消融数据显示，移除重排序后上下文精确率从 0.7734 降至 0.6055（-22%），MRR 从 0.5736 降至 0.4267（-26%），表明重排序对最终答案质量的影响大于检索方式的差异。稠密+稀疏混合检索提供了多路互补的候选池，为重排序阶段提供更丰富的输入。
2. **两阶段检索**：先进行宽泛检索（top_k × 3），再通过重排序模型精确筛选 top-k，兼顾召回率与精确率
3. **语义缓存阈值 0.95**：高阈值避免缓存误命中，同时仍能捕获同义改写查询
4. **单例模型加载**：BGE-M3 和重排序模型仅加载一次并在请求间共享，避免 GPU 显存抖动
5. **LLM-as-judge 评估指标**：使用 LLM 裁判评估忠实度/相关性/正确性，提供比词汇重叠指标更细粒度的评分。支持通过 `EVAL_JUDGE_MODEL` 环境变量指定独立裁判模型，缓解「生成模型 = 评分模型」的自我偏好偏置。
6. **对话历史注入**：同一 session 的历史 Q&A 以标准 OpenAI chat 消息格式注入 system prompt 与当前消息之间，默认取最近 5 轮；LLM 可直接理解指代和追问，无需前端额外处理上下文
