# MedAgent — 医疗 AI Agent

[![Python 3.10+](https://img.shields.io/badge/Python-3.10+-blue)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115+-green)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

**MedAgent** 是一个面向医疗场景的智能问答 Agent 系统，基于多引擎 RAG + 分层记忆 + ReAct 推理构建。系统从"查询-检索-生成"的流水线进化为"感知-记忆-推理-行动"的智能体循环。
> **数据集来源**：[Open-KG](http://data.openkg.cn/dataset/disease-information) · [cMedQA2](https://github.com/zhangsheng93/cMedQA2)  
> **参考项目**：[RAGQnASystem](https://github.com/honeyandme/RAGQnASystem) · [mem0](https://github.com/mem0ai/mem0)

---


## 架构概览

```
┌─────────────────────────────────────────────────────────────────┐
│              Frontend (静态文件 HTML/JS)                          │
├─────────────────────────────────────────────────────────────────┤
│                    FastAPI 路由层                                 │
│  /auth  /chat/stream  /sessions  /documents  /health            │
├─────────────────────────────────────────────────────────────────┤
│             MedicalChatService (编排中枢)                         │
│  ┌──────────┐  ┌──────────────┐  ┌──────────┐  ┌────────────┐  │
│  │ Tool 匹配 │  │ QueryRouter  │  │ 记忆系统  │  │ SafetyGuard│  │
│  │ (快速路径)│  │ (4 种模式)   │  │          │  │ (风险分级) │  │
│  └──────────┘  └──────┬───────┘  └──────────┘  └────────────┘  │
│                        ▼                                         │
│            执行模式: Tool / Chat / RAG / ReAct                    │
├─────────────────────────────────────────────────────────────────┤
│                   检索层 (HybridRetriever)                        │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────────┐    │
│  │ KGRetriever│  │ QARetriever│  │ ESBM25    │  │QueryNormalizer│
│  │ (Neo4j KG)│  │(Milvus ANN)│  │(BM25 关键词)│+ Intent      │
│  └──────────┘  └──────────┘  └──────────┘  └──────────────┘    │
│                       │ RRF + Cross-Encoder 重排序               │
│                       ▼                                          │
│              PromptBuilder + ContextAssembler                    │
│         (Schema-Driven 上下文装配, 优先级 + Token 预算管理)      │
├─────────────────────────────────────────────────────────────────┤
│                    生成层 (多 LLM 提供商)                          │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐       │
│  │ DeepSeek  │  │ ZhipuAI  │  │ Qwen     │  │ Ollama   │       │
│  │(API 官方) │  │ (智谱)   │  │ (通义)   │  │ (本地)   │       │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

---

## 核心特性

| 特性 | 说明 |
|------|------|
| **多模式路由** | LLM 语义路由（专用 DeepSeek 客户端）+ 规则回退，4 种执行模式 |
| **三引擎检索** | Neo4j KG（结构化）+ Milvus ANN（语义）+ ES BM25（关键词）并行检索 |
| **RRF 融合** | Dense + Sparse 倒数排名融合，跨源分数叠加 |
| **Cross-Encoder 精排** | RRF 结果二次排序，提升 top-k 准确率 |
| **Schema-Driven 上下文** | 优先级插槽 + 全局 Token 预算裁剪 |
| **分层记忆系统** | STM + LTM + Preference + GraphMemory，含自动 consolidation |
| **PostgreSQL 持久化** | 会话 / LTM 记忆 / 用户偏好统一存储，多租户隔离 |
| **LLM 偏好提取** | DeepSeek 异步提取用户偏好，规则提取作为同步回退 |
| **ReAct 多步推理** | Thought/Action/Observation 循环，最大 6 步，工具注册制 |
| **内置工具包** | 剂量计算、科室导诊、检查指标正常值查询 |
| **分级安全防护** | 红色急诊警告 + 黄色就医提醒 + 检索质量免责声明 |
| **多 LLM 提供商** | DeepSeek / ZhipuAI / Qwen / Ollama，运行时动态切换 |
| **SSE 流式响应** | ThreadPoolExecutor + asyncio.Queue 异步事件流 |
| **优雅降级** | 每个外部组件独立 try/except，不级联故障 |
| **健康追踪** | 全局组件注册表，统一 `/health` 端点 |

---

## 技术栈

| 分类 | 技术 |
|------|------|
| 框架 | FastAPI + Uvicorn |
| 数据库 | PostgreSQL + psycopg2 连接池 |
| 向量库 | Milvus / Zilliz Cloud |
| 关键词检索 | Elasticsearch (BM25) |
| 知识图谱 | Neo4j + py2neo |
| Embedding | BAAI/bge-small-zh-v1.5 (SentenceTransformers) |
| NER | RoBERTa + BiLSTM |
| 重排序 | Cross-Encoder |
| LLM | DeepSeek / ZhipuAI / Qwen / Ollama |
| 数据集 | DiseaseKG (Open-KG), cMedQA2 |

---

## 快速开始

### 环境要求

- Python >= 3.10
- PostgreSQL（必需，会话和记忆持久化）
- Neo4j (可选，KG 检索需要)
- Milvus / Zilliz Cloud (可选，向量检索需要)
- Elasticsearch (可选，BM25 检索需要)

### 安装

```bash
git clone https://github.com/hugswangyu/MedRAG.git
cd MedRAG
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 配置

复制 `.env.example` 为 `.env`，按需配置：

```ini
# LLM 提供商（至少配置一个）
DEEPSEEK_API_KEY=sk-your-key
ZHIPUAI_API_KEY=your-key
QWEN_API_KEY=your-key

# PostgreSQL（必需，替代 JSON 文件持久化）
PG_HOST=localhost
PG_PORT=5432
PG_USER=ragqa
PG_PASSWORD=ragqa123
PG_DATABASE=ragqa_memory

# 外部服务 URI（可选）
NEO4J_URI=http://localhost:7474
MILVUS_HOST=localhost
ES_HOSTS=http://localhost:9200
```

### 初始化数据库

```bash
# 创建数据库
createdb ragqa_memory

# 建表
psql -d ragqa_memory -f scripts/create_tables.sql
```

### 启动

```bash
uvicorn medrag.app.server:app --host 0.0.0.0 --port 8000 --reload
```

---

## 项目结构

```
src/medrag/
├── app/              # FastAPI 路由层
│   ├── api/          #   auth / chat / sessions / documents
│   ├── server.py     #   应用入口
│   ├── schemas.py    #   Pydantic 模型
│   └── ...
├── service/          # MedicalChatService 编排核心
├── retrieval/        # 检索层
│   ├── hybrid_retriever.py  # 多源检索 + RRF 融合
│   ├── router.py            # 双模路由 (LLM + 规则)
│   ├── kg_retriever.py      # Neo4j KG 检索
│   ├── es_retriever.py      # ES BM25 检索
│   └── reranker.py          # Cross-Encoder 重排序
├── vectors/          # 向量检索
│   ├── qa_retriever.py      # Milvus ANN 检索
│   ├── embedding.py         # BGE Embedding 模型
│   └── milvus_client.py     # Milvus 客户端封装
├── memory/           # 分层记忆系统
│   ├── short_term.py       # STM 滑动窗口
│   ├── long_term.py        # LTM 语义召回 + 持久化
│   ├── graph_memory.py     # 图感知记忆
│   ├── preference.py       # 用户偏好提取
│   └── schema.py           # ContextAssembler
├── react/            # ReAct 多步推理引擎
├── rag/              # RAG 流水线
│   ├── prompt_builder.py   # 提示词构建 (双层设计)
│   ├── answer_generator.py # 流式 / 同步生成
│   └── safety_guard.py     # 红/黄分级安全防护
├── tools/            # 内置工具包
│   ├── dosage_calculator.py
│   ├── department_guide.py
│   └── normal_range.py
├── llm/              # LLM 客户端工厂
├── infrastructure/   # 基础服务
│   └── storage/           # 存储后端
│       └── postgres_client.py  # PostgreSQL 持久化（LTM/会话/偏好）
├── ner/              # 命名实体识别
├── config/           # 集中化配置
└── scripts/          # 数据库脚本
    └── create_tables.sql    # 建表（LTM/会话/偏好）
```

---

## 数据流（RAG 模式）

```
用户输入 → 鉴权 → POST /chat/stream (SSE)
  │
  ├─ ToolRegistry.match() → 工具命中? 直接返回
  │
  ├─ QueryRouter.route() → 决定执行模式 + 检索源
  │
  ├─ SafetyGuard.detect_risk() → 红/黄风险标记
  │
  ├─ HybridRetriever.retrieve()
  │     ├─ KGRetriever (Neo4j)
  │     ├─ QARetriever (Milvus ANN)
  │     ├─ ESBM25 (BM25)
  │     └─ RRF 融合 → Cross-Encoder 精排
  │
  ├─ MemorySystem.build_context() → STM + LTM + 偏好
  ├─ ContextAssembler.assemble() → 优先级裁剪
  ├─ PromptBuilder → System Prompt + 上下文
  ├─ AnswerGenerator.generate_stream() → LLM
  └─ SafetyGuard.append_safety_notice() → 返回
```

---

## API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/auth/login` | POST | 用户登录 |
| `/auth/register` | POST | 用户注册 |
| `/chat/stream` | POST | SSE 流式聊天 |
| `/chat/models` | GET | 可用 LLM 模型列表 |
| `/sessions` | GET/POST/DELETE | 会话管理 |
| `/documents` | GET/POST/DELETE | 文档管理 |
| `/health` | GET | 组件健康状态 |

---

## 许可证

[MIT License](LICENSE)
