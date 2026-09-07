---
sidebar_position: 5
title: RAGFlow 架构与源码学习指南
sidebar_label: RAGFlow 架构与源码学习指南
slug: /ragflow_architecture_learning_zh
sidebar_custom_props: {
  categoryIcon: LucideBookOpen
}
---

# RAGFlow 架构与源码学习指南

本文面向希望从“能够启动”进一步走到“能够解释、调试和修改 RAGFlow”的开发者。内容以当前仓库 `v0.27.1` 的源码为准，重点解释 Python 主链路，同时说明正在演进的 Go 实现、前端结构、数据存储及推荐的源码阅读顺序。

运行环境、MinIO 的职责、macOS 启停及版本升级步骤，参见配套文档：[macOS 源码开发与版本升级](./macos_source_development_zh.md)。

本文是整个学习体系的总览和路线图。需要深入源码时，继续阅读以下分册：

| 分册 | 重点内容 |
|---|---|
| [第一册：运行时、配置与数据模型](./ragflow_learning/01_runtime_configuration_and_data.md) | 进程、配置覆盖、四类存储、Peewee 模型和 Service 边界 |
| [第二册：文档摄取与深度解析](./ragflow_learning/02_ingestion_and_parsing.md) | 上传、任务、队列、解析器、Chunk、Embedding 和索引 |
| [第三册：检索、问答与引用](./ragflow_learning/03_retrieval_chat_and_citations.md) | 全文/向量召回、融合、重排、Prompt、流式生成和引用 |
| [第四册：Agent Canvas 与前端](./ragflow_learning/04_agent_and_frontend.md) | React 请求链、后端分流、DSL、组件、变量和工作流事件 |
| [第五册：Go 实现、测试与调试](./ragflow_learning/05_go_testing_and_debugging.md) | Go 分层、Ingestor、CGO、测试体系和跨进程诊断 |

总览用于建立地图，分册用于逐模块精读和动手实验。两者不是重复关系。

> 本项目迭代较快。学习时应以“稳定的职责和调用链”为主，不要依赖某个函数的固定行号。升级后可以用本文给出的类名、函数名和 `rg` 命令重新定位代码。

## 1. 学习目标

完成本文的学习与练习后，应该能够回答以下问题：

1. 一个上传的 PDF 分别在 MySQL、MinIO、Redis 和 Elasticsearch 中留下了什么？
2. API 服务与任务执行器为什么必须分别启动？
3. 文档从原始二进制变成可检索切片，经历了哪些阶段？
4. 一次聊天请求如何完成查询改写、混合检索、重排、提示词组装、模型调用和引用回填？
5. `rag/app` 与 `deepdoc/parser` 为什么不是同一层解析器？
6. Agent Canvas 如何把一份 DSL 变成可执行组件图？
7. 前端如何在 Python、Go 和混合后端之间切换？
8. 修改某类功能时，应该从哪个“拥有该行为”的模块开始？

## 2. 先建立整体心智模型

RAGFlow 不是“在大模型前拼一段向量检索”的单体程序。它同时包含文档资产管理、异步摄取、深度文档解析、搜索索引、RAG 问答、Agent 工作流和多模型适配。

```text
                         ┌─────────────────────┐
                         │ React + Vite 前端    │
                         │ web/                │
                         └──────────┬──────────┘
                                    │ HTTP / SSE
                 ┌──────────────────▼──────────────────┐
                 │ API 层                               │
                 │ Python: Quart, api/                  │
                 │ Go: Gin, cmd/ + internal/            │
                 └───────┬───────────────┬──────────────┘
                         │               │
              元数据读写 │               │ 创建异步任务
                         ▼               ▼
                 ┌─────────────┐  ┌─────────────┐
                 │ MySQL / PG  │  │ Redis / NATS│
                 │ 业务元数据   │  │ 队列与协调   │
                 └─────────────┘  └──────┬──────┘
                                         │
                                  ┌──────▼──────┐
                                  │ Ingest Worker│
                                  │ rag/svr/ 或  │
                                  │ internal/... │
                                  └──┬────────┬──┘
                                     │        │
                            读取原文 │        │ 写切片/向量
                                     ▼        ▼
                               ┌─────────┐ ┌──────────────┐
                               │ MinIO   │ │ Doc Store    │
                               │ 对象存储 │ │ ES/Infinity等│
                               └─────────┘ └──────┬───────┘
                                                  │ 混合检索
                                                  ▼
                                        ┌─────────────────┐
                                        │ RAG / Chat / Agent│
                                        │ 检索、重排、LLM   │
                                        └─────────────────┘
```

### 四类数据不要混淆

| 数据类别 | 默认落点 | 典型内容 | 丢失后的影响 |
|---|---|---|---|
| 业务元数据 | MySQL | 用户、租户、知识库、文档、任务、会话、模型配置 | 页面对象和关联关系丢失 |
| 原始及派生对象 | MinIO | PDF、DOCX、图片、缩略图、附件 | 原文不能下载或重新解析 |
| 检索数据 | Elasticsearch | 切片文本、倒排字段、向量、页码、位置信息 | 已解析文档无法检索，可由原文重新构建 |
| 临时状态与消息 | Redis | 任务队列、进度、锁、取消标记、缓存 | 运行中任务受影响，通常不等同于业务数据永久丢失 |

核心判断是：MySQL 记录“它是什么以及处于什么状态”，MinIO 保存“原始内容是什么”，检索引擎保存“怎样高效找到它”，Redis 负责“现在由谁处理它”。

## 3. 仓库地图

| 目录 | 职责 | 初次学习优先级 |
|---|---|---|
| `api/` | Python API、鉴权、业务服务、Peewee 数据模型 | 高 |
| `rag/` | 文档摄取、切片、检索、LLM 适配、GraphRAG | 最高 |
| `deepdoc/` | PDF、Office、图片等底层解析，OCR、版面和表格识别 | 高 |
| `agent/` | Canvas DSL、组件、工具、工作流运行时 | 中高 |
| `common/` | 配置、存储抽象、检索引擎抽象、公共基础设施 | 高 |
| `web/` | React、TypeScript、Vite 前端 | 中高 |
| `cmd/` | Go 服务和 CLI 入口 | 中 |
| `internal/` | Go API、摄取、解析、Agent、DAO、服务实现 | 中 |
| `docker/` | Compose、镜像、入口脚本及依赖服务配置 | 高 |
| `conf/` | Python/Go 服务配置和本机覆盖配置 | 高 |
| `sdk/` | 对外 SDK | 中低 |
| `test/` | 单元、集成、API、Playwright 和基准测试 | 高 |

推荐遵循“入口 → 服务 → 抽象 → 实现”的方向阅读，不要从大型解析器或模型供应商列表开始逐文件浏览。

## 4. 运行时进程与启动入口

### 4.1 Python 源码模式

当前 macOS 本机开发方式由三类本机进程和四个基础容器组成：

```text
Docker: MySQL + Redis + MinIO + Elasticsearch
本机:   Quart API + Task Executor + Vite
```

主要入口如下：

| 进程 | 入口 | 默认作用 |
|---|---|---|
| Python API | `api/ragflow_server.py` | 初始化配置、数据库、路由与插件，监听 `9380` |
| Python Worker | `rag/svr/task_executor.py` | 消费文档任务，解析、向量化并建立索引 |
| 前端 | `web/src/main.tsx` | 初始化语言和后端类型，渲染 React 应用 |

本机已经验证过的完整命令、端口与健康检查以 [macOS 源码开发与版本升级](./macos_source_development_zh.md) 为准。

### 4.2 Python API 启动过程

从 `api/ragflow_server.py` 依次追踪：

```text
settings.init_settings()
  -> 加载关系库、对象存储、Doc Store、模型等配置
init_web_db()
  -> 建立数据库连接及必要结构
init_web_data()
  -> 初始化基础业务数据
RuntimeConfig.init_env()
  -> 初始化运行环境
GlobalPluginManager.load_plugins()
  -> 加载插件
app.run(...)
  -> Quart 开始监听
```

`api/apps/__init__.py` 创建 Quart 应用，配置 CORS、Session 和认证，并动态扫描多个 `_app.py` 与 `restful_apis/*.py` 注册路由。因此，查找接口时应搜索 URL 片段、HTTP 装饰器或函数名，而不是期待一个手写的总路由表。

### 4.3 Go 服务模式

`cmd/ragflow_server.go` 是 Go 的统一入口，一个二进制按参数运行不同角色：

```bash
./bin/ragflow_server --admin
./bin/ragflow_server --api
./bin/ragflow_server --ingestor
./bin/ragflow_server --syncer
```

其中 API 使用 Gin，路由集中在 `internal/router/router.go`；业务依赖在入口中组装后注入 Handler 和 Service。默认 Go API、Admin 端口分别是 `9384`、`9383`。Go 本机构建涉及 CGO 和原生解析库，不应绕过仓库的 `build.sh`。

## 5. 配置加载与基础设施抽象

### 5.1 配置优先级

Python 配置入口是 `common/config_utils.py`：

```text
conf/service_conf.yaml
  -> 再用 conf/local.service_conf.yaml 覆盖
  -> 环境变量占位符与运行配置参与解析
```

本地覆盖是顶层配置块的浅覆盖。若在 `local.service_conf.yaml` 中定义 `mysql:`，应写全本机实际需要的 MySQL 子项，避免误以为它会逐字段深度合并。

`common/settings.py::init_settings()` 是理解基础设施装配最重要的函数之一，它会：

1. 选择 MySQL 或 PostgreSQL 等元数据库；
2. 选择 Elasticsearch、Infinity、OpenSearch、OceanBase、SeekDB、GaussDB 或 SereneDB 等 Doc Store；
3. 选择 MinIO、S3、OSS、Azure、GCS 或 OpenDAL 等对象存储；
4. 创建全局 `STORAGE_IMPL`、`docStoreConn`、`retriever` 和 `kg_retriever` 等对象。

### 5.2 两个关键抽象

对象存储通过 `settings.STORAGE_IMPL` 使用。业务代码只关心 `put/get/remove` 等能力，不应直接绑定 MinIO 客户端。

检索存储通过 `common/doc_store/doc_store_base.py::DocStoreConnection` 抽象。它统一定义索引创建、搜索、插入、更新、删除和结果读取，并用下列表达式描述查询：

- `MatchTextExpr`：全文查询；
- `MatchDenseExpr`：稠密向量查询；
- `MatchSparseExpr`：稀疏向量查询；
- `MatchTensorExpr`：张量匹配；
- `FusionExpr`：多路结果融合。

学习新后端时，先看它如何实现 `DocStoreConnection`，再看业务层如何调用，效率远高于直接钻进驱动细节。

## 6. 核心数据模型

关系模型集中在 `api/db/db_models.py`，服务层位于 `api/db/services/`。

```text
User ──< UserTenant >── Tenant
                        │
                        ├──< Knowledgebase
                        │       │
                        │       └──< Document ──< Task
                        │               │
                        │               └── File2Document >── File
                        │
                        ├──< Dialog ──< Conversation
                        ├──< TenantLLM / ModelProvider / ModelInstance
                        └──< UserCanvas / Memory / API Token
```

### 关键模型

| 模型 | 含义 |
|---|---|
| `Tenant` | 数据与模型配置的主要隔离边界 |
| `Knowledgebase` | 知识库/数据集，持有解析器、Embedding 等配置 |
| `Document` | 文档元数据及解析状态，不保存完整文件二进制 |
| `File` | 文件管理器中的目录与文件节点 |
| `File2Document` | 文件对象与知识库文档的关联 |
| `Task` | 文档解析任务，可按页或表格范围拆分 |
| `Dialog` | 聊天助手配置，包括知识库、模型、阈值和提示词 |
| `Conversation` | 具体会话与消息历史 |
| `UserCanvas` | Agent/Dataflow 的图 DSL |

不要把 `Document` 与检索引擎中的“切片文档”混为一谈。前者是一条业务记录，后者通常是一组以 chunk ID 为主键的索引记录。

## 7. 文档摄取：最值得先读通的调用链

### 7.1 上传阶段

公开 REST 上传入口位于 `api/apps/restful_apis/document_api.py::upload_document`：

```text
HTTP 上传
  -> 校验用户、租户和知识库
  -> settings.STORAGE_IMPL.put(...)
     将原始二进制写入 MinIO 等对象存储
  -> DocumentService
     创建 Document 元数据
  -> FileService / File2DocumentService
     建立文件管理关联
```

此时通常只是“文件已存在”，尚未生成可检索向量。

### 7.2 创建与分发任务

解析入口包括：

- `POST /api/v1/documents/ingest`；
- `POST /api/v1/datasets/<dataset_id>/documents/parse`。

它们最终进入 `DocumentService.run(...)` 和 `api/db/services/task_service.py`：

1. 根据文件类型和页数拆分任务；
2. 写入 `Task` 表；
3. 更新 `Document` 的运行状态和进度；
4. 通过 `REDIS_CONN.queue_product(...)` 发布任务。

大文档被拆成多个 Task，是并发、断点处理和进度汇总的基础。调试“任务一直等待”时，应同时检查 MySQL 的任务状态、Redis 队列和 Worker 日志。

### 7.3 Worker 消费与解析

`rag/svr/task_executor.py` 的主循环通过 `queue_consumer` 消费任务。核心阶段是：

```text
collect()
  -> 从 Redis 取消息并加载 Task
build_chunks()
  -> 根据 File2Document 找到对象地址
  -> STORAGE_IMPL.get() 读取原始文件
  -> FACTORY[parser_id].chunk() 解析和切片
内容增强
  -> 分词、关键词、问题、元数据、位置等字段
embedding()
  -> Embedding 模型批量编码
  -> 生成 q_<维度>_vec 字段
insert_chunks()
  -> DocStoreConnection.insert() 写入检索引擎
更新 Document / Task
  -> 切片数、Token 数、进度和最终状态
```

`FACTORY` 将数据集的 `parser_id` 映射到 `rag/app/*.py`。这里是为不同文档语义选择切片策略的第一入口。

### 7.4 `rag/app` 与 `deepdoc` 的分层

这两个目录非常容易被初学者混淆：

```text
rag/app/*.py
  面向知识库的高层“解析模板/切片策略”
  例如 naive、paper、book、laws、qa、table、resume
          │
          ▼
deepdoc/parser/*.py + deepdoc/vision/*
  面向文件格式的底层内容提取
  例如 PDF、DOCX、Excel、PPT、HTML、Markdown、OCR、版面和表格识别
```

高层策略决定“怎样组织成适合检索的 chunk”，底层解析器负责“怎样从文件中可靠地提取文字、表格、图片和位置信息”。

建议先读最通用的 `rag/app/naive.py`，再向下跟踪它实际选用的文件解析器。不要一开始阅读体积最大的 PDF 版面识别实现。

### 7.5 切片索引记录

典型切片包含：

| 字段 | 用途 |
|---|---|
| `id` | 由内容和文档 ID 等生成的稳定切片标识 |
| `doc_id` / `kb_id` | 关联业务文档和知识库 |
| `content_with_weight` | 用于展示和模型上下文的主要内容 |
| `content_ltks` | 分词后的全文检索字段 |
| `title_tks` | 标题全文字段 |
| `important_kwd` / `question_tks` | 关键词和问题增强字段 |
| `q_<dim>_vec` | Embedding 向量，字段名包含向量维度 |
| `page_num_int` / `position_int` | 页码和原文定位 |
| `available_int` | 是否参与检索 |

`rag/nlp/search.py::index_name(uid)` 默认产生 `ragflow_<tenant_id>`。索引以租户为主要边界，`kb_id` 和 `doc_id` 再作为过滤条件。

## 8. 检索链路

检索核心位于 `rag/nlp/search.py::Dealer`。

### 8.1 查询构造

`Dealer.search()` 组合以下部分：

1. `kb_id`、`doc_id`、可用状态等过滤条件；
2. `FulltextQueryer` 生成的全文查询；
3. Embedding 模型产生的查询向量；
4. `MatchDenseExpr` 发起余弦相似度检索；
5. `FusionExpr` 融合全文与向量结果；
6. 根据后端能力执行 Elasticsearch、Infinity 等不同实现。

### 8.2 召回与重排

`Dealer.retrieval()` 在搜索结果上进一步完成：

```text
问题
  ├── 词法召回（全文/BM25）
  └── 语义召回（向量/KNN）
          │
          ▼
       融合候选
          │
          ▼
  内置混合打分或 Rerank 模型
          │
          ▼
  相似度阈值 + top_n 截断
          │
          ▼
  chunks + doc_aggs
```

几个配置不要混为一谈：

- `top_k`：向量候选池规模；
- `top_n`：最终送给后续流程的结果规模；
- `similarity_threshold`：最低相似度门槛；
- `vector_similarity_weight`：语义向量相对于词法匹配的权重；
- `rerank_mdl`：可选的专用重排模型。

排查“明明有内容却搜不到”时，按过滤条件、切片是否已索引、查询分词、Embedding 模型一致性、候选池、阈值、重排依次缩小范围。

## 9. 聊天与 RAG 生成链路

REST 入口位于 `api/apps/restful_apis/chat_api.py::session_completion`，主要服务逻辑位于 `api/db/services/dialog_service.py::async_chat`。

```text
聊天请求
  -> 加载 Dialog、Conversation、模型和知识库配置
  -> 处理多轮问题改写、跨语言或关键词增强
  -> Dealer.retrieval() 混合检索和重排
  -> 可选 TOC、父子切片、Web Search、Knowledge Graph 增强
  -> kb_prompt() 把切片组织成知识上下文
  -> 组装 system prompt + history + user message
  -> LLMBundle 调用模型，流式或非流式返回
  -> 检查/生成引用标记，关联实际 chunks
  -> 持久化会话并返回 answer + reference
```

### 9.1 模型适配层

模型调用主要分布在：

- `api/db/services/llm_service.py`：按租户和模型配置构造 `LLMBundle`；
- `rag/llm/chat_model.py`：聊天模型；
- `rag/llm/embedding_model.py`：向量模型；
- `rag/llm/rerank_model.py`：重排模型；
- `rag/llm/cv_model.py`：视觉模型；
- `rag/llm/sequence2txt_model.py`：语音识别等序列转文本；
- `rag/llm/tts_model.py`：语音合成。

业务层依赖“聊天、向量化、重排”这些能力，而非某一个固定厂商。增加供应商前，先确认现有模型抽象和配置模型是否已能表达其能力。

### 9.2 引用不是简单显示检索结果

引用处理发生在生成答案之后。若模型没有产生可识别的引用标记，代码会按需取得切片向量，通过 `Dealer.insert_citations()` 将答案片段与知识切片匹配，再返回实际引用。因而“答案正确但引用不对”要同时检查检索结果、提示词引用要求和生成后的引用匹配。

## 10. Agent Canvas

Agent 的核心不是一个无限循环的聊天函数，而是一份图 DSL 及其运行时。

`agent/canvas.py` 中的 `Graph` 和 `Canvas` 负责：

1. 读取并规范化 DSL；
2. 根据 `component_name` 动态定位组件类；
3. 校验每个组件的参数；
4. 实例化节点并维护上下游、执行路径和全局变量；
5. 处理会话历史、文件、取消、恢复和事件流；
6. 逐节点执行并通过异步生成器输出工作流事件。

简化后的 DSL 结构如下：

```json
{
  "components": {
    "begin": {
      "obj": {"component_name": "Begin", "params": {}},
      "upstream": [],
      "downstream": ["retrieval_0"]
    }
  },
  "history": [],
  "path": [],
  "retrieval": [],
  "globals": {
    "sys.query": "",
    "sys.user_id": "",
    "sys.files": []
  }
}
```

组件分为三类理解最清晰：

- 控制流：`begin`、`switch`、`iteration`、`loop`、`exit_loop`；
- 数据与生成：`llm`、`message`、变量与列表/字符串/数据操作；
- 外部能力：`invoke`、`browser`、`agent_with_tools` 以及 `agent/tools/` 中的工具。

`agent/component/__init__.py::component_class()` 会在 `agent.component`、`agent.tools` 和 `rag.flow` 中动态寻找类。新增组件时，前端节点定义、参数序列化、后端类名和输出变量必须保持一致。

## 11. 前端架构

前端位于 `web/`，主要技术栈为 React、TypeScript、Vite、React Router 和 TanStack Query。

```text
web/src/main.tsx
  -> 初始化国际化
  -> 请求 /api/v1/language 判断后端语言
  -> 渲染 App
web/src/app.tsx
  -> QueryClient、主题、提示框、RouterProvider
web/src/routes.tsx
  -> 页面路由和懒加载
web/src/pages/
  -> 页面与场景 UI
web/src/hooks/
  -> React Query 与业务状态封装
web/src/services/
  -> API 服务函数
web/src/utils/next-request.ts
  -> 新请求封装
```

`web/src/utils/request.ts` 已标记为废弃，新代码优先使用 `next-request.ts`，不要扩大旧请求层的使用面。

### Python、Go 和混合代理

`web/vite.config.ts` 根据 `API_PROXY_SCHEME` 选择：

| 值 | 行为 |
|---|---|
| `python` | `/api` 和 `/v1` 主要代理到 Python `9380` |
| `go` | 主要代理到 Go `9384`，Admin 到 `9383` |
| `hybrid` | 按 URL 规则把已迁移接口发往 Go，其余保留 Python |

应用首次渲染前，`web/src/utils/backend-runtime.ts` 会请求 `/api/v1/language` 并缓存结果，使组件能够选择对应实现。调试接口时务必先确认实际代理目标，避免在错误的后端进程中打断点。

## 12. Go 与 Python 的关系

当前仓库不是简单的“Python 主程序加几个 Go 工具”。Go 目录已经覆盖 API、Admin、摄取、解析、Agent、DAO、存储及 CLI，但仍与 Python 路径并存并持续演进。

| Python 路径 | Go 对应方向 |
|---|---|
| `api/apps` | `internal/handler` + `internal/router` |
| `api/db/services` | `internal/service` + `internal/dao` |
| `rag/svr/task_executor.py` | `internal/ingestion` |
| `rag/app` / `deepdoc` | `internal/parser` / `internal/deepdoc` |
| `agent` | `internal/agent` |
| `common/doc_store` | `internal/engine` |

学习顺序建议：

1. 先读通当前本机实际运行的 Python 主链路；
2. 选一个明确场景，例如“创建数据集”或“文档摄取”；
3. 对照 Go 的 Router → Handler → Service → DAO/Engine；
4. 用 `web/vite.config.ts` 的 hybrid 路由确认哪些接口正在由 Go 接管；
5. 不把两套实现误解成必须永久兼容的稳定架构，当前代码倾向收敛到单一路径。

Go 测试和构建应使用仓库脚本：

```bash
bash build.sh --test ./internal/path/to/package/...
bash build.sh --go
```

原生依赖、CGO 和平台说明见 `internal/development.md`。

## 13. 调试方法

### 13.1 用一个请求贯穿系统

最有效的学习方式是选一个小 PDF，记录它的 `tenant_id`、`kb_id`、`doc_id` 和 `task_id`，让这些 ID 成为跨系统的关联键：

```text
浏览器 Network
  -> API 日志中的 doc_id
  -> MySQL Document / Task
  -> Redis 任务消息
  -> Worker 日志
  -> Elasticsearch 中相同 doc_id 的 chunks
  -> 聊天 reference 返回的 chunk id
```

### 13.2 推荐断点

第一轮只设置少量断点：

1. `document_api.py::upload_document`；
2. `DocumentService.run`；
3. `task_executor.py::build_chunks`；
4. `task_executor.py::embedding`；
5. `task_executor.py::insert_chunks`；
6. `search.py::Dealer.retrieval`；
7. `dialog_service.py::async_chat`。

### 13.3 常用源码定位

```bash
# 查接口
rg -n 'documents/ingest|chat/completions' api internal

# 查某个模型的调用者
rg -n 'DocumentService\.run|Dealer\(|async_chat\(' api rag

# 查对象存储读写
rg -n 'STORAGE_IMPL\.(put|get|remove)' api rag agent

# 查队列生产与消费
rg -n 'queue_product|queue_consumer' api rag

# 查前端请求 URL
rg -n '/api/v1/|/v1/' web/src/services web/src/hooks

# 查 Go 路由
rg -n '\.(GET|POST|PUT|DELETE)\(' internal/router
```

### 13.4 故障定位顺序

| 现象 | 优先检查 |
|---|---|
| 上传失败 | API 日志、MinIO 连通性、文件大小、对象存储配置 |
| 一直等待解析 | Worker 是否运行、Redis 队列、Task 状态 |
| 解析失败 | Worker 异常、原文件能否从 MinIO 读取、对应 parser |
| 解析完成但搜不到 | Doc Store 健康、chunk 数、Embedding 字段、过滤条件和阈值 |
| 问答无知识 | Dialog 是否绑定知识库、检索返回、提示词是否包含知识变量 |
| 引用错误 | 最终 chunks、引用提示词、引用回填过程 |
| 前端接口 404 | `API_PROXY_SCHEME`、Vite 代理和实际后端语言 |

## 14. 测试体系

### Python 与前端

```bash
# Python：先运行最窄范围
uv run pytest test/unit_test/rag/path/to/test_file.py
ruff check path/to/changed_file.py

# 前端
cd web
npm run lint
npm run type-check
npm run test
npm run build
```

测试目录的主要层次：

- `test/unit_test/`：Python 单元测试；
- `test/testcases/`：Web、REST、SDK 和 Admin API 测试；
- `test/integration/`：需要真实基础设施的集成测试；
- `test/playwright/`：浏览器端到端测试；
- `test/benchmark/`：性能或质量基准。

### Go 测试层级

| 层级 | Build Tag | 外部依赖 |
|---|---|---|
| Unit | 无 | 不应依赖真实服务，但需要构建脚本接入原生库 |
| Integration | `integration` | 单个真实服务 |
| E2E | `e2e` | 完整跨组件链路 |
| Manual | `manual` | 很慢或昂贵，只在本机显式运行 |

```bash
bash build.sh --test ./internal/...
bash build.sh --test-integration ./internal/path/...
bash build.sh --test-e2e
```

## 15. 六阶段学习路线

### 阶段一：运行与观察（1～2 天）

目标：不看实现也能画出进程和存储拓扑。

实践：

1. 启动所有服务并逐个做健康检查；
2. 上传一个两页 PDF；
3. 观察 MinIO 对象、MySQL 文档记录、Task 进度和 ES 切片数；
4. 停掉 Worker 再触发解析，验证任务为何等待；
5. 恢复 Worker，观察任务完成。

验收：能够解释每个进程停止后会影响哪一段能力。

### 阶段二：API 与数据模型（2～3 天）

目标：掌握 Route → Service → Model 的基本结构。

阅读：

1. `api/ragflow_server.py`；
2. `api/apps/__init__.py`；
3. `document_api.py::upload_document`；
4. `api/db/db_models.py` 中的知识库、文档、任务和会话模型；
5. 对应的 `api/db/services/`。

实践：为一个只读接口增加日志或写一个针对 Service 的小测试。

### 阶段三：文档摄取（3～5 天）

目标：完整讲清上传到索引的链路。

阅读：

1. `DocumentService.run` 与任务创建；
2. `task_executor.py` 的消费循环；
3. `build_chunks`、`embedding`、`insert_chunks`；
4. `rag/app/naive.py`；
5. 它调用的一个 `deepdoc/parser` 实现。

实践：给一个纯文本或 Markdown 文档增加可观察字段，并验证它出现在索引中。

### 阶段四：检索与问答（3～5 天）

目标：能解释检索参数如何改变结果。

阅读：

1. `DocStoreConnection`；
2. `rag/nlp/search.py::Dealer.search`；
3. `Dealer.retrieval` 与重排；
4. `dialog_service.py::async_chat`；
5. `rag/llm` 中实际使用的一种聊天和 Embedding 模型。

实践：固定问题和知识库，分别调整阈值、向量权重、`top_k`、`top_n`，记录召回和答案变化。

### 阶段五：前端与 Agent（4～6 天）

目标：能从画布节点或页面操作追到后端。

阅读：

1. `web/src/routes.tsx`；
2. 一个页面对应的 hook 和 service；
3. `agent/canvas.py`；
4. `agent/component/base.py`；
5. `begin`、`llm`、`switch` 和一个工具组件。

实践：制作 Begin → Retrieval/LLM → Message 的最小工作流，观察 DSL、运行事件和变量传播。

### 阶段六：Go 对照阅读（持续）

目标：识别迁移边界，能在 Go 服务中定位等价职责。

阅读：

1. `cmd/ragflow_server.go`；
2. `internal/router/router.go`；
3. 某个 Handler → Service → DAO；
4. `internal/ingestion`；
5. `internal/parser` 与 `internal/agent`。

实践：通过 `python`、`hybrid`、`go` 三种前端代理方案调用同一类只读接口，对比响应头、行为和日志。

## 16. 推荐阅读清单

按顺序阅读以下文件，能够用较少代码覆盖主干架构：

1. `conf/service_conf.yaml`
2. `common/config_utils.py`
3. `common/settings.py::init_settings`
4. `api/ragflow_server.py`
5. `api/apps/__init__.py`
6. `api/db/db_models.py`
7. `api/apps/restful_apis/document_api.py`
8. `api/db/services/task_service.py`
9. `rag/svr/task_executor.py`
10. `rag/app/naive.py`
11. `common/doc_store/doc_store_base.py`
12. `rag/nlp/search.py`
13. `api/db/services/dialog_service.py::async_chat`
14. `api/db/services/llm_service.py`
15. `agent/canvas.py`
16. `agent/component/base.py`
17. `web/src/main.tsx`、`app.tsx` 和 `routes.tsx`
18. `web/src/utils/next-request.ts`
19. `cmd/ragflow_server.go`
20. `internal/router/router.go`

每读一个文件，只记录四件事：输入、输出、外部依赖、失败方式。不要逐行抄注释。

## 17. 修改功能时从哪里开始

| 需求 | 首先定位 |
|---|---|
| 新增上传/文档接口 | `api/apps/restful_apis/document_api.py` 或 Go Router/Handler |
| 调整业务数据规则 | `api/db/services/` 或 `internal/service` |
| 新增解析模板 | `rag/app`，必要时再改 `deepdoc` |
| 支持新文件格式 | `deepdoc/parser` / `internal/parser` |
| 修改切片字段 | Worker 构建 chunk 的路径 + Doc Store schema |
| 调整召回或融合 | `rag/nlp/search.py` + Doc Store 实现 |
| 支持新模型供应商 | `rag/llm` + 模型配置服务 |
| 新增 Agent 节点 | `agent/component`、Agent API、前端画布定义 |
| 新增页面 | `web/src/pages` → hooks → services → routes |
| 调整本机服务 | `conf`、`docker` 和启动脚本 |

修改时应从真正拥有该行为的抽象开始，避免在 API、兼容包装或前端临时补丁中复制同一规则。

## 18. 术语表

| 术语 | 本项目中的含义 |
|---|---|
| Dataset / Knowledgebase / KB | 知识库，部分 API 和代码沿用不同命名 |
| Document | 知识库中的文档业务记录 |
| File | 文件管理器中的对象或目录节点 |
| Chunk | 从文档提取、切分并写入检索引擎的最小检索单元 |
| Parser ID | 面向知识库的高层解析/切片策略标识 |
| Doc Store | 统一的检索存储抽象，不等同于关系数据库 |
| Embedding | 将文本编码为向量，用于语义相似度搜索 |
| Rerank | 对初次召回的候选进行更精细排序 |
| Canvas | 以 DSL 表示并执行的 Agent/数据流组件图 |
| Tenant | 用户资源、知识库和模型配置的主要隔离范围 |

## 19. 最终自测题

不看本文，尝试独立回答：

1. 为什么仅启动 API 和前端，上传可以成功但解析不会完成？
2. 删除 Elasticsearch 数据与删除 MinIO 数据的恢复方式有何不同？
3. `parser_id=paper` 最终为什么仍可能调用 PDF 底层解析器？
4. 为什么更换 Embedding 模型后通常需要重新解析或重建索引？
5. `top_k`、`top_n`、阈值和向量权重分别在哪个阶段生效？
6. 一个聊天答案的引用是在哪个阶段产生的？
7. 前端请求返回 404 时，如何判断请求去了 Python 还是 Go？
8. 新增 Agent 组件为何不能只增加一个 Python 类？

如果能够结合源码、日志和实际数据回答这些问题，就已经从“会运行 RAGFlow”进入了“能修改 RAGFlow”的阶段。
