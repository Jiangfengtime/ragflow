---
sidebar_position: 20
title: 第二十册：SDK、连接器与质量验证
sidebar_label: 20. SDK、连接器与质量验证
---

# 第二十册：SDK、连接器与质量验证

本册把“从应用调用 RAGFlow”与“持续同步外部数据”接到前面的摄取、检索和问答主链路，并建立可复查的质量验证方法。

以下 SDK 和连接器调用链以当前 Python 实现为准。相同 REST 路径不等于 Go 与 Python 的所有响应字段、校验和行为完全一致，双后端验证应分别记录结果。

## 1. 先区分四种“完成”

| 完成信号 | 证明什么 | 没有证明什么 |
|---|---|---|
| HTTP/业务响应成功 | 请求通过本次接口处理 | 后台解析已完成 |
| `SyncLogs` 为 `DONE` | 本次同步协程结束 | 每份文档都解析成功 |
| Document 为 `DONE` | 文档解析流程报告完成 | Chunk 完整、引用准确、召回有效 |
| 答案返回 | 问答链路产生了输出 | 答案有证据、没有越权或幻觉 |

不要用一个成功状态替代全部验证。应用需要保存对象 ID、检查后台终态，再检查实际 Chunk、召回和引用。

## 2. SDK 是 REST 客户端，不是另一个摄取引擎

源码入口：`sdk/python/ragflow_sdk/ragflow.py:RAGFlow`。

构造函数把 `base_url` 与 `/api/{version}` 拼接，默认 `version="v1"`，并添加 `Authorization: Bearer <api_key>`。

因此 `base_url` 应表示部署的服务根地址，不应已经包含 `/api/v1`；源码没有替应用规范化所有重复斜杠或代理前缀。

SDK 的 `get/post/put/patch/delete` 使用同步 `requests`。当前包装层没有为这些请求传入 `timeout`，不是 `asyncio` 客户端，也没有自动提供整个业务操作的截止时间。

真实路径是：

```text
应用调用 SDK 方法
  → RAGFlow 的 HTTP 包装方法
  → /api/v1/... REST 路由
  → 认证、资源权限与参数校验
  → 业务 Service / 数据库 / 存储
  → 需要后台处理时创建 Task 并投递队列
```

当前 SDK 对应的 Python HTTP 路由位于 `api/apps/restful_apis/`，不要按 SDK 名称假设存在 `api/apps/sdk/` 目录。

### 2.1 常用 SDK 方法与 HTTP 契约

下表路径均省略统一的 `/api/v1` 前缀：

| SDK 方法 | HTTP | 主要对象/结果 |
|---|---|---|
| `RAGFlow.create_dataset()` | `POST /datasets` | `DataSet` |
| `DataSet.upload_documents()` | `POST /datasets/{id}/documents` | `list[Document]` |
| `DataSet.list_documents()` | `GET /datasets/{id}/documents` | `data.docs` 转成对象列表 |
| `DataSet.async_parse_documents()` | `POST /datasets/{id}/chunks` | 提交后台解析 |
| `DataSet.async_cancel_parse_documents()` | `DELETE /datasets/{id}/chunks` | 请求取消解析 |
| `Document.list_chunks()` | `GET /datasets/{id}/documents/{doc_id}/chunks` | Chunk 列表 |
| `RAGFlow.retrieve()` | `POST /retrieval` | `data.chunks` 转成 `list[Chunk]` |
| `Chat.create_session()` | `POST /chats/{id}/sessions` | `Session` |
| `Session.ask()`（Chat） | `POST /chats/{id}/completions` | 逐次 `yield Message` |

对象 ID 才是后续操作的定位依据：Dataset ID、Document ID、Chunk ID、Chat ID、Session ID 不能互相替代。文件显示名也不能替代 Document ID。

SDK 多数方法检查响应 JSON 的 `code`，失败时抛异常。应用仍需区分连接错误、HTTP 状态、业务 `code` 和后台文档失败，而不是只捕获一种错误。

### 2.2 SDK 对象不是完整的原始 JSON

`DataSet`、`Document`、`Chunk` 构造函数会删除未声明的响应字段。

例如 `Chunk` 声明了正文、关键词、问题、文档 ID、相似度和 `positions`，但当前没有声明文档元数据字段。

`RAGFlow.retrieve()` 只返回 Chunk 对象列表，不返回原始响应中的整个 `ranks`、文档聚合或其他扩展字段；其参数也没有覆盖 REST 的全部选项。

应用需要新增 REST 字段时，先检查 SDK 参数及对象字段，再决定使用当前 HTTP 包装层读取原始响应，或完善 SDK；不能把服务端有字段等同于 SDK 已完整暴露。

## 3. 上传、解析与异步状态

### 3.1 本地上传调用链

`DataSet.upload_documents()` 接收包含 `display_name` 与二进制 `blob` 的字典列表，把每项编码成同名 `file` multipart 字段。

Python 端链路为：

```text
document_api.upload_document
  → _upload_local_documents
  → FileService.upload_document
  → 原文件写入对象存储
  → Document、File、File2Document 等关系写入
  → 返回 Document ID
```

这一步主要完成文件落地和文档登记。得到 Document 对象不表示正文已经被分词、向量化或写入 Chunk 索引。

### 3.2 普通解析调用链

当前 SDK 的 `async_parse_documents()` 对应 `chunk_api.py:parse`：

```text
POST /datasets/{id}/chunks
  → 校验知识库与文档权限
  → 重置解析进度及相关计数，清理已有解析产物
  → 获取文件存储位置
  → queue_tasks
  → 关系数据库 Task + Redis 消息
  → Task Executor / TaskHandler
  → parser、Chunk 后处理、Embedding、Doc Store
```

其中 `async` 表示“提交后不等待服务器后台任务”，不是 Python 的 `async def`；调用 SDK 提交请求本身仍会阻塞等待 HTTP 响应。

若知识库设置了 `pipeline_id`，该解析接口会拒绝请求并提示使用 `/documents/ingest`。不能用同一个 SDK 解析方法覆盖普通 parser 与所有 Dataflow 摄取场景。

批量请求也不能假设为事务：当前解析路由逐个处理文档，可能已经为合法文档排队，随后才返回其他 ID 的错误。重试前应查明已提交对象的状态。

### 3.3 轮询时需要显式判断成功与失败

`DataSet.parse_documents()` 在提交后调用 `_get_documents_status()` 轮询，当前实现：

- 用 `list_documents(id=...)` 逐个查询；
- 每轮未完成时等待一秒；
- `DONE`、`FAIL`、`CANCEL` 都被视为终态；
- `progress >= 1` 也会被记成 `DONE`；
- 返回包含 ID、状态、Chunk 数和 token 数的元组列表；
- 查询异常被当作暂时没有结果继续重试；
- 没有整个轮询的总超时。

因此“函数返回”不等于“全部成功”，Document 被删除或一直查不到时也可能持续等待。

生产应用宜由自己的任务控制层规定截止时间、退避策略和取消机制，并同时检查：

1. 本次上传的全部目标 ID 是否仍存在；
2. 是否出现 `FAIL` 或 `CANCEL`；
3. `progress_msg` 是否能解释失败；
4. `chunk_count` 是否符合预期；
5. 完成后检索是否能读到实际 Chunk。

取消是请求后台停止，不应当作“已经完全回滚”。清理、重试和索引可见性仍需验证。

## 4. 召回结果与答案引用的返回契约

### 4.1 检索接口与模型调用

`chunk_api.py:retrieval_test` 会校验知识库/文档范围、处理过滤条件、解析 Embedding 和可选 Rerank 模型，然后调用 `settings.retriever.retrieval()`。

普通检索并非完全离线：查询 Embedding 本身可能需要外部模型服务。开启关键词提取、跨语言改写、目录增强或 KG，还可能增加 Chat 模型调用。

当前 SDK 发送 `top_k`；REST 内部使用 `knn_top_k` 并接受该输入。报告参数时应记录实际客户端和服务端取值，不能把不同层的候选数与最终返回条数混为一谈。

REST 返回前会移除向量，并把索引字段映射为接口字段，例如：

| 索引/内部字段 | REST 字段 |
|---|---|
| `chunk_id` | `id` |
| `content_with_weight` | `content` |
| `doc_id` | `document_id` |
| `important_kwd` | `important_keywords` |
| `question_kwd` | `questions` |
| `kb_id` | `dataset_id` |

服务端可按请求补充文档元数据，但当前 `RAGFlow.retrieve()` 的参数及 `Chunk` 对象并没有完整暴露这一扩展契约。

### 4.2 `Session.ask()` 是生成器

即使 `stream=False`，`ask()` 也通过 `yield` 返回一条最终 `Message`，不是直接返回答案字符串。

流式时 SDK 读取 SSE 的 `data:` 内容，跳过非消息事件，再把数据转换成 `Message`。应用不能假设每个 SSE 事件或完整 Agent trace 都保留在 SDK 对象中。

`_structure_answer()` 取答案正文，并把 `reference.chunks` 放入 `Message.reference`；不会把整个原始 `reference` 连同文档聚合原样返回。

引用验证至少检查：

- 引用条目的 Document ID 属于允许访问的知识库；
- Chunk 内容确实支持对应答案句子；
- 页码/`positions` 能回到原始文档位置；
- 流式中间值与最终消息的引用不要重复累积；
- 没有证据的问题能否合理拒答。

GraphRAG 等合成上下文不一定有普通文档级页码定位。它们应单独验证证据范围，不能把所有返回条目都当成可定位的原文引用。

## 5. 连接器配置与知识库绑定

连接器实现目录是 `common/data_source/`，不是 `rag/connectors/`。

`common/data_source/__init__.py` 的 `CONNECTOR_BY_SOURCE` 与 `build_connector_for_source()` 选择具体来源实现；`interfaces.py` 定义轮询、checkpoint、指纹与 slim 文档枚举等能力。

当前 RAGFlow SDK 没有一套对应的 Connector 对象 CRUD 包装。连接器配置与绑定应检查 REST/前端调用及实际服务方法。

### 5.1 三类关系

| 数据 | 作用 |
|---|---|
| `Connector` | 来源类型、配置、所属 tenant、频率与状态 |
| `Connector2Kb` | 某连接器关联某知识库及是否自动解析 |
| `SyncLogs` | 某连接器/知识库的一次同步或删除检查、时间窗、计数与错误 |

连接器可以关联不同知识库，同步身份与删除范围都必须考虑知识库 ID。

相关 REST 入口在 `connector_api.py`，均省略 `/api/v1` 前缀：

- `POST /connectors`：创建连接器；
- `PATCH /connectors/{id}`：更新轮询配置、取消或重新调度；
- `GET /connectors`、`GET /connectors/{id}`：列表与详情；
- `POST /connectors/{id}/test`：校验指定来源配置；
- `POST /connectors/{id}/rebuild`：针对 `kb_id` 重建；
- `DELETE /connectors/{id}`：取消任务并删除连接器记录。

`test` 入口调用具体连接器的 `validate_connector_settings()`，可能访问真实外部来源；不是仅检查 JSON 结构的离线动作。

连接器详情/配置包含来源凭据的可能性很高，不应把完整响应、配置或连接串打印到学习日志。

知识库更新路径 `dataset_api_service.update_dataset()` 调用 `Connector2KbService.link_connectors()`。新增绑定会安排立即同步；`auto_parse` 控制同步落地后是否继续解析。

## 6. 后台同步与摄取队列不是同一个调度器

主循环入口是 `rag/svr/sync_data_source.py:dispatch_tasks`。

```text
SyncLogs 的到期 SCHEDULE 记录
  → dispatch_tasks 查询关系数据库
  → func_factory 选择来源同步实现
  → 读取远端文档/增量游标
  → 统一外部 ID、内容、metadata、fingerprint
  → duplicate_and_parse
  → FileService.upload_document
  → 若 auto_parse：DocumentService.run
  → 后续解析 Task / Redis / Worker
```

同步阶段依赖 SQL 中的 `SyncLogs`，不是直接从解析 Redis 队列消费。解析 Worker 正常运行，也不代表 Data Sync 进程已经启动。

当前 `docker/launch_backend_service.sh` 有 `data_sync` 选项，Docker entrypoint 默认启用 Data Sync；`start_local.sh` 不启动该进程。排查时按实际使用的启动脚本确认，不要只看 API 能否访问。

`refresh_freq`、`prune_freq` 按分钟参与到期判断，单次执行还有 `timeout_secs`。同步与 prune 是不同 `task_type`，不能把刷新频率当成删除检查频率。

`_CursorPersistingSyncBase` 会在批次无失败、没有解析提交错误时持久化游标；部分批次错误可以被捕获并记录后继续处理。

这里检查的是同步/提交结果，不是等待所有后台解析完成。`new_docs_indexed` 等日志计数也按收到的批次文档数累加，不能直接理解为唯一新增文档数或已完成的 Chunk 数。

排障需要同时关联 Connector ID、知识库 ID、SyncLog ID、Document ID 与解析 Task ID。

## 7. 本地上传与连接器同步的共性、差异

两种入口最终复用 `FileService.upload_document()`，再通过 Document/Task 主链路解析，原文件与 Chunk 的存储边界相同。

| 维度 | 本地上传 | 连接器同步 |
|---|---|---|
| 发起者 | 应用/用户提交 bytes | 定时同步器读取来源 |
| 身份 | 通常新建 UUID | 根据知识库、连接器、外部 ID 解析 |
| 变化检测 | 无来源指纹时哈希实际 bytes | 可由来源提供 fingerprint |
| 附加信息 | 上传字段与后续元数据接口 | 外部 DTO 的 metadata、更新时间等 |
| 自动解析 | 显式调用解析接口 | 由绑定的 `auto_parse` 控制 |
| 删除 | 应用显式删除 Document | 可独立启用完整快照 prune |

`resolve_connector_doc_id()` 首次同步使用包含 `kb_id:connector_id:external_id` 的哈希 ID；已有候选 ID 只有属于当前知识库来源范围才会复用，避免同一连接器跨知识库共享文档 ID。

### 7.1 三种“哈希”不要混淆

| 字段/机制 | 比较对象 | 用途 |
|---|---|---|
| 连接器 fingerprint | 来源的变更标识 | 判断是否需要下载/更新 |
| `Document.content_hash` | fingerprint，或文件 bytes 的 xxhash | 判断同一文档内容是否变化 |
| `Task.digest` | 解析配置、文档 ID、页/行范围等 | 判断重跑能否复用已完成 Task 的 Chunk |

fingerprint 是来源定义的相等性标记，不是统一的正文密码学摘要，也不是内容真实性或权限证明。

例如 Blob 来源把对象 ETag 归一化为指纹。`_BlobLikeBase` 可以先列 key/指纹，命中已有 `content_hash` 时跳过正文下载；没有变更不能自动证明所有 metadata 或远端权限也未变更。

`FileService.upload_document()` 对已有同 ID 文档比较 `content_hash`，只把变更项返回给后续解析；但普通既有文档分支仍可能写入对象存储。因此“跳过解析”不等于整个同步路径零 I/O。

重复同步验证应检查 Document ID 稳定、未变化内容不重复解析、修改后确实更新，并单独检查重命名与元数据变化；不能仅看日志里的批次计数。

## 8. 删除、解除绑定与重建的边界

| 操作 | 当前文档行为 | 重点风险 |
|---|---|---|
| 解除 Connector2Kb 绑定 | 取消对应活动同步，不删除文档 | 残留文档仍可检索 |
| 删除 Connector | 取消任务并删除 Connector 记录，不负责删除文档 | 不等于删除全部关联资料 |
| `rebuild(kb_id, connector_id, ...)` | 删除该知识库内该来源文档，再安排重建 | 破坏性操作，需要确认精确范围 |
| 删除同步 prune | 根据完整 slim 快照删除该范围内失踪文档 | 远端枚举范围/权限变化可能导致误删 |
| Document 删除接口 | 调用 `FileService.delete_docs()` | 需验证文件、Task、Chunk 与关系清理 |

prune 需要 `config.sync_deleted_files`，到期调度还受 `prune_freq` 影响；来源必须能提供完整 slim 文档快照。

`_collect_prune_snapshot()` 失败返回 `None` 时不会按空集合清理；成功得到空列表则意味着来源范围内没有任何文档，可以删除当前绑定范围全部旧文档。

这是必须区分的两个状态：“无法枚举”与“可靠地枚举为空”。测试应覆盖权限不足、网络失败、分页遗漏、空来源、解除绑定与删除并发。

来源 DTO 中出现 `external_access` 也不代表远端 ACL 已完整映射到 RAGFlow 文档授权。普通同步主链路没有提供这种统一保证，应按实际来源与消费代码核对。

## 9. 建立四层质量验证

### 9.1 解析质量

固定一组 PDF、扫描件、表格、Markdown/HTML、纯文本与异常文件，逐份核对：

1. 原文关键段落、标题、表格值是否保留；
2. OCR 是否有乱码、重复段或缺页；
3. Chunk 是否断在不合理位置，父子层级是否正确；
4. 页码、bbox/`positions` 与原文是否一致；
5. 分词、关键词、问题字段是否按配置生成；
6. 文档元数据是否写入正确的元数据索引；
7. 修改后重跑是否清理失效 Chunk。

Chunk 数增加不一定更好。应比较有效信息覆盖、边界、token 分布与引用可定位性。

### 9.2 召回质量

准备包含问题、允许范围、相关 Document/Chunk 与预期证据的固定样本集，先独立调用 retrieval，不混入答案生成评分。

明确相关性标注的粒度：文档命中和 Chunk 命中是不同指标；同一文档的十个重复片段不能算十个独立正确证据。

- `Recall@k`：前 k 条覆盖的相关项数 / 标注相关项数；
- `Precision@k`：前 k 条相关项数 / k（不足 k 条时明确分母约定）；
- `MRR`：各问题首个相关结果排名倒数的平均值；
- 范围正确性：是否返回越权或过滤之外的文档；
- 无答案样本：是否仍返回高分但无关的片段。

固定文档版本、Embedding/Rerank 模型、索引后端、候选数和过滤条件，每次只改变一项配置，并记录失败样本，不只记录平均值。

### 9.3 模型调用与答案质量

分别验证 Embedding、Rerank、Chat 的实例配置、模型类型、超时/限流处理和用量；不要把 Chat 连通当作 Embedding 也可用。

开启关键词/问题生成、跨语言、KG、目录增强等能力时，记录额外模型请求、延迟和成本。

答案检查包括事实正确、引用支持、重要证据遗漏、拒答、指令注入与敏感信息泄露。模型评审可以辅助，但需保留可人工复查的原文证据。

### 9.4 端到端运行质量

验证上传/同步 → 解析 → 索引 → 检索 → 问答整个过程的成功率、失败原因、首 token 延迟、总延迟，以及删除后检索是否仍残留。

除了正常路径，还应覆盖取消、重试、Worker 重启、来源限流、重复同步、游标持久化失败及删除检查失败。

## 10. 当前已有测试与评估能力的边界

`TaskHandler._run_evaluation()` 当前仅调用进度回调并写入 `Evaluation task placeholder`，没有实现完整评测。存在 `evaluation` 分派不能证明系统已经计算 Recall、答案正确率或自动质量门禁。

`test/benchmark/` 提供 HTTP 压测/性能测量：`metrics.py` 统计检索耗时、首 token 与总耗时的平均值、p50/p90/p95 等。这些性能结果不是检索相关性和答案忠实度评分。

benchmark 会调用真实服务，部分流程包含创建知识库、上传、解析和模型请求；不能作为默认离线文档校验命令。

### 10.1 可复查的现有测试路径

| 测试路径 | 已有检查重点 | 边界 |
|---|---|---|
| `test/unit_test/api/db/services/test_connector_doc_id.py` | ID 稳定、知识库隔离与归属 | 不证明真实来源完整同步 |
| `test/unit_test/common/test_blob_connector_fingerprint.py` | ETag 指纹、假 S3、未变更跳过下载 | 不连接真实对象存储 |
| `test/unit_test/rag/test_sync_data_source.py` | 空批次、prune 快照、增量游标与失败不推进 | 主要通过 monkeypatch 隔离服务 |
| `test/unit_test/rag/nlp/test_tokenize_chunks.py` | Chunk 分词相关行为 | 不证明模型或真实索引可用 |
| `test/unit_test/deepdoc/parser/test_pdf_parser_table_coordinates.py` | 表格位置处理 | 不等于全套真实 PDF/OCR 验收 |
| `test/unit_test/deepdoc/parser/test_pdf_garbled_detection.py` | 乱码检测与过滤 | 重型依赖被 mock |
| `test/testcases/test_sdk_api/test_file_management_within_dataset/test_parse_documents.py` | SDK 解析与状态契约 | 上层 fixture 会操作真实服务 |
| `test/benchmark/metrics.py` | 延迟统计实现 | 不是质量 gold set |

Python 的 `p0/p1/p2/p3` 是优先级标记，不表示离线/集成分层。`testpaths=["test"]` 也不意味着直接运行全目录没有外部影响。

运行前读目标测试及其上层 `conftest.py`：SDK fixture 会创建和清理资源；`test/unit_test/conftest.py` 在 NLTK 数据缺失时可能下载资源。

本册不把这些测试路径组合成默认全套执行命令。先准备依赖，明确服务与测试租户，再按第八册的验证原则选最窄目标；Go 的 build tag 分层另见第五册。

## 11. 一次可复查的最小验收

建议在明确授权的隔离测试知识库中执行，并先声明哪些步骤会访问来源或模型、哪些操作会删除数据。

1. 记录源码版本、后端类型、知识库 ID、解析配置与模型 ID，不记录密钥。
2. 上传或同步一份包含已知答案、表格与定位信息的小文档。
3. 保存 Document ID，提交解析，检查全部目标 ID 的终态与错误信息。
4. 核对 Chunk 内容、关键词/问题、位置和文档元数据。
5. 用固定问题检索，保留候选、分数与相关性标注。
6. 用 Chat 会话提问，核对最终答案和引用支持关系。
7. 对连接器重复同步，验证 ID 稳定及不重复解析；修改来源后确认更新。
8. 在精确限定的测试范围验证取消、失败、删除与 prune，不以重建代替诊断。
9. 清理仅本次创建的资源，确认检索与引用不再访问已删除文档。
10. 汇总正确性、失败样本、延迟与成本，区分源码检查、mock 测试和真实运行证据。

一个测试通过，只能证明它覆盖的契约；完整验收必须能解释“哪份文档、哪次同步、哪个 Task、哪些证据、哪个模型”共同产生了答案。

## 12. 源码阅读入口

- SDK：`sdk/python/ragflow_sdk/ragflow.py` 与 `modules/dataset.py`、`document.py`、`chunk.py`、`session.py`。
- REST：`api/apps/restful_apis/document_api.py`、`chunk_api.py`、`connector_api.py`。
- 绑定与同步日志：`api/db/services/connector_service.py`。
- 文件、解析与 digest：`api/db/services/file_service.py`、`document_service.py`、`task_service.py`。
- 同步运行：`rag/svr/sync_data_source.py`。
- 来源能力：`common/data_source/__init__.py`、`interfaces.py`、`models.py` 与各来源实现。
- 评估分派：`rag/svr/task_executor_refactor/task_handler.py`。
- 测试配置与性能统计：`pyproject.toml`、`test/benchmark/`。

扩展与数据安全检查继续参考第九册；GraphRAG、RAPTOR 与 Dataflow 细节参考第十一册。
