---
sidebar_position: 2
title: 第二册：文档摄取与深度解析
sidebar_label: 文档摄取与深度解析
---

# 第二册：文档摄取与深度解析

本册沿一个文档从 HTTP 上传一直追踪到切片写入 Doc Store，是理解 RAGFlow 最重要的一条源码主线。

## 1. 摄取状态机

```text
上传原文
  -> Document 已创建
  -> 请求 ingest/parse
  -> 创建一个或多个 Task
  -> Task 进入消息队列
  -> Worker 领取任务
  -> 读取原文
  -> 解析与切片
  -> 内容增强
  -> Embedding
  -> 写入 Doc Store
  -> 汇总 Task
  -> 更新 Document 完成状态
```

每个箭头都可能失败。诊断时要先确定停在哪个状态边界。

## 2. 上传入口

REST 入口：

```text
api/apps/restful_apis/document_api.py::upload_document
```

典型步骤：

1. 检查租户是否有数据集访问权限；
2. 校验文件数量、名称、类型和大小；
3. 调用对象存储写入原始二进制；
4. 创建 `Document`；
5. 建立 `File` 与 `Document` 关联；
6. 返回业务对象。

上传成功只表示原文和元数据已经保存，不代表已经产生 chunk。

### 为什么先存原文

异步 Worker 需要在 API 请求结束后独立读取文件；解析失败时也需要原文重试。将原文交给对象存储，使 Worker 不依赖 API 进程的内存或临时目录。

## 3. 解析触发入口

常见入口：

```text
POST /api/v1/documents/ingest
POST /api/v1/datasets/<dataset_id>/documents/parse
```

接口会检查当前运行状态、清理或保留旧任务和旧 chunk，并调用 `DocumentService.run(...)`。

阅读 `DocumentService.run` 时关注：

- 它从 Document 和 Knowledgebase 合并了哪些配置；
- 哪些解析状态不允许重复启动；
- 是否需要删除旧索引；
- 如何计算任务数量和表格编号；
- 任务创建失败后 Document 状态如何处理。

## 4. 任务拆分

`api/db/services/task_service.py` 负责把一个逻辑文档变成物理任务。

拆分依据包括：

- PDF 页范围；
- 大型表格行范围；
- parser 类型；
- 已有任务是否可复用；
- 优先级与队列后缀。

Task 至少携带：文档、知识库、租户、页范围、解析配置、语言、优先级和摘要标识。

### 为什么保留 digest

任务摘要可用于判断配置或输入是否真正变化，从而避免复用不兼容结果。学习缓存时要同时看“命中了什么”和“哪些配置参与摘要”。

## 5. 发布与消费

生产端调用：

```python
REDIS_CONN.queue_product(queue_name, message=task)
```

消费端位于 `rag/svr/task_executor.py`：

```python
REDIS_CONN.queue_consumer(queue_name, group, consumer)
```

消费者组允许多个 Worker 协作处理。任务存在于数据库和消息系统两处：数据库记录长期状态，消息触发实际执行。

排查队列问题时记录：

1. 生产使用的 queue name；
2. Worker 监听的 task type 和 suffix；
3. consumer group 与 consumer name；
4. 消息是否 pending；
5. 数据库 Task 是否仍被认为 unfinished。

## 6. Worker 主循环

`task_executor.py` 启动时解析 `-i` 与 `-t`，初始化配置、存储、Doc Store 与模型，然后进入异步消费循环。

`collect()` 将 Redis 消息与数据库 Task 合并为执行上下文，并识别普通文档、GraphRAG、RAPTOR、Dataflow、Memory 等任务类型。

不要假设 Worker 只做传统文档切片；同一个执行器已经承载多种后台任务。

## 7. `FACTORY`：高层解析策略分派

`task_executor.py::FACTORY` 将 `parser_id` 映射到 `rag/app` 模块，例如：

| parser_id | 主要场景 |
|---|---|
| `naive` | 通用文档 |
| `paper` | 论文结构 |
| `book` | 书籍 |
| `presentation` | 演示文稿 |
| `manual` | 手册 |
| `laws` | 法律文本 |
| `qa` | 问答对 |
| `table` | 结构化表格 |
| `resume` | 简历 |
| `picture` | 图片 |
| `audio` | 音频 |
| `email` | 邮件 |

这些是知识组织策略，不是简单的扩展名映射。

## 8. `build_chunks()` 逐段阅读

### 8.1 输入保护

先检查文件大小、任务取消状态和 parser 配置。超限文件会更新失败进度而不是继续占用解析资源。

### 8.2 读取对象

```text
File2DocumentService.get_storage_address(doc_id)
  -> bucket + object name
settings.STORAGE_IMPL.get(bucket, name)
  -> binary
```

日志中仍可能使用 “MinIO” 作为历史性描述，但实际调用的是统一对象存储抽象。

### 8.3 合并解析配置

任务携带文档配置，部分能力还来自 Knowledgebase 配置。例如表格列角色需要在执行前合并。调试“页面配置明明存在但 Worker 没采用”时，应观察合并后的配置，而非只看 API 入参。

### 8.4 调用 chunker

```python
chunker.chunk(
    filename,
    binary=binary,
    from_page=...,
    to_page=...,
    lang=...,
    parser_config=...,
)
```

CPU 密集或阻塞解析通过线程池从异步事件循环中隔离，并受并发 limiter 约束。

### 8.5 标准化 chunk

解析器返回的结果会补充：

- `doc_id`、`kb_id`、租户信息；
- 稳定 chunk ID；
- 创建时间；
- 页码、位置与顺序；
- 图片对象 ID；
- 用于全文检索的 token 字段。

PDF outline 等文档级信息不能简单留在第一块 chunk 中，会被提取并持久化为文档元数据。

## 9. DeepDoc 分层

`rag/app/naive.py` 根据扩展名或 MIME 选择更底层解析器：

```text
高层：决定 chunk 的语义边界、标题权重、合并策略
  ↓
格式层：PDF/DOCX/XLSX/PPT/HTML/Markdown/Text
  ↓
视觉层：OCR、Layout、Table Structure、坐标恢复
```

### PDF 为什么复杂

PDF 更接近“带坐标的绘图指令”而不是语义文档。解析需要解决：

- 字符阅读顺序；
- 多栏布局；
- 页眉页脚；
- 表格单元格；
- 扫描页 OCR；
- 图片与文字关系；
- 标题层级；
- 原文位置回溯。

因此 PDF Parser 同时包含普通文本路径与视觉增强路径。修改 PDF 行为时，要先确认测试文件走的是哪条路径。

## 10. Chunk 边界

通用策略会考虑：

- `chunk_token_num` 目标长度；
- delimiter；
- overlap；
- 标题和章节边界；
- 表格、图片等不可随意切开的结构；
- 页面位置和父子关系。

Chunk 不是越小越好：太小会丢语境，太大则降低召回精度并占用模型上下文。应通过真实问答集合评估，而不是只看平均长度。

## 11. 内容增强

基础文本之外，Worker 还可能生成：

- 关键词；
- 潜在问题；
- 标签；
- 文档名/标题 token；
- 元数据字段；
- GraphRAG 实体关系；
- 父子切片和摘要。

增强可能调用 LLM，因此摄取速度、成本和失败面会随配置变化。诊断性能时要把“文件解析耗时”和“模型增强耗时”分开统计。

## 12. Embedding

`task_executor.py::embedding()` 的关键行为：

1. 为每个 chunk 选择问题文本或正文；
2. 清理表格标签和空白内容；
3. 批量调用 `mdl.encode`；
4. 把标题向量与正文向量按 `filename_embd_weight` 混合；
5. 写入 `q_<vector_size>_vec`。

字段名包含维度，所以更换向量模型可能改变 schema。旧 chunk 与新查询使用不同模型或维度时，不能期待可靠结果，应重新建立索引。

## 13. 写入 Doc Store

`insert_chunks()` 分批调用：

```text
settings.docStoreConn.insert(
  chunks,
  index_name(tenant_id),
  dataset_id
)
```

写入前需要确保索引存在并具有正确向量维度。失败时会更新进度并尽力处理已写入部分，取消任务也需要清理对应 chunk。

### 最终汇总

同一文档的多个 Task 完成后，系统汇总 chunk 数、token 数与进度，更新 Document。不要用单个 Task 成功推断整个 Document 已完成。

## 14. 解析质量评估

至少从四个层面检查：

| 层面 | 问题 |
|---|---|
| 提取 | 文字、表格、图片是否漏失或乱码 |
| 结构 | 标题、段落、阅读顺序是否正确 |
| 切片 | 边界是否保留完整语义 |
| 可检索性 | 真实问题能否召回对应 chunk |

仅检查页面显示的解析预览，不足以证明检索质量。

## 15. 实验：比较两种切片参数

选择有明确章节结构的文档：

1. 记录默认配置生成的 chunk 数和长度分布；
2. 降低 `chunk_token_num` 并增加 overlap；
3. 重新解析；
4. 用 10 个固定问题做 retrieval test；
5. 比较正确 chunk 的排名、上下文完整性和总 token；
6. 写出哪一种配置更适合该文档类型及原因。

## 16. 本册检查点

1. 为什么 Document 与 Task 是一对多？
2. `rag/app` 和 `deepdoc/parser` 分别拥有哪类决策？
3. Worker 从哪里获得原始二进制？
4. Embedding 维度为什么进入字段名？
5. 页面显示“完成”之前需要汇总哪些任务状态？
