---
sidebar_position: 11
title: 第十一册：GraphRAG、RAPTOR 与高级数据流水线
sidebar_label: 11. 高级 RAG 与数据流水线
---

# 第十一册：GraphRAG、RAPTOR 与高级数据流水线

标准 RAG 主链路是“文档 → Chunk → Embedding → 混合检索”。当前仓库还支持图谱、层级摘要、Dataflow、知识编译、Memory 和外部连接器。这些能力复用 Task、Redis、Worker 和 Doc Store，但使用不同 `task_type` 分派。

## 1. 特殊任务的统一入口

`PipelineTaskType` 主要值：

| 类型 | 运行时 `task_type` | 用途 |
|---|---|---|
| `Parse` | 普通/`dataflow*` | 文档解析 |
| `RAPTOR` | `raptor` | 层级摘要树 |
| `GraphRAG` | `graphrag` | 实体、关系和社区图谱 |
| `Mindmap` | `mindmap` | 思维导图相关任务 |
| `Memory` | `memory` | 长期记忆处理 |
| `Wiki` | `wiki` | 知识编译/Wiki |
| `Skill` | `skill` | 语料到技能结构 |
| `StructureGraph` 等 | `structure_graph` 等 | 知识库级结构合并 |

`TaskHandler.handle_task()` 在初始化上下文和模型后按 `task_type` 分派；没有特殊类型时才进入标准 `_run_standard_chunking`。

## 2. 为什么有 `GRAPH_RAPTOR_FAKE_DOC_ID`

普通 Task 属于一个真实 Document。GraphRAG、知识库级 RAPTOR、Wiki 等任务可能同时处理整个知识库的多个文档，因此不能自然绑定单一 Document。

`queue_raptor_o_graphrag_tasks()` 可使用固定的 `GRAPH_RAPTOR_FAKE_DOC_ID`：

```text
Task.doc_id = graph_raptor_x
Redis payload.doc_ids = 真实参与文档列表
Task.task_type = graphrag / raptor / wiki / ...
```

它是知识库级任务的占位 ID，不代表 MinIO 中有一份名为 `graph_raptor_x` 的文件。权限、日志和进度代码遇到它时会走特殊分支。

## 3. GraphRAG 构建链路

高层流程：

```text
知识库已有普通 Chunk
  → 创建 task_type=graphrag
  → Worker 加载目标 doc_ids 的 Chunk
  → LLM/Embedding 提取实体与关系
  → 合并、去重和计算图结构特征
  → 生成 entity / relation / community 等记录
  → 写入同一 Doc Store 的专用字段分区
  → 更新 Knowledgebase.graphrag_task_id/finish_at
```

相关位置：

- `api/db/services/document_service.py:queue_raptor_o_graphrag_tasks`
- `rag/svr/task_executor_refactor/task_handler.py:_run_graphrag`
- `rag/graphrag/`
- `Knowledgebase.graphrag_task_id`

图谱记录与普通 Chunk 可以位于同一租户索引，通过 `knowledge_graph_kwd`、`compile_kwd` 等字段区分。

## 4. GraphRAG 查询链路

`common/settings.py` 初始化：

```python
kg_retriever = KGSearch(docStoreConn)
```

当 `use_kg=true`：

```text
标准 Dealer.retrieval
  → 得到普通 Chunk
KGSearch.retrieval
  → LLM 把问题改写为实体与答案类型关键词
  → 向量检索相关实体
  → 按类型检索实体
  → 向量检索相关关系
  → 读取实体的 n-hop 路径
  → 用相似度和 PageRank/权重排序
  → 在 token 预算内组装图谱上下文
  → 把图检索结果加入普通召回结果
```

因此项目支持基于预构建图关系的多跳信息使用。它不是在每次问题上对整个原始文档图做无限制图遍历；可用路径来自构建阶段保存的实体关系和 `n_hop_with_weight`，查询阶段再按问题筛选、加权和截断。

## 5. `KGSearch.retrieval` 参数

| 参数 | 含义 |
|---|---|
| `question` | 查询文本 |
| `tenant_ids` | 目标 tenant 索引 |
| `kb_ids` | 知识库过滤 |
| `emb_mdl` | 实体/关系语义检索模型 |
| `llm` | 查询改写模型 |
| `max_token` | 图谱上下文预算 |
| `ent_topn` | 最终实体数 |
| `rel_topn` | 最终关系数 |
| `comm_topn` | 社区结果数 |
| `ent_sim_threshold` | 实体相似度阈值 |
| `rel_sim_threshold` | 关系相似度阈值 |

图谱质量取决于构建模型、实体归一化、关系抽取和文档覆盖，不只取决于查询参数。

## 6. RAPTOR 是什么

RAPTOR 在普通 Chunk 之上递归聚类并生成摘要节点，使查询可以同时命中细节和高层主题。

```text
叶子 Chunk
  → Embedding/聚类
  → 每簇生成摘要
  → 摘要再次 Embedding
  → 继续聚类形成更高层
  → 层级节点写入 Doc Store
```

当前代码支持：

- 知识库级 RAPTOR：使用 fake doc id 和多个 `doc_ids`；
- 文档级 RAPTOR：普通解析末尾按 parser config 自动创建，Task 绑定真实 `doc_id`。

相关位置：

- `rag/svr/task_executor_refactor/raptor_service.py`
- `queue_per_doc_raptor_task`
- `Knowledgebase.raptor_task_id`

RAPTOR 不是知识图谱：它建立语义摘要层级，没有实体—关系图的显式边。

## 7. 标准 RAG、RAPTOR 与 GraphRAG 对比

| 能力 | 标准 Chunk RAG | RAPTOR | GraphRAG |
|---|---|---|---|
| 基础单元 | 文本 Chunk | Chunk + 层级摘要 | 实体、关系、社区 |
| 擅长 | 局部事实、关键词/语义匹配 | 跨段主题、全局概括 | 实体关系、多跳线索 |
| 构建成本 | 低到中 | 中到高 | 高 |
| 查询成本 | 低 | 中 | 中到高，含改写 LLM |
| 主要风险 | 切片断裂 | 摘要失真 | 抽取和实体合并错误 |

它们可以组合使用，但组合越多，索引成本、排障面和上下文预算越复杂。

## 8. Dataflow 文档解析

`Document.pipeline_id` 非空时，`DocumentService.run()` 不走内置 `queue_tasks()`，而是进入 `queue_dataflow()`。

```text
Document.pipeline_id
  → queue_dataflow
  → MySQL Task(task_type=dataflow...)
  → Redis
  → TaskHandler._run_dataflow
  → Canvas/DSL 定义的节点
  → 解析、转换和写入
```

Dataflow 允许用户用流水线定义摄取步骤。Debug 时需要同时检查：Document 配置、Task payload、Canvas DSL、组件输入输出和 Pipeline Operation Log。

## 9. 知识编译与 `compile_kwd`

高级流水线会把派生产品写入 Doc Store，并用 `compile_kwd` 区分：

```text
普通可检索 Chunk        通常无 compile_kwd
Wiki 页面               wiki_page
Wiki 实体/关系          wiki_entity / wiki_relation
Skill 节点              skill
Skill 聚合              skill_all
结构/时间线/导航        对应专用 compile_kwd
```

这些记录不是原始 Document Chunk，但仍需要：

- `kb_id` 隔离；
- 来源 `source_doc_ids/source_chunk_ids`；
- 删除文档时增量更新或重建；
- 查询时决定包含还是排除。

`search_datasets` 的 `include_knowledge_compilation=false` 会通过 `must_not exists compile_kwd` 排除编译产物。

## 10. Wiki 增量生成

`dataset_wiki_generator.py` 的职责包括：

- 加载目标文档 Chunk；
- Map/Reduce 或增量提取；
- 生成页面、主题、实体和关系；
- 保存 `source_doc_ids/source_chunk_ids`；
- 删除失效派生记录；
- 写入 Wiki 页面和页面图。

它不是把 Markdown 文件写进 MinIO 的简单导出，而是将结构化编译产物保存进 Doc Store，供 Artifact 页面和后续查询读取。

## 11. Skill 与结构合并

`dataset_skill_generator.py` 将语料编译为递归技能树：

- 每个节点一条 `compile_kwd=skill` 记录；
- 整棵树一条 `compile_kwd=skill_all` 聚合记录。

`dataset_structure_merger.py` 处理知识库级的结构图、思维导图、时间线、Session Graph 等合并任务。`TaskHandler` 通过 `is_structure_merge_task()` 统一分派。

这些任务常使用知识库级 fake doc id，进度应从 Task 和 Knowledgebase 对应 task_id/finish_at 字段观察。

## 12. Memory 任务

Memory 任务使用 `task_type=memory...` 分派，不依赖普通 Document 的全部字段。消费者在 `collect()` 中对 Memory 类型使用专门加载逻辑，`TaskHandler` 也在标准知识库初始化前识别它。

学习时区分：

- Conversation：一次助手/Agent 的会话历史；
- Memory：从消息中抽取并长期保存的记忆单元；
- Dataset：用户上传并解析的知识库。

三者可以在回答时共同提供上下文，但生命周期和存储模型不同。

## 13. 外部连接器

`Connector` 和 `Connector2Kb` 描述外部数据源及其知识库绑定。连接器通常经历：

```text
保存连接配置
  → 周期/手工 Sync Task
  → 拉取远端对象及变更
  → 转换为 File/Document 或下载任务
  → 复用标准摄取/Dataflow
  → 记录 SyncLogs
  → 远端删除时执行 prune
```

连接器的关键难点不是下载接口，而是：

- 增量游标和幂等；
- 重命名/移动/删除；
- 远端权限变化；
- 凭据刷新；
- 限流与重试；
- 一个对象不能重复创建多个 Document。

## 14. 高级任务的状态观察

| 证据 | 看什么 |
|---|---|
| `Task.task_type` | 实际进入哪条分支 |
| `Task.doc_id` | 真实文档还是 fake id |
| Redis queue suffix | common、raptor、graphrag 等 |
| `Knowledgebase.*_task_id` | 当前知识库级任务 |
| `*_task_finish_at` | 最近完成时间 |
| Pipeline Operation Log | DSL、阶段和错误 |
| ES `compile_kwd` | 派生产品类型 |
| `source_doc_ids` | 派生产物来源 |

## 15. 多跳检索的正确理解

回答“RAGFlow 是否支持多跳”时要分三类：

1. 普通混合检索：一次查询召回多个 Chunk，本身不是图多跳。
2. GraphRAG：使用实体关系和预计算 n-hop 路径，属于图增强、多跳信息检索。
3. Agent/查询分解：模型把复杂问题拆成多个子问题并多轮调用检索，也能完成逻辑上的多跳，但机制不同于图遍历。

评估多跳能力时，应测试最终答案是否引用了正确的多份证据，而不是只确认 `use_kg` 开关打开。

## 16. 高级功能的成本与质量

| 成本 | 来源 |
|---|---|
| 构建时间 | 多轮 LLM、Embedding、聚类、图合并 |
| 存储 | 派生 Chunk、实体、关系、摘要和图 |
| 查询延迟 | 查询改写、多路检索和上下文组装 |
| 更新复杂度 | 原文变化后的增量重算和来源清理 |
| 质量风险 | 幻觉摘要、错误实体合并、错误关系 |

上线前应为标准 RAG、RAPTOR 和 GraphRAG 分别建立评测集，并验证更新/删除后的派生数据一致性。

## 17. 调试入口

```bash
rg 'queue_raptor_o_graphrag_tasks|queue_per_doc_raptor_task' api
rg '_run_graphrag|_run_raptor|run_wiki_incremental' rag/svr
rg 'class KGSearch|n_hop_with_weight' rag/graphrag
rg 'compile_kwd|source_chunk_ids|source_doc_ids' rag/advanced_rag rag/svr
rg 'class Connector|class Connector2Kb|class SyncLogs' api/db
```

断点顺序：创建特殊 Task → Redis payload → `TaskHandler.handle_task` 分派 → 对应 Service → Doc Store insert → task_id/finish_at 更新。

## 18. 本册检查点

- 能区分标准 RAG、RAPTOR 和 GraphRAG。
- 能解释 fake doc id 与真实 `doc_ids` 的关系。
- 能说明 GraphRAG 如何使用实体、关系和 n-hop 路径。
- 能说明 Dataflow 为什么绕过内置 `queue_tasks`。
- 能解释 `compile_kwd` 与普通 Chunk 的区别。
- 能按 task_type 定位 Wiki、Skill、Memory 和结构合并任务。
