---
sidebar_position: 11
title: 第十一册：GraphRAG、RAPTOR 与高级数据流水线
sidebar_label: 11. 高级 RAG 与数据流水线
---

# 第十一册：GraphRAG、RAPTOR 与高级数据流水线

标准 RAG 主链路是“文档 → Chunk → Embedding → 混合检索”。当前仓库还支持图谱、层级摘要、Dataflow、知识编译、Memory 和外部连接器。后台计算能力复用 Task、Redis、Worker 和 Doc Store，并按 `task_type` 分派；外部连接器的同步阶段则由独立进程按 SQL 中的 SyncLogs 调度，落地后才可继续投递解析 Task。

## 1. 特殊任务的统一入口

`PipelineTaskType` 主要值：

| 类型 | 运行时 `task_type` | 用途 |
|---|---|---|
| `Parse` | 普通/`dataflow*` | 文档解析 |
| `RAPTOR` | `raptor` | 层级摘要树 |
| `GraphRAG` | `graphrag` | 实体、关系和社区图谱 |
| `Mindmap` | `mindmap` | 思维导图相关任务 |
| `Memory` | `memory` | 长期记忆处理 |
| `ARTIFACT`（枚举值为 `Wiki`） | `wiki` | 知识编译/Wiki |
| `Skill` | `skill` | 语料到技能结构 |
| `StructureGraph` 等 | `structure_graph` 等 | 知识库级结构合并 |

当前消费入口调用 `TaskHandler.handle()`：Memory 在模型/索引初始化之前处理并返回，Dataflow 和其他摄取任务在绑定 Embedding、初始化索引后分派；普通文档进入 `_run_standard_chunking()`。此外还有 `reembedding`、`clone` 等独立任务分支。任务类型的公开枚举名、保存值和 Worker 分派字符串需要分别查看，不能仅按枚举名推断行为。

其中 `mindmap` 和 `evaluation` 分派目前只写占位完成进度；页面回答生成思维导图的 `gen_mindmap()` 是另一条查询服务链路。存在任务枚举或路由入口，不足以证明该分支已经实现完整内容处理。

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
  → LLM 提取实体与关系，Embedding 编码图谱内容
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

图谱记录与普通 Chunk 可以位于同一租户索引。GraphRAG 主要以 `knowledge_graph_kwd=entity/relation/community_report/graph/subgraph` 区分；Wiki/结构编译产物则用 `compile_kwd`。即使都能展示为图，也不能把 Wiki 实体/关系记录直接当成 `KGSearch` 所查询的 GraphRAG 记录。

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

构建侧 `rag/graphrag/utils.py::n_neighbor()` 默认枚举两跳路径，`set_graph()` 调用 `graph_node_to_chunk()` 为实体保存这些路径及每步权重；查询侧将其展开为关系对，再与语义召回关系合并排序。当前“多跳”支持的证据是这些具体路径和消费过程，不能泛化成任意跳数的在线推理保证。

`KGSearch.retrieval()` 返回一条合成上下文 Chunk，内容由实体、关系及社区报告组成，`doc_id=""`、`chunk_id` 临时生成。它提供关系证据，但没有直接把每条关系映射成普通文档的页码引用。当前方法仅按 tenant/`kb_ids` 过滤，未消费普通召回的 `doc_ids`/元数据文档范围；开启 KG 时要分别验证普通证据与图证据的范围。

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

还要区分任务范围与输出形态。知识库级 Task 可以按配置 `scope=file` 为各真实文档分别生成摘要，也可按 `scope=dataset` 生成 fake doc id 下的全库摘要。普通 `_generate_raptor()` 默认 `is_tree=false`，输出 `raptor_kwd=raptor` 摘要行及 `raptor_layer_int`；显式 `is_tree=true` 时可输出单条 `raptor_kwd=raptor_tree` JSON 树且 `available_int=0`。`build_doc_tree()` 返回树字典而不自行写索引，供编译模板继续处理；`compile_kwd=raptor_graph` 则是单独保存的展示记录，同样不可用作普通可用 Chunk 召回。

查询端还有实际边界：`Dealer.retrieval()` 会用真实 `Document` 表校验候选的 `doc_id`。fake doc id 下的摘要即使已写入索引，也可能被这一步移除；检查全库 RAPTOR 是否参与答案时，需要同时检查输出记录、可用标记和删除残留过滤后的候选。

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
  → Redis common 队列
  → TaskHandler._run_dataflow
  → DataflowService.run_dataflow
  → rag.flow.pipeline.Pipeline.run 执行 DSL 节点
  → 输出归一化/必要时 Embedding/字段整理
  → ChunkService.insert_chunks
  → 文档计数与 Pipeline Operation Log
```

Dataflow 允许用户用流水线定义摄取步骤。Debug 时需要同时检查：Document 配置、Task payload、Canvas DSL、组件输入输出和 Pipeline Operation Log。

Python 节点实现位于 `rag/flow/`，包括 File、Parser、TokenChunker/TitleChunker、Extractor、Tokenizer 等。当前 `Pipeline.run()` 按执行 `path` 调用节点，将前一节点 `output()` 作为下一节点输入，再追加 downstream；实际调度需要看这个方法，不能只因为继承了 `agent.canvas.Graph` 就假设拥有所有 Agent 执行语义。

Worker 对节点最终输出的处理是明确的：

| Pipeline 输出/字段 | 后续行为 |
|---|---|
| `chunks` 或 `json` | 作为 Chunk 列表继续处理 |
| `markdown/text/html` | 包成带 `text` 的 Chunk |
| 不含 `q_<dim>_vec` | Worker 按知识库模型补做 Embedding |
| `questions` | 转成 `question_kwd/question_tks` |
| `keywords` | 转成 `important_kwd/important_tks` |
| `summary` | 缺少正文分词字段时用其生成 `content_ltks/content_sm_ltks` |
| `metadata` | 汇总到文档元数据，再从 Chunk 中移除该临时键 |
| `positions` | 转成索引位置字段 |

这些转换支持内容增强，但 Worker 不会因为看到一个任意 JSON 就自动为它生成关键词/问题/摘要；内容是否产生取决于 DSL 中的实际节点及配置。调试任务使用 `CANVAS_DEBUG_DOC_ID`，执行后记录输出并在 Embedding/落库前返回；`dataflow_rerun` 则从 Pipeline Operation Log 的 DSL 装载已保存流程，而普通 `dataflow` 从 `UserCanvas` 装载。

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

这个开关仅排除有 `compile_kwd` 的记录，不会自动排除 `knowledge_graph_kwd` 或 `raptor_kwd` 记录。普通检索还检查 `available_int`，所以“编译产物存在”“能在 Artifact 页面显示”“被回答召回”是三个分别需要验证的状态。

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

Memory 任务使用 `task_type=memory` 分派，不依赖普通 Document 的全部字段。消费者在 `collect()` 中对 Memory 类型使用专门加载逻辑，`TaskHandler.handle()` 也在标准知识库初始化前调用 `handle_save_to_memory_task()` 并返回。

学习时区分：

- Conversation：一次助手/Agent 的会话历史；
- Memory：从消息中抽取并长期保存的记忆单元；
- Dataset：用户上传并解析的知识库。

三者可以在回答时共同提供上下文，但生命周期和存储模型不同。

## 13. 外部连接器

`Connector` 和 `Connector2Kb` 描述外部数据源及其知识库绑定。连接器通常经历：

```text
保存连接配置
  → 创建或调度 SyncLogs
  → Data Sync 查询到期记录
  → 拉取远端对象及变更
  → 下载并复用文件/文档落地逻辑
  → auto_parse 开启时投递解析 Task
  → 更新 SyncLogs（不等待所有解析完成）
  → 独立调度 prune，可靠的完整快照用于检测远端删除
```

SDK 调用契约、同步指纹、解除绑定与删除边界，以及验证方法见[第二十册](./20_SDK连接器与质量验证.md)。

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

断点顺序：创建特殊 Task → Redis payload → `TaskHandler.handle` 分派 → 对应 Service → Doc Store insert → task_id/finish_at 更新。

## 18. 本册检查点

- 能区分标准 RAG、RAPTOR 和 GraphRAG。
- 能解释 fake doc id 与真实 `doc_ids` 的关系。
- 能说明 GraphRAG 如何使用实体、关系和 n-hop 路径。
- 能说明 Dataflow 为什么绕过内置 `queue_tasks`。
- 能解释 `compile_kwd` 与普通 Chunk 的区别。
- 能按 task_type 定位 Wiki、Skill、Memory 和结构合并任务。

## 19. 源码复核入口

| 关注点 | 当前源码 |
|---|---|
| 公开任务枚举 | `common/constants.py::PipelineTaskType` |
| 实际 Worker 分派 | `rag/svr/task_executor_refactor/task_handler.py::TaskHandler.handle` |
| GraphRAG 路径生成与图记录 | `rag/graphrag/utils.py::n_neighbor / set_graph / graph_node_to_chunk` |
| 图上下文与过滤边界 | `rag/graphrag/search.py::KGSearch.retrieval` |
| RAPTOR scope、摘要与树 | `rag/svr/task_executor_refactor/raptor_service.py::RaptorService` |
| Dataflow 任务与节点调度 | `api/db/services/task_service.py::queue_dataflow`、`rag/flow/pipeline.py::Pipeline.run` |
| Dataflow 输出增强/落库 | `rag/svr/task_executor_refactor/dataflow_service.py::DataflowService.run_dataflow / _process_chunks` |
