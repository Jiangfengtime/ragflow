---
sidebar_position: 3
title: 第三册：检索、问答与引用
sidebar_label: 检索、问答与引用
---

# 第三册：检索、问答与引用

摄取解决“把文档变成可检索数据”，本册解决“怎样从这些数据中找到证据并生成可追溯答案”。

## 1. 一次 RAG 请求的阶段

```text
原始问题
  -> 会话与助手配置
  -> 问题改写/跨语言/关键词增强
  -> 权限与数据集过滤
  -> 词法召回 + 向量召回
  -> 融合与重排
  -> 阈值和数量裁剪
  -> 可选 TOC/父子块/KG/Web 增强
  -> 知识上下文格式化
  -> Prompt 预算控制
  -> LLM 流式生成
  -> 引用识别或回填
  -> 保存会话与统计
```

回答质量是整条链路的乘积。不能只通过更换 Chat Model 解决召回错误。

## 2. Doc Store 抽象

`common/doc_store/doc_store_base.py::DocStoreConnection` 统一定义：

- 健康检查；
- 索引创建与删除；
- `search`；
- `get`、`insert`、`update`、`delete`；
- total、fields、highlight、aggregation 等结果读取；
- 部分后端的 SQL 能力。

业务层以表达式描述意图：

| 表达式 | 意义 |
|---|---|
| `MatchTextExpr` | 全文/关键词匹配 |
| `MatchDenseExpr` | 稠密向量相似度 |
| `MatchSparseExpr` | 稀疏向量匹配 |
| `MatchTensorExpr` | 多向量或张量匹配 |
| `FusionExpr` | 多路分数融合 |
| `OrderByExpr` | 明确排序 |

不同后端可以用自身 DSL 实现同一意图，但分数尺度和能力边界并不保证完全一致。

## 3. `Dealer.search()`

`rag/nlp/search.py::Dealer` 是 Python 检索主入口。

### 3.1 过滤条件

`get_filters()` 从请求中提取 `kb_ids`、`doc_ids`、`available_int` 等条件。过滤先于相似度排序，它是“0 或 1”的候选资格判断。

最常见的零召回原因之一不是向量质量，而是：

- 错误的 tenant index；
- kb_id 不匹配；
- doc_ids 范围为空；
- chunk 被标记为不可用；
- 文档已删除但残留索引或反向情况。

### 3.2 词法查询

`FulltextQueryer.question()` 对问题分词并构造全文表达式。词法检索擅长：

- 产品型号；
- 人名、缩写、错误码；
- 精确术语；
- 文档中原样出现的关键词。

### 3.3 向量查询

`get_vector()` 调用 `emb_mdl.encode_queries`，根据维度选择 `q_<dim>_vec`，以 cosine 构造 `MatchDenseExpr`。

向量检索擅长语义近似，但必须满足：

1. 查询和 chunk 使用兼容的 Embedding 模型；
2. 维度一致；
3. 文本预处理合理；
4. 候选池足够大。

### 3.4 候选池与分页

`knn_top_k` 与 `knn_num_candidates` 控制近邻搜索候选；页面 `size` 控制返回记录数。候选池太小可能让后续重排没有机会看到正确证据。

## 4. 混合检索

混合检索不是把 BM25 分数和 cosine 分数直接相加这么简单。不同后端可能使用加权和、归一化 Pipeline 或结果级融合。

直观模型：

```text
hybrid_score = lexical_weight × lexical_score
             + vector_weight  × vector_score
             + rank_features
```

实际实现还需考虑分数归一化、缺失分支、候选集合并和后端差异。

`vector_similarity_weight` 高：更偏语义相似；低：更偏关键词。它不是“答案正确率”旋钮，应该基于查询类型调优。

## 5. `Dealer.retrieval()`

`retrieval()` 在底层 search 之上承担完整召回策略：

1. 组装请求和字段；
2. 执行候选搜索；
3. 过滤指向已删除 Document 的残留 chunk；
4. 选择 KNN/内置混合重排或专用 Rerank 模型；
5. 应用相似度门槛；
6. 截取 top_n；
7. 生成文档聚合和用于 API 的 chunk 结构。

### 参数边界

| 参数 | 所在阶段 | 调高的主要代价 |
|---|---|---|
| `knn_top_k` | 向量召回 | 搜索成本和候选量 |
| `rerank_candidates_count` | 重排输入 | Rerank 时间/费用 |
| `top_n` | 最终上下文 | LLM 输入 token |
| `similarity_threshold` | 最终过滤 | 调高会降低召回率 |
| `vector_similarity_weight` | 融合/重排 | 可能弱化精确关键词 |

## 6. Rerank

代码提供多种路径：

- `rerank_with_knn`：利用检索引擎返回的 KNN 分数；
- `rerank`：本地结合 token 和向量相似度；
- `rerank_by_model`：使用专用 Rerank 模型。

Cross-encoder 类型的 Rerank 往往更精确，但需要对问题和每个候选共同推理。正确使用方式通常是“大候选召回 + 小规模精排”，不是对整个知识库逐块计算。

## 7. 聊天入口与配置加载

REST 入口：

```text
api/apps/restful_apis/chat_api.py::session_completion
```

核心生成服务：

```text
api/db/services/dialog_service.py::async_chat
```

它会加载：

- Dialog 的知识库与 Prompt 配置；
- Chat、Embedding、Rerank、TTS 等模型；
- Conversation 历史；
- 请求指定的文档范围与引用元数据；
- Web Search、KG、TOC 等开关。

## 8. 问题预处理

进入检索前可能执行：

### 多轮问题改写

“它的保修期呢？”离开历史上下文无法检索。`refine_multiturn` 会把省略指代的问题转换为独立问题。

### 跨语言扩展

查询语言与知识库语言不同，可生成目标语言版本以改善召回。

### 关键词增强

LLM 提取关键词后附加到查询，帮助词法召回。代价是额外模型调用和延迟。

调试时必须记录“原始用户问题”和“最终送入 retriever 的问题”，否则可能在错误对象上分析。

## 9. 检索增强分支

基本 chunks 之外还可能加入：

- `retrieval_by_toc`：利用目录结构寻找相关章节；
- `retrieval_by_children`：将命中块替换或补充为关联子块；
- Web Search：加入互联网结果；
- KG Retrieval：加入实体关系知识；
- SQL Retrieval：针对有字段映射的数据集生成查询。

任何增强都要保留来源和范围，避免不同租户或不同数据集的信息错误混合。

## 10. `kb_prompt()` 与上下文

`kb_prompt()` 把结构化 chunk 转换成模型可读文本。通常保留：

- 文档名称；
- chunk 内容；
- 页码或位置；
- 引用编号；
- 部分元数据。

上下文不是越多越好。大量低相关 chunk 会：

- 稀释正确证据；
- 增加 token 和延迟；
- 诱导模型拼接冲突内容；
- 增大引用匹配难度。

## 11. Prompt 组装和预算

生成消息包含：

```text
system prompt
  + knowledge
  + citation instruction
  + conversation history
  + current user message
  + text/image attachments
```

`message_fit_in()` 按模型上下文限制裁剪消息，并为输出保留空间。若配置了知识但 Prompt 未显式使用 `{knowledge}`，当前代码会在适当条件下把知识追加到 system content，避免“检索到了但模型没看到”。

## 12. 模型适配

`LLMBundle` 根据租户配置解析模型实例，再调用 `rag/llm` 中对应能力。

模型类型应分开理解：

| 类型 | 输入输出 | 在 RAG 中的职责 |
|---|---|---|
| Chat | messages → text | 问题改写与答案生成 |
| Embedding | text → vector | 摄取和查询编码 |
| Rerank | query + docs → scores | 候选精排 |
| CV | image + text → text | 视觉理解 |
| ASR | audio → text | 音频转录 |
| TTS | text → audio | 答案语音化 |

供应商只是实现，能力类型才是业务依赖。

## 13. 流式响应

流式调用通过异步生成器不断产出增量。需要同时处理：

- 首 token 延迟；
- reasoning 与 answer 的区分；
- 客户端断开；
- 中途异常；
- 最终 reference 和 token 统计；
- 会话持久化时机。

只看到前端文本逐字出现，不能证明最终事件和引用已经正确完成。

## 14. 引用生成

模型可能按提示直接生成引用标记。如果没有生成可识别标记，系统会：

1. 按需取回候选 chunk 的向量；
2. 切分或分析答案；
3. 用词法与向量相似度匹配答案片段和 chunk；
4. 插入引用标记；
5. 只返回实际引用到的 reference。

因此引用质量至少依赖：候选证据正确、chunk 粒度合理、答案忠于证据、引用匹配阈值合理。

## 15. 质量诊断矩阵

| 现象 | 可能阶段 | 验证方法 |
|---|---|---|
| 正确 chunk 完全没出现 | 召回 | 直接调用 retrieval test，放宽阈值 |
| 正确 chunk 排名很低 | 融合/重排 | 比较词法、向量、rerank 分数 |
| 证据正确但答案错误 | Prompt/LLM | 检查实际 system prompt 和上下文 |
| 答案正确但引用错 | 引用回填 | 检查最终候选与 marker |
| 多轮第二问失败 | 问题改写 | 打印最终检索 query |
| 延迟高 | 多阶段 | 分别记录改写、Embedding、Search、Rerank、LLM 时间 |

## 16. 实验：建立最小检索评测集

准备 20 个问题，每题记录：

- 标准答案；
- 必须命中的文档和段落；
- 允许的同义表达；
- 不应该使用的冲突文档。

每次实验保存：Recall@K、正确 chunk 排名、最终引用、回答正确性和总延迟。一次只改变一个参数，避免无法归因。

## 17. 本册检查点

1. 过滤、召回、融合、重排和阈值各自解决什么问题？
2. 为什么提高 `top_n` 可能让答案更差？
3. Embedding Model 和 Chat Model 为什么可以来自不同供应商？
4. “检索到了”与“Prompt 使用了知识”如何分别验证？
5. 引用为何需要在生成后再次匹配？
