#
#  Copyright 2024 The InfiniFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.

"""Embedding Service 模块。

提供 [`EmbeddingService`](rag/svr/task_executor_refactor/embedding_service.py:42) 用于向量嵌入操作。"""

from typing import Any, Dict, List, Tuple

import numpy as np
from common import settings
from common.misc_utils import thread_pool_exec
from common.token_utils import truncate
from rag.svr.task_executor_refactor.embedding_utils import EmbeddingUtils
from rag.svr.task_executor_refactor.task_context import TaskContext


class EmbeddingService:
    """Service 用于向量嵌入操作。

    该服务处理：
    - 文本块的批量编码
    - 标题+内容向量组合
    - Embedding模型速率限制

    所有中间结果均通过RecordingContext记录以供比较。"""

    def __init__(
        self,
        ctx: TaskContext,
        embedding_batch_size: int = None,
    ):
        """初始化EmbeddingService。

        参数：
            ctx: TaskContext 包含任务配置和执行资源。
            embedding_batch_size：嵌入操作的批量大小。"""
        self._task_context = ctx

        self._embedding_batch_size = embedding_batch_size or settings.EMBEDDING_BATCH_SIZE

    async def embed_chunks(
        self,
        docs: List[Dict[str, Any]],
        embedding_model,
        parser_config: Dict = None,
    ) -> Tuple[int, int]:
        """嵌入块列表。

        参数：
            docs：要嵌入的块字典列表。
            embedding_model：嵌入模型包（LLMBundle）。
            parser_config：文件名嵌入权重的解析器配置。

        返回：
            (token_count、vector_size) 的元组。"""
        if parser_config is None:
            parser_config = {}

        # Embedding 阶段只对内存中的 Chunk 做字段增强，不负责持久化。
        # titles 通常来自文件名，contents 来自每个 Chunk 的正文。
        titles, contents = EmbeddingUtils.prepare_texts_for_embedding(docs)

        # 文件名只编码一次，再复制到每个 Chunk。后面会按 filename_embd_weight
        # 与正文向量加权合并，使检索同时保留文件名和正文语义。
        tk_count = 0
        if len(titles) > 0 and len(titles) == len(contents):
            async with self._task_context.embed_limiter:
                vts, c = await thread_pool_exec(embedding_model.encode, titles[0:1])
            tts = np.tile(vts[0], (len(contents), 1))
            tk_count += c
        else:
            tts = None

        # 正文按 EMBEDDING_BATCH_SIZE 分批调用模型；embed_limiter 控制所有任务共享的
        # Embedding 并发，thread_pool_exec 防止同步 SDK 阻塞事件循环。
        vects_batches = []
        for i in range(0, len(contents), self._embedding_batch_size):
            batch = contents[i : i + self._embedding_batch_size]
            async with self._task_context.embed_limiter:
                vts, c = await thread_pool_exec(
                    self._batch_encode_wrapper,
                    [truncate(t, embedding_model.max_length - 10) for t in batch],
                    embedding_model,
                )
            vects_batches.append(vts)
            tk_count += c
            if self._task_context.progress_cb:
                self._task_context.progress_cb(prog=0.7 + 0.2 * (i + 1) / len(contents), msg="")

        # 使用 EmbeddingUtils 的堆栈向量
        cnts = EmbeddingUtils.stack_vectors(vects_batches)

        # 最终每个 Chunk 的向量是“文件名向量”和“正文向量”的加权组合，不是简单拼接；
        # filename_embd_weight 越大，查询命中文件名语义时该文档越容易被向量召回。
        # 使用 EmbeddingUtils 组合标题和内容向量
        title_weight = parser_config.get("filename_embd_weight", EmbeddingUtils.DEFAULT_TITLE_WEIGHT)
        vects = EmbeddingUtils.combine_title_content_vectors(tts, cnts, title_weight)

        assert len(vects) == len(docs)

        # 将最终向量附加回每个 docs 元素，字段名包含维度，例如 q_1024_vec。
        # 同一个 ES Chunk 因此同时拥有倒排索引字段（content_ltks 等）和 dense_vector 字段，
        # 后续 BM25 与 KNN 虽查询同一条记录，但使用的是两套不同索引结构。
        # 随后 TaskHandler 才会调用 ChunkService.insert_chunks 将其写入 ES。
        vector_size = EmbeddingUtils.attach_vectors(docs, vects)

        return tk_count, vector_size

    @staticmethod
    def _batch_encode_wrapper(txts: List[str], embedding_model) -> Tuple[np.ndarray, int]:
        """用于批量编码的同步包装器 - 与 thread_pool_exec 一起使用。"""
        return embedding_model.encode(txts)
