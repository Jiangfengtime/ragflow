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

"""
Embedding Service Module.

Provides [`EmbeddingService`](rag/svr/task_executor_refactor/embedding_service.py:42) for vector embedding operations.
"""

from typing import Any, Dict, List, Tuple

import numpy as np
from common import settings
from common.misc_utils import thread_pool_exec
from common.token_utils import truncate
from rag.svr.task_executor_refactor.embedding_utils import EmbeddingUtils
from rag.svr.task_executor_refactor.task_context import TaskContext


class EmbeddingService:
    """Service for vector embedding operations.

    This service handles:
    - Batch encoding of text chunks
    - Title + content vector combination
    - Embedding model rate limiting

    All intermediate results are recorded via RecordingContext for comparison.
    """

    def __init__(
        self,
        ctx: TaskContext,
        embedding_batch_size: int = None,
    ):
        """Initialize EmbeddingService.

        Args:
            ctx: TaskContext containing task configuration and execution resources.
            embedding_batch_size: Batch size for embedding operations.
        """
        self._task_context = ctx

        self._embedding_batch_size = embedding_batch_size or settings.EMBEDDING_BATCH_SIZE

    async def embed_chunks(
        self,
        docs: List[Dict[str, Any]],
        embedding_model,
        parser_config: Dict = None,
    ) -> Tuple[int, int]:
        """Embed a list of chunks.

        Args:
            docs: List of chunk dictionaries to embed.
            embedding_model: The embedding model bundle (LLMBundle).
            parser_config: Parser configuration for filename embedding weight.

        Returns:
            Tuple of (token_count, vector_size).
        """
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

        # Stack vectors using EmbeddingUtils
        cnts = EmbeddingUtils.stack_vectors(vects_batches)

        # 最终每个 Chunk 的向量是“文件名向量”和“正文向量”的加权组合，不是简单拼接；
        # filename_embd_weight 越大，查询命中文件名语义时该文档越容易被向量召回。
        # Combine title and content vectors using EmbeddingUtils
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
        """Synchronous wrapper for batch encoding — used with thread_pool_exec."""
        return embedding_model.encode(txts)
