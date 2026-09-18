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

"""Task 处理程序模块。

提供[`TaskHandler`](rag/svr/task_executor_refactor/task_handler.py:56)作为主入口点
用于使用重构的、可测试的方法处理文档处理任务。"""

import asyncio
import logging
import json

# Wiki/工件编译流水线位于 ``dataset_wiki_generator``；其调度入口是
# ``TaskHandler.run`` 中的 ``task_type == "artifact"`` 分支。
# 文档结构编译辅助函数（CHAIN_KINDS、compile_structure_from_text、
# merge_compiled_structures、validate_and_correct_chain）位于 ``chunk_post_processor``。
import xxhash

from timeit import default_timer as timer
from typing import AsyncIterator, Callable, Dict, List, Optional

from api.db.services.document_service import DocumentService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.compilation_template_group_service import CompilationTemplateGroupService
from api.db.joint_services.memory_message_service import handle_save_to_memory_task
from api.db.joint_services.tenant_model_service import (
    get_tenant_default_model_by_type,
    resolve_model_config,
    get_model_config_by_id,
)
from api.db.services.llm_service import LLMBundle
from api.db.services.task_service import GRAPH_RAPTOR_FAKE_DOC_ID, abort_doc_chunking_counter
from common.constants import LLMType
from common.exceptions import TaskCanceledException
from common.connection_utils import timeout
from common.misc_utils import thread_pool_exec
from rag.nlp import search
from rag.svr.task_executor_refactor.constants import CANVAS_DEBUG_DOC_ID
from rag.svr.task_executor_refactor.chunk_service import ChunkService
from rag.svr.task_executor_refactor.dataflow_service import BillingHook, DataflowService
from rag.svr.task_executor_refactor.embedding_service import EmbeddingService
from rag.svr.task_executor_refactor.post_processor import PostProcessor
from rag.svr.task_executor_refactor.raptor_service import RaptorService
from rag.svr.task_executor_refactor.raptor_utils import delete_raptor_chunks
from rag.svr.task_executor_refactor.recording_context import RecordingContext
from rag.svr.task_executor_refactor.task_context import TaskContext
from rag.graphrag.general.index import run_graphrag_for_kb
from api.db.services.file2document_service import File2DocumentService
from rag.prompts.generator import run_toc_from_text
from common import settings


def _parser_config_compilation_template_ids(parser_config, tenant_id: str) -> list[str]:
    """解析文档的 parser_config 来编译模板 ID
    查找配置的组。如果文档没有，则返回“`[]`”
    组设置或无组都可以解决。"""
    from rag.svr.task_executor_refactor.chunk_post_processor import (
        _parser_config_compilation_template_group_ids,
    )

    template_ids: list[str] = []
    seen: set[str] = set()
    for group_id in _parser_config_compilation_template_group_ids(parser_config):
        for template_id in CompilationTemplateGroupService.resolve_template_ids(group_id, tenant_id):
            if template_id in seen:
                continue
            seen.add(template_id)
            template_ids.append(template_id)
    return template_ids


# Document-结构编译可调参数
# (DOC_STRUCTURE_COMPILE_BATCH_CHUNKS, DOC_STRUCTURE_MERGE_MAX_DOCS,
# STRUCTURE_CHAIN_CORRECTION_TIMEOUT_S) 移至
# ``chunk_post_processor``.

# Wiki / 工件可调参数（``WIKI_MAP_BATCH_CHUNKS``，
# ``WIKI_GRAPH_MAX_CHUNK_IDS_PER_NODE``，提交标题/评论
# 模板）移至“`dataset_wiki_generator`”。

# 语料库→技能编译管道位于
# ``rag.svr.task_executor_refactor.dataset_skill_generator``。它的条目
# 点是 :func:`run_corpus2skill`;该处理程序从
# ``task_type == "skill"`` branch of ``run``如下。


class TaskHandler:
    """文档处理的主要任务处理程序。

    此类协调整个文档处理管道：
    1. Task类型检测（内存、数据流、raptor、graphrag、standard）
    2.模型绑定（嵌入、聊天）
    3、Chunk构建或RAPTOR/GraphRAG执行
    4.Embedding
    5. 索引
    6.后处理（TOC，表元数据）

    所有中间结果均通过RecordingContext记录以供比较。"""

    def __init__(
        self,
        ctx: TaskContext,
        billing_hook: Optional[BillingHook] = None,
    ):
        """初始化TaskHandler。

        参数：
            ctx: TaskContext 包含任务配置和执行资源。
            billing_hook：管道 success/error 回调的可选计费挂钩。"""
        self._task_context = ctx
        self._billing_hook = billing_hook

    @staticmethod
    def _is_standard_chunking_task(task_type: str) -> bool:
        from rag.svr.task_executor_refactor.dataset_structure_merger import (
            STRUCTURE_MERGE_TASK_TYPES,
        )

        task_type = (task_type or "").lower()
        return task_type not in {
            "memory",
            "raptor",
            "graphrag",
            "mindmap",
            "artifact",
            "skill",
            "evaluation",
            "reembedding",
            "clone",
        } | STRUCTURE_MERGE_TASK_TYPES and not task_type.startswith("dataflow")

    # 真正的处理入口
    async def handle_task(self) -> None:
        try:
            await self.handle()  # 进入新版解析实现
        except Exception:
            if self._is_standard_chunking_task(self._task_context.task_type):
                abort_doc_chunking_counter(self._task_context.doc_id)
            raise
        finally:
            task_id = self._task_context.id
            task_tenant_id = self._task_context.tenant_id
            task_dataset_id = self._task_context.kb_id
            task_doc_id = self._task_context.doc_id
            if self._task_context.has_canceled_func(task_id):
                if self._is_standard_chunking_task(self._task_context.task_type):
                    abort_doc_chunking_counter(task_doc_id)
                    try:
                        exists = await thread_pool_exec(
                            settings.docStoreConn.index_exist,
                            search.index_name(task_tenant_id),
                            task_dataset_id,
                        )
                        if exists:
                            ret = await thread_pool_exec(
                                settings.docStoreConn.delete,
                                {"doc_id": task_doc_id},
                                search.index_name(task_tenant_id),
                                task_dataset_id,
                            )
                            self._task_context.recording_context.save_func_return_value("docStoreConn.delete", ret)
                    except Exception as e:
                        logging.exception(f"Remove doc({task_doc_id}) from docStore failed when task({task_id}) canceled, exception: {e}")

    @timeout(60 * 60 * 3, 1)
    async def handle(self) -> None:
        """按 task_type 分流任务；普通上传文档最终进入 _run_standard_chunking_impl。"""
        ctx = self._task_context
        task_type = ctx.task_type
        task_id = ctx.id

        logging.info(
            "任务处理流水线已开始 任务ID=%s 文档ID=%s 知识库ID=%s 任务类型=%s 解析器ID=%s 页面范围=[%s,%s)",
            task_id,
            ctx.doc_id,
            ctx.kb_id,
            task_type or "standard",
            ctx.parser_id,
            ctx.from_page,
            ctx.to_page,
        )

        # 处理内存任务
        if task_type == "memory":
            # 在空运行时忽略 - 重构时 handle_save_to_memory_task 没有变化
            if isinstance(ctx.write_interceptor, RecordingContext):
                logging.info(f"dry run, ignore handle_save_to_memory_task {task_id}")
            else:
                # 实际运行-非空运行
                await handle_save_to_memory_task(ctx.raw_task)
            return

        # 在任何昂贵操作前检查取消标记；处理中各阶段还会重复检查，以便尽快停止并清理。
        if ctx.has_canceled_func(task_id):
            ctx.progress_cb(-1, msg="Task has been canceled.")
            return

        # Language defaults to "Chinese" via TaskContext._DEFAULTS 鈥?safe to bind model directly.
        # Bind嵌入模型（匹配原始do_handle_task顺序：路由前bind + init_kb）
        # 绑定 Embedding 模型
        result = await self._bind_embedding_model()
        if result is None:
            return
        embedding_model, vector_size = result

        with embedding_model:
            # 索引按 tenant_id 命名、按 kb_id 过滤；向量维度来自实际 Embedding 探测结果。
            # _init_kb 只创建/校验索引及 mapping，不会在这里写入任何 Chunk。
            self._init_kb(vector_size)  # 在解析文档前，确保当前租户对应的文档索引已经存在。

            # 处理数据流任务（在init_kb之后，匹配原始行为）
            if task_type == "dataflow" and ctx.doc_id == CANVAS_DEBUG_DOC_ID:
                await self._run_dataflow()
                return

            if task_type.startswith("dataflow"):
                await self._run_dataflow()
                return

            # 路由至适当的处理程序
            from rag.svr.task_executor_refactor.dataset_structure_merger import (
                is_structure_merge_task,
            )

            if task_type == "raptor":
                await self._run_raptor(embedding_model, vector_size)
            elif task_type == "graphrag":
                await self._run_graphrag(embedding_model)
            elif task_type == "mindmap":
                ctx.progress_cb(1, "place holder")
            elif task_type == "wiki":
                from rag.svr.task_executor_refactor.dataset_wiki_generator import (
                    run_wiki_incremental,
                )

                # 从模板配置中解析 Wiki 模式。
                wiki_mode = None
                try:
                    from api.db.services.compilation_template_service import (
                        CompilationTemplateService,
                    )
                    from rag.svr.task_executor_refactor.dataset_wiki_generator import (
                        _parser_config_compilation_template_ids,
                    )

                    pc = self._task_context.parser_config or {}
                    for tid in _parser_config_compilation_template_ids(pc, self._task_context.tenant_id):
                        tpl = CompilationTemplateService.get_saved(tid, self._task_context.tenant_id)
                        cfg = (tpl.get("config") or {}) if tpl else {}
                        if isinstance(cfg, dict) and cfg.get("mode") in ("entity", "topic"):
                            wiki_mode = cfg["mode"]
                            break
                except Exception:
                    pass

                await run_wiki_incremental(
                    self._task_context,
                    embedding_model,
                    self._load_chunks_for_doc,
                    mode=wiki_mode,
                )
            elif task_type == "skill":
                from rag.svr.task_executor_refactor.dataset_skill_generator import (
                    run_corpus2skill,
                )

                await run_corpus2skill(
                    self._task_context,
                    embedding_model,
                    self._load_chunks_for_doc,
                )
            elif is_structure_merge_task(task_type):
                from rag.svr.task_executor_refactor.dataset_structure_merger import (
                    run_structure_merge,
                )

                await run_structure_merge(self._task_context)
            elif task_type == "evaluation":
                await self._run_evaluation()
            elif task_type == "reembedding":
                await self._run_reembedding()
            elif task_type == "clone":
                await self._run_clone()
            else:
                await self._run_standard_chunking(embedding_model, vector_size)

    def _init_kb(self, vector_size: int) -> None:
        """创建或校验当前租户的文档索引及 parser/向量维度相关 mapping。"""
        ctx = self._task_context
        # 同一租户的多个知识库通常共享 ragflow_{tenant_id} 索引，Chunk 再通过 kb_id 隔离；
        # create_idx 内部会处理“已存在”情况，因此每个 Task 进入这里不会重复创建数据。
        idxnm = search.index_name(ctx.tenant_id)  # 根据租户 ID 生成索引名
        parser_id = ctx.parser_id
        # 如果不存在则创建索引
        settings.docStoreConn.create_idx(idxnm, ctx.kb_id, vector_size, parser_id)  # 创建索引

    async def _run_dataflow(self) -> None:
        """运行数据流管道。"""
        dataflow_service = DataflowService(
            ctx=self._task_context,
            billing_hook=self._billing_hook,
        )
        await dataflow_service.run_dataflow()

    async def _run_evaluation(self) -> None:
        """运行评估任务。"""
        ctx = self._task_context
        ctx.progress_cb(1, "Evaluation task placeholder")

    async def _run_reembedding(self) -> None:
        """运行重新嵌入任务。"""
        ctx = self._task_context
        ctx.progress_cb(1, "Reembedding task placeholder")

    async def _run_clone(self) -> None:
        """运行克隆任务。"""
        ctx = self._task_context
        ctx.progress_cb(1, "Clone task placeholder")

    async def _bind_embedding_model(self) -> Optional[tuple]:
        """将嵌入模型绑定到任务。

        返回：
            成功时为 (embedding_model、vector_size) 元组，失败时为 None。"""
        ctx = self._task_context
        task_tenant_id = ctx.tenant_id
        task_embedding_id = ctx.embd_id
        task_language = ctx.language

        try:
            if ctx.tenant_embd_id:
                try:
                    embd_model_config = get_model_config_by_id(task_tenant_id, LLMType.EMBEDDING, ctx.tenant_embd_id)
                except LookupError:
                    embd_model_config = resolve_model_config(task_tenant_id, LLMType.EMBEDDING, task_embedding_id)
            elif task_embedding_id:  # 根据向量模型 ID 获取模型配置。
                embd_model_config = resolve_model_config(task_tenant_id, LLMType.EMBEDDING, task_embedding_id)
            else:
                embd_model_config = get_tenant_default_model_by_type(task_tenant_id, LLMType.EMBEDDING)
            embedding_model = LLMBundle(task_tenant_id, embd_model_config, lang=task_language)
            vts, _ = embedding_model.encode(["ok"])
            return embedding_model, len(vts[0])
        except Exception as e:
            error_message = f"Fail to bind embedding model: {str(e)}"
            ctx.progress_cb(-1, msg=error_message)
            logging.exception(error_message)
            raise

    async def _run_raptor(
        self,
        embedding_model: LLMBundle,
        vector_size: int,
        mark_done: bool = True,
    ) -> None:
        """运行 RAPTOR 摘要生成。"""
        ctx = self._task_context
        task_tenant_id = ctx.tenant_id
        task_dataset_id = ctx.kb_id
        kb_task_llm_id = ctx.kb_parser_config.get("llm_id") or ctx.llm_id

        ok, kb = KnowledgebaseService.get_by_id(task_dataset_id)
        if not ok:
            ctx.progress_cb(prog=-1.0, msg="Cannot found valid dataset for RAPTOR task")
            return

        kb_parser_config = kb.parser_config
        if not kb_parser_config.get("raptor", {}).get("use_raptor", False):
            kb_parser_config.update(
                {
                    "raptor": {
                        "use_raptor": True,
                        "prompt": "Summarize the paragraphs below without inventing facts or changing numbers.\nOutput exactly two parts in the same language as the source:\n1. First line: a concise title only.\n2. Following lines: a concise summary of the content.\nDo not output labels, Markdown headings, bullet points, or any other commentary.\n\nParagraphs:\n{cluster_content}",
                        "max_token": 512,
                        "clustering_threshold": 0.3,
                        "clustering_ratio": 0.5,
                        "max_cluster": 64,
                        "random_seed": 0,
                        "scope": "file",
                    },
                }
            )
            if ctx.write_interceptor:
                update_result = ctx.write_interceptor.intercept("KnowledgebaseService.update_by_id")
            else:
                update_result = KnowledgebaseService.update_by_id(kb.id, {"parser_config": kb_parser_config})

            if not update_result:
                ctx.progress_cb(prog=-1.0, msg="Internal error: Invalid RAPTOR configuration")
                return

        # 猛禽绑定 LLM
        chat_model_config = resolve_model_config(task_tenant_id, LLMType.CHAT, kb_task_llm_id)
        with LLMBundle(task_tenant_id, chat_model_config, lang=ctx.language) as chat_model:
            # 运行 RAPTOR
            raptor_service = RaptorService(ctx=ctx)

            async with ctx.kg_limiter:
                chunks, token_count, raptor_cleanup_chunks = await raptor_service.run_raptor_for_kb(
                    kb_parser_config=kb_parser_config,
                    chat_mdl=chat_model,
                    embd_mdl=embedding_model,
                    vector_size=vector_size,
                    doc_ids=ctx.doc_ids or [ctx.doc_id],
                )

            ctx.recording_context.record("raptor_chunks", chunks)
            ctx.recording_context.record("raptor_token_count", token_count)

            # 插入 RAPTOR 块
            if chunks:
                task_doc_id = (ctx.doc_ids or [ctx.doc_id] or [GRAPH_RAPTOR_FAKE_DOC_ID])[0]
                chunk_service = ChunkService(ctx=ctx)
                insert_result = await chunk_service.insert_chunks(ctx.id, task_tenant_id, task_dataset_id, chunks)  # 写入chunk
                if insert_result:
                    ctx.recording_context.record("insertion_result", "success")
                else:
                    ctx.recording_context.record("insertion_result", "failed")

                # 清理陈旧的 RAPTOR 块
                cleaned_chunks = 0
                for cleanup_doc_id, keep_method in raptor_cleanup_chunks:
                    ret = await self._delete_raptor_chunks(cleanup_doc_id, task_tenant_id, task_dataset_id, keep_method)
                    cleaned_chunks += ret

                if cleaned_chunks:
                    ctx.progress_cb(msg=f"Cleaned up {cleaned_chunks} stale RAPTOR chunks.")

                # 从刚刚构建的每个文档 RAPTOR 树图
                # 插入摘要。 ``chunks`` 中的每个块都携带
                # doc_id 它是在下面编写的（真实的文档ID
                # 范围="file"; GRAPH_RAPTOR_FAKE_DOC_ID 为
                # 数据集范围路径）。我们将每个图行具体化为
                # 与 doc_id 不同，因此数据集结构图
                # 端点可以为每个文档显示一个 RAPTOR 选项卡。
                # 这里的失败是尽力而为——总结是
                # 已经坚持了；该选项卡不会呈现。
                raptor_doc_ids = {str(c.get("doc_id")) for c in chunks if c.get("doc_id")}
                for raptor_doc_id in raptor_doc_ids:
                    try:
                        await raptor_service._persist_raptor_graph_to_es(raptor_doc_id)
                    except Exception:
                        logging.exception(
                            "raptor_graph：kb = %s doc = %s 构建失败",
                            task_dataset_id,
                            raptor_doc_id,
                        )

                # 更新文档统计信息
                if ctx.write_interceptor:
                    ctx.write_interceptor.intercept("DocumentService.increment_chunk_num")
                else:
                    DocumentService.increment_chunk_num(task_doc_id, task_dataset_id, token_count, len(chunks), 0)

            if mark_done:
                ctx.recording_context.record("task_status", "completed")
                ctx.progress_cb(prog=1.0, msg="RAPTOR done")

    async def _run_graphrag(self, embedding_model: LLMBundle) -> None:
        """运行GraphRAG。"""
        ctx = self._task_context
        task_tenant_id = ctx.tenant_id
        task_dataset_id = ctx.kb_id
        kb_task_llm_id = ctx.kb_parser_config.get("llm_id") or ctx.llm_id
        task_language = ctx.language

        ok, kb = KnowledgebaseService.get_by_id(task_dataset_id)
        if not ok:
            ctx.progress_cb(prog=-1.0, msg="Cannot found valid dataset for GraphRAG task")
            return

        kb_parser_config = kb.parser_config
        if not kb_parser_config.get("graphrag", {}).get("use_graphrag", False):
            kb_parser_config.update(
                {
                    "graphrag": {
                        "use_graphrag": True,
                        "entity_types": [
                            "organization",
                            "person",
                            "geo",
                            "event",
                            "category",
                        ],
                        "method": "light",
                        "batch_chunk_token_size": 4096,
                        "retry_attempts": 2,
                        "retry_backoff_seconds": 2.0,
                        "retry_backoff_max_seconds": 60.0,
                        "build_subgraph_timeout_per_chunk_seconds": 300,
                        "build_subgraph_min_timeout_seconds": 600,
                        "merge_timeout_seconds": 180,
                        "resolution_timeout_seconds": 1800,
                        "community_timeout_seconds": 1800,
                        "lock_acquire_timeout_seconds": 600,
                    }
                }
            )
            if ctx.write_interceptor:
                update_result = ctx.write_interceptor.intercept("KnowledgebaseService.update_by_id")
            else:
                update_result = KnowledgebaseService.update_by_id(kb.id, {"parser_config": kb_parser_config})
            if not update_result:
                ctx.progress_cb(prog=-1.0, msg="Internal error: Invalid GraphRAG configuration")
                return

        graphrag_conf = kb_parser_config.get("graphrag", {})
        start_ts = timer()
        chat_model_config = resolve_model_config(task_tenant_id, LLMType.CHAT, kb_task_llm_id)
        with LLMBundle(task_tenant_id, chat_model_config, lang=task_language) as chat_model:
            with_resolution = graphrag_conf.get("resolution", False)
            with_community = graphrag_conf.get("community", False)

            async with ctx.kg_limiter:
                result = await run_graphrag_for_kb(
                    row=ctx.raw_task,
                    doc_ids=ctx.doc_ids,
                    language=task_language,
                    kb_parser_config=kb_parser_config,
                    chat_model=chat_model,
                    embedding_model=embedding_model,
                    callback=ctx.progress_cb,
                    with_resolution=with_resolution,
                    with_community=with_community,
                )
                logging.info(f"GraphRAG task result for task {ctx.raw_task}:\n{result}")

            ctx.recording_context.record("graphrag_result", result)
            ctx.progress_cb(prog=1.0, msg="Knowledge Graph done ({:.2f}s)".format(timer() - start_ts))

    async def _run_standard_chunking(
        self,
        embedding_model: LLMBundle,
        vector_size: int,
    ) -> None:
        ctx = self._task_context
        try:
            await self._run_standard_chunking_impl(embedding_model, vector_size)
        except Exception:
            abort_doc_chunking_counter(ctx.doc_id)
            raise

    async def _run_standard_chunking_impl(
        self,
        embedding_model: LLMBundle,
        vector_size: int,
    ) -> None:
        """运行标准分块管道。"""
        ctx = self._task_context
        task_id = ctx.id
        task_tenant_id = ctx.tenant_id
        task_dataset_id = ctx.kb_id
        task_doc_id = ctx.doc_id
        task_start_ts = timer()

        def on_chunking_start(wait_time):
            nonlocal task_start_ts
            task_start_ts += wait_time

        doc_task_llm_id = ctx.parser_config.get("llm_id") or ctx.llm_id
        ctx.raw_task["llm_id"] = doc_task_llm_id

        # 构建块
        start_ts = timer()
        chunk_service = ChunkService(ctx=ctx)

        # 【普通文档解析主链路】
        # 1. 根据 doc_id 定位并读取对象存储中的完整原文件；
        # 2. 解析原文件、切 Chunk，并将 Chunk 图片保存到对象存储；
        # 3. 调用 Embedding 模型，把向量直接附加到内存中的 Chunk dict；
        # 4. 批量写入配置的 doc engine（当前为 Elasticsearch）；
        # 5. 更新 MySQL 中的 Task Chunk IDs、Document 统计和执行进度；
        # 6. 最后一个分页 Task 执行文档级收尾，然后由外层对 Redis 消息 XACK。

        # File2DocumentService 将 doc_id 转换成对象存储 bucket/object key。
        # 此处读取的是上传阶段保存的完整原文件；Chunk 文本不从 MinIO 读取。
        bucket, name = File2DocumentService.get_storage_address(doc_id=ctx.doc_id)
        binary = await self._get_storage_binary(bucket, name)
        if binary is None:
            raise FileNotFoundError(f"Can not find file <{ctx.name}> from minio. Could you try it again.")
        # build_chunks 返回的是尚未生成向量、尚未写入 ES 的内存对象。
        chunks = await chunk_service.build_chunks(binary, on_chunking_start)
        logging.info(
            "文档切片已生成 任务ID=%s 文档ID=%s 切片数量=%d",
            task_id,
            task_doc_id,
            len(chunks),
        )
        ctx.recording_context.record("chunks", chunks)
        chunk_ids = [c.get("id") for c in chunks if isinstance(c, dict) and "id" in c]
        ctx.recording_context.record("chunk_ids_count", len(chunk_ids))

        logging.info("Build document {}: {:.2f}s".format(ctx.name, timer() - start_ts))

        if not chunks:
            ctx.progress_cb(msg=f"No chunk built from {ctx.name}")
            if not await self._run_document_post_chunking_if_last(
                embedding_model,
                vector_size,
                task_start_ts,
                0,
                0,
            ):
                return
            task_time_cost = timer() - task_start_ts
            ctx.recording_context.record("task_status", "completed")
            ctx.progress_cb(prog=1.0, msg="Task done ({:.2f}s)".format(task_time_cost))
            return

        ctx.progress_cb(msg="Generate {} chunks".format(len(chunks)))

        # 嵌入块
        start_ts = timer()
        embedding_service = EmbeddingService(ctx=ctx)
        try:
            # 对 Chunk 文本批量编码，并写入形如 q_{vector_size}_vec 的字段；
            # 此处仍只修改内存中的 chunks，下一阶段才真正写入 ES。
            token_count, vector_size = await embedding_service.embed_chunks(chunks, embedding_model, ctx.parser_config)
        except TaskCanceledException:
            raise
        except Exception as e:
            error_message = "Generate embedding error:{}".format(str(e))
            ctx.progress_cb(-1, error_message)
            logging.exception(error_message)
            raise

        logging.info(
            "向量生成完成 任务ID=%s 文档ID=%s 切片数量=%d 令牌数量=%d 向量维度=%d",
            task_id,
            task_doc_id,
            len(chunks),
            token_count,
            vector_size,
        )

        ctx.recording_context.record("token_count", token_count)
        ctx.recording_context.record("vector_size", vector_size)
        progress_message = "Embedding chunks ({:.2f}s)".format(timer() - start_ts)
        logging.info(progress_message)
        ctx.progress_cb(msg=progress_message)

        toc_thread = None
        if ctx.parser_id.lower() == "naive" and ctx.parser_config.get("toc_extraction", False):
            toc_thread = asyncio.create_task(asyncio.to_thread(self._build_toc, ctx, chunks, ctx.progress_cb))

        # 插入块
        chunk_count = len(set([chunk["id"] for chunk in chunks]))
        start_ts = timer()

        chunk_service = ChunkService(ctx=ctx)

        if ctx.has_canceled_func(task_id):
            abort_doc_chunking_counter(task_doc_id)
            ctx.progress_cb(-1, msg="Task has been canceled.")
            return

        # chunks 此时同时包含文本检索字段和向量字段；insert_chunks 分批写入 ES，
        # 并把已成功写入的 Chunk ID checkpoint 到 MySQL Task 记录。
        insert_result = await chunk_service.insert_chunks(task_id, task_tenant_id, task_dataset_id, chunks)

        if not insert_result:
            ctx.recording_context.record("insertion_result", "failed")
            abort_doc_chunking_counter(task_doc_id)
            return
        ctx.recording_context.record("insertion_result", "success")
        logging.info(
            "文档切片已写入检索存储 任务ID=%s 文档ID=%s 知识库ID=%s 切片数量=%d",
            task_id,
            task_doc_id,
            task_dataset_id,
            chunk_count,
        )

        # ES 已写成功后处理表格元数据；TOC 解析可与 Embedding/索引并行，完成后作为独立
        # TOC Chunk 再写入 doc store。普通 Chunk 的主写入不会等待 TOC 生成才开始。
        # 后处理
        post_processor = PostProcessor(ctx=ctx)
        await post_processor.process_table_parser_metadata(task_doc_id, chunks)

        ctx.progress_cb(msg="Indexing done ({:.2f}s).".format(timer() - start_ts))

        toc_chunk = await self._process_toc_thread(toc_thread)
        if toc_chunk:
            ctx.recording_context.record("toc_chunk", [toc_chunk])
            await post_processor.insert_toc_chunk(toc_chunk, chunk_service)

        if ctx.has_canceled_func(task_id):
            abort_doc_chunking_counter(task_doc_id)
            ctx.progress_cb(-1, msg="Task has been canceled.")
            return

        # Task.progress/chunk_ids 与 Document.chunk_num/token_num 是两层状态：前者描述当前分页
        # Task，后者累加整份文档统计；一个 PDF 的多个 Task 都会贡献到同一 Document。
        # 更新文档统计信息
        if ctx.write_interceptor:
            ctx.write_interceptor.intercept("DocumentService.increment_chunk_num")
        else:  # 更新 Document 的 Chunk 数和 Token 数
            DocumentService.increment_chunk_num(task_doc_id, task_dataset_id, token_count, chunk_count, 0)

        if not await self._run_document_post_chunking_if_last(
            embedding_model,
            vector_size,
            task_start_ts,
            len(chunks),
            token_count,
        ):
            return

        task_time_cost = timer() - task_start_ts
        ctx.recording_context.record("task_status", "completed")
        ctx.progress_cb(prog=1.0, msg="Task done ({:.2f}s)".format(task_time_cost))

        logging.info("Chunk doc({}), page({}-{}), chunks({}), token({}), elapsed:{:.2f}".format(ctx.name, ctx.from_page, ctx.to_page, len(chunks), token_count, task_time_cost))

    async def _run_document_post_chunking_if_last(
        self,
        embedding_model: LLMBundle,
        vector_size: int,
        task_start_ts: float,
        chunks_len: int,
        token_count: int,
    ) -> bool:
        """瘦委托器。管道位于
        ``rag.svr.task_executor_refactor.chunk_post_processor``。"""
        from rag.svr.task_executor_refactor.chunk_post_processor import (
            run_document_post_chunking_if_last,
        )

        return await run_document_post_chunking_if_last(
            self,
            embedding_model,
            vector_size,
            task_start_ts,
            chunks_len,
            token_count,
        )

    async def _process_toc_thread(self, toc_thread):
        try:
            if toc_thread:
                return await toc_thread
            else:
                return None
        finally:
            if toc_thread is not None and not toc_thread.done():
                toc_thread.cancel()

    @classmethod
    async def _get_storage_binary(cls, bucket: str, name: str) -> bytes:
        from common import settings

        """从存储中获取二进制文件。"""
        return await thread_pool_exec(settings.STORAGE_IMPL.get, bucket, name)

    @staticmethod
    async def _load_chunks_for_doc(
        tenant_id: str,
        kb_id: str,
        doc_id: str,
        batch_size: int = 500,
    ) -> AsyncIterator[List[Dict]]:
        """从文档存储中一次一批地流式传输文档的块。

        异步生成器可连续生成多达``batch_size``
        chunks. Order is pushed to the doc store via
        ``OrderByExpr().asc("page_num_int").asc("top_int")`` so callers do
        not need to re-sort. Rows with a ``compile_kwd`` marker (artifact
        pages, structure entities, etc.) are filtered out defensively.

        Memory is bounded by ``batch_size``：最多一个页面已具体化
        一次，如此长的文档不会使工人的堆膨胀。"""
        from common.doc_store.doc_store_base import OrderByExpr

        index_nm = search.index_name(tenant_id)
        if not settings.docStoreConn.index_exist(index_nm, kb_id):
            return

        select_fields = [
            "id",
            "doc_id",
            "content_with_weight",
            "page_num_int",
            "top_int",
            "compile_kwd",
        ]
        order_by = OrderByExpr()
        order_by.asc("chunk_order_int")
        order_by.asc("page_num_int")
        order_by.asc("top_int")

        offset = 0
        while True:
            try:
                res = await thread_pool_exec(
                    settings.docStoreConn.search,
                    select_fields,
                    [],
                    {
                        "doc_id": [doc_id],
                        "available_int": 1,
                        # 编译将其输出写回到相同的
                        # 文档索引。排除查询中的这些行
                        # 他们无法更改偏移分页，而这
                        # 任务仍在流式传输源块。
                        "must_not": {"exists": "compile_kwd"},
                    },
                    [],
                    order_by,
                    offset,
                    batch_size,
                    index_nm,
                    [kb_id],
                )
                field_map = settings.docStoreConn.get_fields(res, select_fields)
            except Exception:
                logging.exception("load_chunks_for_doc：无法加载 doc=%s 的块", doc_id)
                return
            if not field_map:
                # 恢复由旧的 doc-page-source upsert 损坏的行，这
                # 更新了共享“`doc_id`”的每一行并标记了源块
                # 与“`compile_kwd=wiki_doc_page_source`”。正品追踪
                # 行没有块体； MAP 行不可用。来源
                # 行仍然可用并保留其内容，因此它们可以
                # 无需根据 ids 进行猜测即可识别。
                try:
                    recovery_fields = [*select_fields, "available_int"]
                    recovered_batch: List[Dict] = []
                    recovery_offset = 0
                    recovery_page_size = 1000
                    while True:
                        recovery_res = await thread_pool_exec(
                            settings.docStoreConn.search,
                            recovery_fields,
                            [],
                            {"doc_id": [doc_id], "available_int": 1},
                            [],
                            order_by,
                            recovery_offset,
                            recovery_page_size,
                            index_nm,
                            [kb_id],
                        )
                        recovery_rows = settings.docStoreConn.get_fields(recovery_res, recovery_fields) or {}
                        for row_id, recovery_row in recovery_rows.items():
                            marker = recovery_row.get("compile_kwd")
                            if isinstance(marker, (list, tuple)):
                                marker = marker[0] if marker else ""
                            content = recovery_row.get("content_with_weight") or ""
                            if marker != "wiki_doc_page_source" or not content:
                                continue
                            await thread_pool_exec(
                                settings.docStoreConn.update,
                                {"id": row_id},
                                {"remove": "compile_kwd"},
                                index_nm,
                                kb_id,
                            )
                            recovered_batch.append(
                                {
                                    "id": row_id,
                                    "doc_id": recovery_row.get("doc_id") or doc_id,
                                    "content_with_weight": content,
                                    "page_num_int": recovery_row.get("page_num_int", 0),
                                    "top_int": recovery_row.get("top_int", 0),
                                }
                            )
                        if len(recovery_rows) < recovery_page_size:
                            break
                        recovery_offset += recovery_page_size
                    if recovered_batch:
                        logging.warning(
                            "load_chunks_for_doc：恢复的 %d 源块被错误标记为 wiki_doc_page_source doc=%s",
                            len(recovered_batch),
                            doc_id,
                        )
                        yield recovered_batch
                except Exception:
                    logging.exception("load_chunks_for_doc：doc = %s 的恢复查询失败", doc_id)
                return

            batch: List[Dict] = []
            for row_id, row in field_map.items():
                if row.get("compile_kwd"):
                    continue
                batch.append(
                    {
                        "id": row_id,
                        "doc_id": row.get("doc_id") or doc_id,
                        "content_with_weight": row.get("content_with_weight") or "",
                        "page_num_int": row.get("page_num_int", 0),
                        "top_int": row.get("top_int", 0),
                    }
                )
            if batch:
                yield batch
            if len(field_map) < batch_size:
                return
            offset += batch_size

    @classmethod
    def _build_toc(cls, ctx: TaskContext, docs: List[Dict], progress_cb: Callable) -> Optional[Dict]:
        """构建目录。"""
        progress_cb(msg="Start to generate table of content ...")
        chat_model_config = resolve_model_config(ctx.tenant_id, LLMType.CHAT, ctx.llm_id)
        with LLMBundle(ctx.tenant_id, chat_model_config, lang=ctx.language) as chat_mdl:
            docs = sorted(
                docs,
                key=lambda d: (
                    d.get("page_num_int", 0)[0] if isinstance(d.get("page_num_int", 0), list) else d.get("page_num_int", 0),
                    d.get("top_int", 0)[0] if isinstance(d.get("top_int", 0), list) else d.get("top_int", 0),
                ),
            )

            # NOTE：asyncio.run() 在工作线程中创建一个新的事件循环
            # （该方法通过asyncio.to_thread调用），这是
            # 用于在线程上下文中桥接同步 -> 异步的模式。
            toc: list[dict] = asyncio.run(run_toc_from_text([d["content_with_weight"] for d in docs], chat_mdl, progress_cb))
            logging.info("------------ T O C -------------\n" + json.dumps(toc, ensure_ascii=False, indent="  "))

            for ii, item in enumerate(toc):
                try:
                    chunk_val = item.pop("chunk_id", None)
                    if chunk_val is None or str(chunk_val).strip() == "":
                        logging.warning(f"Index {ii}: chunk_id is missing or empty. Skipping.")
                        continue
                    curr_idx = int(chunk_val or -1)
                    if curr_idx >= len(docs):
                        logging.error(f"Index {ii}: chunk_id {curr_idx} exceeds docs length {len(docs)}.")
                        continue
                    item["ids"] = [docs[curr_idx]["id"]]
                    if ii + 1 < len(toc):
                        next_chunk_val = toc[ii + 1].get("chunk_id", "")
                        if str(next_chunk_val).strip() != "":
                            next_idx = int(next_chunk_val)
                            for jj in range(curr_idx + 1, min(next_idx + 1, len(docs))):
                                item["ids"].append(docs[jj]["id"])
                        else:
                            logging.warning(f"Index {ii + 1}: next chunk_id is empty, range fill skipped.")
                except (ValueError, TypeError) as e:
                    logging.error(f"Index {ii}: Data conversion error - {e}")
                except Exception as e:
                    logging.exception(f"Index {ii}: Unexpected error - {e}")

            if toc:
                import copy

                d = copy.deepcopy(docs[-1])
                d["content_with_weight"] = json.dumps(toc, ensure_ascii=False)
                d["toc_kwd"] = "toc"
                d["available_int"] = 0
                d["page_num_int"] = [100000000]
                d["id"] = xxhash.xxh64((d["content_with_weight"] + str(d["doc_id"])).encode("utf-8", "surrogatepass")).hexdigest()
                return d
            return None

    async def _delete_raptor_chunks(self, doc_id: str, tenant_id: str, kb_id: str, keep_method: Optional[str]) -> int:
        """删除 RAPTOR 块。"""
        if self._task_context.write_interceptor:
            return self._task_context.write_interceptor.intercept("delete_raptor_chunks")
        else:
            return await delete_raptor_chunks(doc_id, tenant_id, kb_id, keep_method)
