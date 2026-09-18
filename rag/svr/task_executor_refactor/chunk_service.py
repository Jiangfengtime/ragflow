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

"""Chunk Service 模块。

提供 [`ChunkService`](rag/svr/task_executor_refactor/chunk_service.py:50) 用于文档分块，
后处理（关键字、问题、元数据、标签）、MinIO 上传以及将块插入到文档存储中。

该模块通过委托来协调块构建管道：
- [`chunk_builder`](rag/svr/task_executor_refactor/chunk_builder.py)：解析器选择和文档分块
- [`chunk_post_processor`](rag/svr/task_executor_refactor/chunk_post_processor.py): 后处理函数"""

import asyncio
import copy
import logging
from datetime import datetime
from functools import partial
from timeit import default_timer as timer
from typing import Any, Dict, List

import xxhash
from common import settings
from common.connection_utils import timeout
from common.constants import PAGERANK_FLD, TAG_FLD
from common.misc_utils import thread_pool_exec
from common.float_utils import normalize_overlapped_percent
from api.db.services.document_service import DocumentService
from api.db.services.task_service import TaskService
from rag.nlp import search
from rag.svr.task_executor_refactor.constants import GRAPH_RAPTOR_FAKE_DOC_ID
from rag.svr.task_executor_refactor.task_context import TaskContext
from rag.utils.base64_image import image2id

# 重新导出以实现向后兼容
from rag.svr.task_executor_refactor.chunk_builder import (
    get_parser,
    run_chunking,
    extract_outline,
)
from rag.svr.task_executor_refactor.chunk_post_processor import (
    extract_keywords,
    generate_questions,
    generate_metadata,
    apply_built_in_metadata,
    apply_tags,
)


def apply_document_availability(chunks: List[Dict[str, Any]], status) -> int:
    """当文档状态禁用时，标记普通源块 available_int=0。

    已编译的产品 (compile_kwd) 被保留 - 它们已经不可搜索。
    返回标记的普通源块的数量。"""
    if str(status if status is not None else "1") != "0":
        return 0
    stamped = 0
    for ck in chunks:
        if ck.get("compile_kwd"):
            continue
        ck["available_int"] = 0
        stamped += 1
    return stamped


def apply_source_chunks_document_availability(chunks: List[Dict[str, Any]]) -> None:
    """将每个源文档的状态应用于其普通（非 RAPTOR）块。

    混合RAPTOR批次可以携带来自多个文档的块；每个都盖上印章
    组从其自己的文档状态而不是批次中的第一个 doc_id 。"""
    source_chunks_by_doc_id: Dict[str, List[Dict[str, Any]]] = {}
    for ck in chunks:
        if ck.get("raptor_kwd"):
            continue
        doc_id = ck.get("doc_id")
        if not doc_id or doc_id == GRAPH_RAPTOR_FAKE_DOC_ID:
            continue
        source_chunks_by_doc_id.setdefault(doc_id, []).append(ck)

    for doc_id, source_chunks in source_chunks_by_doc_id.items():
        ok, doc = DocumentService.get_by_id(doc_id)
        if not ok or doc is None:
            continue
        status = getattr(doc, "status", "1")
        stamped = apply_document_availability(source_chunks, status)
        if stamped:
            logging.info(
                "文档 %s 已禁用；在 %d 普通源块上标记 available_int=0",
                doc_id,
                stamped,
            )


class ChunkService:
    """Service 用于文档分块和后处理。

    该服务处理：
    - 通过解析器模块进行 Document 分块（委托给 chunk_builder）
    - MinIO 上传块图像
    - 关键词提取（委托chunk_post_processor）
    - 问题生成（委托给chunk_post_processor）
    - 元数据生成（委托给chunk_post_processor）
    - 内容标记（委托给chunk_post_processor）
    - 目录生成
    - Chunk 插入文档存储

    所有中间结果均通过RecordingContext记录以供比较。"""

    def __init__(
        self,
        ctx: TaskContext,
    ):
        """初始化ChunkService。

        参数：
            ctx: TaskContext 包含任务配置和执行资源。"""
        self._task_context = ctx

    @timeout(60 * 80, 1)
    async def build_chunks(
        self,
        storage_binary: bytes,  # 从对象存储读取的完整原文件二进制。
        on_chunking_start=None,  # 取得 Chunk 并发许可后的回调，用于排除排队耗时。
    ) -> List[Dict[str, Any]]:
        """从文档二进制构建块。

        这是块构建的主要入口点。它精心策划：
        1. File尺寸验证
        2.解析器选择和分块（委托给chunk_builder）
        3.轮廓提取（委托chunk_builder）
        4.MinIO上传
        5、后处理（委托chunk_post_processor）

        参数：
            storage_binary：文档的二进制内容。

        返回：
            准备嵌入的块字典列表。"""
        ctx = self._task_context
        # 验证文件大小
        # 这里使用的是 MySQL 文档记录中的 size，不是重新计算 len(storage_binary)
        if ctx.size > settings.DOC_MAXIMUM_SIZE:
            self._progress(prog=-1, msg="File size exceeds( <= %dMb )" % (int(settings.DOC_MAXIMUM_SIZE / 1024 / 1024)))
            self._task_context.recording_context.record("file_size_exceeded", True)
            return []
        ctx.recording_context.record("file_size_exceeded", False)
        ctx.recording_context.record("parser_id", ctx.parser_id)

        # 获取解析器
        # 根据 parser_id 选择解析器
        chunker = get_parser(ctx.parser_id)

        # 记录配置进行比较
        chunk_config = {
            "parser_id": ctx.parser_id,
            "chunk_token_num": ctx.parser_config.get("chunk_token_num", 128),
            "overlapped_percent": normalize_overlapped_percent(ctx.parser_config.get("overlapped_percent", 0)),
            "delimiter": ctx.parser_config.get("delimiter", "\n!?。；！？"),
            "from_page": ctx.from_page,
            "to_page": ctx.to_page,
            "language": ctx.language,
            "layout_recognizer": ctx.parser_config.get("layout_recognizer"),
        }
        # 记录当前任务的 Chunk 配置, 不落地, 只用于对比
        ctx.recording_context.record("chunk_config", chunk_config)

        # 调用 parser 模块的 chunk()。解析器是同步且可能消耗大量 CPU，因此 run_chunking
        # 会在 chunk_limiter 保护下把它放入线程池。返回的 cks 只是内存中的原始 Chunk。
        cks = await run_chunking(chunker, storage_binary, ctx, on_chunking_start)

        # 记录原始块
        self._task_context.recording_context.record("raw_chunks", cks)

        # PDF parser 可能把目录临时放在首个 Chunk 的 __outline__ 中；这里取出并写入
        # MySQL 文档元数据，避免 __outline__ 作为普通 Chunk 字段进入 ES。
        # 摘录大纲（委托）
        await extract_outline(cks, ctx)

        # 给每个 Chunk 补充 doc_id/kb_id、稳定 Chunk ID 和时间字段；如果 Chunk 携带图片，
        # 图片写入对象存储，Chunk 中只保留可用于反查图片的 img_id。
        docs = await self._prepare_docs_and_upload(cks)

        # 准备后记录文档
        self._task_context.recording_context.record("docs_after_prep", docs)

        # 以下后处理均发生在 Embedding 之前，因为 important_kwd/question_kwd/tag_kwd 等字段
        # 既会参与最终 ES 文档，也可能影响后续文本拼接与查询排序。是否执行由 parser_config 控制。
        # 后处理（委托给chunk_post_processor）
        if ctx.parser_config.get("auto_keywords", 0):
            await extract_keywords(docs, ctx)
        keywords = [d for d in docs if d.get("important_kwd")]
        self._task_context.recording_context.record("keywords_extracted", keywords)

        if ctx.parser_config.get("auto_questions", 0):
            await generate_questions(docs, ctx)
        questions = [d for d in docs if d.get("question_kwd")]
        self._task_context.recording_context.record("questions_generated", questions)

        if ctx.parser_config.get("enable_metadata", False) and (ctx.parser_config.get("metadata") or ctx.parser_config.get("built_in_metadata")):
            await generate_metadata(docs, ctx)
            apply_built_in_metadata(ctx)
        metadata_list = [d for d in docs if d.get("metadata_obj")]
        self._task_context.recording_context.record("metadata_list_generated", metadata_list)

        if ctx.kb_parser_config.get("tag_kb_ids", []):
            await apply_tags(docs, ctx)
        tags_applied = [d for d in docs if d.get(TAG_FLD)]
        self._task_context.recording_context.record("tags_applied", tags_applied)

        # 记录最终块
        self._task_context.recording_context.record("final_chunks", docs)
        final_chunk_ids = [c.get("id") for c in docs if isinstance(c, dict) and "id" in c]
        self._task_context.recording_context.record("final_chunk_ids_count", len(final_chunk_ids))

        return docs

    async def _prepare_docs_and_upload(self, cks: List[Dict]) -> List[Dict]:
        """准备文档并将图片上传到MinIO。"""
        ctx = self._task_context
        docs = []
        doc = {"doc_id": ctx.doc_id, "kb_id": str(ctx.kb_id)}
        if ctx.pagerank:
            doc[PAGERANK_FLD] = int(ctx.pagerank)

        st = timer()

        @timeout(60)
        async def upload_to_minio(document, chunk):
            try:
                d = copy.deepcopy(document)
                d.update(chunk)
                # Chunk ID 由“正文+doc_id”稳定计算：同一文档相同内容重跑时得到相同 ID，
                # ES bulk index 会覆盖该 _id；不同文档即使正文相同也不会冲突。
                d["id"] = xxhash.xxh64((chunk["content_with_weight"] + str(d["doc_id"])).encode("utf-8", "surrogatepass")).hexdigest()
                d["create_time"] = str(datetime.now()).replace("T", " ")[:19]
                d["create_timestamp_flt"] = datetime.now().timestamp()

                if d.get("img_id"):
                    docs.append(d)
                    return

                if not d.get("image"):
                    _ = d.pop("image", None)
                    d["img_id"] = ""
                    docs.append(d)
                    return
                # 只有图片二进制写入 MinIO；Chunk 文本和向量稍后进入 ES。
                # image2id 会用 kb_id/chunk_id 生成可反查的 img_id，并从 ES 文档中移除大块 image 数据。
                await image2id(d, partial(settings.STORAGE_IMPL.put, tenant_id=ctx.tenant_id), d["id"], ctx.kb_id)
                docs.append(d)
            except Exception:
                logging.exception("Saving image of chunk {}/{}/{} got exception".format(ctx.location, ctx.name, d["id"]))
                raise

        tasks = []
        for ck in cks:
            tasks.append(asyncio.create_task(upload_to_minio(doc, ck)))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"MINIO PUT({ctx.name}) got exception: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        el = timer() - st
        logging.info("MINIO PUT({}) cost {:.3f} s".format(ctx.name, el))
        return docs

    def _progress(self, prog=None, msg=None):
        """进度回调帮助程序。"""
        if prog is not None or msg is not None:
            self._task_context.progress_cb(prog=prog, msg=msg)

    # =========================================================================
    # 插入 Service 方法（从 insert_service.py 合并）
    # =========================================================================

    async def insert_chunks(
        self,
        task_id: str,
        task_tenant_id: str,
        task_dataset_id: str,
        chunks: List[Dict[str, Any]],
        doc_bulk_size: int = None,
    ) -> bool:
        """将块插入文档存储。

        参数：
            task_id：Task 标识符。
            task_tenant_id：租户ID。
            task_dataset_id：Dataset/knowledge底座ID。
            chunks：要插入的块字典列表。
            doc_bulk_size：文档存储插入的批量大小。

        返回：
            如果所有块均已成功插入，则为 True，否则为 False。"""
        # 这里才进入 Chunk 的持久化阶段。docs/chunks 在此前始终只存在于 Worker 内存中。
        # docStoreConn 是统一抽象；当前 DOC_ENGINE=elasticsearch 时实际调用 ESConnection.insert。
        doc_bulk_size = doc_bulk_size or settings.DOC_BULK_SIZE
        logging.info(
            "文档切片开始写入检索存储 任务ID=%s 文档ID=%s 知识库ID=%s 切片数量=%d 批次大小=%d",
            task_id,
            self._task_context.doc_id,
            task_dataset_id,
            len(chunks),
            doc_bulk_size,
        )

        self._apply_document_availability(chunks)

        # 某些 parser 会产生 mom/mom_with_weight（父级摘要）。父 Chunk 单独写入 doc store，
        # 子 Chunk 通过 mom_id 关联；父 Chunk available_int=0，不参与普通直接召回。
        # 创建母块（汇总块）
        mothers = self._create_mother_chunks(chunks)

        # 插入母块
        if not await self._insert_mother_chunks(task_id, task_tenant_id, task_dataset_id, mothers, doc_bulk_size):
            return False

        # 插入主块
        inserted = await self._insert_main_chunks(task_id, task_tenant_id, task_dataset_id, chunks, doc_bulk_size)
        logging.info(
            "文档切片写入检索存储完成 任务ID=%s 文档ID=%s 切片数量=%d 是否成功=%s",
            task_id,
            self._task_context.doc_id,
            len(chunks),
            inserted,
        )
        return inserted

    def _apply_document_availability(self, chunks: List[Dict[str, Any]]) -> None:
        """当所属文档被禁用（状态 = 0）时隐藏源块。

        在解析之前禁用仅更新MySQL；否则以后的插入会
        默认为 available_int=1 并保持可检索。"""
        doc_id = self._task_context.doc_id
        if not doc_id or doc_id == GRAPH_RAPTOR_FAKE_DOC_ID:
            return
        ok, doc = DocumentService.get_by_id(doc_id)
        if not ok or doc is None:
            return
        stamped = apply_document_availability(chunks, getattr(doc, "status", "1"))
        if stamped:
            logging.info(
                "文档 %s 已禁用；在 %d 普通源块上标记 available_int=0",
                doc_id,
                stamped,
            )

    @classmethod
    def _create_mother_chunks(cls, chunks: List[Dict]) -> List[Dict]:
        """从汇总字段创建母块。

        母块是单独存储的 summary/abstract 块。"""
        mothers = []
        mother_ids = set()

        for ck in chunks:
            mom = ck.get("mom") or ck.get("mom_with_weight") or ""
            if not mom:
                continue

            mom_id = xxhash.xxh64(mom.encode("utf-8")).hexdigest()
            ck["mom_id"] = mom_id

            if mom_id in mother_ids:
                continue

            mother_ids.add(mom_id)
            mom_ck = copy.deepcopy(ck)
            mom_ck["id"] = mom_id
            mom_ck["content_with_weight"] = mom
            mom_ck["available_int"] = 0

            # 仅保留必要字段
            allowed_fields = ["id", "content_with_weight", "doc_id", "docnm_kwd", "kb_id", "available_int", "position_int", "create_timestamp_flt", "page_num_int", "top_int"]
            for fld in list(mom_ck.keys()):
                if fld not in allowed_fields:
                    del mom_ck[fld]

            mothers.append(mom_ck)

        return mothers

    async def _insert_mother_chunks(
        self,
        task_id: str,
        task_tenant_id: str,
        task_dataset_id: str,
        mothers: List[Dict],
        doc_bulk_size: int,
    ) -> bool:
        """批量插入母块。"""
        for b in range(0, len(mothers), doc_bulk_size):
            await self._intercept_doc_store_insert(mothers[b : b + doc_bulk_size], search.index_name(task_tenant_id), task_dataset_id, refresh=False)

            if self._task_context.has_canceled_func(task_id):
                self._task_context.progress_cb(-1, msg="Task has been canceled.")
                return False

        return True

    async def _intercept_doc_store_delete(self, condition: dict, index_name: str, task_dataset_id: str) -> Any:
        if self._task_context.write_interceptor:
            return self._task_context.write_interceptor.intercept("docStoreConn.delete")
        else:
            return await thread_pool_exec(settings.docStoreConn.delete, condition, index_name, task_dataset_id)

    async def _intercept_doc_store_insert(self, chunks: list, index_name: str, task_dataset_id: str, refresh: str | bool = "wait_for") -> Any:
        if self._task_context.write_interceptor:
            if self._task_context.doc_id == GRAPH_RAPTOR_FAKE_DOC_ID:  # 猛禽 - 不确定
                return self._task_context.write_interceptor.intercept("docStoreConn.insert", [])
            return self._task_context.write_interceptor.intercept("docStoreConn.insert")
        else:
            # 同步 doc engine 客户端放入线程池，避免阻塞 asyncio；返回空列表表示该批次成功。
            return await thread_pool_exec(settings.docStoreConn.insert, chunks, index_name, task_dataset_id, refresh)

    async def _insert_main_chunks(
        self,
        task_id: str,
        task_tenant_id: str,
        task_dataset_id: str,
        chunks: List[Dict],
        doc_bulk_size: int,
    ) -> bool:
        """批量插入主块并进行取消处理。"""
        # 定期保留任务块 IDs，而不是每个批量请求一次。
        # 这使任务可恢复，同时避免一个 MySQL 事务
        # 每个小文档存储批次。
        checkpoint_batches = max(1, 256 // doc_bulk_size)
        last_checkpoint = 0
        for b in range(0, len(chunks), doc_bulk_size):
            # 每批同时写入 Chunk 文本、分词字段、页码/位置、doc_id/kb_id 和 Embedding 向量。
            doc_store_result = await self._intercept_doc_store_insert(chunks[b : b + doc_bulk_size], search.index_name(task_tenant_id), task_dataset_id, refresh=False)

            if self._task_context.has_canceled_func(task_id):
                # 回滚部分 RAPTOR 摘要插入
                await self._rollback_raptor_chunks(task_id, task_tenant_id, task_dataset_id, chunks, b, doc_bulk_size)
                self._task_context.progress_cb(-1, msg="Task has been canceled.")
                return False

            if b % 128 == 0:
                self._task_context.progress_cb(prog=0.8 + 0.1 * (b + 1) / len(chunks), msg="")

            # docStoreConn.insert 的约定与常见 API 相反：成功返回空列表，非空列表是失败详情。
            if doc_store_result:
                error_message = f"Insert chunk error: {doc_store_result}, please check log file and Elasticsearch/Infinity status!"
                self._task_context.progress_cb(-1, msg=error_message)
                raise Exception(error_message)

            batch_end = min(b + doc_bulk_size, len(chunks))
            is_last_batch = batch_end == len(chunks)
            # 定期把已经成功写入的 Chunk ID 保存到 MySQL Task.chunk_ids：Worker 中途崩溃时
            # 可识别已完成部分；若 checkpoint 写库失败，则删除本轮已插入 Chunk 保持一致性。
            if is_last_batch or batch_end - last_checkpoint >= checkpoint_batches * doc_bulk_size:
                chunk_ids = [chunk["id"] for chunk in chunks[:batch_end]]
                if not await self._update_task_chunk_ids(task_id, chunk_ids):
                    # 失败回滚
                    await self._rollback_insertion(task_tenant_id, task_dataset_id, chunk_ids)
                    self._task_context.progress_cb(-1, msg=f"Chunk updates failed since task {task_id} is unknown.")
                    return False
                last_checkpoint = batch_end

        # 批量写入期间 refresh=False，最后统一 refresh，避免每批刷新 ES segment 的开销；
        # refresh 完成后新 Chunk 才能稳定地被随后的检索请求看到。
        refresh_idx = getattr(settings.docStoreConn, "refresh_idx", None)
        if callable(refresh_idx):
            await thread_pool_exec(refresh_idx, search.index_name(task_tenant_id))

        return True

    async def _rollback_raptor_chunks(
        self,
        task_id: str,
        task_tenant_id: str,
        task_dataset_id: str,
        chunks: List[Dict],
        up_to_batch: int,
        doc_bulk_size: int,
    ):
        """取消后回滚部分RAPTOR 摘要插入。"""
        raptor_ids = [c["id"] for c in chunks[: up_to_batch + doc_bulk_size] if c.get("raptor_kwd") == "raptor"]

        if raptor_ids:
            try:
                await self._intercept_doc_store_delete({"id": raptor_ids}, search.index_name(task_tenant_id), task_dataset_id)
                logging.info(
                    "insert_chunks：取消后回滚%d部分RAPTOR块（任务= %s）",
                    len(raptor_ids),
                    task_id,
                )
            except Exception:
                logging.exception(
                    "insert_chunks：取消后无法回滚部分 RAPTOR 块（任务 = %s）",
                    task_id,
                )

    async def _update_task_chunk_ids(self, task_id: str, chunk_ids: List[str]) -> bool:
        """更新任务记录中的块 IDs。"""
        from peewee import DoesNotExist

        try:
            if self._task_context.write_interceptor:
                if self._task_context.doc_id == GRAPH_RAPTOR_FAKE_DOC_ID:
                    self._task_context.write_interceptor.intercept("TaskService.update_chunk_ids", True)
                else:
                    self._task_context.write_interceptor.intercept("TaskService.update_chunk_ids")
            else:
                # 保存当前 Task 已写入的 Chunk ID
                TaskService.update_chunk_ids(task_id, " ".join(chunk_ids))
            return True
        except DoesNotExist:
            logging.warning(f"do_handle_task update_chunk_ids failed since task {task_id} is unknown.")
            return False

    async def _rollback_insertion(
        self,
        task_tenant_id: str,
        task_dataset_id: str,
        chunk_ids: List[str],
    ):
        """通过删除块和图像来回滚插入。"""
        await self._intercept_doc_store_delete({"id": chunk_ids}, search.index_name(task_tenant_id), task_dataset_id)

        # 删除关联图像
        tasks = []
        for chunk_id in chunk_ids:
            tasks.append(asyncio.create_task(self._delete_image(task_dataset_id, chunk_id)))

        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"delete_image failed: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

    async def _delete_image(self, kb_id: str, chunk_id: str):
        """从存储中删除块的图像。"""
        try:
            async with self._task_context.minio_limiter:
                settings.STORAGE_IMPL.delete(kb_id, chunk_id)
        except Exception:
            logging.exception(f"Deleting image of chunk {chunk_id} got exception")
            raise
