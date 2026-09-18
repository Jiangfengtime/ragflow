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

"""Chunk 构建器模块。

提供解析器工厂和文档分块逻辑：
- 解析器模块注册和选择
- Document 通过解析器分块
- PDF轮廓提取"""

import logging
from timeit import default_timer as timer
from typing import Dict, List

from common.constants import ParserType
from common.exceptions import TaskCanceledException
from common.misc_utils import thread_pool_exec
from rag.svr.task_executor_refactor.task_context import TaskContext

from api.db.services.doc_metadata_service import DocMetadataService
from common.metadata_utils import update_metadata_to
from rag.utils.table_es_metadata import merge_table_parser_config_from_kb


def get_parser(parser_id: str):
    """通过ID获取解析器模块。

    参数：
        parser_id：解析器标识符。

    返回：
        给定解析器 ID 的解析器模块。"""
    from rag.app import laws, paper, presentation, manual, qa, table, book, resume, picture, naive, one, audio, email, tag

    # parser_id 决定“如何把原文件变成结构化 Chunk”，不是文件后缀的简单映射；
    # 上传阶段已经根据文件类型和知识库配置选好 parser_id，Worker 在这里取得对应模块。
    factory = {
        "general": naive,
        ParserType.NAIVE.value: naive,
        ParserType.PAPER.value: paper,
        ParserType.BOOK.value: book,
        ParserType.PRESENTATION.value: presentation,
        ParserType.MANUAL.value: manual,
        ParserType.LAWS.value: laws,
        ParserType.QA.value: qa,
        ParserType.TABLE.value: table,
        ParserType.RESUME.value: resume,
        ParserType.PICTURE.value: picture,
        ParserType.ONE.value: one,
        ParserType.AUDIO.value: audio,
        ParserType.EMAIL.value: email,
        ParserType.KG.value: naive,
        ParserType.TAG.value: tag,
    }
    return factory[parser_id.lower()]


async def run_chunking(
    chunker,
    binary: bytes,
    ctx: TaskContext,
    on_chunking_start=None,
) -> List[Dict]:
    """通过解析器运行文档分块。

    参数：
        chunker：要使用的解析器模块。
        二进制：文档的二进制内容。
        ctx: TaskContext 包含任务配置。

    返回：
        块字典列表。"""
    st = timer()
    try:
        # 表格解析允许知识库级字段角色配置覆盖文档配置；普通文档通常保持原 parser_config。
        # 合并表解析器配置
        parser_config = merge_table_parser_config_from_kb(ctx.raw_task)

        chunking_wait_started_at = timer()
        async with ctx.chunk_limiter:
            if on_chunking_start:
                on_chunking_start(timer() - chunking_wait_started_at)
            # parser.chunk() 同时完成格式解析/OCR或版面识别（取决于 parser）以及切分。
            # 返回值仍在内存中，通常包含 content_with_weight、content_ltks、页码和位置，
            # 此时还没有稳定 Chunk ID、Embedding 向量，也尚未写入 ES。
            cks = await thread_pool_exec(
                chunker.chunk,
                ctx.name,
                binary=binary,
                from_page=ctx.from_page,
                to_page=ctx.to_page,
                lang=ctx.language,
                callback=ctx.progress_cb,
                kb_id=ctx.kb_id,
                parser_config=parser_config,
                tenant_id=ctx.tenant_id,
            )
        logging.info("Chunking({}) {}/{} done".format(timer() - st, ctx.location, ctx.name))
        ctx.recording_context.record("parser_config_after_merge", parser_config)
        return cks
    except TaskCanceledException:
        raise
    except Exception as e:
        ctx.progress_cb(-1, msg="Internal server error while chunking: %s" % str(e).replace("'", ""))
        logging.exception("Chunking {}/{} got exception".format(ctx.location, ctx.name))
        raise


async def extract_outline(cks: List[Dict], ctx: TaskContext) -> None:
    """提取并保留 PDF 大纲（如果存在）。

    参数：
        cks：块字典列表。
        ctx: TaskContext 包含任务配置。"""
    outline_data = cks[0].get("__outline__") if cks else None
    ctx.recording_context.record("outline_data", outline_data)

    if cks and cks[0].get("__outline__"):
        outline = cks[0].pop("__outline__")
        try:
            if ctx.write_interceptor:
                ctx.write_interceptor.intercept("DocMetadataService.update_document_metadata")
            else:
                temp_doc = DocMetadataService.get_document_metadata(ctx.doc_id) or {}
                DocMetadataService.update_document_metadata(ctx.doc_id, update_metadata_to({"outline": outline}, temp_doc))

            logging.info("PDF大纲已持久化 条目数=%d 文档ID=%s", len(outline), ctx.doc_id)
        except Exception as e:
            logging.warning("无法保留文档 %s 的 PDF 大纲：%s", ctx.doc_id, e)
