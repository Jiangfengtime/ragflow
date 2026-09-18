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

"""Chunk 后处理器模块。

提供块的后处理功能：
- 关键词提取
- 问题生成
- 元数据生成
- 内容标记"""

import asyncio
import json
import logging
import random
import re
from datetime import datetime
from timeit import default_timer as timer

from api.db.joint_services.tenant_model_service import resolve_model_config
from api.db.services.doc_metadata_service import DocMetadataService
from api.db.services.llm_service import LLMBundle
from common import settings
from common.constants import TAG_FLD, LLMType
from common.metadata_utils import turn2jsonschema, update_metadata_to
from rag.graphrag.utils import get_llm_cache, get_tags_from_cache, set_llm_cache, set_tags_to_cache
from rag.nlp import rag_tokenizer
from rag.prompts.generator import content_tagging, gen_metadata, keyword_extraction, question_proposal
from rag.svr.task_executor_refactor.task_context import TaskContext


# Elasticsearch 关键字字段拒绝 UTF-8 编码超过的术语
# 32766 字节。分割超大术语，因此摄取永远不会失败，因为
# 格式错误的 LLM 响应产生了一个巨大的“关键字”。
_ES_KEYWORD_MAX_TERM_BYTES = 32766


def _sanitize_keyword_term(term: str) -> list[str]:
    """返回适合 Elasticsearch 关键字字段的关键字片段。

    如果“`term`”足够小，则按原样返回。否则就是
    在字符边界处截断，因此 UTF-8 编码永远不会超过
    ES 关键字限制。这可以避免损坏多字节字符
    切片原始字节。"""
    term = term.strip()
    if not term:
        return []
    term_byte_length = len(term.encode("utf-8"))
    if term_byte_length <= _ES_KEYWORD_MAX_TERM_BYTES:
        return [term]

    logging.warning(
        "清理超大关键字术语（%d 字节，限制 %d）",
        term_byte_length,
        _ES_KEYWORD_MAX_TERM_BYTES,
    )
    length = 0
    end = 0
    for index, character in enumerate(term):
        character_bytes = len(character.encode("utf-8"))
        if length + character_bytes > _ES_KEYWORD_MAX_TERM_BYTES:
            end = index
            break
        length += character_bytes
    else:
        end = len(term)
    truncated = term[:end].rstrip()
    if not truncated:
        return []
    return [truncated]


async def extract_keywords(docs: list[dict], ctx: TaskContext) -> None:
    """提取块的关键字。

    参数：
        docs：要处理的块字典列表。
        ctx: TaskContext 包含任务配置。"""
    chat_limiter = ctx.chat_limiter

    st = timer()
    ctx.progress_cb(msg="Start to generate keywords for every chunk ...")
    chat_model_config = resolve_model_config(ctx.tenant_id, LLMType.CHAT, ctx.llm_id)
    with LLMBundle(ctx.tenant_id, chat_model_config, lang=ctx.language) as chat_model:

        async def doc_keyword_extraction(chat_mdl, d, topn):
            cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], "keywords", {"topn": topn})
            if not cached:
                if ctx.has_canceled_func(ctx.id):
                    ctx.progress_cb(-1, msg="Task has been canceled.")
                    return
                async with chat_limiter:
                    cached = await keyword_extraction(chat_mdl, d["content_with_weight"], topn)
                set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, "keywords", {"topn": topn})
            if cached:
                d["important_kwd"] = [kw for k in re.split(r"[,，;；、\r\n]+", cached) for kw in _sanitize_keyword_term(k)]
                d["important_tks"] = rag_tokenizer.tokenize(" ".join(d["important_kwd"]))
            return

        tasks = []
        for doc in docs:
            tasks.append(asyncio.create_task(doc_keyword_extraction(chat_model, doc, ctx.parser_config["auto_keywords"])))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"Error in doc_keyword_extraction: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        ctx.progress_cb(msg=f"Keywords generation {len(docs)} chunks completed in {timer() - st:.2f}s")


async def generate_questions(docs: list[dict], ctx: TaskContext) -> None:
    """生成块的问题。

    参数：
        docs：要处理的块字典列表。
        ctx: TaskContext 包含任务配置。"""
    chat_limiter = ctx.chat_limiter

    st = timer()
    ctx.progress_cb(msg="Start to generate questions for every chunk ...")
    chat_model_config = resolve_model_config(ctx.tenant_id, LLMType.CHAT, ctx.llm_id)
    with LLMBundle(ctx.tenant_id, chat_model_config, lang=ctx.language) as chat_model:

        async def doc_question_proposal(chat_mdl, d, topn):
            cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], "question", {"topn": topn})
            if not cached:
                if ctx.has_canceled_func(ctx.id):
                    ctx.progress_cb(-1, msg="Task has been canceled.")
                    return
                async with chat_limiter:
                    cached = await question_proposal(chat_mdl, d["content_with_weight"], topn)
                set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, "question", {"topn": topn})
            if cached:
                d["question_kwd"] = cached.split("\n")
                d["question_tks"] = rag_tokenizer.tokenize("\n".join(d["question_kwd"]))

        tasks = []
        for doc in docs:
            tasks.append(asyncio.create_task(doc_question_proposal(chat_model, doc, ctx.parser_config["auto_questions"])))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error("doc_question_proposal 中的错误", exc_info=e)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        ctx.progress_cb(msg=f"Question generation {len(docs)} chunks completed in {timer() - st:.2f}s")


def build_metadata_config(parser_config: dict) -> list:
    """从 parser_config 构建元数据配置。

    提取并标准化``metadata`` and ``built_in_metadata`` from the
    parser configuration into a single list or dict that is passed to the LLM
    cache and generation functions.

    This should be called once per `ZXQKEEP00 090005ZXQ` invocation — the result
    is identical for every chunk within the same document parse session so
    extracting it avoids rebuilding inside the per-chunk async task.

    Args:
        parser_config: Configuration dict from the parser, expected to contain
            ``metadata`` (dict or list) and optionally ``built_in_metadata``
            （元数据项字典列表）。

    返回：
        表示合并的元数据配置的列表或字典。"""
    metadata_conf = parser_config.get("metadata", [])
    built_in_metadata = list(parser_config.get("built_in_metadata") or [])
    if isinstance(metadata_conf, dict):
        if not isinstance(metadata_conf.get("properties"), dict):
            metadata_conf = {"type": "object", "properties": {}}
        if built_in_metadata:
            metadata_conf = {
                **metadata_conf,
                "properties": {
                    **metadata_conf.get("properties", {}),
                    **turn2jsonschema(built_in_metadata).get("properties", {}),
                },
            }
    elif isinstance(metadata_conf, list):
        metadata_conf = metadata_conf + built_in_metadata
    else:
        metadata_conf = built_in_metadata
    return metadata_conf


async def generate_metadata(docs: list[dict], ctx: TaskContext) -> None:
    """生成块的元数据。

    参数：
        docs：要处理的块字典列表。
        ctx: TaskContext 包含任务配置。"""
    chat_limiter = ctx.chat_limiter

    st = timer()
    ctx.progress_cb(msg="Start to generate meta-data for every chunk ...")
    chat_model_config = resolve_model_config(ctx.tenant_id, LLMType.CHAT, ctx.llm_id)
    with LLMBundle(ctx.tenant_id, chat_model_config, lang=ctx.language) as chat_model:
        metadata_conf = build_metadata_config(ctx.parser_config)

        async def gen_metadata_task(chat_mdl, d):
            cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], "metadata", metadata_conf)
            if not cached:
                if ctx.has_canceled_func(ctx.id):
                    ctx.progress_cb(-1, msg="Task has been canceled.")
                    return
                async with chat_limiter:
                    cached = await gen_metadata(chat_mdl, turn2jsonschema(metadata_conf), d["content_with_weight"])
                set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, "metadata", metadata_conf)
            if cached:
                d["metadata_obj"] = cached

        tasks = []
        for doc in docs:
            tasks.append(asyncio.create_task(gen_metadata_task(chat_model, doc)))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error("gen_metadata 中的错误", exc_info=e)
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise

        metadata = {}
        for doc in docs:
            if "metadata_obj" in doc:
                metadata = update_metadata_to(metadata, doc["metadata_obj"])
                del doc["metadata_obj"]
        if metadata:
            existing_meta = DocMetadataService.get_document_metadata(ctx.doc_id)
            existing_meta = existing_meta if isinstance(existing_meta, dict) else {}
            metadata = update_metadata_to(metadata, existing_meta)
            if ctx.write_interceptor:
                ctx.write_interceptor.intercept("DocMetadataService.update_document_metadata")
            else:
                DocMetadataService.update_document_metadata(ctx.doc_id, metadata)
        ctx.progress_cb(msg=f"Metadata generation {len(docs)} chunks completed in {timer() - st:.2f}s")


def apply_built_in_metadata(ctx: TaskContext) -> None:
    built_in_meta_config = ctx.parser_config.get("built_in_metadata", [])
    if not built_in_meta_config:
        return

    built_in_meta = {}
    for item in built_in_meta_config:
        key = item.get("key", "")
        if key == "update_time":
            built_in_meta["update_time"] = str(datetime.now()).replace("T", " ")[:19]
        elif key == "file_name":
            built_in_meta["file_name"] = ctx.name
    if built_in_meta:
        existing_meta = DocMetadataService.get_document_metadata(ctx.doc_id)
        existing_meta = existing_meta if isinstance(existing_meta, dict) else {}
        existing_meta = update_metadata_to(existing_meta, built_in_meta)
        if ctx.write_interceptor:
            ctx.write_interceptor.intercept("DocMetadataService.update_document_metadata")
        else:
            DocMetadataService.update_document_metadata(ctx.doc_id, existing_meta)


async def apply_tags(docs: list[dict], ctx: TaskContext) -> None:
    """将标签应用于块。

    参数：
        docs：要处理的块字典列表。
        ctx: TaskContext 包含任务配置。"""
    chat_limiter = ctx.chat_limiter

    ctx.progress_cb(msg="Start to tag for every chunk ...")
    kb_ids = ctx.kb_parser_config["tag_kb_ids"]
    tenant_id = ctx.tenant_id
    topn_tags = ctx.kb_parser_config.get("topn_tags", 3)
    S = 1000
    st = timer()
    examples = []
    all_tags = get_tags_from_cache(kb_ids)
    if not all_tags:
        all_tags = settings.retriever.all_tags_in_portion(tenant_id, kb_ids, S)
        set_tags_to_cache(kb_ids, all_tags)
    else:
        all_tags = json.loads(all_tags)
    chat_model_config = resolve_model_config(tenant_id, LLMType.CHAT, ctx.llm_id)
    with LLMBundle(ctx.tenant_id, chat_model_config, lang=ctx.language) as chat_model:
        docs_to_tag = []
        for doc in docs:
            if ctx.has_canceled_func(ctx.id):
                ctx.progress_cb(-1, msg="Task has been canceled.")
                return
            if settings.retriever.tag_content(tenant_id, kb_ids, doc, all_tags, topn_tags=topn_tags, S=S) and len(doc.get(TAG_FLD, [])) > 0:
                examples.append({"content": doc["content_with_weight"], TAG_FLD: doc[TAG_FLD]})
            else:
                docs_to_tag.append(doc)

        async def doc_content_tagging(chat_mdl, d, topn_tags):
            cached = get_llm_cache(chat_mdl.llm_name, d["content_with_weight"], all_tags, {"topn": topn_tags})
            if not cached:
                if ctx.has_canceled_func(ctx.id):
                    ctx.progress_cb(-1, msg="Task has been canceled.")
                    return
                picked_examples = random.choices(examples, k=2) if len(examples) > 2 else examples
                if not picked_examples:
                    picked_examples.append({"content": "This is an example", TAG_FLD: {"example": 1}})
                async with chat_limiter:
                    cached = await content_tagging(
                        chat_mdl,
                        d["content_with_weight"],
                        all_tags,
                        picked_examples,
                        topn_tags,
                    )
                if cached:
                    cached = json.dumps(cached)
            if cached:
                set_llm_cache(chat_mdl.llm_name, d["content_with_weight"], cached, all_tags, {"topn": topn_tags})
                d[TAG_FLD] = json.loads(cached)

        tasks = []
        for doc in docs_to_tag:
            tasks.append(asyncio.create_task(doc_content_tagging(chat_model, doc, topn_tags)))
        try:
            await asyncio.gather(*tasks, return_exceptions=False)
        except Exception as e:
            logging.error(f"Error tagging docs: {e}")
            for t in tasks:
                t.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)
            raise
        ctx.progress_cb(msg=f"Tagging {len(docs)} chunks completed in {timer() - st:.2f}s")


def count_with_key(docs: list[dict], key: str) -> int:
    """统计具有特定键的文档。

    参数：
        docs：块字典列表。
        key：要检查的密钥。

    返回：
        具有密钥的文档计数。"""
    return sum(1 for d in docs if d.get(key))


# =====================================================================
# Document 分块后管道
# ---------------------------------------------------------------------
# 从“`task_handler`”中提取，以保持处理程序类较小。
# 公共入口点为:func:`run_document_post_chunking_if_last`；
# 下面的所有内容都是从那里调用（传递）的：
#   run_document_post_chunking_if_last
#     ├─ run_document_structure_compile
#     │    ├─ run_tree_templates
#     │    │    ├─ load_chunks_with_vec
#     │    │    ├─ rechunk_doc_by_tree
#     │    │    └─ raptor_tree_to_graph
# │ └─（通过每个模板的聊天模型进行流式编译）
# └─ handler._run_raptor ← 留在处理程序上
#
# 所有条目都将 ``handler`` (``TaskHandler``) 作为第一个参数，因此
# 他们可以到达处理程序的``_task_context``, ``_run_raptor``，并且
# ``_load_chunks_for_doc`` 没有循环导入。
# =====================================================================

from collections.abc import Callable

import numpy as np

from api.db.services.compilation_template_group_service import (
    CompilationTemplateGroupService,
)
from api.db.services.document_service import DocumentService
from api.db.services.task_service import (
    abort_doc_chunking_counter,
    clear_doc_chunking_counter,
    credit_doc_chunking_task,
    is_doc_chunking_aborted,
)
from common.misc_utils import thread_pool_exec
from common.token_utils import num_tokens_from_string

# ----- 可调参数 ------------------------------------------------------
# 结构编译批处理/合并刷新/链校正可调参数
# 和非树编译核心移至
# ``rag.advanced_rag.knowlege_compile.runner`` so the ``rag.flow`` 编译器
# 组件可以共享它们。此处重新导出以实现向后兼容性。
from rag.advanced_rag.knowlege_compile.runner import (
    DOC_STRUCTURE_COMPILE_BATCH_CHUNKS,
    DOC_STRUCTURE_MERGE_MAX_DOCS,  # noqa: F401
    STRUCTURE_CHAIN_CORRECTION_TIMEOUT_S,  # noqa: F401
    load_active_templates,
    run_structure_compile_over_batches,
)
from rag.nlp import search

# ----- parser_config 助手 -----------------------------------------


def _parser_config_compilation_template_group_ids(parser_config) -> list[str]:
    def _normalize(raw) -> list[str]:
        if isinstance(raw, str):
            raw = [raw]
        if not isinstance(raw, list):
            return []
        ids: list[str] = []
        seen: set[str] = set()
        for gid in raw:
            if not isinstance(gid, str):
                continue
            gid = gid.strip()
            if gid and gid not in seen:
                seen.add(gid)
                ids.append(gid)
        return ids

    if not isinstance(parser_config, dict):
        return []
    if "compilation_template_group_id" in parser_config:
        return _normalize(parser_config.get("compilation_template_group_id"))
    ext = parser_config.get("ext")
    if isinstance(ext, dict):
        return _normalize(ext.get("compilation_template_group_id"))
    return []


def _parser_config_compilation_template_ids(parser_config, tenant_id: str) -> list[str]:
    template_ids: list[str] = []
    seen: set[str] = set()
    for group_id in _parser_config_compilation_template_group_ids(parser_config):
        for template_id in CompilationTemplateGroupService.resolve_template_ids(
            group_id,
            tenant_id,
        ):
            if template_id in seen:
                continue
            seen.add(template_id)
            template_ids.append(template_id)
    return template_ids


def _resolve_ingestion_chat_llm_id(ctx) -> str:
    """选择知识编译模板的摄取模型 ID。"""
    doc_cfg = getattr(ctx, "parser_config", None) or {}
    if isinstance(doc_cfg, dict):
        did = doc_cfg.get("llm_id")
        if isinstance(did, str) and did.strip():
            return did.strip()
    return ctx.llm_id


# ----- 进度助手------------------------------------------------------------


def cap_done_progress(progress_cb: Callable) -> Callable:
    """包装一个进度回调，以便任何“`prog >= 1`` gets clamped to
    ``0.99`` — the final ``1.0`”保留给拥有的调用者
    任务的终止状态。"""

    def capped_progress(*args, **kwargs):
        args = list(args)
        if args:
            prog = args[0]
            if isinstance(prog, (int, float)) and not isinstance(prog, bool) and prog >= 1:
                args[0] = 0.99
        if "prog" in kwargs:
            prog = kwargs["prog"]
            if isinstance(prog, (int, float)) and not isinstance(prog, bool) and prog >= 1:
                kwargs["prog"] = 0.99
        return progress_cb(*args, **kwargs)

    return capped_progress


# -----树助手------------------------------------------------


def raptor_tree_to_graph(tree: dict) -> dict:
    """项目 RAPTOR 树字典（来自``Raptor(is_tree=True)``) onto
    the ``{entities, relations}`` shape the document-structure graph
    endpoint already serves for ``page_index``类行。"""
    entities: list[dict] = []
    relations: list[dict] = []

    def _collapse_unary(node: dict) -> dict:
        """折叠仅包裹一个子节点的树节点。"""
        collapsed = dict(node)
        collapsed["children"] = [_collapse_unary(child) for child in node.get("children") or [] if isinstance(child, dict)]

        while len(collapsed["children"]) == 1:
            child = collapsed["children"][0]
            parent_title = collapsed.get("title") or ""
            child_title = child.get("title") or ""
            parent_description = collapsed.get("description") or parent_title
            child_description = child.get("description") or child_title

            descriptions = [str(parent_description)]
            if child_title and child_title != parent_title and child_title not in child_description:
                descriptions.append(str(child_title))
            if child_description and child_description not in descriptions:
                descriptions.append(str(child_description))

            source_chunk_ids = []
            for source in (collapsed.get("source_chunk_ids") or [], child.get("source_chunk_ids") or []):
                for chunk_id in source:
                    if isinstance(chunk_id, str) and chunk_id and chunk_id not in source_chunk_ids:
                        source_chunk_ids.append(chunk_id)

            collapsed["description"] = "\n\n".join(descriptions)
            if source_chunk_ids:
                collapsed["source_chunk_ids"] = source_chunk_ids
            collapsed["children"] = child.get("children") or []

        return collapsed

    tree = _collapse_unary(tree) if isinstance(tree, dict) else tree

    def _walk(node: dict, parent_id: str | None) -> None:
        if not isinstance(node, dict):
            return
        title = node.get("title") or ""
        node_id = title
        ent: dict = {
            "name": node_id,
            "type": "tree_node",
            "description": node.get("description", title),
            "mention_count": 1,
        }
        src_ids = node.get("source_chunk_ids")
        if isinstance(src_ids, list) and src_ids:
            ent["source_chunk_ids"] = [s for s in src_ids if isinstance(s, str) and s]
        entities.append(ent)
        # 摘要及其子级偶尔会收到相同的 LLM 生成的内容
        # 标题。它们仍然是有效的树节点，但一定不能成为自循环
        # 当树投影到图关系时。
        if parent_id is not None and parent_id != node_id:
            relations.append({"from": parent_id, "to": node_id, "type": "child"})
        for child in node.get("children") or []:
            _walk(child, node_id)

    _walk(tree, None)
    return {"entities": entities, "relations": relations}


async def rewrite_duplicate_tree_names(tree: dict, chat_mdl) -> None:
    """仅重写描述不同的重复树标题。"""
    from rag.advanced_rag.knowlege_compile._common import knowledge_compile_gen_conf
    from rag.prompts.generator import gen_json

    groups: dict[str, list[tuple[dict, str, str]]] = {}

    def _walk(node: dict, path: tuple[int, ...]) -> None:
        if not isinstance(node, dict):
            return
        title = str(node.get("title") or "").strip()
        if title:
            description = str(node.get("description") or title).strip()
            node_key = ".".join(str(index) for index in path)
            groups.setdefault(title, []).append((node, node_key, description))
        for index, child in enumerate(node.get("children") or []):
            _walk(child, (*path, index))

    _walk(tree, (0,))
    for title, candidates in groups.items():
        descriptions = {description for _, _, description in candidates}
        if len(candidates) < 2 or len(descriptions) < 2:
            continue

        items = [{"id": node_key, "description": description} for _, node_key, description in candidates]
        prompt = (
            "The following tree nodes currently have the same title but describe different content. "
            "Give each node a concise, distinct human-readable title. Preserve the original language, "
            "do not add numbering unless necessary, and return only a JSON array of objects with the "
            "same ids and a name field.\n\n"
            f"Current title: {title}\n"
            f"Nodes: {json.dumps(items, ensure_ascii=False)}"
        )
        try:
            result = await gen_json(
                "You rename duplicate tree node titles for display.",
                prompt,
                chat_mdl,
                gen_conf=knowledge_compile_gen_conf(chat_mdl, {"temperature": 0.0}),
            )
        except Exception:
            logging.exception("树模板：标题 = %s 的重复标题重写失败", title)
            continue

        rewrites = {}
        if isinstance(result, list):
            rewrites = {str(item.get("id")): str(item.get("name")).strip() for item in result if isinstance(item, dict) and item.get("id") and str(item.get("name") or "").strip()}
        for node, node_key, _ in candidates:
            new_title = rewrites.get(node_key)
            if new_title:
                node["title"] = new_title

    # LLM 被要求提供不同的名称，但强制执行该合同
    # 在图形使用标题作为关系端点之前确定。
    used_names: dict[str, int] = {}

    def _ensure_unique(node: dict) -> None:
        if not isinstance(node, dict):
            return
        title = str(node.get("title") or "").strip()
        if title:
            occurrence = used_names.get(title, 0) + 1
            used_names[title] = occurrence
            if occurrence > 1:
                node["title"] = f"{title} ({occurrence})"
        for child in node.get("children") or []:
            _ensure_unique(child)

    _ensure_unique(tree)


async def load_chunks_with_vec(
    tenant_id: str,
    kb_id: str,
    doc_id: str,
    vctr_nm: str,
) -> list[tuple[str, "np.ndarray", str]]:
    """翻阅此文档的块，提取内容 + 矢量 +
    chunk_id，形状为“`RaptorService.build_doc_tree`` expects.
    Mirrors the streaming ``_load_chunks_for_doc`”装载机，但带有
    预先选择的向量场。"""
    from common.doc_store.doc_store_base import OrderByExpr

    index_nm = search.index_name(tenant_id)
    if not settings.docStoreConn.index_exist(index_nm, kb_id):
        return []
    select_fields = ["id", "doc_id", "content_with_weight", "compile_kwd", vctr_nm]
    order_by = OrderByExpr()
    order_by.asc("page_num_int")
    order_by.asc("top_int")

    out: list[tuple[str, np.ndarray, str]] = []
    offset = 0
    PAGE = 500
    while True:
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields,
                [],
                {
                    "doc_id": [doc_id],
                    "available_int": 1,
                    "must_not": {"exists": "compile_kwd"},
                },
                [],
                order_by,
                offset,
                PAGE,
                index_nm,
                [kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields)
        except Exception:
            logging.exception(
                "树模板：无法加载 doc=%s 的块",
                doc_id,
            )
            break
        if not field_map:
            break
        for row_id, row in field_map.items():
            if row.get("compile_kwd"):
                continue
            text = row.get("content_with_weight") or ""
            vec = row.get(vctr_nm)
            if not text or vec is None:
                continue
            try:
                arr = np.asarray(vec, dtype=np.float32)
            except Exception:
                continue
            if arr.size == 0:
                continue
            out.append((text, arr, str(row_id)))
        if len(field_map) < PAGE:
            break
        offset += PAGE
    return out


async def rechunk_doc_by_tree(
    handler,
    tree: dict,
    template_id: str,
    embedding_model,
) -> None:
    """将每个叶簇的源块合并为一个
    替换块并重写树的叶簇
    ``source_chunk_ids`` in-place. Original chunks are soft-deleted
    via ``available_int=0`` and stamped with ``superseded_by_chunk_id``。"""
    from datetime import datetime

    from common.misc_utils import get_uuid

    ctx = handler._task_context

    cluster_id_map: dict[int, tuple[dict, list[str]]] = {}

    def _is_terminal(node: object) -> bool:
        return isinstance(node, dict) and not (node.get("children") or [])

    def _walk(node: object) -> None:
        if not isinstance(node, dict):
            return
        children = node.get("children") or []
        if children and all(_is_terminal(c) for c in children):
            src_ids: list[str] = []
            seen: set[str] = set()
            for c in children:
                for cid in c.get("source_chunk_ids") or []:
                    if isinstance(cid, str) and cid and cid not in seen:
                        seen.add(cid)
                        src_ids.append(cid)
            for cid in node.get("source_chunk_ids") or []:
                if isinstance(cid, str) and cid and cid not in seen:
                    seen.add(cid)
                    src_ids.append(cid)
            if src_ids:
                cluster_id_map[id(node)] = (node, src_ids)
        else:
            for c in children:
                _walk(c)

    _walk(tree)
    if not cluster_id_map:
        return

    all_source_ids = sorted({sid for _, ids in cluster_id_map.values() for sid in ids})

    from common.doc_store.doc_store_base import OrderByExpr

    index_nm = search.index_name(ctx.tenant_id)
    if not settings.docStoreConn.index_exist(index_nm, ctx.kb_id):
        return

    vctr_nm = "q_%d_vec" % len(embedding_model.encode(["x"])[0][0])
    select_fields = [
        "id",
        "doc_id",
        "kb_id",
        "content_with_weight",
        "page_num_int",
        "top_int",
        "position_int",
        "docnm_kwd",
        "title_tks",
        "title_sm_tks",
        "available_int",
    ]
    try:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            select_fields,
            [],
            {"id": all_source_ids, "available_int": 1},
            [],
            OrderByExpr(),
            0,
            len(all_source_ids) + 16,
            index_nm,
            [ctx.kb_id],
        )
        field_map = settings.docStoreConn.get_fields(res, select_fields)
    except Exception:
        logging.exception(
            "重新分块：无法加载 doc=%s 模板=%s 的源块",
            ctx.doc_id,
            template_id,
        )
        return
    if not field_map:
        return

    chunks_by_id: dict[str, dict] = {str(rid): {**row, "id": str(rid)} for rid, row in field_map.items()}

    merged_rows: list[dict] = []
    cluster_new_id: dict[int, str] = {}

    for node_id_int, (node, src_ids) in cluster_id_map.items():
        cluster_chunks = [chunks_by_id[c] for c in src_ids if c in chunks_by_id]
        if not cluster_chunks:
            continue

        def _sort_key(c: dict) -> tuple:
            pages = c.get("page_num_int") or [0]
            tops = c.get("top_int") or [0]
            return (
                min(pages) if pages else 0,
                min(tops) if tops else 0,
                c.get("id") or "",
            )

        cluster_chunks.sort(key=_sort_key)

        merged_content = "\n\n".join((c.get("content_with_weight") or "") for c in cluster_chunks).strip()
        if not merged_content:
            continue
        page_union = sorted({p for c in cluster_chunks for p in (c.get("page_num_int") or [])})
        top_union = sorted({t for c in cluster_chunks for t in (c.get("top_int") or [])})

        base = dict(cluster_chunks[0])
        new_id = get_uuid()
        cluster_new_id[node_id_int] = new_id

        base.update(
            {
                "id": new_id,
                "content_with_weight": merged_content,
                "content_ltks": rag_tokenizer.tokenize(merged_content),
                "page_num_int": page_union,
                "top_int": top_union,
                "available_int": 1,
                "rechunk_kwd": "tree",
                "rechunked_from_template_id": template_id,
                "rechunked_from_chunk_ids": [c.get("id") for c in cluster_chunks if c.get("id")],
                "token_num": num_tokens_from_string(merged_content),
                "create_time": str(datetime.now()).replace("T", " ")[:19],
                "create_timestamp_flt": datetime.now().timestamp(),
            }
        )
        base["content_sm_ltks"] = rag_tokenizer.fine_grained_tokenize(base["content_ltks"])
        merged_rows.append(base)

    if not merged_rows:
        return

    contents = [r["content_with_weight"] for r in merged_rows]
    try:
        vectors, _ = embedding_model.encode(contents)
    except Exception:
        logging.exception(
            "重新分块：doc=%s 模板=%s 嵌入失败",
            ctx.doc_id,
            template_id,
        )
        return
    for row, vec in zip(merged_rows, vectors):
        try:
            row[vctr_nm] = np.asarray(vec, dtype=np.float32).tolist()
        except Exception:
            logging.exception(
                "重新分块：矢量转换失败；跳行 %s",
                row.get("id"),
            )
            row[vctr_nm] = None
    merged_rows = [r for r in merged_rows if r.get(vctr_nm) is not None]
    if not merged_rows:
        return

    try:
        await thread_pool_exec(
            settings.docStoreConn.insert,
            merged_rows,
            index_nm,
            ctx.kb_id,
        )
    except Exception:
        logging.exception(
            "重新分块： doc=%s 模板=%s 插入失败",
            ctx.doc_id,
            template_id,
        )
        return

    for node_id_int, new_chunk_id in cluster_new_id.items():
        node, _ = cluster_id_map[node_id_int]
        node["source_chunk_ids"] = [new_chunk_id]
        for child in node.get("children") or []:
            if isinstance(child, dict):
                child["source_chunk_ids"] = [new_chunk_id]

    for node_id_int, new_chunk_id in cluster_new_id.items():
        _, src_ids = cluster_id_map[node_id_int]
        for cid in src_ids:
            try:
                await thread_pool_exec(
                    settings.docStoreConn.update,
                    {"id": cid},
                    {
                        "available_int": 0,
                        "superseded_by_chunk_id": new_chunk_id,
                    },
                    index_nm,
                    ctx.kb_id,
                )
            except Exception:
                logging.exception(
                    "重新分块：块 = %s 的软删除失败（合并 = %s）",
                    cid,
                    new_chunk_id,
                )


async def run_tree_templates(
    handler,
    templates: list[tuple[str, dict]],
    chat_mdl_by_tid: dict[str, "LLMBundle"],
    embedding_model,
    doc_name: str,
) -> None:
    """运行``tree``-kind compilation templates for the current
    doc. Each pair runs RAPTOR with ``is_tree=True`` via
    ``RaptorService.build_doc_tree`` and persists a single graph row
    via ``_struct_upsert_graph_json``。"""
    from rag.advanced_rag.knowlege_compile.structure import _struct_upsert_graph_json
    from rag.svr.task_executor_refactor.raptor_service import RaptorService

    ctx = handler._task_context
    progress_cb = ctx.progress_cb

    try:
        doc_id = ctx.doc_id
    except Exception:
        doc_id = getattr(ctx, "_task", {}).get("doc_id") if hasattr(ctx, "_task") else None
    if not doc_id:
        logging.warning("树模板：任务上下文中没有 doc_id；跳绳")
        return

    vctr_nm = "q_%d_vec" % len(embedding_model.encode(["x"])[0][0])
    chunks = await load_chunks_with_vec(
        ctx.tenant_id,
        ctx.kb_id,
        doc_id,
        vctr_nm,
    )
    if not chunks:
        progress_cb(msg=f"tree-template: doc {doc_id} has no chunks; skipping")
        return

    raptor_service = RaptorService(ctx)

    for idx, (template_id, parser_cfg) in enumerate(templates):
        raptor_cfg = (parser_cfg or {}).get("raptor") or {}
        raptor_config = {
            "prompt": raptor_cfg.get("prompt") or "Please write a concise summary of the following texts:\n{cluster_content}",
            "max_token": int(raptor_cfg.get("max_token") or 512),
            "threshold": float(raptor_cfg.get("threshold") or 0.1),
            "random_seed": int(raptor_cfg.get("random_seed") or 0),
            "max_cluster": int(raptor_cfg.get("max_cluster") or 64),
            "ext": raptor_cfg.get("ext") or {},
        }
        progress_cb(
            msg=f"tree-template ({idx + 1}/{len(templates)}): building tree for doc={doc_id}",
        )
        try:
            tree = await raptor_service.build_doc_tree(
                chunks=chunks,
                raptor_config=raptor_config,
                chat_mdl=chat_mdl_by_tid[template_id],
                embd_mdl=embedding_model,
                max_errors=3,
            )
        except Exception:
            logging.exception(
                "树模板 %s：文档 %s 的 RAPTOR 构建失败",
                template_id,
                doc_id,
            )
            continue
        if tree is None:
            logging.info(
                "树模板 %s：没有为文档 %s 生成树",
                template_id,
                doc_id,
            )
            continue

        if bool((raptor_cfg or {}).get("rechunk")):
            try:
                await rechunk_doc_by_tree(
                    handler=handler,
                    tree=tree,
                    template_id=template_id,
                    embedding_model=embedding_model,
                )
            except Exception:
                logging.exception(
                    "树模板 %s：文档 %s 的重新分块失败；具有原始块 ID 的持久树",
                    template_id,
                    doc_id,
                )

        await rewrite_duplicate_tree_names(tree, chat_mdl_by_tid[template_id])
        graph = raptor_tree_to_graph(tree)
        try:
            await _struct_upsert_graph_json(
                graph,
                ctx.tenant_id,
                ctx.kb_id,
                doc_id,
                doc_name,
                compile_kwd="tree",
                compilation_template_id=template_id,
            )
        except Exception:
            logging.exception(
                "树模板 %s：文档 %s 的图形更新插入失败",
                template_id,
                doc_id,
            )
            continue

        # 在图形节点之后保留每个文档 nav_doc，因此解析
        # 文件生成 nav_doc，其中 FULL 实体描述为
        # graph_content -- 无需单独运行 GENERATE NAVIGATION
        # 并且无需更改导航聚类输入。 `title`继续使用
        # 树[“标题”]（upsert_dataset_nav_doc用于从`tree`派生
        # 此更改之前为
        # ），因此解析簇标题逻辑未更改。
        # 仅当图形实际包含实体时才执行此操作，否则跳过
        # （当RAPTOR什么也没产生时，避免空graph_content）。
        try:
            if graph.get("entities"):
                from rag.advanced_rag.knowlege_compile.dataset_nav import (
                    build_nav_graph_text,
                    upsert_dataset_nav_doc,
                )

                _, nav_graph_text = build_nav_graph_text(graph)
                await upsert_dataset_nav_doc(
                    ctx.tenant_id,
                    ctx.kb_id,
                    doc_id,
                    {"title": tree.get("title"), "graph_text": nav_graph_text},
                    embd_mdl=embedding_model,
                    chat_mdl=chat_mdl_by_tid[template_id],
                )
        except Exception:
            logging.exception(
                "树模板 %s：文档 %s 的 dataset_nav 更新插入失败",
                template_id,
                doc_id,
            )

        progress_cb(
            msg=f"tree-template ({idx + 1}/{len(templates)}): persisted {len(graph['entities'])} node(s), {len(graph['relations'])} edge(s) for doc {doc_id}",
        )


async def run_document_structure_compile(handler, embedding_model: LLMBundle) -> None:
    """对非工件运行文档范围的知识编译
    模板。流式传输文档的块（通过
    ``handler._load_chunks_for_doc``) and fans each batch out to every
    configured non-artifact template, flushing accumulators through
    ``merge_compiled_structures`` at :data:`DOC_STRUCTURE_MERGE_MAX_DOCSZXQKEEP00 650005ZXQ`synthesis.enabled``,
    runs ``wiki_plan_from_reduction`` + ``wiki_refine_from_plan``到
    生成综合输出（wiki 页面、精华段落等）。
    Compile_kwd 和 REFINE 提示符是从模板配置中读取的。"""
    from api.apps.restful_apis.chunk_api import _compilation_template_kind

    ctx = handler._task_context
    found, document = DocumentService.get_by_id(ctx.doc_id)
    doc_name = document.name if found and document else ""
    template_ids = _parser_config_compilation_template_ids(ctx.parser_config, ctx.tenant_id)
    if not template_ids:
        return

    active_templates = load_active_templates(template_ids, ctx.tenant_id)
    if not active_templates:
        return

    chat_llm_id = _resolve_ingestion_chat_llm_id(ctx)
    try:
        cfg = resolve_model_config(ctx.tenant_id, LLMType.CHAT, chat_llm_id)
        chat_mdl = LLMBundle(ctx.tenant_id, cfg, lang=ctx.language)
    except Exception:
        logging.exception("document_structure_compile：无法解析摄取聊天模型 %s", chat_llm_id)
        return
    chat_mdl_by_tid = {template_id: chat_mdl for template_id, _ in active_templates}

    tree_templates: list[tuple[str, dict]] = []
    non_tree_templates: list[tuple[str, dict]] = []
    for tid, cfg in active_templates:
        if _compilation_template_kind((cfg or {}).get("kind")) == "tree":
            tree_templates.append((tid, cfg))
        else:
            non_tree_templates.append((tid, cfg))

    if tree_templates:
        await run_tree_templates(
            handler,
            tree_templates,
            chat_mdl_by_tid,
            embedding_model,
            doc_name,
        )

    if not non_tree_templates:
        return

    async def _stream_doc_batches():
        async for batch in handler._load_chunks_for_doc(
            ctx.tenant_id,
            ctx.kb_id,
            ctx.doc_id,
            batch_size=DOC_STRUCTURE_COMPILE_BATCH_CHUNKS,
        ):
            yield batch

    await run_structure_compile_over_batches(
        active_templates=non_tree_templates,
        chat_mdl_by_tid=chat_mdl_by_tid,
        embedding_model=embedding_model,
        tenant_id=ctx.tenant_id,
        kb_id=ctx.kb_id,
        doc_id=ctx.doc_id,
        doc_name=doc_name,
        language=ctx.language,
        chunk_batches=_stream_doc_batches(),
        progress_cb=ctx.progress_cb,
        cancel_check=lambda: ctx.has_canceled_func(ctx.id),
        record=ctx.recording_context.record,
    )


async def run_document_post_chunking_if_last(
    handler,
    embedding_model: LLMBundle,
    vector_size: int,
    task_start_ts: float,
    chunks_len: int,
    token_count: int,
) -> bool:
    """Gate：仅文档的最后一个分块任务运行后处理。
    返回“`True`` if the caller may proceed to its own terminal
    progress update, ``False`` if the task was cancelled.

    The pass runs :func:`run_document_structure_compile` and
    ``handler._run_raptor`”同时——他们读取相同的块
    但写入不相交的 ES 行。"""
    ctx = handler._task_context
    task_id = ctx.id
    task_doc_id = ctx.doc_id

    if ctx.has_canceled_func(task_id):
        abort_doc_chunking_counter(task_doc_id)
        ctx.progress_cb(-1, msg="Task has been canceled.")
        return False

    chunking_aborted = is_doc_chunking_aborted(task_doc_id)
    remaining_chunking_tasks = 0 if ctx.write_interceptor else credit_doc_chunking_task(task_doc_id, task_id)
    # 一个 PDF 可拆成多个并行 Task。Redis 文档级计数器每完成一个分页 Task 减一；
    # 非最后一个 Task 只结束自身，只有减到 0 的 Task 才执行一次文档级结构编译/RAPTOR，
    # 防止每个页段重复生成整份文档的衍生数据。
    if remaining_chunking_tasks != 0:
        if chunking_aborted:
            logging.info(
                "在任务 %s 到达后处理之前，文档 %s 的分块已中止；跳过文档终结器。",
                task_doc_id,
                task_id,
            )
        elif remaining_chunking_tasks is not None and remaining_chunking_tasks < 0:
            logging.warning(
                "任务 %s 后，文档 %s 的分块计数器丢失或过期；跳过后处理以避免重复的终结器。",
                task_doc_id,
                task_id,
            )
        else:
            logging.info(
                "Chunk 文档（%s），页面（%s-%s），块（%s），代币(%s)，已过去：%.2f;在后处理之前等待 %s 分块任务",
                ctx.name,
                ctx.from_page,
                ctx.to_page,
                chunks_len,
                token_count,
                timer() - task_start_ts,
                remaining_chunking_tasks,
            )
        return True

    async def _maybe_run_raptor():
        raptor_cfg = (ctx.parser_config or {}).get("raptor") or {}
        if not raptor_cfg.get("do_raptor"):
            return
        try:
            ok_doc, doc_obj = DocumentService.get_by_id(task_doc_id)
            if ok_doc and doc_obj is not None:
                ctx.progress_cb(msg="Starting RAPTOR task.")
                await handler._run_raptor(embedding_model, vector_size, mark_done=False)
            else:
                logging.warning(
                    "raptor：无法解析文档 %s 来对每个文档任务进行排队",
                    task_doc_id,
                )
        except Exception:
            logging.exception(
                "raptor：无法对文档 %s 的每个文档任务进行排队",
                task_doc_id,
            )

    original_progress_cb = getattr(ctx, "_progress_cb", None)
    if original_progress_cb is not None:
        ctx._progress_cb = cap_done_progress(original_progress_cb)
    try:
        await asyncio.gather(
            run_document_structure_compile(handler, embedding_model),
            _maybe_run_raptor(),
        )
    finally:
        if original_progress_cb is not None:
            ctx._progress_cb = original_progress_cb
        clear_doc_chunking_counter(task_doc_id)

    if ctx.has_canceled_func(task_id):
        abort_doc_chunking_counter(task_doc_id)
        ctx.progress_cb(-1, msg="Task has been canceled.")
        return False
    return True
