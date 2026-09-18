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
#
import logging
import random
from datetime import datetime
from time import monotonic

import xxhash
from peewee import fn, Case, JOIN

from api.constants import IMG_BASE64_PREFIX, FILE_NAME_LEN_LIMIT
from api.db import PIPELINE_SPECIAL_PROGRESS_FREEZE_TASK_TYPES, FileType, UserTenantRole, CanvasCategory
from api.db.db_models import DB, Document, Knowledgebase, Task, Tenant, UserTenant, File2Document, File, UserCanvas, User
from api.db.db_utils import bulk_insert_into_db
from api.db.services.common_service import CommonService, retry_deadlock_operation
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.doc_metadata_service import DocMetadataService

from common import settings
from common.constants import ParserType, StatusEnum, TaskStatus, SVR_CONSUMER_GROUP_NAME, MAXIMUM_TASK_PAGE_NUMBER
from common.doc_store.doc_store_base import OrderByExpr
from common.misc_utils import get_uuid
from common.time_utils import current_timestamp, get_format_time

from rag.nlp import search
from rag.utils.redis_conn import REDIS_CONN


class DocumentService(CommonService):
    model = Document

    @classmethod
    @DB.connection_context()
    def get_disabled_doc_ids_by_kb_id(cls, kb_id) -> set[str]:
        return {str(doc_id) for (doc_id,) in cls.model.select(cls.model.id).where((cls.model.kb_id == kb_id) & (cls.model.status == "0")).tuples()}

    @classmethod
    def get_cls_model_fields(cls):
        return [
            cls.model.id,
            cls.model.thumbnail,
            cls.model.kb_id,
            cls.model.parser_id,
            cls.model.pipeline_id,
            cls.model.parser_config,
            cls.model.source_type,
            cls.model.type,
            cls.model.created_by,
            cls.model.name,
            cls.model.location,
            cls.model.size,
            cls.model.token_num,
            cls.model.chunk_num,
            cls.model.progress,
            cls.model.progress_msg,
            cls.model.process_begin_at,
            cls.model.process_duration,
            cls.model.suffix,
            cls.model.run,
            cls.model.status,
            cls.model.create_time,
            cls.model.create_date,
            cls.model.update_time,
            cls.model.update_date,
        ]

    @classmethod
    @DB.connection_context()
    def get_list(cls, kb_id, page_number, items_per_page, orderby, desc, keywords, id, name, suffix=None, run=None, doc_ids=None):
        fields = cls.get_cls_model_fields()
        docs = (
            cls.model.select(*[*fields, UserCanvas.title])
            .join(File2Document, on=(File2Document.document_id == cls.model.id))
            .join(File, on=(File.id == File2Document.file_id))
            .join(UserCanvas, on=((cls.model.pipeline_id == UserCanvas.id) & (UserCanvas.canvas_category == CanvasCategory.DataFlow.value)), join_type=JOIN.LEFT_OUTER)
            .where(cls.model.kb_id == kb_id)
        )
        if id:
            docs = docs.where(cls.model.id == id)
        if name:
            docs = docs.where(cls.model.name == name)
        if keywords:
            docs = docs.where(fn.LOWER(cls.model.name).contains(keywords.lower()))
        if doc_ids is not None:
            docs = docs.where(cls.model.id.in_(doc_ids))
        if suffix:
            docs = docs.where(cls.model.suffix.in_(suffix))
        if run:
            docs = docs.where(cls.model.run.in_(run))
        if desc:
            docs = docs.order_by(cls.model.getter_by(orderby).desc())
        else:
            docs = docs.order_by(cls.model.getter_by(orderby).asc())

        count = docs.count()
        docs = docs.paginate(page_number, items_per_page)

        docs_list = list(docs.dicts())
        doc_ids_on_page = [doc["id"] for doc in docs_list]
        metadata_map = DocMetadataService.get_metadata_for_documents(doc_ids_on_page, kb_id) if doc_ids_on_page else {}
        for doc in docs_list:
            doc["meta_fields"] = metadata_map.get(doc["id"], {})
        return docs_list, count

    @classmethod
    @DB.connection_context()
    def check_doc_health(cls, tenant_id: str, filename):
        import os

        MAX_FILE_NUM_PER_USER = int(os.environ.get("MAX_FILE_NUM_PER_USER", 0))
        if 0 < MAX_FILE_NUM_PER_USER <= DocumentService.get_doc_count(tenant_id):
            raise RuntimeError("Exceed the maximum file number of a free user!")
        if len(filename.encode("utf-8")) > FILE_NAME_LEN_LIMIT:
            raise RuntimeError("Exceed the maximum length of file name!")
        return True

    @classmethod
    @DB.connection_context()
    def get_by_kb_id(cls, kb_id, page_number, items_per_page, orderby, desc, keywords, run_status, types, suffix, name=None, doc_ids=None, return_empty_metadata=False):
        fields = cls.get_cls_model_fields()
        if keywords:
            docs = (
                cls.model.select(*[*fields, UserCanvas.title.alias("pipeline_name"), User.nickname])
                .join(File2Document, on=(File2Document.document_id == cls.model.id))
                .join(File, on=(File.id == File2Document.file_id))
                .join(UserCanvas, on=(cls.model.pipeline_id == UserCanvas.id), join_type=JOIN.LEFT_OUTER)
                .join(User, on=(cls.model.created_by == User.id), join_type=JOIN.LEFT_OUTER)
                .where((cls.model.kb_id == kb_id), (fn.LOWER(cls.model.name).contains(keywords.lower())))
            )
        else:
            docs = (
                cls.model.select(*[*fields, UserCanvas.title.alias("pipeline_name"), User.nickname])
                .join(File2Document, on=(File2Document.document_id == cls.model.id))
                .join(UserCanvas, on=(cls.model.pipeline_id == UserCanvas.id), join_type=JOIN.LEFT_OUTER)
                .join(File, on=(File.id == File2Document.file_id))
                .join(User, on=(cls.model.created_by == User.id), join_type=JOIN.LEFT_OUTER)
                .where(cls.model.kb_id == kb_id)
            )
        if doc_ids is not None:
            docs = docs.where(cls.model.id.in_(doc_ids))
        if run_status:
            docs = docs.where(cls.model.run.in_(run_status))
        if types:
            docs = docs.where(cls.model.type.in_(types))
        if suffix:
            docs = docs.where(cls.model.suffix.in_(suffix))
        if name:
            docs = docs.where(cls.model.name == name)

        if return_empty_metadata:
            metadata_map = DocMetadataService.get_metadata_for_documents(None, kb_id)
            doc_ids_with_metadata = set(metadata_map.keys())
            if doc_ids_with_metadata:
                docs = docs.where(cls.model.id.not_in(doc_ids_with_metadata))

        count = docs.count()
        if desc:
            docs = docs.order_by(cls.model.getter_by(orderby).desc())
        else:
            docs = docs.order_by(cls.model.getter_by(orderby).asc())

        if page_number and items_per_page:
            docs = docs.paginate(page_number, items_per_page)

        docs_list = list(docs.dicts())
        if return_empty_metadata:
            for doc in docs_list:
                doc["meta_fields"] = {}
        else:
            doc_ids_on_page = [doc["id"] for doc in docs_list]
            metadata_map = DocMetadataService.get_metadata_for_documents(doc_ids_on_page, kb_id) if doc_ids_on_page else {}
            for doc in docs_list:
                doc["meta_fields"] = metadata_map.get(doc["id"], {})
        return docs_list, count

    @classmethod
    @DB.connection_context()
    def get_filter_by_kb_id(cls, kb_id, keywords, run_status, types, suffix):
        """返回：
        {
            "suffix"：{
                "ppt": 1,
                "doxc"：2
            },
            "run_status"：{
             "1": 2,
             "2"：2
            }
            "metadata"：{
                "key1"：{
                 "key1_value1": 1,
                 "key1_value2": 2,
                },
                "key2"：{
                 "key2_value1"：2，
                 "key2_value2": 1,
                },
            }
        }，总计
        其中 "1" => RUNNING，"2" => CANCEL"""
        fields = cls.get_cls_model_fields()
        if keywords:
            query = (
                cls.model.select(*fields)
                .join(File2Document, on=(File2Document.document_id == cls.model.id))
                .join(File, on=(File.id == File2Document.file_id))
                .where((cls.model.kb_id == kb_id), (fn.LOWER(cls.model.name).contains(keywords.lower())))
            )
        else:
            query = cls.model.select(*fields).join(File2Document, on=(File2Document.document_id == cls.model.id)).join(File, on=(File.id == File2Document.file_id)).where(cls.model.kb_id == kb_id)

        if run_status:
            query = query.where(cls.model.run.in_(run_status))
        if types:
            query = query.where(cls.model.type.in_(types))
        if suffix:
            query = query.where(cls.model.suffix.in_(suffix))

        rows = query.select(cls.model.run, cls.model.suffix, cls.model.id)
        total = rows.count()

        suffix_counter = {}
        run_status_counter = {}
        metadata_counter = {}
        empty_metadata_count = 0

        doc_ids = [row.id for row in rows]
        metadata = {}
        if doc_ids:
            try:
                metadata = DocMetadataService.get_metadata_for_documents(doc_ids, kb_id)
            except Exception as e:
                logging.warning(f"Failed to fetch metadata from ES/Infinity: {e}")

        for row in rows:
            suffix_counter[row.suffix] = suffix_counter.get(row.suffix, 0) + 1
            run_status_counter[str(row.run)] = run_status_counter.get(str(row.run), 0) + 1
            meta_fields = metadata.get(row.id, {})
            if not meta_fields:
                empty_metadata_count += 1
                continue
            has_valid_meta = False
            for key, value in meta_fields.items():
                values = value if isinstance(value, list) else [value]
                for vv in values:
                    if vv is None:
                        continue
                    if isinstance(vv, str) and not vv.strip():
                        continue
                    sv = str(vv)
                    if key not in metadata_counter:
                        metadata_counter[key] = {}
                    metadata_counter[key][sv] = metadata_counter[key].get(sv, 0) + 1
                    has_valid_meta = True
            if not has_valid_meta:
                empty_metadata_count += 1

        metadata_counter["empty_metadata"] = {"true": empty_metadata_count}
        return {
            "suffix": suffix_counter,
            "run_status": run_status_counter,
            "metadata": metadata_counter,
        }, total

    @classmethod
    @DB.connection_context()
    def get_parsing_status_by_kb_ids(cls, kb_ids: list[str]) -> dict[str, dict[str, int]]:
        """返回按数据集分组的聚合文档解析状态计数 (kb_id)。

        对于每个 kb_id，计算每个运行状态存储桶中的文档：
            - unstart_count（运行== "0"）
            - running_count（运行== "1"）
            - cancel_count（运行== "2"）
            - done_count（运行== "3"）
            - fail_count（运行== "4"）

        返回一个由 kb_id、e.g 键控的字典。
            {"kb-abc"：{"unstart_count": 10, "running_count": 2, ...}，...}"""
        if not kb_ids:
            return {}

        status_field_map = {
            TaskStatus.UNSTART.value: "unstart_count",
            TaskStatus.RUNNING.value: "running_count",
            TaskStatus.CANCEL.value: "cancel_count",
            TaskStatus.DONE.value: "done_count",
            TaskStatus.FAIL.value: "fail_count",
        }

        empty_status = {v: 0 for v in status_field_map.values()}
        result: dict[str, dict[str, int]] = {kb_id: dict(empty_status) for kb_id in kb_ids}

        rows = (
            cls.model.select(
                cls.model.kb_id,
                cls.model.run,
                fn.COUNT(cls.model.id).alias("cnt"),
            )
            .where(cls.model.kb_id.in_(kb_ids))
            .group_by(cls.model.kb_id, cls.model.run)
            .dicts()
        )

        for row in rows:
            kb_id = row["kb_id"]
            run_val = str(row["run"])
            field_name = status_field_map.get(run_val)
            if field_name and kb_id in result:
                result[kb_id][field_name] = int(row["cnt"])

        return result

    @classmethod
    @DB.connection_context()
    def count_by_kb_id(cls, kb_id, keywords, run_status, types):
        if keywords:
            docs = cls.model.select().where((cls.model.kb_id == kb_id), (fn.LOWER(cls.model.name).contains(keywords.lower())))
        else:
            docs = cls.model.select().where(cls.model.kb_id == kb_id)

        if run_status:
            docs = docs.where(cls.model.run.in_(run_status))
        if types:
            docs = docs.where(cls.model.type.in_(types))

        count = docs.count()

        return count

    @classmethod
    @DB.connection_context()
    def get_total_size_by_kb_id(cls, kb_id, keywords="", run_status=[], types=[]):
        query = cls.model.select(fn.COALESCE(fn.SUM(cls.model.size), 0)).where(cls.model.kb_id == kb_id)

        if keywords:
            query = query.where(fn.LOWER(cls.model.name).contains(keywords.lower()))
        if run_status:
            query = query.where(cls.model.run.in_(run_status))
        if types:
            query = query.where(cls.model.type.in_(types))

        return int(query.scalar()) or 0

    @classmethod
    @DB.connection_context()
    def get_all_doc_ids_by_kb_ids(cls, kb_ids):
        fields = [cls.model.id, cls.model.kb_id]
        docs = cls.model.select(*fields).where(cls.model.kb_id.in_(kb_ids))
        docs = docs.order_by(cls.model.create_time.asc())
        # 可能会因为深度分页导致查询慢，稍后优化
        offset, limit = 0, 100
        res = []
        while True:
            doc_batch = docs.offset(offset).limit(limit)
            _temp = list(doc_batch.dicts())
            if not _temp:
                break
            res.extend(_temp)
            offset += limit
        return res

    @classmethod
    @DB.connection_context()
    def list_doc_headers_by_kb_and_source_type(cls, kb_id, source_type, page_size=500):
        fields = [cls.model.id, cls.model.kb_id, cls.model.source_type, cls.model.name]
        docs = (
            cls.model.select(*fields)
            .where(
                cls.model.kb_id == kb_id,
                cls.model.source_type == source_type,
            )
            .order_by(cls.model.create_time.asc())
        )
        offset = 0
        res = []
        while True:
            doc_batch = docs.offset(offset).limit(page_size)
            _temp = list(doc_batch.dicts())
            if not _temp:
                break
            res.extend(_temp)
            offset += page_size
        return res

    @classmethod
    @DB.connection_context()
    def list_id_content_hash_map_by_kb_and_source_type(cls, kb_id, source_type, page_size=500):
        """返回 {doc_id: content_hash} 连接器的现有文档。

        由指纹绕过路径用来决定哪些键可以跳过
        重新获取 - 如果连接器的列表指纹等于 content_hash，
        自上次同步以来，主体没有改变。

        按 create_time 排序，因此 LIMIT/OFFSET 分页在以下情况下稳定
        并发写入；如果没有这个，页面边界可能会丢失或重复
        行和生成的映射将默默地错过条目。"""
        fields = [cls.model.id, cls.model.content_hash]
        docs = (
            cls.model.select(*fields)
            .where(
                cls.model.kb_id == kb_id,
                cls.model.source_type == source_type,
            )
            .order_by(cls.model.create_time.asc())
        )
        offset = 0
        result: dict[str, str] = {}
        while True:
            batch = list(docs.offset(offset).limit(page_size).dicts())
            if not batch:
                break
            for row in batch:
                result[row["id"]] = row.get("content_hash") or ""
            offset += page_size
        return result

    @classmethod
    @DB.connection_context()
    def get_all_docs_by_creator_id(cls, creator_id):
        fields = [cls.model.id, cls.model.kb_id, cls.model.token_num, cls.model.chunk_num, Knowledgebase.tenant_id]
        docs = cls.model.select(*fields).join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id)).where(cls.model.created_by == creator_id)
        docs = docs.order_by(cls.model.create_time.asc())
        # 可能会因深度分页导致查询慢，稍后优化
        offset, limit = 0, 100
        res = []
        while True:
            doc_batch = docs.offset(offset).limit(limit)
            _temp = list(doc_batch.dicts())
            if not _temp:
                break
            res.extend(_temp)
            offset += limit
        return res

    @classmethod
    @DB.connection_context()
    def insert(cls, doc):
        if not cls.save(**doc):
            raise RuntimeError("Database error (Document)!")
        if not KnowledgebaseService.atomic_increase_doc_num_by_id(doc["kb_id"]):
            raise RuntimeError("Database error (Knowledgebase)!")
        return Document(**doc)

    @classmethod
    @DB.connection_context()
    def remove_document(cls, doc, tenant_id):
        from api.db.services.task_service import TaskService, cancel_all_task_of

        if not cls.delete_document_and_update_kb_counts(doc.id):
            return True

        chunk_index_name = search.index_name(tenant_id)
        chunk_index_exists = settings.docStoreConn.index_exist(chunk_index_name, doc.kb_id)

        # 首先使用 task_service.py 中预设的功能取消所有正在运行的任务 --- 在 Redis 中设置取消标志
        try:
            cancel_all_task_of(doc.id)
            logging.info(f"Cancelled all tasks for document {doc.id}")
        except Exception as e:
            logging.warning(f"Failed to cancel tasks for document {doc.id}: {e}")

        # 从数据库中删除任务
        try:
            TaskService.filter_delete([Task.doc_id == doc.id])
        except Exception as e:
            logging.warning(f"Failed to delete tasks for document {doc.id}: {e}")

        # 删除块映像（非关键，记录并继续）
        try:
            if chunk_index_exists:
                cls.delete_chunk_images(doc, tenant_id)
        except Exception as e:
            logging.warning(f"Failed to delete chunk images for document {doc.id}: {e}")

        # 删除缩略图（非关键，记录并继续）
        try:
            if doc.thumbnail and not doc.thumbnail.startswith(IMG_BASE64_PREFIX):
                if settings.STORAGE_IMPL.obj_exist(doc.kb_id, doc.thumbnail):
                    settings.STORAGE_IMPL.rm(doc.kb_id, doc.thumbnail)
        except Exception as e:
            logging.warning(f"Failed to delete thumbnail for document {doc.id}: {e}")

        # 在之前修剪此文档's line from the KB's树类导航
        # 下面的
        # 宽 doc_id 删除删除了定位所需的 nav_doc 行
        # 并更新其父集群。
        try:
            from rag.advanced_rag.knowlege_compile.dataset_nav import (
                remove_dataset_nav_doc_sync,
            )

            remove_dataset_nav_doc_sync(tenant_id, doc.kb_id, doc.id)
        except Exception as e:
            logging.warning(
                f"Failed to prune dataset_nav for document {doc.id}: {e}",
            )

        # 从文档存储中删除块 - 这很关键，记录错误
        try:
            settings.docStoreConn.delete({"doc_id": doc.id}, chunk_index_name, doc.kb_id)
        except Exception as e:
            logging.error(f"Failed to delete chunks from doc store for document {doc.id}: {e}")

        # 记录增量结构合并幽灵清理的文档删除。
        # 在 doc_id 扫描之后运行，因此标记（存储在
        # deleted_doc_id 以避免匹配相同的扫描）幸存。
        try:
            from rag.svr.task_executor_refactor.dataset_structure_merger import (
                record_doc_deletion,
            )

            record_doc_deletion(tenant_id, doc.kb_id, doc.id)
        except Exception as e:
            logging.warning(
                f"Failed to record doc deletion for structure merge: {e}",
            )

        # 本文档输入的 wiki/artifact 产品的参考计数清理
        # （非关键，记录并继续）。其他文档共享的产品
        # 幸存；该文档独自拥有的一项已被删除。
        try:
            if chunk_index_exists:
                cls.remove_wiki_products(doc, tenant_id)
        except Exception as e:
            logging.warning(f"Failed to clean up artifact products for document {doc.id}: {e}")

        # 删除文档元数据（非关键，记录并继续）
        try:
            DocMetadataService.delete_document_metadata(doc.id, doc.kb_id, tenant_id)
        except Exception as e:
            logging.warning(f"Failed to delete metadata for document {doc.id}: {e}")

        # 清理知识图参考（非关键，记录并继续）
        try:
            if chunk_index_exists:
                graph_source = settings.docStoreConn.get_fields(
                    settings.docStoreConn.search(["source_id"], [], {"kb_id": doc.kb_id, "knowledge_graph_kwd": ["graph"]}, [], OrderByExpr(), 0, 1, chunk_index_name, [doc.kb_id]),
                    ["source_id"],
                )
                if len(graph_source) > 0 and doc.id in list(graph_source.values())[0]["source_id"]:
                    settings.docStoreConn.update(
                        {"kb_id": doc.kb_id, "knowledge_graph_kwd": ["entity", "relation", "graph", "subgraph", "community_report"], "source_id": doc.id},
                        {"remove": {"source_id": doc.id}},
                        chunk_index_name,
                        doc.kb_id,
                    )
                    settings.docStoreConn.update({"kb_id": doc.kb_id, "knowledge_graph_kwd": ["graph"]}, {"removed_kwd": "Y"}, chunk_index_name, doc.kb_id)
                    settings.docStoreConn.delete(
                        {"kb_id": doc.kb_id, "knowledge_graph_kwd": ["entity", "relation", "graph", "subgraph", "community_report"], "must_not": {"exists": "source_id"}},
                        chunk_index_name,
                        doc.kb_id,
                    )
        except Exception as e:
            logging.warning(f"Failed to cleanup knowledge graph for document {doc.id}: {e}")

        return True

    @classmethod
    @DB.connection_context()
    def delete_chunk_images(cls, doc, tenant_id):
        page = 0
        page_size = 1000
        while True:
            chunks = settings.docStoreConn.search(["img_id"], [], {"doc_id": doc.id}, [], OrderByExpr(), page * page_size, page_size, search.index_name(tenant_id), [doc.kb_id])
            chunk_ids = settings.docStoreConn.get_doc_ids(chunks)
            if not chunk_ids:
                break
            for cid in chunk_ids:
                if settings.STORAGE_IMPL.obj_exist(doc.kb_id, cid):
                    settings.STORAGE_IMPL.rm(doc.kb_id, cid)
            page += 1

    @classmethod
    def remove_wiki_products(cls, doc, tenant_id):
        """KB 范围的 wiki/artifact 产品的引用计数清理
        删除文档时在文档存储中。

        每个派生的工件行（页面、实体、关系、草稿、
        主题，reduce/plan 聚合）携带``source_doc_ids`` list
        of the documents that contributed to it. On delete we detach
        ``doc.id`` from that list and drop the row only when this document
        was its sole contributor — a product shared by other docs
        survives. ``wiki_map_extract`` resume rows are 1:1 with a
        document and are removed directly by ``doc_id``。

        compile_kwd 集是从 wiki 生成器中提取的，所以很新
        自动覆盖工件行类型（单一来源
        真相）。删除不是热路径，因此模块导入成本为
        这里可以接受。"""
        from rag.svr.task_executor_refactor.dataset_wiki_generator import (
            WIKI_MAP_COMPILE_KWD,
            WIKI_DERIVED_COMPILE_KWDS,
        )

        index = search.index_name(tenant_id)
        if not settings.docStoreConn.index_exist(index, doc.kb_id):
            return

        # 1. 每个文档 MAP 简历行由真实的 doc_id 键入。
        settings.docStoreConn.delete(
            {"compile_kwd": [WIKI_MAP_COMPILE_KWD], "doc_id": doc.id},
            index,
            doc.kb_id,
        )

        # 2. 派生 KB 范围的行：通过 source_doc_ids 进行引用计数。
        # 读取该文档贡献的每一行，并将其分区为行
        # 独资拥有（按 id 删除）与与其他文档共享的行
        # （分离此文档）。先读书——而不是盖毯子
        # ``must_not exists`` 扫描 — 避免删除合法的行
        # 不携带 source_doc_ids。
        derived_kwds = list(WIKI_DERIVED_COMPILE_KWDS)
        select_fields = ["id", "source_doc_ids"]
        sole_owner_ids: list[str] = []
        shared_seen = False
        offset = 0
        page_size = 1000
        while True:
            res = settings.docStoreConn.search(
                select_fields,
                [],
                {"compile_kwd": derived_kwds, "source_doc_ids": [doc.id]},
                [],
                OrderByExpr(),
                offset,
                page_size,
                index,
                [doc.kb_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields) or {}
            if not field_map:
                break
            for row_id, row in field_map.items():
                raw = row.get("source_doc_ids")
                if isinstance(raw, str):
                    owners = [raw] if raw else []
                elif isinstance(raw, list):
                    owners = [d for d in raw if isinstance(d, str) and d]
                else:
                    owners = []
                if any(d != doc.id for d in owners):
                    shared_seen = True
                else:
                    sole_owner_ids.append(row_id)
            if len(field_map) < page_size:
                break
            offset += page_size

        # 删除该文档独占的行（按id批量删除）。
        for i in range(0, len(sole_owner_ids), page_size):
            settings.docStoreConn.delete(
                {"id": sole_owner_ids[i : i + page_size]},
                index,
                doc.kb_id,
            )

        # 将此文档与仍由其他人拥有的行分离。过滤器
        # 保证source_doc_ids包含doc.id，所以商店的
        # 列表删除是安全的；上面已经删除的任何唯一所有者行都是
        # 根本不匹配。
        if shared_seen:
            settings.docStoreConn.update(
                {"compile_kwd": derived_kwds, "source_doc_ids": doc.id},
                {"remove": {"source_doc_ids": doc.id}},
                index,
                doc.kb_id,
            )

        # 3. 清理doc_page_source 跟踪行（新的增量设计）。
        try:
            doc_page_kwd = "wiki_doc_page_source"
            res = settings.docStoreConn.search(
                ["id"],
                [],
                {"compile_kwd": [doc_page_kwd], "doc_id": [doc.id]},
                [],
                OrderByExpr(),
                0,
                10,
                index,
                doc.kb_id,
            )
            if settings.docStoreConn.get_fields(res, ["id"]):
                settings.docStoreConn.delete(
                    {"compile_kwd": [doc_page_kwd], "doc_id": [doc.id]},
                    index,
                    doc.kb_id,
                )
        except Exception:
            logging.exception(
                "DocumentService.remove_wiki_products：文档 %s 的 doc_page_source 清理失败",
                doc.id,
            )

    @classmethod
    @DB.connection_context()
    def get_newly_uploaded(cls):
        fields = [
            cls.model.id,
            cls.model.kb_id,
            cls.model.parser_id,
            cls.model.parser_config,
            cls.model.name,
            cls.model.type,
            cls.model.location,
            cls.model.size,
            Knowledgebase.tenant_id,
            Tenant.embd_id,
            Tenant.img2txt_id,
            Tenant.asr_id,
            cls.model.update_time,
        ]
        docs = (
            cls.model.select(*fields)
            .join(Knowledgebase, on=(cls.model.kb_id == Knowledgebase.id))
            .join(Tenant, on=(Knowledgebase.tenant_id == Tenant.id))
            .where(
                cls.model.status == StatusEnum.VALID.value,
                ~(cls.model.type == FileType.VIRTUAL.value),
                cls.model.progress == 0,
                cls.model.update_time >= current_timestamp() - 1000 * 600,
                cls.model.run == TaskStatus.RUNNING.value,
            )
            .order_by(cls.model.update_time.asc())
        )
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def get_unfinished_docs(cls):
        fields = [cls.model.id, cls.model.process_begin_at, cls.model.parser_config, cls.model.progress_msg, cls.model.run, cls.model.parser_id]
        unfinished_task_query = Task.select(Task.doc_id).where((Task.progress >= 0) & (Task.progress < 1))
        docs_with_non_failed_tasks = Task.select(Task.doc_id).where(Task.progress >= 0).distinct()

        docs = cls.model.select(*fields).where(
            cls.model.status == StatusEnum.VALID.value,
            ~(cls.model.type == FileType.VIRTUAL.value),
            ((cls.model.run.is_null(True)) | (cls.model.run != TaskStatus.CANCEL.value)),
            (
                ((cls.model.progress < 1) & (cls.model.progress > 0))
                | (cls.model.id.in_(unfinished_task_query))
                | ((cls.model.progress == -1) & (cls.model.run == TaskStatus.FAIL.value) & (cls.model.id.in_(docs_with_non_failed_tasks)))
            ),
        )  # 包括 GraphRAG/RAPTOR/思维导图；重新同步失败的文档
        return list(docs.dicts())

    @classmethod
    @DB.connection_context()
    def increment_chunk_num(cls, doc_id, kb_id, token_num, chunk_num, duration):
        """在文档及其知识库上自动添加 chunk/token 计数器。"""
        with DB.atomic():
            num = (
                cls.model.update(
                    token_num=cls.model.token_num + token_num,
                    chunk_num=cls.model.chunk_num + chunk_num,
                    process_duration=cls.model.process_duration + duration,
                )
                .where((cls.model.id == doc_id) & (cls.model.kb_id == kb_id))
                .execute()
            )
            if num == 0:
                logging.error(
                    "increment_chunk_num：没有匹配的文档 文档ID=%s 知识库ID=%s token_num=%s chunk_num=%s 持续时间=%s",
                    doc_id,
                    kb_id,
                    token_num,
                    chunk_num,
                    duration,
                )
                raise LookupError("Document not found which is supposed to be there")
            num = (
                Knowledgebase.update(
                    token_num=Knowledgebase.token_num + token_num,
                    chunk_num=Knowledgebase.chunk_num + chunk_num,
                )
                .where(Knowledgebase.id == kb_id)
                .execute()
            )
            if num == 0:
                logging.error(
                    "increment_chunk_num：没有匹配的知识库 知识库ID=%s for 文档ID=%s token_num=%s chunk_num=%s 持续时间=%s",
                    kb_id,
                    doc_id,
                    token_num,
                    chunk_num,
                    duration,
                )
                raise LookupError("Knowledgebase not found which is supposed to be there")
        return num

    @classmethod
    @DB.connection_context()
    def decrement_chunk_num(cls, doc_id, kb_id, token_num, chunk_num, duration):
        """以原子方式减去文档及其知识库上的 chunk/token 计数器。"""
        with DB.atomic():
            num = (
                cls.model.update(
                    token_num=cls.model.token_num - token_num,
                    chunk_num=cls.model.chunk_num - chunk_num,
                    process_duration=cls.model.process_duration + duration,
                )
                .where((cls.model.id == doc_id) & (cls.model.kb_id == kb_id))
                .execute()
            )
            if num == 0:
                raise LookupError("Document not found which is supposed to be there")
            num = (
                Knowledgebase.update(
                    token_num=Knowledgebase.token_num - token_num,
                    chunk_num=Knowledgebase.chunk_num - chunk_num,
                )
                .where(Knowledgebase.id == kb_id)
                .execute()
            )
            if num == 0:
                logging.error(
                    "decrement_chunk_num：没有匹配的知识库 知识库ID=%s for 文档ID=%s token_num=%s chunk_num=%s 持续时间=%s",
                    kb_id,
                    doc_id,
                    token_num,
                    chunk_num,
                    duration,
                )
                raise LookupError("Knowledgebase not found which is supposed to be there")
        return num

    @classmethod
    @retry_deadlock_operation()
    @DB.connection_context()
    def delete_document_and_update_kb_counts(cls, doc_id) -> bool:
        """以原子方式删除文档行并更新 KB 计数器。

        如果此调用删除了文档，则返回 True；如果是，则返回 False
        已被并发请求删除（幂等）。"""
        with DB.atomic():
            doc = (
                cls.model.select(
                    cls.model.id,
                    cls.model.kb_id,
                    cls.model.token_num,
                    cls.model.chunk_num,
                )
                .where(cls.model.id == doc_id)
                .for_update()
                .get_or_none()
            )
            if doc is None:
                return False
            deleted = cls.model.delete().where(cls.model.id == doc_id).execute()
            if not deleted:
                return False
            Knowledgebase.update(
                token_num=Knowledgebase.token_num - doc.token_num,
                chunk_num=Knowledgebase.chunk_num - doc.chunk_num,
                doc_num=Knowledgebase.doc_num - 1,
            ).where(Knowledgebase.id == doc.kb_id).execute()
        return True

    @classmethod
    @DB.connection_context()
    def clear_chunk_num(cls, doc_id):
        """已弃用：使用 delete_document_and_update_kb_counts 代替。
        对于 GraphRAG、RAPTOR 和思维导图任务，当 keep_progress=True 时，"""
        doc = cls.model.get_by_id(doc_id)
        assert doc, "Can't find document in database."

        num = (
            Knowledgebase.update(token_num=Knowledgebase.token_num - doc.token_num, chunk_num=Knowledgebase.chunk_num - doc.chunk_num, doc_num=Knowledgebase.doc_num - 1)
            .where(Knowledgebase.id == doc.kb_id)
            .execute()
        )
        return num

    @classmethod
    @DB.connection_context()
    def clear_chunk_num_when_rerun(cls, doc_id):
        doc = cls.model.get_by_id(doc_id)
        assert doc, "Can't find document in database."

        num = (
            Knowledgebase.update(
                token_num=Knowledgebase.token_num - doc.token_num,
                chunk_num=Knowledgebase.chunk_num - doc.chunk_num,
            )
            .where(Knowledgebase.id == doc.kb_id)
            .execute()
        )
        return num

    @classmethod
    @DB.connection_context()
    def get_tenant_id(cls, doc_id):
        docs = cls.model.select(Knowledgebase.tenant_id).join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id)).where(cls.model.id == doc_id, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return None
        return docs[0]["tenant_id"]

    @classmethod
    @DB.connection_context()
    def get_knowledgebase_id(cls, doc_id):
        docs = cls.model.select(cls.model.kb_id).where(cls.model.id == doc_id)
        docs = docs.dicts()
        if not docs:
            return None
        return docs[0]["kb_id"]

    @classmethod
    @DB.connection_context()
    def get_tenant_id_by_name(cls, name):
        docs = cls.model.select(Knowledgebase.tenant_id).join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id)).where(cls.model.name == name, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return None
        return docs[0]["tenant_id"]

    @classmethod
    @DB.connection_context()
    def accessible(cls, doc_id, user_id):
        e, doc = cls.get_by_id(doc_id)
        if not e:
            return False
        return KnowledgebaseService.accessible(doc.kb_id, user_id)

    @classmethod
    @DB.connection_context()
    def accessible4deletion(cls, doc_id, user_id):
        docs = (
            cls.model.select(cls.model.id)
            .join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id))
            .join(UserTenant, on=((UserTenant.tenant_id == Knowledgebase.created_by) & (UserTenant.user_id == user_id)))
            .where(cls.model.id == doc_id, UserTenant.status == StatusEnum.VALID.value, ((UserTenant.role == UserTenantRole.NORMAL) | (UserTenant.role == UserTenantRole.OWNER)))
            .paginate(0, 1)
        )
        docs = docs.dicts()
        if not docs:
            return False
        return True

    @classmethod
    @DB.connection_context()
    def get_embd_id(cls, doc_id):
        docs = cls.model.select(Knowledgebase.embd_id).join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id)).where(cls.model.id == doc_id, Knowledgebase.status == StatusEnum.VALID.value)
        docs = docs.dicts()
        if not docs:
            return None
        return docs[0]["embd_id"]

    @classmethod
    @DB.connection_context()
    def get_tenant_embd_id(cls, doc_id):
        docs = (
            cls.model.select(Knowledgebase.tenant_embd_id).join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id)).where(cls.model.id == doc_id, Knowledgebase.status == StatusEnum.VALID.value)
        )
        docs = docs.dicts()
        if not docs:
            return None
        return docs[0]["tenant_embd_id"]

    @classmethod
    @DB.connection_context()
    def get_chunking_config(cls, doc_id):
        configs = (
            cls.model.select(
                cls.model.id,
                cls.model.kb_id,
                cls.model.parser_id,
                cls.model.parser_config,
                cls.model.size,
                cls.model.content_hash,
                Knowledgebase.language,
                Knowledgebase.embd_id,
                Tenant.id.alias("tenant_id"),
                Tenant.img2txt_id,
                Tenant.asr_id,
                Tenant.llm_id,
            )
            .join(Knowledgebase, on=(cls.model.kb_id == Knowledgebase.id))
            .join(Tenant, on=(Knowledgebase.tenant_id == Tenant.id))
            .where(cls.model.id == doc_id)
        )
        configs = configs.dicts()
        if not configs:
            return None
        return configs[0]

    @classmethod
    @DB.connection_context()
    def get_doc_id_by_doc_name(cls, doc_name):
        fields = [cls.model.id]
        doc_id = cls.model.select(*fields).where(cls.model.name == doc_name)
        doc_id = doc_id.dicts()
        if not doc_id:
            return None
        return doc_id[0]["id"]

    @classmethod
    @DB.connection_context()
    def get_doc_ids_by_doc_names(cls, doc_names):
        if not doc_names:
            return []

        query = cls.model.select(cls.model.id).where(cls.model.name.in_(doc_names))
        return list(query.scalars().iterator())

    @classmethod
    @DB.connection_context()
    def get_thumbnails(cls, docids):
        fields = [cls.model.id, cls.model.kb_id, cls.model.thumbnail]
        return list(cls.model.select(*fields).where(cls.model.id.in_(docids)).dicts())

    @classmethod
    @DB.connection_context()
    def update_parser_config(cls, id, config):
        if not config:
            return
        e, d = cls.get_by_id(id)
        if not e:
            raise LookupError(f"Document({id}) not found.")

        def dfs_update(old, new):
            for k, v in new.items():
                if k not in old:
                    old[k] = v
                    continue
                if isinstance(v, dict) and isinstance(old[k], dict):
                    dfs_update(old[k], v)
                else:
                    old[k] = v

        dfs_update(d.parser_config, config)
        if not config.get("raptor") and d.parser_config.get("raptor"):
            del d.parser_config["raptor"]
        cls.update_by_id(id, {"parser_config": d.parser_config})

    @classmethod
    @DB.connection_context()
    def get_doc_count(cls, tenant_id):
        docs = cls.model.select(cls.model.id).join(Knowledgebase, on=(Knowledgebase.id == cls.model.kb_id)).where(Knowledgebase.tenant_id == tenant_id)
        return len(docs)

    @classmethod
    @DB.connection_context()
    def begin2parse(cls, doc_id, keep_progress=False):
        info = {
            "progress_msg": "Task is queued...",
            "process_begin_at": get_format_time(),
        }
        if not keep_progress:
            info["progress"] = random.random() * 1 / 100.0
            info["run"] = TaskStatus.RUNNING.value
            # 将文档保持在 DONE 状态

        cls.model.update(info).where((cls.model.id == doc_id) & ((cls.model.run.is_null(True)) | (cls.model.run != TaskStatus.CANCEL.value))).execute()

    @classmethod
    @DB.connection_context()
    def update_progress(cls):
        docs = cls.get_unfinished_docs()

        cls._sync_progress(docs)

    @classmethod
    @DB.connection_context()
    def update_progress_immediately(cls, docs: list[dict]):
        if not docs:
            return

        cls._sync_progress(docs)

    @classmethod
    @DB.connection_context()
    def _sync_progress(cls, docs: list[dict]):
        from api.db.services.task_service import TaskService

        for d in docs:
            try:
                tsks = TaskService.query(doc_id=d["id"], order_by=Task.create_time)
                if not tsks:
                    continue
                msg = []
                prg = 0
                finished = True
                bad = 0
                e, doc = DocumentService.get_by_id(d["id"])
                status = doc.run  # TaskStatus.RUNNING.value
                if status == TaskStatus.CANCEL.value:
                    continue
                doc_progress = doc.progress if doc and doc.progress else 0.0
                special_task_running = False
                priority = 0
                # 按优先级计算该文档自己的尚未启动的任务，以便
                # 它们可以从 "tasks ahead in the queue" 图中排除
                # 为匹配优先级队列。
                own_queued_by_priority = {}
                for t in tsks:
                    task_type = (t.task_type or "").lower()
                    if task_type in PIPELINE_SPECIAL_PROGRESS_FREEZE_TASK_TYPES:
                        special_task_running = True
                    if 0 <= t.progress < 1:
                        finished = False
                    if t.progress == -1:
                        bad += 1
                    if (t.progress or 0) == 0:
                        own_queued_by_priority[t.priority] = own_queued_by_priority.get(t.priority, 0) + 1
                    prg += t.progress if t.progress >= 0 else 0
                    if (t.progress_msg or "").strip():
                        msg.append(t.progress_msg)
                    priority = max(priority, t.priority)
                prg /= len(tsks)
                if finished and bad:
                    prg = -1
                    status = TaskStatus.FAIL.value
                elif finished:
                    prg = 1
                    status = TaskStatus.DONE.value
                elif not finished:
                    status = TaskStatus.RUNNING.value

                # 仅适用于特殊任务和已解析的文档以及未完成的
                freeze_progress = special_task_running and doc_progress >= 1 and not finished
                msg = "\n".join(sorted(msg))
                begin_at = d.get("process_begin_at")
                if not begin_at:
                    begin_at = datetime.now()
                    # 后备
                    cls.update_by_id(d["id"], {"process_begin_at": begin_at})

                info = {"process_duration": max(datetime.timestamp(datetime.now()) - begin_at.timestamp(), 0), "run": status}
                if prg != 0 and not freeze_progress:
                    info["progress"] = prg
                if msg:
                    info["progress_msg"] = msg
                    if msg.endswith("created task graphrag") or msg.endswith("created task raptor") or msg.endswith("created task mindmap"):
                        # 在同一个文档中排除本文档自己的排队任务
                        # 优先级队列：它们不是"ahead"本身，它们
                        # ARE 正在等待的工作。
                        queue_ahead = max(0, get_queue_length(priority) - own_queued_by_priority.get(priority, 0))
                        info["progress_msg"] += "\n%d tasks are ahead in the queue..." % queue_ahead
                else:
                    queue_ahead = max(0, get_queue_length(priority) - own_queued_by_priority.get(priority, 0))
                    info["progress_msg"] = "%d tasks are ahead in the queue..." % queue_ahead
                info["update_time"] = current_timestamp()
                info["update_date"] = get_format_time()
                (cls.model.update(info).where((cls.model.id == d["id"]) & ((cls.model.run.is_null(True)) | (cls.model.run != TaskStatus.CANCEL.value))).execute())
            except Exception as e:
                if str(e).find("'0'") < 0:
                    logging.exception("取任务异常")

    @classmethod
    @DB.connection_context()
    def get_kb_doc_count(cls, kb_id):
        return cls.model.select().where(cls.model.kb_id == kb_id).count()

    @classmethod
    @DB.connection_context()
    def get_all_kb_doc_count(cls):
        result = {}
        rows = cls.model.select(cls.model.kb_id, fn.COUNT(cls.model.id).alias("count")).group_by(cls.model.kb_id)
        for row in rows:
            result[row.kb_id] = row.count
        return result

    @classmethod
    @DB.connection_context()
    def do_cancel(cls, doc_id):
        try:
            _, doc = DocumentService.get_by_id(doc_id)
            return doc.run == TaskStatus.CANCEL.value or doc.progress < 0
        except Exception:
            pass
        return False

    @classmethod
    @DB.connection_context()
    def knowledgebase_basic_info(cls, kb_id: str) -> dict[str, int]:
        # 取消：运行 == "2"
        cancelled = cls.model.select(fn.COUNT(1)).where((cls.model.kb_id == kb_id) & (cls.model.run == TaskStatus.CANCEL)).scalar()
        downloaded = cls.model.select(fn.COUNT(1)).where(cls.model.kb_id == kb_id, cls.model.source_type != "local").scalar()

        row = (
            cls.model.select(
                # 已完成：进度 == 1
                fn.COALESCE(fn.SUM(Case(None, [(cls.model.progress == 1, 1)], 0)), 0).alias("finished"),
                # 失败：进度 == -1
                fn.COALESCE(fn.SUM(Case(None, [(cls.model.progress == -1, 1)], 0)), 0).alias("failed"),
                # 处理：0 <= 进度 < 1
                fn.COALESCE(
                    fn.SUM(
                        Case(
                            None,
                            [
                                (((cls.model.progress == 0) | ((cls.model.progress > 0) & (cls.model.progress < 1))), 1),
                            ],
                            0,
                        )
                    ),
                    0,
                ).alias("processing"),
            )
            .where((cls.model.kb_id == kb_id) & ((cls.model.run.is_null(True)) | (cls.model.run != TaskStatus.CANCEL)))
            .dicts()
            .get()
        )

        return {"processing": int(row["processing"]), "finished": int(row["finished"]), "failed": int(row["failed"]), "cancelled": int(cancelled), "downloaded": int(downloaded)}

    @classmethod
    def run(cls, tenant_id: str, doc: dict, kb_table_num_map: dict):
        """根据文档配置选择任务创建方式。

        自定义 pipeline 文档进入 queue_dataflow；普通文档先通过 doc_id 查询
        File2Document/Document 得到对象存储地址，再由 queue_tasks 按页或按行拆分任务。
        这里仍然只是在“生产任务”，真正解析由独立 Task Executor 完成。

        参数：
            tenant_id: 文档所属租户；用于定位 ES 索引与模型配置。
            doc: Document.to_dict() 结果，至少包含 id/kb_id/parser_id/parser_config/location。
            kb_table_num_map: 同一次批量 ingest 共享的表格知识库计数缓存，避免重复查库。
        """
        from api.db.services.task_service import queue_dataflow, queue_tasks
        from api.db.services.file2document_service import File2DocumentService

        doc["tenant_id"] = tenant_id
        doc_parser = doc.get("parser_id", ParserType.NAIVE)
        if doc_parser == ParserType.TABLE:
            kb_id = doc.get("kb_id")
            if not kb_id:
                return
            if kb_id not in kb_table_num_map:
                count = DocumentService.count_by_kb_id(kb_id=kb_id, keywords="", run_status=[TaskStatus.DONE], types=[])
                kb_table_num_map[kb_id] = count
                if kb_table_num_map[kb_id] <= 0:
                    KnowledgebaseService.delete_field_map(kb_id)
        # 分流点：有 pipeline_id 的文档交给画布定义的 Dataflow；否则走内置 parser。
        # 两条路径都会先创建持久化 Task，再通过 Redis 唤醒独立 Worker。
        if doc.get("pipeline_id", ""):
            queue_dataflow(tenant_id, flow_id=doc["pipeline_id"], task_id=get_uuid(), doc_id=doc["id"])
        else:
            # 根据 doc_id 找到对象存储的 bucket 和 object key。
            bucket, name = File2DocumentService.get_storage_address(doc_id=doc["id"])
            # 创建 MySQL Task 记录并把未完成任务投递到 Redis Stream。
            queue_tasks(doc, bucket, name, 0)


def queue_raptor_o_graphrag_tasks(sample_doc, ty, priority, fake_doc_id="", doc_ids=None):
    """您可以提供fake_doc_id来绕过知识库级别的任务限制。
    （可选）指定 doc_ids 列表以确定哪些文档参与该任务。"""
    if doc_ids is None:
        doc_ids = []
    assert ty in [
        "graphrag",
        "raptor",
        "mindmap",
        "wiki",
        "skill",
        # KB 范围的结构图合并任务类型（重建 dataset_graph 行）。
        "structure_graph",
        "structure_mindmap",
        "timeline",
        "session_graph",
        "session_essence",
        "structure",
    ], f"unsupported task type '{ty}'"

    chunking_config = DocumentService.get_chunking_config(sample_doc["id"])
    hasher = xxhash.xxh64()
    for field in sorted(chunking_config.keys()):
        hasher.update(str(chunking_config[field]).encode("utf-8"))

    def new_task():
        return {
            "id": get_uuid(),
            "doc_id": fake_doc_id,
            "from_page": MAXIMUM_TASK_PAGE_NUMBER,
            "to_page": MAXIMUM_TASK_PAGE_NUMBER,
            "task_type": ty,
            "progress_msg": datetime.now().strftime("%H:%M:%S") + " created task " + ty,
            "begin_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    task = new_task()
    for field in ["doc_id", "from_page", "to_page"]:
        hasher.update(str(task.get(field, "")).encode("utf-8"))
    hasher.update(ty.encode("utf-8"))
    task["digest"] = hasher.hexdigest()
    bulk_insert_into_db(Task, [task], True)

    task["doc_ids"] = doc_ids
    DocumentService.begin2parse(task["doc_id"], keep_progress=True)
    assert REDIS_CONN.queue_product(settings.get_svr_queue_name(priority, ty), message=task), "Can't access Redis. Please check the Redis' status."
    return task["id"]


def queue_per_doc_raptor_task(doc, priority):
    """对文档范围的 RAPTOR 任务进行排队。

    与 :func:`queue_raptor_o_graphrag_tasks` 不同（这是 KB 范围的
    和用途``GRAPH_RAPTOR_FAKE_DOC_ID`` as the task's ``doc_id`` so it
    fans out across the dataset). Here the task's `ZXQKEEP00 610007ZXQ` is the real
    document id, so ``TaskHandler._run_raptor`` runs only on this doc's
    chunks and the RAPTOR summaries it produces are scoped to this doc.

    Triggered automatically at the tail of standard chunking when the
    doc's ``parser_config["raptor"]["use_raptor"]``是真的。否
    跨任务重复数据删除——在一个分块任务执行中，这个助手是
    最多调用一次，这是调用者需要的唯一不变量。"""
    chunking_config = DocumentService.get_chunking_config(doc["id"])
    hasher = xxhash.xxh64()
    for field in sorted(chunking_config.keys()):
        hasher.update(str(chunking_config[field]).encode("utf-8"))

    task = {
        "id": get_uuid(),
        "doc_id": doc["id"],
        "from_page": MAXIMUM_TASK_PAGE_NUMBER,
        "to_page": MAXIMUM_TASK_PAGE_NUMBER,
        "task_type": "raptor",
        "progress_msg": datetime.now().strftime("%H:%M:%S") + " created task raptor",
        "begin_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    for field in ["doc_id", "from_page", "to_page"]:
        hasher.update(str(task[field]).encode("utf-8"))
    hasher.update(b"raptor")
    task["digest"] = hasher.hexdigest()
    bulk_insert_into_db(Task, [task], True)

    # Redis消息携带“`doc_ids`”给下游消费者
    # （TaskHandler._run_raptor 读取）。与假文档相同
    # 路径's convention so we don't 必须对执行器进行特殊处理。
    task["doc_ids"] = [doc["id"]]
    assert REDIS_CONN.queue_product(
        settings.get_svr_queue_name(priority, "raptor"),
        message=task,
    ), "Can't access Redis. Please check the Redis' status."
    return task["id"]


# 用于真正排队任务积压的短期按优先级缓存，因此
# 每个文档进度同步不会对每个文档发出 COUNT 查询
# 每个周期。按优先级键入（无表示 "all priorities"）。
_PENDING_TASK_COUNT_CACHE = {}
_PENDING_TASK_COUNT_TTL_SECONDS = 3.0


def get_pending_task_count(priority=None):
    """统计真正仍在等待处理的任务。

    当任务尚未开始时（进度 == 0），则计为 "waiting"，并且
    它的文件既没有被取消也没有失败。我们故意做NOT要求
    文档为 RUNNING，进度在 [0, 1)：特殊任务 (graphrag/
    raptor/mindmap）通过“`begin2parse(keep_progress=True)`` while the
    document's own progress may already be 1, so requiring RUNNING/progress<1
    would undercount them and wrongly drop the cap to 0 while Redis lag is still
    non-zero. Only cancelled documents (run == CANCEL) and failed ones
    (progress < 0) are excluded, plus soft-deleted (invalid) documents.

    When ``priority`”进行排队，仅计算以该优先级排队的任务，
    因此该数字与它所限制的每个优先级 Redis 队列保持一致。

    当无法确定计数时返回 None，因此调用者可以回退
    原始 Redis 流滞后。"""
    now = monotonic()
    cached = _PENDING_TASK_COUNT_CACHE.get(priority)
    if cached and cached.get("expire_at", 0.0) > now:
        return cached["value"]
    try:
        query = (
            Task.select(fn.COUNT(Task.id))
            .join(Document, on=(Task.doc_id == Document.id))
            .where((Task.progress == 0) & ((Document.run.is_null(True)) | (Document.run != TaskStatus.CANCEL.value)) & (Document.progress >= 0) & (Document.status == StatusEnum.VALID.value))
        )
        if priority is not None:
            query = query.where(Task.priority == priority)
        count = int(query.scalar() or 0)
    except Exception:
        logging.exception("get_pending_task_count 失败")
        return None
    _PENDING_TASK_COUNT_CACHE[priority] = {"value": count, "expire_at": now + _PENDING_TASK_COUNT_TTL_SECONDS}
    return count


def get_queue_length(priority, suffix="common"):
    """返回处理队列中前面有多少任务。

    Redis 流消费者组“`lag`”对每条未包含的消息进行计数
    尚未传递给任务执行者，包括其任务已执行的消息
    已经 cancelled/stopped. 这些消息只会停止计数一次
    执行器碰巧读取了它们，因此在用户停止解析后，延迟可以
    无限期地膨胀，解析似乎永远挂起
    （"N tasks are ahead in the queue..."）。

    为了保持数字的真实性，原始延迟受到任务数量的限制
    那些确实仍在数据库中等待的数据，它可以自我修复
    工作被取消或完成的那一刻。"""
    group_info = REDIS_CONN.queue_info(settings.get_svr_queue_name(priority, suffix), SVR_CONSUMER_GROUP_NAME)
    lag = int(group_info.get("lag", 0) or 0) if group_info else 0

    # Redis 中没有任何内容排队：无论 DB 积压如何，答案都是 0，因此
    # 短路以避免每个进度同步周期出现 COUNT/JOIN。
    if lag <= 0:
        return 0

    pending = get_pending_task_count(priority)
    if pending is None:
        return lag
    return min(lag, pending)
