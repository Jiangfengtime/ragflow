#
#  Copyright 2026 The InfiniFlow Authors. All Rights Reserved.
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
import json
import logging
import os
import re

from api.db.db_models import Connector2Kb, Document, File, SyncLogs
from api.db.joint_services.tenant_model_service import get_composite_model_name_by_ids, resolve_model_config, resolve_model_id
from api.db.services.connector_service import Connector2KbService, SyncLogsService
from api.db.services.document_service import DocumentService, queue_raptor_o_graphrag_tasks
from api.db.services.file2document_service import File2DocumentService
from api.db.services.file_service import FileService
from api.db.services.knowledgebase_service import KnowledgebaseService, validate_dataset_embedding_models
from api.db.services.task_service import GRAPH_RAPTOR_FAKE_DOC_ID, TaskService
from api.db.services.tenant_model_service import TenantModelService
from api.db.services.user_service import TenantService, UserService, UserTenantService
from api.utils.api_utils import deep_merge, get_parser_config, remap_dictionary_keys, verify_embedding_availability
from common import settings
from common.constants import PAGERANK_FLD, FileSource, LLMType, RetCode, StatusEnum, TaskStatus
from common.misc_utils import thread_pool_exec, thread_pool_exec_long_time
from rag.advanced_rag.knowlege_compile.wiki import WIKI_PAGE_COMPILE_KWD

# KB 范围的结构图合并索引类型。每个（重新）构建“`dataset_graph`”
# 通过 ``rebuild_dataset_structure_graph_json`` 获取一种结构类型的
# 行；的
# task_type 等于 index_type，KB 任务 ID 列为“`<type>_task_id`”。
# 该值为可通过``_resolve_dataset_structure_kind``解析的友好类型
# （定义如下），“`structure`”除外，它是全部合并变体。
_STRUCTURE_INDEX_TYPE_TO_KIND = {
    "structure_graph": "graph",
    "structure_mindmap": "mindmap",
    "timeline": "timeline",
    "session_graph": "session_graph",
    "session_essence": "session_essence",
    "structure": None,  # merge-all：重建每个数据集合并类型
}
_STRUCTURE_INDEX_TYPES = frozenset(_STRUCTURE_INDEX_TYPE_TO_KIND)

_VALID_INDEX_TYPES = {"graph", "raptor", "mindmap", "wiki", "skill"} | set(_STRUCTURE_INDEX_TYPES)

_INDEX_TYPE_TO_TASK_TYPE = {
    "graph": "structure_graph",
    "raptor": "raptor",
    "mindmap": "structure_mindmap",
    "wiki": "wiki",
    "skill": "skill",
    # 结构合并类型携带自己的 task_type (== index_type)，因此
    # 执行器可以从任务体中解析出要合并哪种类型。
    **{t: t for t in _STRUCTURE_INDEX_TYPES},
}

_INDEX_TYPE_TO_TASK_ID_FIELD = {
    "graph": "graphrag_task_id",
    "raptor": "raptor_task_id",
    "mindmap": "mindmap_task_id",
    "wiki": "wiki_task_id",
    "skill": "skill_task_id",
    **{t: f"{t}_task_id" for t in _STRUCTURE_INDEX_TYPES},
}

_INDEX_TYPE_TO_DISPLAY_NAME = {
    "graph": "Graph",
    "raptor": "RAPTOR",
    "mindmap": "Mindmap",
    "wiki": "Wiki",
    "skill": "Skill",
    "structure_graph": "Structure Graph",
    "structure_mindmap": "Structure Mindmap",
    "timeline": "Timeline",
    "session_graph": "Session Graph",
    "session_essence": "Session Essence",
    "structure": "Structure",
}


async def create_dataset(tenant_id: str, req: dict):
    """创建一个新数据集。

    ：参数 tenant_id：租户 ID
    :param req: 数据集创建请求
    :return: (成功,结果)或(成功,error_message)"""
    # Knowledgebase 只保存知识库级配置。创建时没有 Document、Task 或检索索引；
    # 索引会在第一批文档解析时由任务执行器按需创建。
    # 提取附加参数的 ext 字段
    ext_fields = req.pop("ext", {})

    # 将 auto_metadata_config（如果提供）映射到 parser_config 结构
    auto_meta = req.pop("auto_metadata_config", {})
    if auto_meta:
        parser_cfg = req.get("parser_config") or {}
        fields = []
        for f in auto_meta.get("fields", []):
            fields.append(
                {
                    "name": f.get("name", ""),
                    "type": f.get("type", ""),
                    "description": f.get("description"),
                    "examples": f.get("examples"),
                    "restrict_values": f.get("restrict_values", False),
                }
            )
        parser_cfg["metadata"] = fields
        parser_cfg["enable_metadata"] = auto_meta.get("enabled", True)
        req["parser_config"] = parser_cfg
    req.update(ext_fields)

    e, create_dict = KnowledgebaseService.create_with_name(name=req.pop("name", None), tenant_id=tenant_id, parser_id=req.pop("parser_id", None), **req)

    if not e:
        return False, create_dict

    # 插入嵌入模型(embd id)
    ok, t = TenantService.get_by_id(tenant_id)
    if not ok:
        return False, "Tenant not found"
    if not create_dict.get("embd_id"):
        create_dict["embd_id"] = t.embd_id
    else:
        ok, err = verify_embedding_availability(create_dict["embd_id"], tenant_id)
        if not ok:
            return False, err

    if not KnowledgebaseService.save(**create_dict):
        return False, "Failed to save dataset"
    ok, k = KnowledgebaseService.get_by_id(create_dict["id"])
    if not ok:
        return False, "Dataset created failed"
    response_data = remap_dictionary_keys(k.to_dict())
    logging.info(
        "知识库已创建 租户ID=%s 知识库ID=%s 解析器ID=%s 向量模型ID=%s 权限=%s",
        tenant_id,
        create_dict["id"],
        create_dict.get("parser_id"),
        create_dict.get("embd_id"),
        create_dict.get("permission"),
    )
    return True, response_data


def _delete_datasets_sync(tenant_id: str, ids: list = None, delete_all: bool = False):
    """删除数据集。

    ：参数 tenant_id：租户 ID
    :param ids: 数据集列表 IDs
    :param delete_all: 是否删除租户的所有数据集（如果没有提供ids）
    :return: (成功,结果)或(成功,error_message)"""
    kb_id_instance_pairs = []
    if not ids:
        if not delete_all:
            return True, {"success_count": 0}
        else:
            ids = [kb.id for kb in KnowledgebaseService.query(tenant_id=tenant_id)]

    logging.info("知识库删除已开始 租户ID=%s 知识库ID列表=%s 是否全部删除=%s", tenant_id, ids, delete_all)

    error_kb_ids = []
    for kb_id in ids:
        kb = KnowledgebaseService.get_or_none(id=kb_id, tenant_id=tenant_id)
        if kb is None:
            error_kb_ids.append(kb_id)
            continue
        kb_id_instance_pairs.append((kb_id, kb))
    if len(error_kb_ids) > 0:
        return False, f"""User '{tenant_id}' lacks permission for datasets: '{", ".join(error_kb_ids)}'"""

    errors = []
    success_count = 0
    for kb_id, kb in kb_id_instance_pairs:
        # 删除知识库是跨存储清理：先处理 Document/对象关系，再删除 Doc Store 索引，
        # 最后删除 Knowledgebase、连接器和同步日志。中途失败会进入 errors 返回给调用方。
        # 在接触其文档之前取消此数据集的排队同步。
        # 任务仅在 SCHEDULE 时才被拾取，因此取消停止
        # 每一次尚未开始的运行；已在运行的同步不是
        # 可中断，这就是下面的搁浅行扫描所涵盖的内容。
        SyncLogsService.filter_update(
            [SyncLogs.kb_id == kb_id, SyncLogs.status.in_([TaskStatus.SCHEDULE, TaskStatus.RUNNING])],
            {"status": TaskStatus.CANCEL},
        )

        for doc in DocumentService.query(kb_id=kb_id):
            if not DocumentService.remove_document(doc, tenant_id):
                errors.append(f"Remove document '{doc.id}' error for dataset '{kb_id}'")
                continue
            f2d = File2DocumentService.get_by_document_id(doc.id)
            if f2d:
                FileService.filter_delete(
                    [
                        File.source_type == FileSource.KNOWLEDGEBASE,
                        File.id == f2d[0].file_id,
                    ]
                )
            else:
                # 正常上传通过 FileService.add_file_from_kb 创建 File2Document 行。
                # 缺失行通常意味着 stale/partial 数据（e.g. 链接已删除，
                # 插入后文件链接或旧行失败）。删除仍在继续。
                logging.warning(
                    "delete_datasets：数据集%s中的文档%s没有File2Document行；跳过链接文件删除",
                    doc.id,
                    kb_id,
                )
            File2DocumentService.delete_by_document_id(doc.id)
        FileService.filter_delete([File.source_type == FileSource.KNOWLEDGEBASE, File.type == "folder", File.name == kb.name])

        # 该数据集的删除索引
        try:
            from rag.nlp import search

            idxnm = search.index_name(kb.tenant_id)
            settings.docStoreConn.delete_idx(idxnm, kb_id)
        except Exception as e:
            errors.append(f"Failed to drop index for dataset {kb_id}: {e}")

        if not KnowledgebaseService.delete_by_id(kb_id):
            errors.append(f"Delete dataset error for {kb_id}")
            continue

        # 仅在数据集真正消失后才取消数据源连接，因此
        # 上面删除失败，留下的数据集仍然链接并且仍然
        # 可同步。这些行留在后面，使连接器调度程序保持排队
        # 与不再解析的 kb_id 同步，以及任何文档，例如
        # 运行写入的数据比其数据集的寿命更长 - 稍后报告的不可见行
        # 针对接下来链接的任何数据集的跨 KB id 冲突。
        Connector2KbService.filter_delete([Connector2Kb.kb_id == kb_id])
        SyncLogsService.filter_delete([SyncLogs.kb_id == kb_id])

        # 扫描每个文档循环看不到的任何内容，包括行
        # 由删除开始时已在运行的同步写入。
        stranded = DocumentService.filter_delete([Document.kb_id == kb_id])
        if stranded:
            logging.warning("delete_datasets：删除了数据集 %s 的 %s 搁浅文档行", stranded, kb_id)

        success_count += 1
        logging.info("单个知识库删除完成 租户ID=%s 知识库ID=%s", tenant_id, kb_id)

    if not errors:
        logging.info("知识库删除完成 租户ID=%s 成功数=%d 失败数=0", tenant_id, success_count)
        return True, {"success_count": success_count}

    error_message = f"Successfully deleted {success_count} datasets, {len(errors)} failed. Details: {'; '.join(errors)[:128]}..."
    if success_count == 0:
        logging.warning("知识库删除完成 租户ID=%s 成功数=0 失败数=%d", tenant_id, len(errors))
        return False, error_message

    logging.warning("知识库删除完成 租户ID=%s 成功数=%d 失败数=%d", tenant_id, success_count, len(errors))
    return True, {"success_count": success_count, "errors": errors[:5]}


async def delete_datasets(tenant_id: str, ids: list = None, delete_all: bool = False):
    return await thread_pool_exec_long_time(_delete_datasets_sync, tenant_id, ids, delete_all)


def get_dataset(dataset_id: str, tenant_id: str):
    """获取单个数据集。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    :return: (成功,结果)或(成功,error_message)"""
    if not dataset_id:
        return False, 'Lack of "Dataset ID"'

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, f"User '{tenant_id}' lacks permission for dataset '{dataset_id}'"

    ok, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not ok:
        return False, "Invalid Dataset ID"

    response_data = remap_dictionary_keys(kb.to_dict())
    response_data["size"] = DocumentService.get_total_size_by_kb_id(dataset_id)
    response_data["connectors"] = list(Connector2KbService.list_connectors(dataset_id))
    return True, response_data


def get_ingestion_summary(dataset_id: str, tenant_id: str):
    """获取数据集的摄取摘要。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    :return: (成功,结果)或(成功,error_message)"""
    if not dataset_id:
        return False, 'Lack of "Dataset ID"'

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, f"User '{tenant_id}' lacks permission for dataset '{dataset_id}'"

    ok, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not ok:
        return False, "Invalid Dataset ID"

    status = DocumentService.get_parsing_status_by_kb_ids([dataset_id]).get(dataset_id, {})
    return True, {
        "doc_num": kb.doc_num,
        "chunk_num": kb.chunk_num,
        "token_num": kb.token_num,
        "status": status,
    }


async def update_dataset(tenant_id: str, dataset_id: str, req: dict):
    """更新数据集。

    ：参数 tenant_id：租户 ID
    ：参数dataset_id：数据集ID
    :param req: 数据集更新请求
    :return: (成功,结果)或(成功,error_message)"""
    if not req:
        return False, "no properties were modified"

    kbs = KnowledgebaseService.get_kb_by_id(dataset_id, tenant_id)
    if not kbs:
        return False, f"User '{tenant_id}' lacks permission for dataset '{dataset_id}'"

    kb = KnowledgebaseService.get_or_none(id=dataset_id)
    if kb is None:
        return False, "Invalid Dataset ID"

    # 提取附加参数的 ext 字段
    ext_fields = req.pop("ext", {})

    # 将 auto_metadata_config 映射到 parser_config（如果存在）
    auto_meta = req.pop("auto_metadata_config", {})
    if auto_meta:
        parser_cfg = req.get("parser_config") or {}
        fields = []
        for f in auto_meta.get("fields", []):
            fields.append(
                {
                    "name": f.get("name", ""),
                    "type": f.get("type", ""),
                    "description": f.get("description"),
                    "examples": f.get("examples"),
                    "restrict_values": f.get("restrict_values", False),
                }
            )
        parser_cfg["metadata"] = fields
        parser_cfg["enable_metadata"] = auto_meta.get("enabled", True)
        req["parser_config"] = parser_cfg

    # 将 ext 字段与 req 合并
    req.update(ext_fields)

    # 从请求中提取连接器
    connectors = []
    if "connectors" in req:
        connectors = req["connectors"]
        del req["connectors"]

    if req.get("parser_config"):
        # 将 parent_child 配置压平为 children_delimiter 用于执行层
        pc = req["parser_config"].get("parent_child", {})
        if pc.get("use_parent_child"):
            req["parser_config"]["children_delimiter"] = pc.get("children_delimiter", "\n")
            req["parser_config"]["enable_children"] = pc.get("use_parent_child", True)
        else:
            req["parser_config"]["children_delimiter"] = ""
            req["parser_config"]["enable_children"] = False
            req["parser_config"]["parent_child"] = {}

        parser_config = req["parser_config"]
        req_ext_fields = parser_config.pop("ext", {})
        parser_config.update(req_ext_fields)
        req["parser_config"] = deep_merge(kb.parser_config, parser_config)

    if (chunk_method := req.get("parser_id")) and chunk_method != kb.parser_id:
        if not req.get("parser_config"):
            req["parser_config"] = get_parser_config(chunk_method, None)
    elif "parser_config" in req and not req["parser_config"]:
        del req["parser_config"]

    if kb.pipeline_id and req.get("parser_id") and not req.get("pipeline_id"):
        # 转用 parser_id，删除旧的 pipeline_id
        req["pipeline_id"] = ""

    if "name" in req and req["name"].lower() != kb.name.lower():
        exists = KnowledgebaseService.get_or_none(name=req["name"], tenant_id=kb.tenant_id, status=StatusEnum.VALID.value)
        if exists:
            return False, f"Dataset name '{req['name']}' already exists"

    if "embd_id" in req:
        if not req["embd_id"]:
            req["embd_id"] = kb.embd_id
        ok, err = verify_embedding_availability(req["embd_id"], tenant_id)
        if not ok:
            return False, err
        ok, _ = TenantModelService.get_by_id(req["embd_id"])
        if ok:
            req["tenant_embd_id"] = req["embd_id"]
        else:
            req["tenant_embd_id"] = resolve_model_id(tenant_id, LLMType.EMBEDDING, req["embd_id"])

    if "pagerank" in req and req["pagerank"] != kb.pagerank:
        if os.environ.get("DOC_ENGINE", "elasticsearch") == "infinity":
            return False, "'pagerank' can only be set when doc_engine is elasticsearch"

        if req["pagerank"] > 0:
            from rag.nlp import search

            settings.docStoreConn.update({"kb_id": kb.id}, {PAGERANK_FLD: req["pagerank"]}, search.index_name(kb.tenant_id), kb.id)
        else:
            # Elasticsearch 要求 PAGERANK_FLD 为非零！
            from rag.nlp import search

            settings.docStoreConn.update({"exists": PAGERANK_FLD}, {"remove": PAGERANK_FLD}, search.index_name(kb.tenant_id), kb.id)
    req.pop("parse_type", None)

    if not KnowledgebaseService.update_by_id(kb.id, req):
        return False, "Update dataset error.(Database error)"

    # 修改 parser/embedding 配置只影响后续解析。已存在的 Chunk 不会在这里自动重建；
    # 若要应用新配置，需要由文档 ingest/rerun 链路显式重解析。
    logging.info(
        "知识库配置已更新 租户ID=%s 知识库ID=%s 变更字段=%s 解析配置是否变化=%s 向量模型是否变化=%s",
        tenant_id,
        dataset_id,
        sorted(req.keys()),
        "parser_id" in req or "parser_config" in req or "pipeline_id" in req,
        "embd_id" in req or "tenant_embd_id" in req,
    )

    ok, k = KnowledgebaseService.get_by_id(kb.id)
    if not ok:
        return False, "Dataset updated failed"

    # 将连接器链接到数据集
    errors = Connector2KbService.link_connectors(kb.id, [conn for conn in connectors], tenant_id)
    if errors:
        logging.error("链接 KB 错误：%s", errors)

    response_data = remap_dictionary_keys(k.to_dict())
    response_data["connectors"] = connectors
    return True, response_data


def list_datasets(tenant_id: str, args: dict):
    """列出数据集。

    ：参数 tenant_id：租户 ID
    :param args: 查询参数
    :return: (成功,结果)或(成功,error_message)"""
    kb_id = args.get("id")
    kb_ids = args.get("ids")
    name = args.get("name")
    page = int(args.get("page", 1))
    page_size = int(args.get("page_size", 30))
    ext_fields = args.get("ext", {})
    parser_id = ext_fields.get("parser_id")
    keywords = ext_fields.get("keywords", "")
    orderby = args.get("orderby", "create_time")
    desc_arg = args.get("desc", "true")
    if isinstance(desc_arg, str):
        desc = desc_arg.lower() != "false"
    elif isinstance(desc_arg, bool):
        desc = desc_arg
    else:
        # 未知类型，默认为 True
        desc = True

    if kb_id and kb_ids:
        return False, f"Should not provide both 'id':{kb_id} and 'ids'{kb_ids}"
    if kb_id:
        kbs = KnowledgebaseService.get_kb_by_id(kb_id, tenant_id)
        if not kbs:
            return False, f"User '{tenant_id}' lacks permission for dataset '{kb_id}'"

    if name:
        kbs = KnowledgebaseService.get_kb_by_name(name, tenant_id)
        if not kbs:
            return False, f"User '{tenant_id}' lacks permission for dataset '{name}'"
    owner_ids = [owner_id.strip() for owner_id in ext_fields.get("owner_ids", []) if isinstance(owner_id, str) and owner_id.strip()]
    if owner_ids:
        tenants = TenantService.get_joined_tenants_by_user_id(tenant_id)
        allowed_tenant_ids = {m["tenant_id"] for m in tenants}
        allowed_tenant_ids.add(tenant_id)
        tenant_ids = [owner_id for owner_id in owner_ids if owner_id in allowed_tenant_ids]
        query_user_id = tenant_id if tenant_id in tenant_ids else ""
    else:
        tenants = TenantService.get_joined_tenants_by_user_id(tenant_id)
        tenant_ids = [m["tenant_id"] for m in tenants]
        query_user_id = tenant_id
    if kb_ids:
        accessible_ids = KnowledgebaseService.get_accessible_ids([m["tenant_id"] for m in tenants], tenant_id, kb_ids)
        denied_ids = [kb_id for kb_id in kb_ids if kb_id not in accessible_ids]
        if denied_ids:
            logging.warning("用户 '%s' 缺乏数据集的权限：'%s'", tenant_id, ", ".join(denied_ids))
        kb_ids = [kb_id for kb_id in kb_ids if kb_id in accessible_ids]
        if not kb_ids:
            return True, {"data": [], "total": 0}
    kbs, total = KnowledgebaseService.get_list(tenant_ids, query_user_id, page, page_size, orderby, desc, kb_id, name, keywords, parser_id, kb_ids)
    users = UserService.get_by_ids([m["tenant_id"] for m in kbs])
    user_map = {m.id: m.to_dict() for m in users}

    ips_arg = args.get("include_parsing_status", False)
    if isinstance(ips_arg, str):
        include_parsing_status = ips_arg.lower() not in ("false", "0", "")
    elif isinstance(ips_arg, bool):
        include_parsing_status = ips_arg
    else:
        include_parsing_status = bool(ips_arg)

    status_by_kb = {}
    if include_parsing_status and kbs:
        status_by_kb = DocumentService.get_parsing_status_by_kb_ids([kb["id"] for kb in kbs])

    response_data_list = []
    for kb in kbs:
        user_dict = user_map.get(kb["tenant_id"], {})
        kb.update({"nickname": user_dict.get("nickname", ""), "tenant_avatar": user_dict.get("avatar", "")})
        if status_by_kb:
            kb["parsing_status"] = status_by_kb.get(kb["id"], {})
        response_data_list.append(remap_dictionary_keys(kb))

    embed_model_names = get_composite_model_name_by_ids([m["embedding_model"] for m in response_data_list])
    for response_data in response_data_list:
        response_data["embedding_model_name"] = embed_model_names.get(response_data["embedding_model"], "")
    return True, {"data": response_data_list, "total": total}


def list_dataset_filters(tenant_id: str):
    tenants = TenantService.get_joined_tenants_by_user_id(tenant_id)
    tenant_ids = [m["tenant_id"] for m in tenants]
    owners = KnowledgebaseService.get_owner_filter(tenant_ids, tenant_id)
    return True, {"filter": {"owner": owners}, "total": sum(owner["count"] for owner in owners)}


async def get_knowledge_graph(dataset_id: str, tenant_id: str):
    """获取数据集的知识图。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    :return: (成功,结果)或(成功,error_message)"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    req = {"kb_id": [dataset_id], "knowledge_graph_kwd": ["graph"]}

    obj = {"graph": {}, "mind_map": {}}
    from rag.nlp import search

    if not settings.docStoreConn.index_exist(search.index_name(kb.tenant_id), dataset_id):
        return True, obj
    sres = await settings.retriever.search(req, search.index_name(kb.tenant_id), [dataset_id])
    if not len(sres.ids):
        return True, obj

    for id in sres.ids[:1]:
        ty = sres.field[id]["knowledge_graph_kwd"]
        try:
            content_json = json.loads(sres.field[id]["content_with_weight"])
        except Exception:
            continue

        obj[ty] = content_json

    if "nodes" in obj["graph"]:
        obj["graph"]["nodes"] = sorted(obj["graph"]["nodes"], key=lambda x: x.get("pagerank", 0), reverse=True)[:256]
        if "edges" in obj["graph"]:
            node_id_set = {o["id"] for o in obj["graph"]["nodes"]}
            filtered_edges = [o for o in obj["graph"]["edges"] if o["source"] != o["target"] and o["source"] in node_id_set and o["target"] in node_id_set]
            obj["graph"]["edges"] = sorted(filtered_edges, key=lambda x: x.get("weight", 0), reverse=True)[:128]
    return True, obj


def delete_knowledge_graph(dataset_id: str, tenant_id: str):
    """删除数据集的知识图。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    :return: (成功,结果)或(成功,error_message)"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)
    from rag.graphrag.phase_markers import clear_phase_markers
    from rag.nlp import search

    settings.docStoreConn.delete({"knowledge_graph_kwd": ["graph", "subgraph", "entity", "relation", "community_report"]}, search.index_name(kb.tenant_id), dataset_id)
    # 擦除图形会使用于的任何阶段完成标记无效
    # 短路解决/恢复时的社区检测。
    clear_phase_markers(dataset_id)
    KnowledgebaseService.update_by_id(
        kb.id,
        {"graphrag_task_id": "", "graphrag_task_finish_at": None},
    )

    return True, True


def run_index(dataset_id: str, tenant_id: str, index_type: str):
    """为数据集运行索引任务 (graph/raptor/mindmap)。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    ：参数 index_type："graph"、"raptor"、"mindmap" 之一
    :return: (成功,结果)或(成功,error_message)"""
    if index_type not in _VALID_INDEX_TYPES:
        return False, f"Invalid index type '{index_type}'. Must be one of {sorted(_VALID_INDEX_TYPES)}"

    if not dataset_id:
        return False, 'Lack of "Dataset ID"'
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"

    ok, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not ok:
        return False, "Invalid Dataset ID"

    task_type = _INDEX_TYPE_TO_TASK_TYPE[index_type]
    task_id_field = _INDEX_TYPE_TO_TASK_ID_FIELD[index_type]
    display_name = _INDEX_TYPE_TO_DISPLAY_NAME[index_type]

    existing_task_id = getattr(kb, task_id_field, None)
    if existing_task_id:
        ok, task = TaskService.get_by_id(existing_task_id)
        if not ok:
            logging.warning(f"A valid {display_name} task id is expected for Dataset {dataset_id}")

        if task and task.progress not in [-1, 1]:
            return False, f"Task {existing_task_id} in progress with status {task.progress}. A {display_name} Task is already running."

    documents, _ = DocumentService.get_by_kb_id(
        kb_id=dataset_id,
        page_number=0,
        items_per_page=0,
        orderby="create_time",
        desc=False,
        keywords="",
        run_status=[],
        types=[],
        suffix=[],
    )
    # 禁用文档不得参与数据集级结构
    # 重建。让它们远离扇出任务本身；此次合并还
    # 应用数据库支持的过滤器作为深度防御。
    documents = [document for document in documents if str(document.get("status", "1")) != "0"]
    if not documents:
        return False, f"No documents in Dataset {dataset_id}"

    sample_document = documents[0]
    document_ids = [document["id"] for document in documents]

    task_id = queue_raptor_o_graphrag_tasks(sample_doc=sample_document, ty=task_type, priority=0, fake_doc_id=GRAPH_RAPTOR_FAKE_DOC_ID, doc_ids=list(document_ids))

    if not KnowledgebaseService.update_by_id(kb.id, {task_id_field: task_id}):
        logging.warning(f"Cannot save {task_id_field} for Dataset {dataset_id}")

    return True, {"task_id": task_id}


def trace_index(dataset_id: str, tenant_id: str, index_type: str):
    """跟踪数据集的索引任务 (graph/raptor/mindmap)。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    ：参数 index_type："graph"、"raptor"、"mindmap" 之一
    :return: (成功,结果)或(成功,error_message)"""
    if index_type not in _VALID_INDEX_TYPES:
        return False, f"Invalid index type '{index_type}'. Must be one of {sorted(_VALID_INDEX_TYPES)}"

    if not dataset_id:
        return False, 'Lack of "Dataset ID"'

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"

    ok, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not ok:
        return False, "Invalid Dataset ID"

    task_id_field = _INDEX_TYPE_TO_TASK_ID_FIELD[index_type]
    task_id = getattr(kb, task_id_field, None)
    if not task_id:
        return True, {}

    ok, task = TaskService.get_by_id(task_id)
    if not ok:
        return True, {}

    return True, task.to_dict()


def list_tags(dataset_id: str, tenant_id: str):
    """列出数据集的标签。

    ：参数 dataset_id：数据集 ID
    ：参数 tenant_id：租户 ID
    :return: (成功,结果)或(成功,error_message)"""
    if not dataset_id:
        return False, 'Lack of "Dataset ID"'

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"

    tenants = UserTenantService.get_tenants_by_user_id(tenant_id)
    tags = []
    for tenant in tenants:
        tags += settings.retriever.all_tags(tenant["tenant_id"], [dataset_id])
    return True, tags


def aggregate_tags(dataset_ids: list[str], tenant_id: str):
    """跨多个数据集聚合标签。

    :param dataset_ids: 数据集列表 IDs
    ：参数 tenant_id：租户 ID
    :return: (成功,结果)或(成功,error_message)"""
    if not dataset_ids:
        return False, 'Lack of "dataset_ids"'

    for dataset_id in dataset_ids:
        if not KnowledgebaseService.accessible(dataset_id, tenant_id):
            return False, f"No authorization for dataset '{dataset_id}'"

    dataset_ids_by_tenant = {}
    for dataset_id in dataset_ids:
        ok, kb = KnowledgebaseService.get_by_id(dataset_id)
        if not ok:
            return False, f"Invalid Dataset ID '{dataset_id}'"
        dataset_ids_by_tenant.setdefault(kb.tenant_id, []).append(dataset_id)

    merged = {}
    for kb_tenant_id, kb_ids in dataset_ids_by_tenant.items():
        for tag, count in settings.retriever.all_tags(kb_tenant_id, kb_ids):
            merged[tag] = merged.get(tag, 0) + count

    return True, [{"value": tag, "count": count} for tag, count in merged.items()]


def get_flattened_metadata(dataset_ids: list[str], tenant_id: str):
    """获取数据集的扁平化元数据。

    :param dataset_ids: 数据集列表 IDs
    ：参数 tenant_id：租户 ID
    :return: (成功,结果)或(成功,error_message)"""
    if not dataset_ids:
        return False, 'Lack of "dataset_ids"'

    for dataset_id in dataset_ids:
        if not KnowledgebaseService.accessible(dataset_id, tenant_id):
            return False, f"No authorization for dataset '{dataset_id}'"

    from api.db.services.doc_metadata_service import DocMetadataService

    return True, DocMetadataService.get_flatted_meta_by_kbs(dataset_ids)


def get_auto_metadata(dataset_id: str, tenant_id: str):
    """获取数据集的自动元数据配置。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    :return: (成功,结果)或(成功,error_message)"""
    kb = KnowledgebaseService.get_or_none(id=dataset_id, tenant_id=tenant_id)
    if kb is None:
        return False, f"User '{tenant_id}' lacks permission for dataset '{dataset_id}'"
    parser_cfg = kb.parser_config or {}
    return True, {"metadata": parser_cfg.get("metadata") or [], "built_in_metadata": parser_cfg.get("built_in_metadata") or []}


async def update_auto_metadata(dataset_id: str, tenant_id: str, cfg: dict):
    """更新数据集的自动元数据配置。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    :param cfg: 自动元数据配置
    :return: (成功,结果)或(成功,error_message)"""
    kb = KnowledgebaseService.get_or_none(id=dataset_id, tenant_id=tenant_id)
    if kb is None:
        return False, f"User '{tenant_id}' lacks permission for dataset '{dataset_id}'"

    parser_cfg = kb.parser_config or {}
    parser_cfg["metadata"] = cfg.get("metadata")
    parser_cfg["built_in_metadata"] = cfg.get("built_in_metadata")

    if not KnowledgebaseService.update_by_id(kb.id, {"parser_config": parser_cfg}):
        return False, "Update auto-metadata error.(Database error)"

    return True, cfg


def delete_tags(dataset_id: str, tenant_id: str, tags: list[str]):
    """从数据集中删除标签。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    :param Tags: 要删除的标签列表
    :return: (成功,结果)或(成功,error_message)"""
    if not dataset_id:
        return False, 'Lack of "Dataset ID"'

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"

    ok, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not ok:
        return False, "Invalid Dataset ID"

    from rag.nlp import search

    for t in tags:
        updated = settings.docStoreConn.update({"tag_kwd": t, "kb_id": [dataset_id]}, {"remove": {"tag_kwd": t}}, search.index_name(kb.tenant_id), dataset_id)
        if callable(getattr(settings.docStoreConn, "db_type", None)) and settings.docStoreConn.db_type() == "gaussdb" and not updated:
            return False, "Failed to update dataset tags in document store"

    return True, {}


def list_ingestion_logs(
    dataset_id: str,
    tenant_id: str,
    page: int,
    page_size: int,
    orderby: str,
    desc: bool,
    operation_status: list = None,
    create_date_from: str = None,
    create_date_to: str = None,
    log_type: str = "dataset",
    keywords: str = None,
):
    """列出数据集的摄取日志。

    ：参数 dataset_id：数据集 ID
    ：参数 tenant_id：租户 ID
    :param page: 页码
    :param page_size: 每页的项目
    :param orderby: 按字段排序
    :param desc: 降序
    :param operation_status: 按运行状态过滤
    :param create_date_from: 过滤器开始日期
    ：参数 create_date_to：过滤器结束日期
    ：参数 log_type："dataset" 或 "file"
    :param keywords: 文件日志的搜索关键字
    :return: (成功,结果)或(成功,error_message)"""
    if not dataset_id:
        return False, 'Lack of "Dataset ID"'

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"

    from api.db.services.pipeline_operation_log_service import PipelineOperationLogService

    allowed_log_types = {"dataset", "file"}
    if log_type not in allowed_log_types:
        logging.warning(
            "list_ingestion_logs 无效 log_type：dataset_id=%s 租户ID=%s log_type=%s",
            dataset_id,
            tenant_id,
            log_type,
        )
        return False, 'Invalid "log_type", expected "dataset" or "file"'

    logging.info(
        "list_ingestion_logs: dataset_id=%s 租户ID=%s log_type=%s 页=%s page_size=%s",
        dataset_id,
        tenant_id,
        log_type,
        page,
        page_size,
    )

    if log_type == "file":
        logs, total = PipelineOperationLogService.get_file_logs_by_kb_id(dataset_id, page, page_size, orderby, desc, keywords, operation_status or [], None, None, create_date_from, create_date_to)
    else:
        logs, total = PipelineOperationLogService.get_dataset_logs_by_kb_id(dataset_id, page, page_size, orderby, desc, operation_status or [], create_date_from, create_date_to, keywords)
    return True, {"total": total, "logs": logs}


def get_ingestion_log(dataset_id: str, tenant_id: str, log_id: str):
    """获取单个摄取日志。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    ：参数 log_id：日志 ID
    :return: (成功,结果)或(成功,error_message)"""
    if not dataset_id:
        return False, 'Lack of "Dataset ID"'

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"

    from api.db.services.pipeline_operation_log_service import PipelineOperationLogService

    # 返回完整记录（包括`dsl`）所以前端dataflow-result
    # 页面可以渲染管道时间轴和块。文件级字段集
    # 是数据集级字段的超集，因此对两者都有效
    # 数据集级 (graph/raptor/mindmap) 和每个文件日志。
    fields = PipelineOperationLogService.get_file_logs_fields()
    log = PipelineOperationLogService.model.select(*fields).where((PipelineOperationLogService.model.id == log_id) & (PipelineOperationLogService.model.kb_id == dataset_id)).first()
    if not log:
        return False, "Log not found"

    result = log.to_dict()
    # 此处明确：数据流结果页面需要完整的 DSL 有效负载
    # 重建时间线和右侧解析器视图。一些序列化路径
    # 可以省略 Peewee 模型字典中的 JSON 字段，因此请将其保留在此处。
    result["dsl"] = log.dsl or {}
    return True, result


def delete_index(dataset_id: str, tenant_id: str, index_type: str, wipe: bool = True):
    """删除数据集的索引任务（graph/raptor/mindmap）。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    ：参数 index_type："graph"、"raptor"、"mindmap" 之一
    ：参数擦除：当为真（默认）时，持久的人工制品（图形行，
        raptor 摘要）已从文档存储中删除，并且任何 GraphRAG
        阶段完成标记被清除。  传递 False 取消
        运行任务，同时保持先前的进度，以便可以恢复。
    :return: (成功,结果)或(成功,error_message)"""
    if index_type not in _VALID_INDEX_TYPES:
        return False, f"Invalid index type '{index_type}'. Must be one of {sorted(_VALID_INDEX_TYPES)}"

    if not dataset_id:
        return False, 'Lack of "Dataset ID"'

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"

    ok, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not ok:
        return False, "Invalid Dataset ID"

    task_id_field = _INDEX_TYPE_TO_TASK_ID_FIELD[index_type]
    task_finish_at_field = f"{task_id_field.replace('_task_id', '_task_finish_at')}"
    task_id = getattr(kb, task_id_field, None)

    logging.info("delete_index：数据集=%s index_type=%s擦除=%s", dataset_id, index_type, wipe)

    if task_id:
        from rag.utils.redis_conn import REDIS_CONN

        try:
            REDIS_CONN.set(f"{task_id}-cancel", "x")
        except Exception as e:
            logging.exception(e)
        TaskService.delete_by_id(task_id)

    if wipe and index_type == "graph":
        from rag.graphrag.phase_markers import clear_phase_markers
        from rag.nlp import search

        settings.docStoreConn.delete({"knowledge_graph_kwd": ["graph", "subgraph", "entity", "relation", "community_report"]}, search.index_name(kb.tenant_id), dataset_id)
        # 擦除图形会使用于的任何阶段完成标记无效
        # 短路解决/恢复时的社区检测。
        clear_phase_markers(dataset_id)
        logging.info("delete_index：清除了数据集 = %s 的 GraphRAG 伪影和相位标记", dataset_id)
    elif wipe and index_type == "raptor":
        from rag.nlp import search

        settings.docStoreConn.delete({"raptor_kwd": ["raptor"]}, search.index_name(kb.tenant_id), dataset_id)
    elif wipe and index_type == "skill":
        from rag.nlp import search

        settings.docStoreConn.delete({"compile_kwd": ["skill", "skill_all"]}, search.index_name(kb.tenant_id), dataset_id)
    elif wipe and index_type in _STRUCTURE_INDEX_TYPES:
        from rag.nlp import search

        # 擦除合并的 KB 范围元数据和 entity/relation 行
        # 请求的种类（所有种类为合并所有 "structure" 类型）。这
        # 每个文档 entity/relation 合并读取的行保持不变。
        friendly = _STRUCTURE_INDEX_TYPE_TO_KIND.get(index_type)
        resolved_kind = _resolve_dataset_structure_kind(friendly) if friendly else None
        conditions = [
            {"knowledge_graph_kwd": ["dataset_graph"]},
            {"knowledge_graph_kwd": ["entity", "relation"], "scope_kwd": ["dataset"]},
        ]
        for condition in conditions:
            if resolved_kind:
                condition["compilation_template_kind_kwd"] = [resolved_kind]
            settings.docStoreConn.delete(condition, search.index_name(kb.tenant_id), dataset_id)

    KnowledgebaseService.update_by_id(kb.id, {task_id_field: "", task_finish_at_field: None})
    return True, {}


def rename_tag(dataset_id: str, tenant_id: str, from_tag: str, to_tag: str):
    """重命名数据集中的标签。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    :param from_tag: 原始标签名称
    :param to_tag: 新标签名称
    :return: (成功,结果)或(成功,error_message)"""
    if not dataset_id:
        return False, 'Lack of "Dataset ID"'

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"

    ok, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not ok:
        return False, "Invalid Dataset ID"

    from rag.nlp import search

    updated = settings.docStoreConn.update(
        {"tag_kwd": from_tag, "kb_id": [dataset_id]}, {"remove": {"tag_kwd": from_tag.strip()}, "add": {"tag_kwd": to_tag}}, search.index_name(kb.tenant_id), dataset_id
    )
    if callable(getattr(settings.docStoreConn, "db_type", None)) and settings.docStoreConn.db_type() == "gaussdb" and not updated:
        return False, "Failed to update dataset tags in document store"

    return True, {"from": from_tag, "to": to_tag}


async def search(dataset_id: str, tenant_id: str, req: dict):
    """数据集中的 Search（检索测试）。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    :param req: 搜索请求
    :return: (成功,结果)或(成功,error_message)"""
    from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type
    from api.db.services.doc_metadata_service import DocMetadataService
    from api.db.services.llm_service import LLMBundle
    from api.db.services.search_service import SearchService
    from api.db.services.user_service import UserTenantService
    from common.constants import LLMType
    from common.metadata_utils import apply_meta_data_filter
    from rag.app.tag import label_question
    from rag.prompts.generator import cross_languages, keyword_extraction

    logging.debug(
        "搜索（数据集=%s，租户=%s，question_len=%s）",
        dataset_id,
        tenant_id,
        len(req.get("question", "")),
    )

    page = int(req.get("page", 1))
    size = int(req.get("page_size") or req.get("size", 30))
    rerank_candidates_count = int(req.get("rerank_candidates_count", 64))
    question = req.get("question", "")
    doc_ids = req.get("doc_ids", [])
    use_kg = req.get("use_kg", False)
    similarity_threshold = float(req.get("similarity_threshold", 0.0))
    vector_similarity_weight = float(req.get("vector_similarity_weight", 0.3))
    knn_top_k = max(1, min(int(req.get("knn_top_k", 1024)), 2048))
    knn_num_candidates = int(req.get("knn_num_candidates", 2048))
    langs = req.get("cross_languages", [])

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        logging.warning("搜索访问被拒绝：数据集=%s 租户=%s", dataset_id, tenant_id)
        return False, "Only owner of dataset authorized for this operation."

    e, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not e:
        logging.warning("搜索数据集未找到：数据集=%s\n未找到", dataset_id)
        return False, "Dataset not found!"

    if doc_ids is not None and not isinstance(doc_ids, list):
        return False, "`doc_ids` should be a list"
    local_doc_ids = list(doc_ids) if doc_ids else []

    meta_data_filter = {}
    search_id = req.get("search_id", "")
    search_config = {}
    chat_mdl = None
    if search_id:
        search_detail = SearchService.get_detail(search_id)
        if not search_detail:
            logging.warning("搜索配置：搜索应用ID=%s", search_id)
            return False, "Invalid search_id"
        search_config = search_detail.get("search_config", {})
        meta_data_filter = search_config.get("meta_data_filter", {})
        similarity_threshold = float(search_config.get("similarity_threshold", similarity_threshold))
        vector_similarity_weight = float(search_config.get("vector_similarity_weight", vector_similarity_weight))
        knn_top_k = max(1, min(int(search_config.get("top_k", knn_top_k)), 2048))
        rerank_candidates_count = int(search_config.get("rerank_candidates_count", 100))
        use_kg = search_config.get("use_kg", use_kg)
        langs = search_config.get("cross_languages", langs)
        logging.debug(
            "数据集搜索已加载 Search 配置：搜索应用ID=%s dataset_id=%s vector_similarity_weight=%s full_text_weight=%s similarity_threshold=%s knn_top_k=%s",
            search_id,
            dataset_id,
            vector_similarity_weight,
            1 - vector_similarity_weight,
            similarity_threshold,
            knn_top_k,
        )
        if meta_data_filter.get("method") in ["auto", "semi_auto"]:
            chat_id = search_config.get("chat_id", "")
            if chat_id:
                chat_model_config = resolve_model_config(tenant_id, LLMType.CHAT, search_config["chat_id"])
            else:
                chat_model_config = get_tenant_default_model_by_type(tenant_id, LLMType.CHAT)
            chat_mdl = LLMBundle(tenant_id, chat_model_config)
    else:
        meta_data_filter = req.get("meta_data_filter") or {}
        if meta_data_filter.get("method") in ["auto", "semi_auto"]:
            chat_model_config = get_tenant_default_model_by_type(tenant_id, LLMType.CHAT)
            chat_mdl = LLMBundle(tenant_id, chat_model_config)

    if meta_data_filter:
        local_doc_ids = await apply_meta_data_filter(
            meta_data_filter,
            None,
            question,
            chat_mdl,
            local_doc_ids,
            kb_ids=[dataset_id],
            metas_loader=lambda: DocMetadataService.get_flatted_meta_by_kbs([dataset_id]),
        )

    tenant_ids = []
    tenants = UserTenantService.query(user_id=tenant_id)
    for tenant in tenants:
        if KnowledgebaseService.query(tenant_id=tenant.tenant_id, id=dataset_id):
            tenant_ids.append(tenant.tenant_id)
            break
    else:
        return False, "Only owner of dataset authorized for this operation."

    _question = question
    if langs:
        _question = await cross_languages(kb.tenant_id, None, _question, langs)
    if kb.embd_id:
        embd_model_config = resolve_model_config(kb.tenant_id, LLMType.EMBEDDING, kb.embd_id)
    else:
        embd_model_config = get_tenant_default_model_by_type(kb.tenant_id, LLMType.EMBEDDING)
    embd_mdl = LLMBundle(kb.tenant_id, embd_model_config)

    rerank_mdl = None
    rerank_id = req.get("rerank_id") or search_config.get("rerank_id")
    if rerank_id:
        rerank_model_config = resolve_model_config(kb.tenant_id, LLMType.RERANK.value, rerank_id)
        rerank_mdl = LLMBundle(kb.tenant_id, rerank_model_config)

    if search_config.get("keyword", req.get("keyword", False)):
        default_chat_model_config = get_tenant_default_model_by_type(kb.tenant_id, LLMType.CHAT)
        chat_mdl = LLMBundle(kb.tenant_id, default_chat_model_config)
        _question += await keyword_extraction(chat_mdl, _question)

    labels = label_question(_question, [kb])
    ranks = await settings.retriever.retrieval(
        _question,
        embd_mdl,
        tenant_ids,
        [dataset_id],
        page,
        size,
        similarity_threshold,
        vector_similarity_weight,
        doc_ids=local_doc_ids,
        knn_top_k=knn_top_k,
        knn_num_candidates=knn_num_candidates,
        rerank_mdl=rerank_mdl,
        rank_feature=labels,
        trace_id=search_id,
        rerank_candidates_count=rerank_candidates_count,
    )

    if use_kg:
        try:
            default_chat_model_config = get_tenant_default_model_by_type(tenant_id, LLMType.CHAT)
            ck = await settings.kg_retriever.retrieval(_question, tenant_ids, [dataset_id], embd_mdl, LLMBundle(kb.tenant_id, default_chat_model_config))
            if ck["content_with_weight"]:
                ranks["chunks"].insert(0, ck)
        except Exception:
            logging.warning("搜索 KG 检索失败：数据集=%s 租户=%s", dataset_id, tenant_id, exc_info=True)
    ranks["chunks"] = settings.retriever.retrieval_by_children(ranks["chunks"], tenant_ids)
    ranks["total"] = len(ranks["chunks"])

    for c in ranks["chunks"]:
        c.pop("vector", None)
    ranks["labels"] = labels

    return True, ranks


def check_embedding(dataset_id: str, tenant_id: str, req: dict):
    """通过对随机块进行采样来检查嵌入模型的兼容性，
    用新模型重新嵌入它们，并计算余弦相似度。

    ：参数dataset_id：数据集ID
    ：参数 tenant_id：租户 ID
    :param req: 请求主体为 embd_id
    :return: (成功,结果)或(成功,error_message)"""
    import random

    import numpy as np

    from api.db.services.llm_service import LLMBundle
    from common.constants import LLMType, RetCode
    from common.doc_store.doc_store_base import OrderByExpr
    from rag.nlp import search

    def _guess_vec_field(src: dict):
        for k in src or {}:
            if k.endswith("_vec"):
                return k
        return None

    def _as_float_vec(v):
        if v is None:
            return []
        if isinstance(v, str):
            return [float(x) for x in v.split("\t") if x != ""]
        if isinstance(v, (list, tuple, np.ndarray)):
            return [float(x) for x in v]
        return []

    def _to_1d(x):
        a = np.asarray(x, dtype=np.float32)
        return a.reshape(-1)

    def _cos_sim(a, b, eps=1e-12):
        a = _to_1d(a)
        b = _to_1d(b)
        na = np.linalg.norm(a)
        nb = np.linalg.norm(b)
        if na < eps or nb < eps:
            return 0.0
        return float(np.dot(a, b) / (na * nb))

    def sample_random_chunks_with_vectors(
        docStoreConn,
        tenant_id: str,
        kb_id: str,
        n: int = 5,
        base_fields=("docnm_kwd", "doc_id", "content_with_weight", "page_num_int", "position_int", "top_int"),
    ):
        index_nm = search.index_name(tenant_id)
        try:
            res0 = docStoreConn.search(
                select_fields=[],
                highlight_fields=[],
                condition={"kb_id": kb_id, "available_int": 1},
                match_expressions=[],
                order_by=OrderByExpr(),
                offset=0,
                limit=1,
                index_names=index_nm,
                knowledgebase_ids=[kb_id],
            )
        except Exception as e:
            if "not_found_exception" in repr(e) or "index_not_found_exception" in repr(e):
                logging.info(
                    "sample_random_chunks_with_vectors：尚未为租户 %s 创建索引 %s；返回空样本集",
                    index_nm,
                    tenant_id,
                )
                return []
            raise
        total = docStoreConn.get_total(res0)
        if total <= 0:
            return []

        n = min(n, total)
        offsets = sorted(random.sample(range(min(total, 1000)), n))
        out = []

        for off in offsets:
            res1 = docStoreConn.search(
                select_fields=list(base_fields),
                highlight_fields=[],
                condition={"kb_id": kb_id, "available_int": 1},
                match_expressions=[],
                order_by=OrderByExpr(),
                offset=off,
                limit=1,
                index_names=index_nm,
                knowledgebase_ids=[kb_id],
            )
            ids = docStoreConn.get_doc_ids(res1)
            if not ids:
                continue

            cid = ids[0]
            full_doc = docStoreConn.get(cid, index_nm, [kb_id]) or {}
            vec_field = _guess_vec_field(full_doc)
            vec_valid = full_doc.get(f"{vec_field}_valid") if vec_field else None
            if callable(getattr(docStoreConn, "db_type", None)) and docStoreConn.db_type() == "gaussdb" and vec_valid is False:
                vec = []
            else:
                vec = _as_float_vec(full_doc.get(vec_field))

            out.append(
                {
                    "chunk_id": cid,
                    "kb_id": kb_id,
                    "doc_id": full_doc.get("doc_id"),
                    "doc_name": full_doc.get("docnm_kwd"),
                    "vector_field": vec_field,
                    "vector_dim": len(vec),
                    "vector": vec,
                    "page_num_int": full_doc.get("page_num_int"),
                    "position_int": full_doc.get("position_int"),
                    "top_int": full_doc.get("top_int"),
                    "content_with_weight": full_doc.get("content_with_weight") or "",
                    "question_kwd": full_doc.get("question_kwd") or [],
                }
            )
        return out

    def _clean(s: str):
        return re.sub(r"</?(table|td|caption|tr|th)( [^<>]{0,12})?>", " ", s or "").strip()

    if not dataset_id:
        return False, 'Lack of "Dataset ID"'

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"

    ok, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not ok:
        return False, "Invalid Dataset ID"

    embd_id = req.get("embd_id", "")
    if not embd_id:
        return False, "`embd_id` is required."

    logging.info("check_embedding：数据集=%s租户=%s embd_id=%s", dataset_id, tenant_id, embd_id)

    ok, err = verify_embedding_availability(embd_id, tenant_id)
    if not ok:
        return False, err

    embd_model_config = resolve_model_config(kb.tenant_id, LLMType.EMBEDDING, embd_id)
    emb_mdl = LLMBundle(kb.tenant_id, embd_model_config)

    n = int(req.get("check_num", 5))
    samples = sample_random_chunks_with_vectors(settings.docStoreConn, tenant_id=kb.tenant_id, kb_id=dataset_id, n=n)
    logging.info("check_embedding：数据集=%s采样=%d块", dataset_id, len(samples))

    results, eff_sims = [], []
    mode = "content_only"
    for ck in samples:
        title = ck.get("doc_name") or "Title"

        txt_in = "\n".join(ck.get("question_kwd") or []) or ck.get("content_with_weight") or ""
        txt_in = _clean(txt_in)
        if not txt_in:
            results.append({"chunk_id": ck["chunk_id"], "reason": "no_text"})
            continue

        if not ck.get("vector"):
            results.append({"chunk_id": ck["chunk_id"], "reason": "no_stored_vector"})
            continue

        try:
            v, _ = emb_mdl.encode([title, txt_in])
            assert len(v[1]) == len(ck["vector"]), f"The dimension ({len(v[1])}) of given embedding model is different from the original ({len(ck['vector'])})"
            sim_content = _cos_sim(v[1], ck["vector"])
            title_w = 0.1
            qv_mix = title_w * v[0] + (1 - title_w) * v[1]
            sim_mix = _cos_sim(qv_mix, ck["vector"])
            sim = sim_content
            mode = "content_only"
            if sim_mix > sim:
                sim = sim_mix
                mode = "title+content"
        except Exception as e:
            return False, f"Embedding failure. {e}"

        eff_sims.append(sim)
        results.append(
            {
                "chunk_id": ck["chunk_id"],
                "doc_id": ck["doc_id"],
                "doc_name": ck["doc_name"],
                "vector_field": ck["vector_field"],
                "vector_dim": ck["vector_dim"],
                "cos_sim": round(sim, 6),
            }
        )

    summary = {
        "kb_id": dataset_id,
        "model": embd_id,
        "sampled": len(samples),
        "valid": len(eff_sims),
        "avg_cos_sim": round(float(np.mean(eff_sims)) if eff_sims else 0.0, 6),
        "min_cos_sim": round(float(np.min(eff_sims)) if eff_sims else 0.0, 6),
        "max_cos_sim": round(float(np.max(eff_sims)) if eff_sims else 0.0, 6),
        "match_mode": mode,
    }

    data = {"summary": summary, "results": results}
    if not eff_sims:
        logging.warning("check_embedding：数据集=%s 没有可比较的块", dataset_id)
        return False, "No embedded chunks are available to compare."
    if summary["avg_cos_sim"] >= 0.9:
        logging.info("check_embedding：数据集=%s 兼容avg_cos_sim=%s 有效=%d", dataset_id, summary["avg_cos_sim"], len(eff_sims))
        return True, data
    logging.warning("check_embedding：数据集=%s not_effective avg_cos_sim=%s 有效=%d", dataset_id, summary["avg_cos_sim"], len(eff_sims))
    return "not_effective", {
        "code": RetCode.NOT_EFFECTIVE,
        "message": "Embedding model switch failed: the average similarity between old and new vectors is below 0.9, indicating incompatible vector spaces.",
        "data": data,
    }


async def search_datasets(tenant_id: str, req: dict):
    """
    Search (retrieval test) across multiple datasets.

    查询主链路（这里只返回相关 Chunk，不负责调用 Chat 模型生成最终答案）：
    请求校验/权限 -> 元数据缩小 doc_id 范围 -> 跨语言/关键词扩展 -> 绑定 Embedding
    与可选 Rerank -> 生成查询标签 -> Dealer.retrieval(BM25+KNN+重排) -> 可选 GraphRAG
    补充 -> 子 Chunk 替换/展开 -> 移除向量后返回前端。

    :param tenant_id: tenant ID
    :param req: search request containing dataset_ids and other params
    :return: (success, result) or (success, error_message)
    """
    from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type
    from api.db.services.doc_metadata_service import DocMetadataService
    from api.db.services.llm_service import LLMBundle
    from api.db.services.search_service import SearchService
    from api.db.services.user_service import UserTenantService
    from common.constants import LLMType
    from common.metadata_utils import apply_meta_data_filter
    from rag.app.tag import label_question
    from rag.prompts.generator import cross_languages, keyword_extraction

    kb_ids = req.get("dataset_ids", [])
    page = int(req.get("page", 1))  # 返回第几页
    size = int(req.get("page_size") or req.get("size", 30))  # 每页数量，范围 1～100
    rerank_candidates_count = int(req.get("rerank_candidates_count", 64))  # 进入最终重排流程的候选数量
    question = req.get("question", "")  # 用户查询文本
    doc_ids = req.get("doc_ids", [])  # 只检索指定文档；空数组表示不限定文档
    use_kg = req.get("use_kg", False)  # 是否额外执行知识图谱检索
    similarity_threshold = float(req.get("similarity_threshold", 0.0))  # 最终相似度过滤阈值
    vector_similarity_weight = float(req.get("vector_similarity_weight", 0.3))  # 向量得分在混合得分中的权重
    knn_top_k = max(1, min(int(req.get("knn_top_k", 1024)), 2048))  # 向量召回最多返回的候选数量
    knn_num_candidates = int(req.get("knn_num_candidates", 2048))  # ES HNSW 搜索时考察的候选规模
    langs = req.get("cross_languages", [])  # 是否将问题转换为其他语言进行跨语言检索

    logging.debug(
        "search_datasets（数据集=%s，租户=%s，question_len=%s）",
        kb_ids,
        tenant_id,
        len(question),
    )

    # 所有数据集的访问检查
    for kb_id in kb_ids:
        # 检查知识库权限
        if not KnowledgebaseService.accessible(kb_id, tenant_id):
            logging.warning("search_datasets 访问被拒绝：数据集=%s 租户=%s", kb_id, tenant_id)
            return False, f"Only owner of dataset {kb_id} authorized for this operation."
    # 读取数据库配置
    kbs = KnowledgebaseService.get_by_ids(kb_ids)
    if not kbs:
        return False, "Datasets not found!"
    # 检查这些知识库使用的 Embedding 模型是否兼容,
    # 跨知识库检索时，向量必须处于相同的向量空间，通常要求使用相同的 Embedding 模型。
    err = validate_dataset_embedding_models(kbs)
    if err:
        return False, err

    if doc_ids is not None and not isinstance(doc_ids, list):
        return False, "`doc_ids` should be a list"
    local_doc_ids = list(doc_ids) if doc_ids else []

    meta_data_filter = {}
    # 使用已保存的 Search 配置
    search_id = req.get("search_id", "")
    search_config = {}
    chat_mdl = None
    if search_id:
        search_detail = SearchService.get_detail(search_id)
        if not search_detail:
            logging.warning("搜索配置未找到：搜索应用ID=%s", search_id)
            return False, "Invalid search_id"
        search_config = search_detail.get("search_config", {})
        # 根据文档元数据过滤检索范围
        meta_data_filter = search_config.get("meta_data_filter", {})
        similarity_threshold = float(search_config.get("similarity_threshold", similarity_threshold))
        vector_similarity_weight = float(search_config.get("vector_similarity_weight", vector_similarity_weight))
        knn_top_k = max(1, min(int(search_config.get("top_k", knn_top_k)), 2048))
        rerank_candidates_count = int(search_config.get("rerank_candidates_count", 100))
        use_kg = search_config.get("use_kg", use_kg)
        langs = search_config.get("cross_languages", langs)
        logging.debug(
            "数据集搜索已加载 Search 配置：搜索应用ID=%s dataset_ids=%s vector_similarity_weight=%s full_text_weight=%s similarity_threshold=%s knn_top_k=%s",
            search_id,
            kb_ids,
            vector_similarity_weight,
            1 - vector_similarity_weight,
            similarity_threshold,
            knn_top_k,
        )
        if meta_data_filter.get("method") in ["auto", "semi_auto"]:
            chat_id = search_config.get("chat_id", "")
            if chat_id:
                chat_model_config = resolve_model_config(tenant_id, LLMType.CHAT, search_config["chat_id"])
            else:
                chat_model_config = get_tenant_default_model_by_type(tenant_id, LLMType.CHAT)
            chat_mdl = LLMBundle(tenant_id, chat_model_config)
    else:
        # 根据文档元数据过滤检索范围
        meta_data_filter = req.get("meta_data_filter") or {}
        if meta_data_filter.get("method") in ["auto", "semi_auto"]:
            chat_model_config = get_tenant_default_model_by_type(tenant_id, LLMType.CHAT)
            chat_mdl = LLMBundle(tenant_id, chat_model_config)

    if meta_data_filter:
        logging.debug("应用元数据过滤器：%s，问题长度：%d，chat_mdl=%s", meta_data_filter, len(question), "None" if chat_mdl is None else "configured")
        # 元数据过滤
        # 元数据过滤先在 MySQL 元数据层得到允许的 doc_id，随后把这些 ID 作为 ES filter；
        # 它不是相关性打分的一部分，而是在召回前缩小搜索空间。
        local_doc_ids = await apply_meta_data_filter(
            meta_data_filter,
            None,
            question,
            chat_mdl,
            local_doc_ids,
            kb_ids=kb_ids,
            metas_loader=lambda: DocMetadataService.get_flatted_meta_by_kbs(kb_ids),
        )

    tenant_ids = []
    # SELECT * FROM user_tenant WHERE user_id = 当前用户ID;
    tenants = UserTenantService.query(user_id=tenant_id)
    for tenant in tenants:
        if any(KnowledgebaseService.query(tenant_id=tenant.tenant_id, id=kb_id) for kb_id in kb_ids):  # 判断知识库是否属于当前租户
            tenant_ids.append(tenant.tenant_id)
            break
    else:
        return False, "Only owner of datasets authorized for this operation."

    kb = kbs[0]
    _question = question
    if langs:
        # 将问题翻译成每个指定语言并与原问题拼接为一个检索字符串；不是分别执行多次 retrieval。
        _question = await cross_languages(kb.tenant_id, None, _question, langs)

    embd_mdl = None
    if kb.embd_id:
        embd_model_config = resolve_model_config(kb.tenant_id, LLMType.EMBEDDING, kb.embd_id)
    else:
        embd_model_config = get_tenant_default_model_by_type(kb.tenant_id, LLMType.EMBEDDING)
    embd_mdl = LLMBundle(kb.tenant_id, embd_model_config)

    rerank_mdl = None
    # 指定 Rerank 模型
    rerank_id = req.get("rerank_id") or search_config.get("rerank_id")
    # 如果勾选了rerank, 则根据rerank_id获取配置 以及加载rerank模型
    if rerank_id:
        rerank_model_config = resolve_model_config(kb.tenant_id, LLMType.RERANK.value, rerank_id)
        rerank_mdl = LLMBundle(kb.tenant_id, rerank_model_config)
    # 如果配置了关键之, 则将关键字追加到问题中
    # 是否使用 LLM 提取关键词并追加到问题
    if search_config.get("keyword", req.get("keyword", False)):
        default_chat_model_config = get_tenant_default_model_by_type(kb.tenant_id, LLMType.CHAT)
        chat_mdl = LLMBundle(kb.tenant_id, default_chat_model_config)
        _question += await keyword_extraction(chat_mdl, _question)

    # 从配置的标签知识库中通过全文搜索聚合 top-N 标签；labels 会作为 rank_feature
    # 同时影响 ES 第一阶段候选排序和 Python 第二阶段最终加分，不是硬过滤条件。
    labels = label_question(_question, kbs)

    # 不记录问题原文、向量或 Chunk 正文，避免把用户数据写入常规服务日志；search_id 可用于
    # 将入口、Dealer.retrieval 和最终返回日志关联起来。
    logging.info(
        "知识库检索已开始 检索ID=%s 租户ID=%s 知识库ID列表=%s 文档过滤数=%d 问题长度=%d 向量权重=%.3f 是否重排=%s 标签数=%d",
        search_id or "-",
        tenant_id,
        kb_ids,
        len(local_doc_ids),
        len(_question),
        vector_similarity_weight,
        bool(rerank_mdl),
        len(labels or {}),
    )

    # 召回
    ranks = await settings.retriever.retrieval(
        _question,  # 最终用于检索的问题，可能已经经过翻译或关键词扩展
        embd_mdl,  # 将问题转换为向量的 Embedding 模型
        tenant_ids,  # 租户 ID，决定查询哪些 ES 索引
        kb_ids,  # 限定查询的知识库
        page,  # 最终结果页码
        size,  # 每页返回多少个 Chunk
        similarity_threshold,  # 最终相似度阈值
        vector_similarity_weight,  # 向量分数占比
        doc_ids=local_doc_ids,  # 可选，只检索指定文档
        knn_top_k=knn_top_k,  # ES KNN 最多召回多少个向量候选
        knn_num_candidates=knn_num_candidates,  # ES 每个分片内部用于近似搜索的候选规模
        rerank_mdl=rerank_mdl,  # 可选的专用重排模型
        rank_feature=labels,  # 标签特征
        trace_id=search_id,  # 仅用于串联当前检索配置/日志，方便排查一次检索链路
        must_not=None if req.get("include_knowledge_compilation", True) else {"exists": "compile_kwd"},  # 排除某些类型的 Chunk
        rerank_candidates_count=rerank_candidates_count,  # 拉回应用层重新打分的候选数量，默认 64
    )

    if use_kg:
        # GraphRAG 是标准 Chunk 召回之后的额外独立检索；成功时把图检索结果插到首位，
        # 失败只记录日志，不影响已得到的普通 BM25/KNN 结果。
        try:
            default_chat_model_config = get_tenant_default_model_by_type(tenant_id, LLMType.CHAT)
            ck = await settings.kg_retriever.retrieval(_question, tenant_ids, kb_ids, embd_mdl, LLMBundle(kb.tenant_id, default_chat_model_config))
            if ck["content_with_weight"]:
                ranks["chunks"].insert(0, ck)
        except Exception:
            logging.warning("search_datasets KG 检索失败：数据集=%s 租户=%s", kb_ids, tenant_id, exc_info=True)
    # 若命中母 Chunk/层级 Chunk，根据 mom_id 等关系转换为最终可展示的子 Chunk。
    ranks["chunks"] = settings.retriever.retrieval_by_children(ranks["chunks"], tenant_ids)

    # 查询向量只供服务端后续引用计算使用，API 返回前移除，避免响应体过大和暴露内部向量。
    for c in ranks["chunks"]:
        c.pop("vector", None)
    ranks["labels"] = labels

    logging.info(
        "知识库检索完成 检索ID=%s 知识库ID列表=%s 命中总数=%d 返回切片数=%d",
        search_id or "-",
        kb_ids,
        ranks.get("total", 0),
        len(ranks["chunks"]),
    )

    return True, ranks


# ---------------------------------------------------------------------------
# 神器（知识汇编）页面
#
# 这三个助手为数据集级 "Artifact" 选项卡提供支持。他们查询行
# 与 TaskHandler 编写的“`compile_kwd="wiki_page"`”
# ``persist_wiki_pages``。他们依赖的模式字段是：
#   slug_kwd, title_kwd, page_type_kwd, content_with_weight,
#   topic_kwd, entity_names_kwd, outlinks_kwd, related_kb_pages_kwd,
#   source_chunk_ids, source_doc_ids
# ---------------------------------------------------------------------------

_SKILL_COMPILE_KWD = "skill"
_SKILL_ALL_COMPILE_KWD = "skill_all"


def _compiled_index_or_none(tenant_id: str, kb_id: str):
    """当租户索引存在时返回(index_name, search_module)，
    否则``None``。避免对 ES 索引未包含的全新租户收取 500 美元
    尚未创建。"""
    from rag.nlp import search as _rag_search

    index_nm = _rag_search.index_name(tenant_id)
    if not settings.docStoreConn.index_exist(index_nm, kb_id):
        return None
    return index_nm, _rag_search


def _wiki_index_or_none(tenant_id: str, kb_id: str):
    return _compiled_index_or_none(tenant_id, kb_id)


def _compilation_template_kind(kind) -> str:
    if not isinstance(kind, str):
        return ""
    normalized = kind.strip().lower().replace("-", "_")
    if normalized in {"pageindex", "page_index", "knowledge_graph"}:
        return "timeline"
    return normalized


def _scalar(raw, default=""):
    """Infinity ``get_fields`` returns every ``*_kwd`` field as a list (split
    on ``###``), even single scalar values like ``slug_kwd=["entity/foo"]``。
    将预期为标量标识符的字段值规范化回
    第一个非空元素。"""
    if isinstance(raw, (list, tuple)):
        for item in raw:
            if item not in (None, ""):
                return item
        return default
    return raw if raw not in (None, "") else default


def _string_list(raw) -> list[str]:
    """规范化本机数组和旧版 JSON/Infinity 字符串字段。"""
    if isinstance(raw, (list, tuple, set)):
        values = raw
    elif isinstance(raw, str):
        value = raw.strip()
        if not value:
            return []
        try:
            decoded = json.loads(value)
        except (json.JSONDecodeError, TypeError):
            decoded = None
        if isinstance(decoded, list):
            values = decoded
        else:
            values = value.split("###")
    else:
        return []

    result: list[str] = []
    seen: set[str] = set()
    for item in values:
        if not isinstance(item, str):
            continue
        item = item.strip()
        if item and item not in seen:
            seen.add(item)
            result.append(item)
    return result


def _normalize_compilation_template_group_ids(raw) -> list[str]:
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []
    ids: list[str] = []
    seen: set[str] = set()
    for group_id in raw:
        if not isinstance(group_id, str):
            continue
        group_id = group_id.strip()
        if group_id and group_id not in seen:
            seen.add(group_id)
            ids.append(group_id)
    return ids


def _extract_pipeline_compiler_group_ids(dsl) -> list[str]:
    if isinstance(dsl, str):
        try:
            dsl = json.loads(dsl)
        except Exception:
            return []
    if not isinstance(dsl, dict):
        return []
    components = dsl.get("components")
    if not isinstance(components, dict):
        return []

    group_ids: list[str] = []
    seen: set[str] = set()
    for component in components.values():
        if not isinstance(component, dict):
            continue
        obj = component.get("obj") if isinstance(component.get("obj"), dict) else {}
        component_name = obj.get("component_name") or component.get("component_name") or component.get("name")
        if not isinstance(component_name, str) or component_name.lower() != "compiler":
            continue
        candidates = [
            obj.get("params") if isinstance(obj.get("params"), dict) else {},
            obj,
            component.get("params") if isinstance(component.get("params"), dict) else {},
            component,
        ]
        for candidate in candidates:
            for key in ("compilation_template_group_ids", "compilation_template_group_id"):
                for group_id in _normalize_compilation_template_group_ids(candidate.get(key)):
                    if group_id not in seen:
                        seen.add(group_id)
                        group_ids.append(group_id)
    return group_ids


def _normalize_template_kind(raw) -> str:
    """小写模板的原始类型 WITHOUT 折叠。

    与 :func:`_compilation_template_kind` （折叠``page_index`` /
    ``knowledge_graph`` into ``timeline`ZXQKEEP00 170007ZXQ`graph`` / ``timeline`` / ``tree``分开。"""
    if not isinstance(raw, str):
        return ""
    return raw.strip().lower().replace("-", "_")


def _template_matches_kind(template: dict | None, accepted: set[str]) -> bool:
    if not isinstance(template, dict):
        return False
    config = template.get("config") if isinstance(template.get("config"), dict) else {}
    raw_kind = config.get("kind") or template.get("kind") or ""
    return _normalize_template_kind(raw_kind) in accepted


def _group_has_template_of_kind(group_id: str, tenant_id: str, accepted: set[str], group_cache: dict[str, bool]) -> bool:
    if group_id in group_cache:
        return group_cache[group_id]
    from api.db.services.compilation_template_group_service import CompilationTemplateGroupService

    group = CompilationTemplateGroupService.get_saved(group_id, tenant_id)
    matched = any(_template_matches_kind(template, accepted) for template in (group or {}).get("templates") or [])
    group_cache[group_id] = matched
    return matched


def _parser_config_has_template_of_kind(parser_config, tenant_id: str, accepted: set[str], template_cache: dict[str, bool]) -> bool:
    from api.db.services.compilation_template_service import CompilationTemplateService
    from rag.svr.task_executor_refactor.chunk_post_processor import _parser_config_compilation_template_ids

    for template_id in _parser_config_compilation_template_ids(parser_config, tenant_id):
        if template_id not in template_cache:
            template_cache[template_id] = _template_matches_kind(CompilationTemplateService.get_saved(template_id, tenant_id), accepted)
        if template_cache[template_id]:
            return True
    return False


def _pipeline_has_compiler_of_kind(
    pipeline_id: str,
    tenant_id: str,
    accepted: set[str],
    pipeline_cache: dict[str, bool],
    group_cache: dict[str, bool],
) -> bool:
    pipeline_id = (pipeline_id or "").strip()
    if not pipeline_id:
        return False
    if pipeline_id in pipeline_cache:
        return pipeline_cache[pipeline_id]

    from api.db.services.canvas_service import UserCanvasService

    ok, canvas = UserCanvasService.get_by_id(pipeline_id)
    if not ok or not canvas:
        pipeline_cache[pipeline_id] = False
        return False

    group_ids = _extract_pipeline_compiler_group_ids(getattr(canvas, "dsl", None))
    matched = any(_group_has_template_of_kind(group_id, tenant_id, accepted, group_cache) for group_id in group_ids)
    pipeline_cache[pipeline_id] = matched
    return matched


def _skill_index_or_none(tenant_id: str, kb_id: str):
    return _compiled_index_or_none(tenant_id, kb_id)


async def has_any_wiki(dataset_id: str, tenant_id: str):
    """用于侧边栏选项卡可见性检查的快速存在探测。

    返回“`(True, {"has": bool})`` on success or ``(False, str)`` on
    auth failure. Runs a ``limit=1`”搜索并仅读取总数。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _wiki_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, {"has": False}
    index_nm, _ = pack

    from common.doc_store.doc_store_base import OrderByExpr

    try:
        res = settings.docStoreConn.search(
            select_fields=["id"],
            highlight_fields=[],
            condition={"compile_kwd": [WIKI_PAGE_COMPILE_KWD]},
            match_expressions=[],
            order_by=OrderByExpr(),
            offset=0,
            limit=1,
            index_names=index_nm,
            knowledgebase_ids=[dataset_id],
        )
    except Exception:
        logging.exception("has_any_wiki：kb = %s 的 docStore 搜索失败", dataset_id)
        return True, {"has": False}

    total = settings.docStoreConn.get_total(res)
    return True, {"has": bool(total)}


# 数据集范围结构类型为 artifacts_structure API 服务。按键是
# 友好命名前端通行证；值是模板的*顶级*类型
# 印在“`dataset_graph`”行上。规范名称映射到自身，因此
# 调用者可以传递任一形式。请注意，这是故意重用 NOT
# ``_compilation_template_kind`` — that helper folds ``knowledge_graph`` 进入
# ``timeline`` 并将在这里合并不同的类型。
_DATASET_STRUCTURE_ROW_KWD = "dataset_graph"
_DATASET_STRUCTURE_KIND_ALIASES = {
    "graph": "knowledge_graph",
    "knowledge_graph": "knowledge_graph",
    "mindmap": "mind_map",
    "mind_map": "mind_map",
    "timeline": "timeline",
    "session_essence": "session_essence",
    "session_graph": "session_graph",
}
_DATASET_STRUCTURE_KIND_TO_INDEX_TYPE = {
    "knowledge_graph": "structure_graph",
    "mind_map": "structure_mindmap",
    "timeline": "timeline",
    "session_essence": "session_essence",
    "session_graph": "session_graph",
}


def _resolve_dataset_structure_kind(kind) -> str | None:
    """将 friendly/canonical 种类字符串映射到存储的顶级种类。"""
    if not isinstance(kind, str):
        return None
    return _DATASET_STRUCTURE_KIND_ALIASES.get(kind.strip().lower().replace("-", "_"))


def delete_dataset_structure(dataset_id: str, tenant_id: str, kind: str, wipe: bool = True):
    """删除一种 artifacts_structure 种类的合并的 KB 范围的结构行。"""
    resolved_kind = _resolve_dataset_structure_kind(kind)
    if not resolved_kind:
        return False, f"Unsupported structure kind: {kind!r}. Expected one of: graph, mindmap, timeline, session_essence, session_graph."
    index_type = _DATASET_STRUCTURE_KIND_TO_INDEX_TYPE[resolved_kind]
    return delete_index(dataset_id, tenant_id, index_type, wipe=wipe)


async def get_dataset_structure(dataset_id: str, tenant_id: str, kind: str, keywords: str = ""):
    """加载一个数据集范围（KB-wide）结构图``kind``.

    ``kind`ZXQKEEP002 80004ZXQ`graph`` / ``mindmap`ZXQK EEP00280008ZXQ`timeline`` /
    ``session_essence`` / ``session_graph``. The `ZXQKEEP00 280015ZXQ`
    blob rows (one per template, written by ``rebuild_dataset_structure_graph_json``)
    are used only to DISCOVER the requested kind's template buckets; each bucket's
    entities/relations are then fetched from the raw KB-wide `ZX QKEEP00280019ZXQ` / ``relation`ZXQKEEP0028002 2ZXQ`doc_id`` filter), since dataset-merge templates dedup those rows
    across documents.

    ``keywords`ZXQKEEP 00280026ZXQ`(True, {"kind": <kind>, "templates": [...]})`` or
    ``(False, message)``关于 auth/validation 故障。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"

    resolved_kind = _resolve_dataset_structure_kind(kind)
    if not resolved_kind:
        return False, f"Unsupported structure kind: {kind!r}. Expected one of: graph, mindmap, timeline, session_essence, session_graph."

    _, kb = KnowledgebaseService.get_by_id(dataset_id)
    empty = {"kind": kind, "templates": []}

    pack = _compiled_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, empty
    index_nm, _ = pack

    from api.apps.services import structure_graph_common as sgc
    from api.db.services.compilation_template_service import CompilationTemplateService
    from api.db.services.tenant_llm_service import TenantLLMService
    from common.doc_store.doc_store_base import OrderByExpr

    keywords = (keywords or "").strip()
    _, active_doc_ids = await _current_dataset_docs(dataset_id)
    disabled_doc_ids = await _disabled_dataset_doc_ids(dataset_id)
    if not active_doc_ids:
        return True, empty
    # 数据集行使用 KB id 作为“`doc_id`”。如果历史行没有
    # 源出处，在文档被禁用并回退时将其排除
    # 到活动文档行，而不是返回可能过时的内容。
    dataset_excluded_doc_ids = disabled_doc_ids | ({dataset_id} if disabled_doc_ids else set())

    def _row_template_id(row: dict) -> str | None:
        raw = row.get("compilation_template_ids")
        if isinstance(raw, list):
            for v in raw:
                if isinstance(v, str) and v.strip():
                    return v.strip()
        if isinstance(raw, str) and raw.strip():
            return raw.strip()
        return None

    # 解析模板的顶级种类+显示名称，已记忆。两者都用于
    # 标签桶并作为在种类标记之前写入的行的后备
    # （它们携带“`compilation_template_ids`` but no ``compilation_template_kind_kwd`”）。
    template_kind_cache: dict[str, str | None] = {}
    template_name_cache: dict[str, str] = {}

    def _template_meta(tid: str | None) -> str | None:
        if not tid:
            return None
        if tid in template_kind_cache:
            return template_kind_cache[tid]
        top_kind = None
        try:
            saved = CompilationTemplateService.get_saved(tid, tenant_id)
            if saved:
                top_kind = (saved.get("kind") or "").strip() or None
                template_name_cache[tid] = saved.get("name") or tid
        except Exception:
            logging.exception("get_dataset_structure：%s 模板查找失败", tid)
        template_kind_cache[tid] = top_kind
        return top_kind

    def _bucket_meta_for(tid: str, row_kind: str = "") -> dict:
        if tid not in template_name_cache:
            _template_meta(tid)
        return {
            "template_id": tid,
            "template_name": template_name_cache.get(tid, tid),
            "kind": row_kind or template_kind_cache.get(tid) or resolved_kind,
        }

    # 发现由结构合并流程写入的知识库级记录。
    # ``run_structure_merge``（位于 dataset_structure_merger）会写入合并后的
    # ``knowledge_graph_kwd="entity"/"relation"`` 记录，并设置
    # ``scope_kwd="dataset"``、``compilation_template_ids`` 和顶层
    # ``compilation_template_kind_kwd``。这里只读取元数据字段，枚举符合请求类型的模板 ID；
    # 后面的 ``build_bucket`` 再读取每个模板对应的完整记录。
    meta_fields = ["id", "compile_kwd", "compilation_template_ids", "compilation_template_kind_kwd"]
    page_size = 1000

    async def _discover_scope_templates(scope_kwd: str) -> tuple[list[str], bool, int]:
        scope_template_ids: list[str] = []
        seen_scope_tid: set[str] = set()
        scope_has_templateless = False
        offset = 0
        pages = 0
        while True:
            pages += 1
            try:
                res = await thread_pool_exec(
                    settings.docStoreConn.search,
                    meta_fields,
                    [],
                    {"knowledge_graph_kwd": ["entity", "relation"], "scope_kwd": [scope_kwd]},
                    [],
                    OrderByExpr(),
                    offset,
                    page_size,
                    index_nm,
                    [dataset_id],
                )
                meta_rows = settings.docStoreConn.get_fields(res, meta_fields) or {}
            except Exception:
                logging.exception("get_dataset_structure：kb = %s 范围 = %s 的 docStore 发现失败", dataset_id, scope_kwd)
                return [], False, pages
            if not meta_rows:
                break
            for row in meta_rows.values():
                tid = _row_template_id(row)
                stamped_kind = row.get("compilation_template_kind_kwd") or ""
                if isinstance(stamped_kind, list):
                    stamped_kind = stamped_kind[0].strip()
                row_kind = stamped_kind or _template_meta(tid) or ""
                if _resolve_dataset_structure_kind(row_kind) != resolved_kind:
                    continue
                if tid:
                    if tid not in seen_scope_tid:
                        seen_scope_tid.add(tid)
                        scope_template_ids.append(tid)
                else:
                    scope_has_templateless = True
            if len(meta_rows) < page_size:
                break
            offset += page_size
        return scope_template_ids, scope_has_templateless, pages

    dataset_template_ids, has_templateless, dataset_pages = await _discover_scope_templates("dataset")
    kind_template_ids: list[str] = list(dataset_template_ids)
    template_scope_by_id: dict[str, str] = {tid: "dataset" for tid in dataset_template_ids}

    doc_template_ids, _, doc_pages = await _discover_scope_templates("doc")
    for tid in doc_template_ids:
        if tid not in template_scope_by_id:
            template_scope_by_id[tid] = "doc"
            kind_template_ids.append(tid)

    # 检测具有 ONLY 旧版 dataset_graph blob 的数据集（无
    # entity/relation 行尚未），因此下面的后备路径可以处理它们。
    if not kind_template_ids and not has_templateless:
        try:
            legacy_check = await thread_pool_exec(
                settings.docStoreConn.search,
                ["id"],
                [],
                {"knowledge_graph_kwd": [_DATASET_STRUCTURE_ROW_KWD]},
                [],
                OrderByExpr(),
                0,
                1,
                index_nm,
                [dataset_id],
            )
            legacy_fm = settings.docStoreConn.get_fields(legacy_check, ["id"]) or {}
            if legacy_fm:
                has_templateless = True
        except Exception:
            pass

    logging.debug(
        "get_dataset_structure：在 %d/%d 页面中发现 %d 数据集模板和 %d 文档模板kb=%s 种类=%s",
        len(dataset_template_ids),
        len(doc_template_ids),
        dataset_pages,
        doc_pages,
        dataset_id,
        resolved_kind,
    )

    # ── 关键字模式：跨种类→匹配子图命名matching/KNN。 ──
    if keywords:
        if not kind_template_ids:
            return True, empty
        try:
            model_config = resolve_model_config(kb.tenant_id, LLMType.EMBEDDING.value, kb.embd_id)
            embd_mdl = TenantLLMService.model_instance(model_config)
        except Exception:
            logging.exception("get_dataset_structure：kb = %s 的嵌入绑定失败", dataset_id)
            return True, empty

        def _scope_for_template(row: dict):
            tid = _row_template_id(row) or ""
            stamped = row.get("compilation_template_kind_kwd") or ""
            if isinstance(stamped, list):
                stamped = stamped[0].strip()
            scope_kwd = template_scope_by_id.get(tid, "dataset") if tid else "dataset"
            meta = _bucket_meta_for(tid, stamped) if tid else {"template_id": f"kind:{resolved_kind}", "template_name": f"kind:{resolved_kind}", "kind": resolved_kind}
            if tid:
                return meta, {"compilation_template_ids": [tid], "scope_kwd": [scope_kwd]}
            return meta, {"compilation_template_kind_kwd": [stamped], "scope_kwd": [scope_kwd]}

        bucket_meta, kw_entities, kw_relations = await sgc.keyword_subgraph(
            index_nm,
            dataset_id,
            embd_mdl,
            {"compilation_template_ids": kind_template_ids, "knowledge_graph_kwd": ["entity"], "scope_kwd": ["dataset", "doc"]},
            keywords,
            _scope_for_template,
            log_ctx=f"kb={dataset_id}",
            excluded_doc_ids=dataset_excluded_doc_ids,
        )
        if resolved_kind in {"knowledge_graph", "mind_map", "timeline"}:
            kw_entities = sgc.filter_entities_with_relations(kw_entities, kw_relations)
            if not kw_entities:
                return True, empty
        if not bucket_meta or (not kw_entities and not kw_relations):
            return True, empty
        bucket = dict(bucket_meta)
        bucket["entities"] = kw_entities
        bucket["relations"] = kw_relations
        return True, {"kind": kind, "templates": [bucket]}

    # ── 正常模式：从原始 KB 宽行中采样每个模板子图。 ──
    templates_out: list[dict] = []
    for tid in kind_template_ids:
        scope_kwd = template_scope_by_id.get(tid, "dataset")
        try:
            scope = {"compilation_template_ids": [tid], "scope_kwd": [scope_kwd]}
            if scope_kwd == "doc":
                scope["doc_id"] = sorted(active_doc_ids)
            entities, relations = await sgc.build_bucket(
                index_nm,
                dataset_id,
                scope,
                excluded_doc_ids=dataset_excluded_doc_ids if scope_kwd == "dataset" else disabled_doc_ids,
            )
        except Exception:
            logging.exception("get_dataset_structure：kb=%s 模板=%s 的存储桶构建失败", dataset_id, tid)
            continue
        if resolved_kind in {"knowledge_graph", "mind_map", "timeline"}:
            entities = sgc.filter_entities_with_relations(entities, relations)
            if not entities:
                continue
        if not entities and not relations:
            continue
        meta = _bucket_meta_for(tid)
        templates_out.append({**meta, "entities": entities, "relations": relations})

    # 传统无模板 dataset_graph 行没有模板 ID 来限定原始范围
    # 行通过，因此直接回退到其 blob 内容（无采样）。罕见——
    # rebuild_dataset_structure_graph_json 始终标记模板 id。
    if has_templateless:
        try:
            res_l = await thread_pool_exec(
                settings.docStoreConn.search,
                ["content_with_weight", "compilation_template_kind_kwd", "compile_kwd"],
                [],
                {"knowledge_graph_kwd": [_DATASET_STRUCTURE_ROW_KWD], "must_not": {"exists": "compilation_template_ids"}},
                [],
                OrderByExpr(),
                0,
                1000,
                index_nm,
                [dataset_id],
            )
            legacy_rows = settings.docStoreConn.get_fields(res_l, ["content_with_weight", "compilation_template_kind_kwd", "compile_kwd"]) or {}
        except Exception:
            logging.exception("get_dataset_structure：kb=%s 的旧版 blob 获取失败", dataset_id)
            legacy_rows = {}
        legacy_bucket = {"template_id": f"kind:{resolved_kind}", "template_name": f"kind:{resolved_kind}", "kind": resolved_kind, "entities": [], "relations": []}
        reconstructed_compile_kwds: set[str] = set()
        for row in legacy_rows.values():
            stamped_kind = row.get("compilation_template_kind_kwd") or ""
            if isinstance(stamped_kind, list):
                stamped_kind = stamped_kind[0].strip()
            if _resolve_dataset_structure_kind((stamped_kind or "").strip()) != resolved_kind:
                continue
            if disabled_doc_ids:
                compile_kwd = _scalar(row.get("compile_kwd"))
                if not compile_kwd or compile_kwd in reconstructed_compile_kwds:
                    continue
                reconstructed_compile_kwds.add(compile_kwd)
                try:
                    entities, relations = await sgc.build_bucket(
                        index_nm,
                        dataset_id,
                        {"compile_kwd": [compile_kwd], "doc_id": sorted(active_doc_ids)},
                        excluded_doc_ids=disabled_doc_ids,
                    )
                except Exception:
                    continue
                legacy_bucket["entities"].extend(entities)
                legacy_bucket["relations"].extend(relations)
                continue
            try:
                graph = json.loads(row.get("content_with_weight") or "{}")
            except Exception:
                continue
            if not isinstance(graph, dict):
                continue
            legacy_bucket["entities"].extend(item for item in (graph.get("entities") or []) if isinstance(item, dict))
            legacy_bucket["relations"].extend(item for item in (graph.get("relations") or []) if isinstance(item, dict))
        if resolved_kind in {"knowledge_graph", "mind_map", "timeline"}:
            legacy_bucket["entities"] = sgc.filter_entities_with_relations(legacy_bucket["entities"], legacy_bucket["relations"])
        if legacy_bucket["entities"] or legacy_bucket["relations"]:
            if resolved_kind not in {"knowledge_graph", "mind_map", "timeline"} or legacy_bucket["entities"]:
                templates_out.append(legacy_bucket)

    return True, {"kind": kind, "templates": templates_out}


# ── artifacts/alteration：根据“`kind`”来源和资格映射 ──
#
# Alteration 是各种共享的纯集运算：
# 已删除 = 涉及（产品出处）− 当前（数据集文档）
# newly_uploaded = 合格（应编译）− 涉及
# 只有两个输入因种类而异：如何收集“`involved`”出处，以及
# 哪些模板类型生成文档“`eligible`”。

# 非折叠模板类型，使文档适合每种 API 类型。
_ALTERATION_ELIGIBLE_TEMPLATE_KINDS = {
    "wiki": {"wiki"},
    "graph": {"knowledge_graph"},
    "mindmap": {"mind_map"},
    "timeline": {"timeline"},
    "tree": {"tree", "page_index"},
}

# 数据集合并结构类型的来源为“`source_doc_ids`”或
# ``doc_ids_kwd`` on ``scope_kwd="dataset"`` 行，通过其标记发现
# 顶级种。
_ALTERATION_KIND_TO_MERGED_ROW_KIND = {
    "graph": "knowledge_graph",
    "mindmap": "mind_map",
    "timeline": "timeline",
}

# 每文档结构类型为每个文档保留一个产品行；出处是
# 该行自己的 ``doc_id`` (no dataset merge, no ``source_doc_ids``). ``tree`` 和
# ``page_index`` write distinct ``compile_kwd`` 值。
_ALTERATION_TREE_COMPILE_KWDS = ["tree", "page_index"]


async def _current_dataset_docs(dataset_id: str):
    """返回数据集的“`(docs, current_doc_id_set)`”。"""
    docs, _ = await thread_pool_exec(
        DocumentService.get_by_kb_id,
        kb_id=dataset_id,
        page_number=0,
        items_per_page=0,
        orderby="create_time",
        desc=False,
        keywords="",
        run_status=[],
        types=[],
        suffix=[],
    )
    docs = [doc for doc in docs if str(doc.get("status", "1")) != "0"]
    docs = docs or []
    current_doc_ids = {str(doc.get("id")) for doc in docs if doc.get("id")}
    return docs, current_doc_ids


async def _disabled_dataset_doc_ids(dataset_id: str) -> set[str]:
    return await thread_pool_exec(DocumentService.get_disabled_doc_ids_by_kb_id, dataset_id)


def _alteration_result(current_doc_ids: set, involved_doc_ids: set, eligible_doc_ids: set) -> dict:
    """构建每个“`kind`”共享的漂移响应字典。"""
    removed_doc_ids = sorted(involved_doc_ids - current_doc_ids)
    newly_uploaded_doc_ids = sorted(eligible_doc_ids - involved_doc_ids)
    return {
        "removed": len(removed_doc_ids),
        "newly_uploaded": len(newly_uploaded_doc_ids),
        "removed_doc_ids": removed_doc_ids,
        "newly_uploaded_doc_ids": newly_uploaded_doc_ids,
        "involved_doc_ids": sorted(involved_doc_ids),
        "eligible_doc_ids": sorted(eligible_doc_ids),
    }


async def _wiki_chunk_alteration(
    tenant_id: str,
    dataset_id: str,
    eligible_doc_ids: set[str],
    involved_doc_ids: set[str],
) -> dict:
    """将活动源块与 Wiki MAP 使用的哈希值进行比较。

    Document级别出处无法检测到编辑、添加或删除
    块。  MAP 简历行已包含由
    编译，因此它们是权威的编译端快照。
    当添加、删除块时，先前涉及的文档会发生更改，
    或者有不同的哈希值。在文档级别删除或禁用文档
    仍然由现有的“`removed`` fields rather than being
    duplicated in ``changed`”表示。"""
    empty = {
        "changed": 0,
        "changed_doc_ids": [],
    }
    if not eligible_doc_ids and not involved_doc_ids:
        return empty

    from rag.advanced_rag.knowlege_compile.wiki import (
        _wiki_compare_chunk_states,
        _wiki_load_active_map_state,
        _wiki_scan_current_chunk_state,
    )

    try:
        current = await _wiki_scan_current_chunk_state(tenant_id, dataset_id, eligible_doc_ids)
        previous = await _wiki_load_active_map_state(tenant_id, dataset_id)
    except Exception as exc:
        logging.exception("变更：无法比较 kb=%s 的 Wiki 块状态", dataset_id)
        raise RuntimeError(f"Failed to compare Wiki chunk state for alteration (kb={dataset_id})") from exc

    delta = _wiki_compare_chunk_states(previous, current)
    changed_doc_ids = {
        str((current.get(chunk_id) or previous.get(chunk_id) or {}).get("doc_id") or "") for chunk_id in delta["new_chunk_ids"] | delta["changed_chunk_ids"] | delta["deleted_chunk_ids"]
    }
    # ``newly_uploaded`` 拥有未促成的合格文件
    # 当前维基； “`removed`”拥有先前涉及的文件
    # 不再符合资格。保持“`changed`”与两个类别不相交。
    changed_doc_ids &= eligible_doc_ids & involved_doc_ids

    return {
        "changed": len(changed_doc_ids),
        "changed_doc_ids": sorted(changed_doc_ids),
    }


def _flatten_provenance_doc_ids(value) -> set[str]:
    """将 source_doc_ids 标准化存储为 JSON 字符串、列表或标量。"""
    if value is None:
        return set()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return set()
        try:
            return _flatten_provenance_doc_ids(json.loads(raw))
        except (json.JSONDecodeError, TypeError):
            return {raw}
    if isinstance(value, (list, tuple, set)):
        result: set[str] = set()
        for item in value:
            result.update(_flatten_provenance_doc_ids(item))
        return result
    return {str(value)}


def _eligible_doc_ids_for_kind(docs, tenant_id: str, kind: str) -> set:
    """parser_config 或管道携带“`kind`”模板的文档 ID。"""
    accepted = _ALTERATION_ELIGIBLE_TEMPLATE_KINDS.get(kind) or set()
    template_cache: dict[str, bool] = {}
    group_cache: dict[str, bool] = {}
    pipeline_cache: dict[str, bool] = {}
    eligible: set[str] = set()
    for doc in docs or []:
        if str(doc.get("status", "1")) == "0":
            continue
        doc_id = str(doc.get("id") or "")
        if not doc_id:
            continue
        parser_config = doc.get("parser_config") or {}
        if _parser_config_has_template_of_kind(parser_config, tenant_id, accepted, template_cache):
            eligible.add(doc_id)
            continue
        if _pipeline_has_compiler_of_kind(doc.get("pipeline_id") or "", tenant_id, accepted, pipeline_cache, group_cache):
            eligible.add(doc_id)
    return eligible


async def _involved_doc_ids_paged(index_nm, dataset_id: str, condition: dict, field: str | list[str], from_list: bool) -> set:
    """对 docStore 搜索进行分页，将出处字段折叠到 doc-id 集中。

    ``from_list`` 将所选字段读取为出处文档 IDs 的列表；
    否则，所选字段是该行自己的标量文档 ID。"""
    from common.doc_store.doc_store_base import OrderByExpr

    involved: set[str] = set()
    fields = [field] if isinstance(field, str) else field
    select_fields = ["id", *fields]
    offset = 0
    page_size = 1000
    while True:
        try:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                select_fields=select_fields,
                highlight_fields=[],
                condition=condition,
                match_expressions=[],
                order_by=OrderByExpr(),
                offset=offset,
                limit=page_size,
                index_names=index_nm,
                knowledgebase_ids=[dataset_id],
            )
            rows = settings.docStoreConn.get_fields(res, select_fields) or {}
        except Exception:
            logging.exception("更改：kb=%s cond=%s 的 docStore 搜索失败", dataset_id, condition)
            rows = {}

        if not rows:
            break
        for row in rows.values():
            if from_list:
                for field_name in fields:
                    involved.update(_flatten_provenance_doc_ids(row.get(field_name)))
            else:
                involved.update(_flatten_provenance_doc_ids(row.get(fields[0])))

        offset += page_size
        total = settings.docStoreConn.get_total(res)
        if not total or offset >= int(total):
            break
    return involved


async def _involved_doc_ids_for_kind(index_nm, dataset_id: str, kind: str) -> set:
    """收集“`kind`”编译产品中的文档 ID。"""
    if kind == "wiki":
        return await _involved_doc_ids_paged(index_nm, dataset_id, {"compile_kwd": [WIKI_PAGE_COMPILE_KWD]}, "source_doc_ids", from_list=True)
    if kind in _ALTERATION_KIND_TO_MERGED_ROW_KIND:
        condition = {
            "knowledge_graph_kwd": ["entity", "relation"],
            "scope_kwd": ["dataset"],
            "compilation_template_kind_kwd": [_ALTERATION_KIND_TO_MERGED_ROW_KIND[kind]],
        }
        involved = await _involved_doc_ids_paged(index_nm, dataset_id, condition, ["source_doc_ids", "doc_ids_kwd"], from_list=True)
        if involved:
            return involved

        # 历史数据集行可能早于出处列。恢复
        # 源文件集由其 template/compile 身份进行修改
        # 仍然报告禁用文档而不是处理每个活动文档
        # 文件为新上传的。
        template_ids = await _involved_doc_ids_paged(index_nm, dataset_id, condition, "compilation_template_ids", from_list=True)
        compile_kwds = await _involved_doc_ids_paged(index_nm, dataset_id, condition, "compile_kwd", from_list=True)
        if not template_ids and not compile_kwds:
            legacy_condition = {
                "knowledge_graph_kwd": [_DATASET_STRUCTURE_ROW_KWD],
                "compilation_template_kind_kwd": [_ALTERATION_KIND_TO_MERGED_ROW_KIND[kind]],
            }
            template_ids = await _involved_doc_ids_paged(index_nm, dataset_id, legacy_condition, "compilation_template_ids", from_list=True)
            compile_kwds = await _involved_doc_ids_paged(index_nm, dataset_id, legacy_condition, "compile_kwd", from_list=True)
        source_condition: dict = {"knowledge_graph_kwd": ["entity", "relation"]}
        if template_ids:
            source_condition["compilation_template_ids"] = sorted(template_ids)
        elif compile_kwds:
            source_condition["compile_kwd"] = sorted(compile_kwds)
        else:
            return set()
        involved = await _involved_doc_ids_paged(index_nm, dataset_id, source_condition, "doc_id", from_list=False)
        involved.discard(dataset_id)
        return involved
    if kind == "tree":
        return await _involved_doc_ids_paged(index_nm, dataset_id, {"compile_kwd": list(_ALTERATION_TREE_COMPILE_KWDS)}, "doc_id", from_list=False)
    return set()


async def _get_alteration(dataset_id: str, tenant_id: str, kind: str):
    """共享驱动程序：“`kind`”产品和数据集之间的文档级漂移。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    ok, kb = KnowledgebaseService.get_by_id(dataset_id)
    if not ok:
        return False, "Invalid Dataset ID"

    docs, current_doc_ids = await _current_dataset_docs(dataset_id)
    eligible_doc_ids = _eligible_doc_ids_for_kind(docs, kb.tenant_id, kind)

    involved_doc_ids: set[str] = set()
    chunk_changes = None
    pack = _compiled_index_or_none(kb.tenant_id, dataset_id)
    if pack is not None:
        index_nm, _ = pack
        involved_doc_ids = await _involved_doc_ids_for_kind(index_nm, dataset_id, kind)
        if kind == "wiki":
            chunk_changes = await _wiki_chunk_alteration(
                kb.tenant_id,
                dataset_id,
                eligible_doc_ids,
                involved_doc_ids,
            )

    # Wiki 会员资格遵循编译资格。禁用文档或
    # 因此，
    # 删除其 Wiki 模板就是删除；再次启用它或
    # 重建后恢复模板使其重新上传。
    alteration_current_doc_ids = eligible_doc_ids if kind == "wiki" else current_doc_ids
    result = _alteration_result(alteration_current_doc_ids, involved_doc_ids, eligible_doc_ids)
    if chunk_changes is not None:
        result.update(chunk_changes)
    return True, result


async def get_wiki_alteration(dataset_id: str, tenant_id: str):
    """返回当前数据集文档和编译的 wiki 出处之间的文档级偏差。"""
    return await _get_alteration(dataset_id, tenant_id, "wiki")


async def get_structure_alteration(dataset_id: str, tenant_id: str, kind: str):
    """返回非 wiki 结构的文档级漂移``kind``.

    ``kind`` is one of `ZXQKEEP004500 04ZXQ` / ``mindmap`` / ``timeline`ZXQKEEP004 50009ZXQ`source_doc_ids``) or ``tree`` (per-document
    products covering `ZXQKEE P00450014ZXQ` + ``page_index``, provenance from ``doc_id``)。"""
    kind = (kind or "").strip().lower()
    if kind not in _ALTERATION_KIND_TO_MERGED_ROW_KIND and kind != "tree":
        return False, f"Unsupported structure kind: {kind!r}. Expected one of: graph, mindmap, timeline, tree."
    return await _get_alteration(dataset_id, tenant_id, kind)


async def list_wiki_pages(
    dataset_id: str,
    tenant_id: str,
    page: int = 1,
    page_size: int = 200,
    page_type: str | None = None,
    topic: str | None = None,
    keywords: str = "",
):
    """列出左侧 2 列列表的工件页面。

    返回 ``(True, {"total", "items": [{slug, title, page_type}, ...]})``.
    Ordering: ``page_type`` ascending, then ``title`` 升序 — 保持
    相同类型的页面在视觉上分组在一起。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _wiki_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, {"total": 0, "items": []}
    index_nm, _ = pack

    from common.doc_store.doc_store_base import OrderByExpr

    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 200), 1000))
    offset = (page - 1) * page_size
    page_type = page_type.strip() if isinstance(page_type, str) else page_type
    topic = topic.strip() if isinstance(topic, str) else topic
    keywords = (keywords or "").strip().casefold()

    condition: dict = {"compile_kwd": [WIKI_PAGE_COMPILE_KWD]}
    if page_type:
        condition["page_type_kwd"] = [page_type]
    if topic:
        condition["topic_kwd"] = [topic]

    order_by = OrderByExpr()
    try:
        # 最常连接的页面在前： outlinks_int = len(outlinks_kwd) 是
        # 是由持久层为这个查询编写的。
        order_by.desc("outlinks_int").asc("title_kwd")
    except Exception:
        # OrderByExpr API 在文档存储后端之间有所不同；降级为
        # 默认顺序而不是 500。
        order_by = OrderByExpr()

    select_fields = [
        "id",
        "slug_kwd",
        "title_kwd",
        "page_type_kwd",
        "topic_kwd",
        "outlinks_int",
        "summary_with_weight",
    ]

    def _to_item(row):
        slug = _scalar(row.get("slug_kwd"))
        if not slug:
            return None
        return {
            "slug": slug,
            "title": _scalar(row.get("title_kwd")) or slug,
            "page_type": _scalar(row.get("page_type_kwd")) or "concept",
            "topic": _scalar(row.get("topic_kwd")) or "",
            "summary": row.get("summary_with_weight") or "",
        }

    try:
        if not keywords:
            res = settings.docStoreConn.search(
                select_fields=select_fields,
                highlight_fields=[],
                condition=condition,
                match_expressions=[],
                order_by=order_by,
                offset=offset,
                limit=page_size,
                index_names=index_nm,
                knowledgebase_ids=[dataset_id],
            )
            field_map = settings.docStoreConn.get_fields(res, select_fields)
            items = [item for row in (field_map or {}).values() if (item := _to_item(row)) is not None]
            total = settings.docStoreConn.get_total(res)
        else:
            # Wiki 列表搜索有意基于元数据。这些领域
            # 可用于现有行，并且在整个行中表现一致
            # 每个文档存储后端，与全文查询不同
            # 后端特定的令牌字段。
            matched_items = []
            batch_size = 1000
            search_offset = 0
            while True:
                res = settings.docStoreConn.search(
                    select_fields=select_fields,
                    highlight_fields=[],
                    condition=condition,
                    match_expressions=[],
                    order_by=order_by,
                    offset=search_offset,
                    limit=batch_size,
                    index_names=index_nm,
                    knowledgebase_ids=[dataset_id],
                )
                field_map = settings.docStoreConn.get_fields(res, select_fields)
                rows = list((field_map or {}).values())
                if not rows:
                    break
                for row in rows:
                    item = _to_item(row)
                    if item is not None and any(keywords in str(item[field]).casefold() for field in ("title", "slug", "summary")):
                        matched_items.append(item)
                if len(rows) < batch_size:
                    break
                search_offset += batch_size
            total = len(matched_items)
            items = matched_items[offset : offset + page_size]
    except Exception:
        logging.exception("list_wiki_pages：kb = %s 的 docStore 搜索失败", dataset_id)
        return True, {"total": 0, "items": []}

    return True, {"total": int(total or 0), "items": items}


async def list_wiki_topics(
    dataset_id: str,
    tenant_id: str,
    page: int = 1,
    page_size: int = 200,
    keywords: str = "",
):
    """列出数据集 Artifact 选项卡的 wiki 主题。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _wiki_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, {"total": 0, "items": []}
    index_nm, _ = pack

    from common.doc_store.doc_store_base import OrderByExpr

    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 200), 1000))
    offset = (page - 1) * page_size
    keywords = (keywords or "").strip().casefold()

    try:
        agg_res = settings.docStoreConn.search(
            select_fields=["id"],
            highlight_fields=[],
            condition={"compile_kwd": [WIKI_PAGE_COMPILE_KWD], "page_type_kwd": ["concept", "entity"]},
            match_expressions=[],
            order_by=OrderByExpr(),
            offset=0,
            limit=0,
            index_names=index_nm,
            knowledgebase_ids=[dataset_id],
            agg_fields=["topic_kwd"],
        )
        buckets = settings.docStoreConn.get_aggregation(agg_res, "topic_kwd")
    except Exception:
        logging.exception("list_wiki_topics：kb=%s 的 docStore 聚合失败", dataset_id)
        return True, {"total": 0, "items": []}

    counts = {t: int(c) for t, c in (buckets or []) if isinstance(t, str) and t and int(c or 0) > 0}
    if not counts:
        return True, {"total": 0, "items": []}

    # 按页数（降序）对主题进行排名，然后是标题以获得稳定的顺序。
    ranked = sorted(
        (
            {
                "topic": t,
                "title": t.rsplit("/", 1)[-1],
                "slug": t,
                "page_count": c,
            }
            for t, c in counts.items()
        ),
        key=lambda x: (-x["page_count"], x["title"].lower()),
    )

    if keywords:
        matching_topics = {item["topic"] for item in ranked if any(keywords in str(item[field]).casefold() for field in ("topic", "title", "slug"))}

        child_fields = ["topic_kwd", "title_kwd", "slug_kwd", "summary_with_weight"]
        batch_size = 1000
        child_offset = 0
        try:
            while True:
                child_res = settings.docStoreConn.search(
                    select_fields=child_fields,
                    highlight_fields=[],
                    condition={"compile_kwd": [WIKI_PAGE_COMPILE_KWD], "page_type_kwd": ["concept", "entity"]},
                    match_expressions=[],
                    order_by=OrderByExpr(),
                    offset=child_offset,
                    limit=batch_size,
                    index_names=index_nm,
                    knowledgebase_ids=[dataset_id],
                )
                child_map = settings.docStoreConn.get_fields(child_res, child_fields)
                child_rows = list((child_map or {}).values())
                if not child_rows:
                    break
                for row in child_rows:
                    topic_name = _scalar(row.get("topic_kwd"))
                    if topic_name and any(keywords in str(_scalar(row.get(field)) or "").casefold() for field in ("title_kwd", "slug_kwd", "summary_with_weight")):
                        matching_topics.add(topic_name)
                if len(child_rows) < batch_size:
                    break
                child_offset += batch_size
        except Exception:
            logging.exception("list_wiki_topics：kb=%s 的子页面关键字查找失败", dataset_id)

        ranked = [item for item in ranked if item["topic"] in matching_topics]

    total = len(ranked)
    items = ranked[offset : offset + page_size]
    return True, {"total": total, "items": items}


async def get_wiki_page(
    dataset_id: str,
    tenant_id: str,
    page_type: str,
    slug: str,
):
    """为右侧 Markdown 查看器获取单个工件页面。

    ``slug`` is the tail after ``<page_type>/`` — i.e. the URL component
    that came from the markdown link ``artifact/<kb_id>/<page_type>/<slug>``.
    The stored `ZXQKEEP00 010006ZXQ` is the full ``<page_type>/<slug>`` form, so we
    reconstruct it before the lookup.

    Returns ``(True, page_dict)`` or ``(True, None)``当没有行匹配时。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _wiki_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, None
    index_nm, _ = pack

    from common.doc_store.doc_store_base import OrderByExpr

    full_slug = f"{page_type}/{slug}" if "/" not in slug else slug
    select_fields = [
        "id",
        "slug_kwd",
        "title_kwd",
        "page_type_kwd",
        "topic_kwd",
        "md_with_weight",
        "content_with_weight",
        "summary_with_weight",
        "entity_names_kwd",
        "outlinks_kwd",
        "related_kb_pages_kwd",
        "source_chunk_ids",
        "source_doc_ids",
    ]
    try:
        res = settings.docStoreConn.search(
            select_fields=select_fields,
            highlight_fields=[],
            condition={
                "compile_kwd": [WIKI_PAGE_COMPILE_KWD],
                "page_type_kwd": [page_type],
                "slug_kwd": [full_slug],
            },
            match_expressions=[],
            order_by=OrderByExpr(),
            offset=0,
            limit=1,
            index_names=index_nm,
            knowledgebase_ids=[dataset_id],
        )
        field_map = settings.docStoreConn.get_fields(res, select_fields)
    except Exception:
        logging.exception(
            "get_wiki_page：搜索 kb=%s slug=%s 失败",
            dataset_id,
            full_slug,
        )
        return True, None

    if not field_map:
        return True, None

    _, row = next(iter(field_map.items()))
    # 增量写入器将页体存储在md_with_weight中；回落到
    # content_with_weight 用于旧路径写入的任何行。
    content_md = row.get("md_with_weight") or row.get("content_with_weight") or ""
    summary = row.get("summary_with_weight") or ""
    return True, {
        "slug": _scalar(row.get("slug_kwd")) or full_slug,
        "title": _scalar(row.get("title_kwd")) or full_slug,
        "page_type": _scalar(row.get("page_type_kwd")) or page_type,
        "topic": _scalar(row.get("topic_kwd")) or "",
        "content_md_rendered": content_md,
        "summary": summary,
        "entity_names": _string_list(row.get("entity_names_kwd")),
        "outlinks": _string_list(row.get("outlinks_kwd")),
        "related_kb_pages": _string_list(row.get("related_kb_pages_kwd")),
        "source_chunk_ids": _string_list(row.get("source_chunk_ids")),
        "source_doc_ids": _string_list(row.get("source_doc_ids")),
    }


async def has_any_skill(dataset_id: str, tenant_id: str):
    """数据集技能侧边栏条目的快速存在探测。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _skill_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, {"has": False}
    index_nm, _ = pack

    from common.doc_store.doc_store_base import OrderByExpr

    try:
        res = settings.docStoreConn.search(
            select_fields=["id"],
            highlight_fields=[],
            condition={"compile_kwd": [_SKILL_ALL_COMPILE_KWD]},
            match_expressions=[],
            order_by=OrderByExpr(),
            offset=0,
            limit=1,
            index_names=index_nm,
            knowledgebase_ids=[dataset_id],
        )
    except Exception:
        logging.exception("has_any_skill：kb=%s 的 docStore 搜索失败", dataset_id)
        return True, {"has": False}

    total = settings.docStoreConn.get_total(res)
    return True, {"has": bool(total)}


async def get_skill_tree(dataset_id: str, tenant_id: str):
    """获取该数据集的一次性递归技能树。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _skill_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, None
    index_nm, _ = pack

    from common.doc_store.doc_store_base import OrderByExpr

    select_fields = ["id", "kb_id", "doc_id", "compile_kwd", "skill_with_weight"]
    try:
        res = settings.docStoreConn.search(
            select_fields=select_fields,
            highlight_fields=[],
            condition={"compile_kwd": [_SKILL_ALL_COMPILE_KWD]},
            match_expressions=[],
            order_by=OrderByExpr(),
            offset=0,
            limit=1,
            index_names=index_nm,
            knowledgebase_ids=[dataset_id],
        )
        field_map = settings.docStoreConn.get_fields(res, select_fields)
    except Exception:
        logging.exception("get_skill_tree：kb = %s 的 docStore 搜索失败", dataset_id)
        return True, None

    if not field_map:
        return True, None

    _, row = next(iter(field_map.items()))
    return True, {
        "id": row.get("id"),
        "kb_id": row.get("kb_id") or dataset_id,
        "doc_id": row.get("doc_id") or dataset_id,
        "compile_kwd": row.get("compile_kwd") or _SKILL_ALL_COMPILE_KWD,
        "skill_with_weight": json.loads(row.get("skill_with_weight")) or [],
    }


async def delete_skills(dataset_id: str, tenant_id: str):
    """删除所有已编译的技能行(``skill`` + ``skill_all``) for a dataset.

    Returns ``(True, {"deleted": <n>})`` on success. When the tenant index does
    not exist yet there is nothing to delete, so it succeeds with ``0``."""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _skill_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, {"deleted": 0}
    index_nm, _ = pack

    try:
        deleted = settings.docStoreConn.delete(
            {"compile_kwd": [_SKILL_COMPILE_KWD, _SKILL_ALL_COMPILE_KWD]},
            index_nm,
            dataset_id,
        )
    except Exception:
        logging.exception("delete_skills：kb = %s 的 docStore 删除失败", dataset_id)
        return False, "Failed to delete skills."

    # 清除技能编译标记，以便数据集反映 "no skill"
    # （并且稍后的重新编译不会因过时的任务 ID 而短路）。
    try:
        KnowledgebaseService.update_by_id(kb.id, {"skill_task_id": "", "skill_task_finish_at": None})
    except Exception:
        logging.exception("delete_skills：kb=%s 的技能任务标记清除失败", dataset_id)

    return True, {"deleted": int(deleted or 0)}


def _parse_list_field(value):
    """规范化搜索结果“`_kwd`` field to a Python list.

    Infinity stores keyword fields as ``###`”连接的字符串；
    ES / OpenSearch / OceanBase 保留本机列表。"""
    if isinstance(value, list):
        return [v for v in value if v]
    if isinstance(value, str) and value:
        return [v for v in value.split("###") if v]
    return []


def _walk_skill_tree(tree: list[dict], target_kwd: str, parent_kwd: str | None = None):
    """通过``skill_all`` tree JSON.

    Returns ``(parent_kwd, found_node)`` or ``(None, None)``时进行深度优先搜索*target_kwd*
    不存在于树中。"""
    for node in tree:
        if node.get("skill_kwd") == target_kwd:
            return parent_kwd, node
        children = node.get("children_kwd", [])
        if children:
            result = _walk_skill_tree(children, target_kwd, node.get("skill_kwd"))
            p, n = result
            if n is not None:
                return p, n
    return None, None


def _collect_tree_descendants(node: dict) -> set[str]:
    """从 *node* 的子树返回所有 ``skill_kwd`` 值（递归）。"""
    result: set[str] = set()
    for child in node.get("children_kwd", []):
        kwd = child.get("skill_kwd", "")
        if kwd:
            result.add(kwd)
        result.update(_collect_tree_descendants(child))
    return result


def _prune_skill_tree(tree: list[dict], deleted_kwds: set[str]) -> list[dict]:
    """删除 *deleted_kwds* 中包含 ``skill_kwd`` 的节点（及其子树）。"""
    result = []
    for node in tree:
        if node.get("skill_kwd") in deleted_kwds:
            continue
        children = node.get("children_kwd", [])
        node["children_kwd"] = _prune_skill_tree(children, deleted_kwds)
        result.append(node)
    return result


async def delete_skill(dataset_id: str, tenant_id: str, skill_kwd: str):
    """删除已编译的技能节点及其后代。

    1. 步行``skill_all`` tree to identify the target node, its parent,
       and all descendant ``skill_kwd`` values.
    2. Delete every matching `ZXQKEEP00 180004ZXQ` row (self + subtree).
    3. If a parent exists, remove the deleted node from its ``children_kwd``.
    4. Prune the ``skill_all``树并重写聚合行。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _skill_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, {"deleted": 0}
    index_nm, _ = pack

    # ------------------------------------------------------------------
    # 1.读取当前skill_all树
    # ------------------------------------------------------------------
    from common.doc_store.doc_store_base import OrderByExpr

    _ALL_SELECT = ["id", "kb_id", "doc_id", "compile_kwd", "skill_with_weight", "available_int"]
    try:
        res = settings.docStoreConn.search(
            select_fields=_ALL_SELECT,
            highlight_fields=[],
            condition={"compile_kwd": [_SKILL_ALL_COMPILE_KWD]},
            match_expressions=[],
            order_by=OrderByExpr(),
            offset=0,
            limit=1,
            index_names=index_nm,
            knowledgebase_ids=[dataset_id],
        )
        field_map = settings.docStoreConn.get_fields(res, _ALL_SELECT)
    except Exception:
        logging.exception("delete_skill：读取skill_all失败kb=%s技能=%s", dataset_id, skill_kwd)
        return False, "Failed to read skill tree."

    if not field_map:
        return True, {"deleted": 0}
    _, skill_all_row = next(iter(field_map.items()))

    raw_tree = skill_all_row.get("skill_with_weight", "[]")
    tree = json.loads(raw_tree) if isinstance(raw_tree, str) else raw_tree
    tree = tree if isinstance(tree, list) else [tree]

    # ------------------------------------------------------------------
    # 2. 定位目标节点并收集后代
    # ------------------------------------------------------------------
    parent_kwd, found_node = _walk_skill_tree(tree, skill_kwd)
    if found_node is None:
        # Fallback：树中未跟踪节点；无论如何尝试直接删除。
        try:
            deleted = settings.docStoreConn.delete(
                {"compile_kwd": [_SKILL_COMPILE_KWD], "skill_kwd": [skill_kwd]},
                index_nm,
                dataset_id,
            )
        except Exception:
            logging.exception("delete_skill：回退删除失败kb = %s技能= %s", dataset_id, skill_kwd)
            return False, "Failed to delete skill."
        return True, {"deleted": int(deleted or 0)}

    descendant_kwds = _collect_tree_descendants(found_node)
    all_kwds = [skill_kwd] + sorted(descendant_kwds)

    # ------------------------------------------------------------------
    # 3.删除compile_kwd="skill"行（自身+所有后代）
    # ------------------------------------------------------------------
    try:
        deleted = settings.docStoreConn.delete(
            {"compile_kwd": [_SKILL_COMPILE_KWD], "skill_kwd": all_kwds},
            index_nm,
            dataset_id,
        )
    except Exception:
        logging.exception("delete_skill：删除失败kb=%s技能=%s kwds=%s", dataset_id, skill_kwd, all_kwds)
        return False, "Failed to delete skill."

    # ------------------------------------------------------------------
    # 4.更新父级的children_kwd
    # ------------------------------------------------------------------
    if parent_kwd:
        _NODE_SELECT = [
            "id",
            "kb_id",
            "doc_id",
            "compile_kwd",
            "skill_kwd",
            "depth_int",
            "children_kwd",
            "source_doc_ids",
            "md_with_weight",
            "available_int",
            "parent_kwd",
        ]
        try:
            res = settings.docStoreConn.search(
                select_fields=_NODE_SELECT,
                highlight_fields=[],
                condition={"compile_kwd": [_SKILL_COMPILE_KWD], "skill_kwd": [parent_kwd]},
                match_expressions=[],
                order_by=OrderByExpr(),
                offset=0,
                limit=1,
                index_names=index_nm,
                knowledgebase_ids=[dataset_id],
            )
            fm = settings.docStoreConn.get_fields(res, _NODE_SELECT)
        except Exception:
            logging.exception("delete_skill：读取父级失败 kb=%s 父级=%s", dataset_id, parent_kwd)
        else:
            if fm:
                _, parent_row = next(iter(fm.items()))
                children = [c for c in _parse_list_field(parent_row.get("children_kwd", [])) if c != skill_kwd]
                parent_row["children_kwd"] = children
                try:
                    settings.docStoreConn.insert([parent_row], index_nm, dataset_id)
                except Exception:
                    logging.exception("delete_skill：更新父级 children_kwd 失败 kb=%s 父级=%s", dataset_id, parent_kwd)

    # ------------------------------------------------------------------
    # 5.修剪并重写skill_all树
    # ------------------------------------------------------------------
    try:
        deleted_set = set(all_kwds)
        pruned_tree = _prune_skill_tree(tree, deleted_set)
        skill_all_row["skill_with_weight"] = json.dumps(pruned_tree, ensure_ascii=False, indent=2)
        settings.docStoreConn.insert([skill_all_row], index_nm, dataset_id)
    except Exception:
        logging.exception("delete_skill：重写skill_all失败kb=%s技能=%s", dataset_id, skill_kwd)
        return False, "Failed to update skill tree."

    return True, {"deleted": int(deleted or 0)}


async def get_skill_page(dataset_id: str, tenant_id: str, skill_kwd: str):
    """获取单个技能节点的完整 Markdown 正文。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _skill_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, None
    index_nm, _ = pack

    from common.doc_store.doc_store_base import OrderByExpr

    select_fields = [
        "id",
        "kb_id",
        "doc_id",
        "compile_kwd",
        "skill_kwd",
        "depth_int",
        "children_kwd",
        "source_doc_ids",
        "md_with_weight",
    ]
    try:
        res = settings.docStoreConn.search(
            select_fields=select_fields,
            highlight_fields=[],
            condition={
                "compile_kwd": [_SKILL_COMPILE_KWD],
                "skill_kwd": [skill_kwd],
            },
            match_expressions=[],
            order_by=OrderByExpr(),
            offset=0,
            limit=1,
            index_names=index_nm,
            knowledgebase_ids=[dataset_id],
        )
        field_map = settings.docStoreConn.get_fields(res, select_fields)
    except Exception:
        logging.exception(
            "get_skill_page：kb=%s技能=%s的docStore搜索失败",
            dataset_id,
            skill_kwd,
        )
        return True, None

    if not field_map:
        return True, None

    _, row = next(iter(field_map.items()))
    return True, {
        "id": row.get("id"),
        "kb_id": row.get("kb_id") or dataset_id,
        "doc_id": row.get("doc_id") or dataset_id,
        "compile_kwd": row.get("compile_kwd") or _SKILL_COMPILE_KWD,
        "skill_kwd": row.get("skill_kwd") or skill_kwd,
        "depth_int": row.get("depth_int") or 0,
        "children_kwd": row.get("children_kwd") or [],
        "source_doc_ids": row.get("source_doc_ids") or [],
        "md_with_weight": row.get("md_with_weight") or "",
    }


# ---------------------------------------------------------------------------
# 数据集导航树（作者：rag/advanced_rag/knowlege_compile/
# dataset_nav.py 为 nav_cluster / nav_doc 行）。分层加载：
# 第一次调用返回顶部簇（父簇 = "root"）；单击集群
# 返回其直接子节点（子集群 + 文档叶）。这些行是
# ``available_int=0`` 因此可以直接通过 docStoreConn.search 读取它们。
# ---------------------------------------------------------------------------

_NAV_COMPILE_KWD = "dataset_nav"
_NAV_ROOT_PARENT = "root"
_NAV_FIELDS = [
    "id",
    "name",
    "type_kwd",
    "content_with_weight",
    "doc_count_int",
    "doc_ids_kwd",
    "doc_id",
    "depth_int",
    "parent_kwd",
]


def _first_str(val) -> str:
    """从可能是 str、list、tuple 或 set 的值中提取第一个字符串。"""
    if isinstance(val, (list, tuple, set)):
        return str(next(iter(val), "") or "")
    return str(val or "")


def _resolve_embd_mdl(kb):
    """解析知识库的嵌入模型，如果失败则为“无”。"""
    from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type
    from api.db.services.llm_service import LLMBundle
    from common.constants import LLMType

    try:
        if kb.embd_id:
            embd_model_config = resolve_model_config(kb.tenant_id, LLMType.EMBEDDING, kb.embd_id)
        else:
            embd_model_config = get_tenant_default_model_by_type(kb.tenant_id, LLMType.EMBEDDING)
        if embd_model_config is None:
            return None
        return LLMBundle(kb.tenant_id, embd_model_config)
    except Exception:
        logging.exception("无法解析 kb=%s 的嵌入模型", kb.id)
        return None


def _nav_item(row: dict) -> dict:
    """将一个导航行塑造成一个 UI 节点：名称、描述、文档计数、类型。"""
    try:
        payload = json.loads(row.get("content_with_weight") or "{}")
    except Exception:
        payload = {}
    row_type = row.get("type_kwd")
    if isinstance(row_type, (list, tuple, set)):
        row_type = next(iter(row_type), None)
    is_cluster = (row_type or payload.get("type")) == "nav_cluster"
    return {
        "name": row.get("name") or "",
        "description": payload.get("description") or "",
        "keywords": list(payload.get("keywords") or []),
        "entities": list(payload.get("entities") or []),
        "graph_content": payload.get("graph_content") or "",
        # doc_id 该节点下的计数：集群计数，或者叶子为 1。
        "doc_count": int(row.get("doc_count_int") or 0) if is_cluster else 1,
        "type": "cluster" if is_cluster else "doc",
        "doc_id": None if is_cluster else (row.get("doc_id") or row.get("name")),
        "has_children": is_cluster,
    }


async def _nav_search(dataset_id: str, tenant_id: str, condition: dict, page: int, page_size: int):
    """运行一个导航树搜索并将命中结果整形为 UI 节点。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _compiled_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, {"total": 0, "items": []}
    index_nm, _ = pack

    from common.doc_store.doc_store_base import OrderByExpr

    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 1000), 2000))
    offset = (page - 1) * page_size

    order_by = OrderByExpr()
    try:
        # 最大的簇优先；叶子（无doc_count_int）落到最后。
        order_by.desc("doc_count_int")
    except Exception:
        order_by = OrderByExpr()

    try:
        res = settings.docStoreConn.search(
            select_fields=_NAV_FIELDS,
            highlight_fields=[],
            condition=condition,
            match_expressions=[],
            order_by=order_by,
            offset=offset,
            limit=page_size,
            index_names=index_nm,
            knowledgebase_ids=[dataset_id],
        )
        field_map = settings.docStoreConn.get_fields(res, _NAV_FIELDS)
    except Exception:
        logging.exception("dataset_nav：kb = %s 的 docStore 搜索失败", dataset_id)
        return True, {"total": 0, "items": []}

    total = settings.docStoreConn.get_total(res)
    items = [_nav_item(row) for row in (field_map or {}).values()]
    return True, {"total": int(total or 0), "items": items}


async def list_nav_clusters(dataset_id: str, tenant_id: str, page: int = 1, page_size: int = 1000, q: str | None = None, top_k: int | None = None):
    """导航树的第一级：没有父级的集群。

    当``q`` is provided, runs a tree-structured search (mode="navigation_tree")
    and returns enriched nav node items in the same ``_nav_item``形状时，
    前端树搜索可以重用此端点。"""
    if q and q.strip():
        success, result = await search_dataset_layers(dataset_id, tenant_id, q.strip(), "navigation_tree", top_k=top_k or 1000)
        if not success:
            return success, result
        result["items"] = await _enrich_nav_items(dataset_id, tenant_id, result.get("items", []))
        return True, {"total": result.get("total", 0), "items": result["items"]}
    condition = {
        "compile_kwd": [_NAV_COMPILE_KWD],
        "type_kwd": ["nav_cluster"],
        "parent_kwd": [_NAV_ROOT_PARENT],
    }
    return await _nav_search(dataset_id, tenant_id, condition, page, page_size)


async def list_nav_children(dataset_id: str, tenant_id: str, name: str, page: int = 1, page_size: int = 1000):
    """节点“`name`”的直接子节点 — 子集群和文档叶。

    一次一层（惰性），因此树按照用户分层加载
    展开每个节点。"""
    if not isinstance(name, str) or not name.strip():
        return True, {"total": 0, "items": []}
    condition = {
        "compile_kwd": [_NAV_COMPILE_KWD],
        "parent_kwd": [name.strip()],
    }
    return await _nav_search(dataset_id, tenant_id, condition, page, page_size)


async def delete_nav(dataset_id: str, tenant_id: str):
    """删除数据集的整个数据集导航树。

    当没有时返回``(True, {"deleted": <n>})``; succeeds with ``0``
    索引还没有。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _compiled_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, {"deleted": 0}
    index_nm, _ = pack

    try:
        deleted = await thread_pool_exec(
            settings.docStoreConn.delete,
            {"compile_kwd": [_NAV_COMPILE_KWD]},
            index_nm,
            dataset_id,
        )
    except Exception:
        logging.exception("delete_nav：kb = %s 的 docStore 删除失败", dataset_id)
        return False, "Failed to delete the navigation tree."

    return True, {"deleted": int(deleted or 0)}


async def delete_nav_node(dataset_id: str, tenant_id: str, name: str):
    """删除一个导航节点（由``name``) and its whole subtree.

    Children reference their parent by ``name`` (``parent_kwd``标识），因此删除一个导航节点
    没有其后代的集群将使它们在树视图中成为孤儿。
    因此，我们自上而下遍历子树并删除其中的每个节点。"""
    if not isinstance(name, str) or not name.strip():
        return True, {"deleted": 0}
    name = name.strip()

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _compiled_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, {"deleted": 0}
    index_nm, _ = pack

    from common.doc_store.doc_store_base import OrderByExpr
    from rag.advanced_rag.knowlege_compile.dataset_nav import (
        _LOCK_BLOCKING_TIMEOUT_S,
        _LOCK_TIMEOUT_S,
        _nav_lock_key,
        _remove_dataset_nav_doc_locked,
    )
    from rag.utils.redis_conn import RedisDistributedLock

    async def search_rows(condition):
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            ["id", "name", "type_kwd", "parent_kwd", "doc_id"],
            [],
            condition,
            [],
            OrderByExpr(),
            0,
            10000,
            index_nm,
            [dataset_id],
        )
        return list((settings.docStoreConn.get_fields(res, ["id", "name", "type_kwd", "parent_kwd", "doc_id"]) or {}).values())

    lock = RedisDistributedLock(
        _nav_lock_key(dataset_id),
        timeout=_LOCK_TIMEOUT_S,
        blocking_timeout=_LOCK_BLOCKING_TIMEOUT_S,
    )
    try:
        await lock.spin_acquire()
    except Exception:
        logging.exception("delete_nav_node：kb = %s 的锁获取失败", dataset_id)
        return False, "Failed to acquire the navigation tree lock."

    try:
        # ES 需要关键字子字段以实现精确的节点名称匹配。 Infinity
        # 具有标量 varchar 名称字段，而 parent_kwd 是精确匹配的。
        name_field = "name.keyword" if settings.DOC_ENGINE.lower() in {"elasticsearch", "opensearch"} else "name"
        target_condition = {"compile_kwd": [_NAV_COMPILE_KWD], name_field: [name]}
        rows = await search_rows(target_condition)
        if not rows and name_field != "name":
            # 使用 older/missing 映射保留此后备安装。
            rows = [row for row in await search_rows({"compile_kwd": [_NAV_COMPILE_KWD]}) if row.get("name") == name]
        if not rows:
            return True, {"deleted": 0}

        # 按稳定行 IDs 收集目标和后代，而不是按名称。
        # 名称是显示值，可能会与 malformed/old 数据发生冲突。
        rows_by_id = {row.get("id"): row for row in rows if row.get("id")}
        frontier = [row.get("name") for row in rows if row.get("name")]
        for _ in range(64):
            if not frontier:
                break
            children = await search_rows(
                {"compile_kwd": [_NAV_COMPILE_KWD], "parent_kwd": frontier},
            )
            frontier = []
            for child in children:
                row_id = child.get("id")
                child_name = child.get("name")
                if row_id and row_id not in rows_by_id:
                    rows_by_id[row_id] = child
                    if child_name:
                        frontier.append(child_name)

        deleted = 0
        # 重用现有的文档删除路径，以便直接父级
        # 更新和空祖先集群一致清理。
        for row in rows_by_id.values():
            if row.get("type_kwd") == "nav_doc" and row.get("doc_id"):
                await _remove_dataset_nav_doc_locked(tenant_id, dataset_id, row["doc_id"])
                deleted += 1

        cluster_ids = [row_id for row_id, row in rows_by_id.items() if row.get("type_kwd") == "nav_cluster" and row_id]
        if cluster_ids:
            deleted_rows = await thread_pool_exec(
                settings.docStoreConn.delete,
                {"compile_kwd": [_NAV_COMPILE_KWD], "id": cluster_ids},
                index_nm,
                dataset_id,
            )
            deleted += int(deleted_rows or 0)

        return True, {"deleted": deleted}
    except Exception:
        logging.exception("delete_nav_node： kb=%s 删除失败 名称=%s", dataset_id, name)
        return False, "Failed to delete the navigation node."
    finally:
        try:
            lock.release()
        except Exception:
            logging.exception("delete_nav_node：kb = %s 的锁定释放失败", dataset_id)


async def generate_nav(
    dataset_id: str,
    tenant_id: str,
    documents: list[dict] | None = None,
):
    """创建整个导航树。

    首先删除任何现有的导航树，然后重建它
    刮擦。当“`documents`` is provided, only those doc→summary pairs
    are used; otherwise all documents in the dataset are auto-discovered
    and inserted into the tree.

    ``documents`` is a list of ``{"doc_id": str, "summary": str,
    "doc_title": str (optional), "source_type": str (optional)}``.

    Returns ``(True, {"deleted": <n>, "upserted": <n>})`”开启时成功。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"

    _, kb = KnowledgebaseService.get_by_id(dataset_id)
    if kb is None:
        return False, "Dataset not found."

    # 解析模型。
    from api.db.joint_services.tenant_model_service import (
        get_tenant_default_model_by_type,
    )
    from api.db.services.llm_service import LLMBundle
    from common.constants import LLMType
    from rag.advanced_rag.knowlege_compile.dataset_nav import (
        build_nav_graph_text,
        upsert_dataset_nav_doc,
    )

    embd_mdl = _resolve_embd_mdl(kb)

    chat_model_config = get_tenant_default_model_by_type(kb.tenant_id, LLMType.CHAT)
    chat_mdl = LLMBundle(kb.tenant_id, chat_model_config)

    # 步骤0：未明确提供时自动发现文档。
    if not documents:
        try:
            # 直接查询数据集中所有文档，无需
            # File / File2Document JOINs 这样每个文件
            # 参与导航树（包括文档
            # 通过 API 创建的
            # 没有文件记录）。
            from api.db.db_utils import DB
            from api.db.services.doc_metadata_service import DocMetadataService

            with DB.connection_context():
                doc_rows = list(
                    DocumentService.model.select(
                        DocumentService.model.id,
                        DocumentService.model.name,
                    )
                    .where(
                        DocumentService.model.kb_id == dataset_id,
                    )
                    .dicts()
                )
            doc_ids = [str(row["id"]) for row in doc_rows]
            metadata_map = DocMetadataService.get_metadata_for_documents(doc_ids, dataset_id) if doc_ids else {}
            all_docs = []
            for row in doc_rows:
                did = str(row["id"])
                all_docs.append(
                    {
                        "id": did,
                        "name": (row.get("name") or ""),
                        "meta_fields": metadata_map.get(did, {}),
                    }
                )

            # 更喜欢 RAPTOR 生成的摘要存储在知识中
            # 图（compile_kwd="tree"，knowledge_graph_kwd="graph"）
            # 以便重建的导航描述与原始内容相匹配
            # 树编译输出。  回退到 meta_fields.title
            # 或图表不存在时的文件名。
            raptor_summaries: dict[str, str] = {}
            pack = _compiled_index_or_none(kb.tenant_id, dataset_id)
            if pack is not None:
                try:
                    index_nm, _ = pack
                    from common.doc_store.doc_store_base import OrderByExpr

                    graph_res = settings.docStoreConn.search(
                        select_fields=["doc_id", "content_with_weight"],
                        highlight_fields=[],
                        condition={"compile_kwd": ["tree"], "knowledge_graph_kwd": ["graph"]},
                        match_expressions=[],
                        order_by=OrderByExpr(),
                        offset=0,
                        limit=10000,
                        index_names=index_nm,
                        knowledgebase_ids=[dataset_id],
                    )
                    graph_map = settings.docStoreConn.get_fields(graph_res, ["doc_id", "content_with_weight"])
                    for row in (graph_map or {}).values():
                        gid = str(row.get("doc_id") or "")
                        if not gid:
                            continue
                        try:
                            graph = json.loads(row.get("content_with_weight") or "{}")
                        except Exception:
                            continue
                        # 构建两者：
                        # - root_summary：根描述的第一行（短，
                        # 为显示标题/描述字段）
                        # - graph_text：来自 ALL 实体的结构化文本
                        # 和关系（用于嵌入、关键字提取、
                        # 实体提取，并存储为graph_content）。
                        # 与解析时路径共享 (run_tree_templates)
                        # 因此两者产生相同、完整的 nav_doc 内容。
                        root_summary, graph_text = build_nav_graph_text(graph)

                        if root_summary:
                            raptor_summaries[gid] = {
                                "title": root_summary,
                                "graph_text": graph_text or root_summary,
                            }
                except Exception:
                    logging.exception("generate_nav：无法读取 kb=%s 的 RAPTOR 图形摘要", dataset_id)

            documents = []
            for d in all_docs:
                doc_id = str(d.get("id", ""))
                if not doc_id:
                    continue
                if doc_id in raptor_summaries:
                    summary = raptor_summaries[doc_id]
                else:
                    meta = d.get("meta_fields") or {}
                    summary = (meta.get("title") or "").strip() or d.get("name", "")
                documents.append(
                    {
                        "doc_id": doc_id,
                        "summary": summary,
                    }
                )

        except Exception:
            logging.exception("generate_nav：无法自动发现 kb=%s 的文档", dataset_id)
            return False, "Failed to auto-discover documents."

    if not documents:
        return False, "No documents found in dataset."

    # 第 1 步：删除整个现有导航树，以便我们开始清理。
    # ``deleted`` 报告删除的*簇*数量（不是 nav_doc 叶子），
    # 因为这是导航树的有意义的单位。
    deleted = 0
    pack = _compiled_index_or_none(kb.tenant_id, dataset_id)
    if pack is not None:
        index_nm, _ = pack
        try:
            from common.doc_store.doc_store_base import OrderByExpr

            # 在擦除树之前对现有簇进行计数。
            count_res = await thread_pool_exec(
                settings.docStoreConn.search,
                ["id"],
                [],
                {"compile_kwd": [_NAV_COMPILE_KWD], "type_kwd": ["nav_cluster"]},
                [],
                OrderByExpr(),
                0,
                10000,
                index_nm,
                [dataset_id],
            )
            deleted = len(settings.docStoreConn.get_fields(count_res, ["id"]) or {})

            await thread_pool_exec(
                settings.docStoreConn.delete,
                {"compile_kwd": [_NAV_COMPILE_KWD]},
                index_nm,
                dataset_id,
            )
        except Exception:
            logging.exception("generate_nav：无法清除 kb=%s 的现有导航", dataset_id)
            return False, "Failed to clear existing navigation tree."

    # 步骤 2：根据提供的文档→摘要对重建树。
    upserted = 0
    for doc in documents or []:
        doc_id = (doc.get("doc_id") or "").strip()
        summary = doc.get("summary")
        # 摘要可以是纯字符串或 RAPTOR 树字典。
        if isinstance(summary, str):
            summary = summary.strip()
        if not doc_id or not summary:
            continue
        try:
            await upsert_dataset_nav_doc(
                tenant_id=kb.tenant_id,
                kb_id=dataset_id,
                doc_id=doc_id,
                summary_or_tree=summary,
                embd_mdl=embd_mdl,
                chat_mdl=chat_mdl,
            )
            upserted += 1
        except Exception:
            logging.exception("generate_nav： doc=%s kb=%s 失败", doc_id, dataset_id)
            return True, {"deleted": deleted, "upserted": upserted, "failed_doc_id": doc_id}

    return True, {"deleted": deleted, "upserted": upserted}


# ---------------------------------------------------------------------------
# 统一探索 Search
# ---------------------------------------------------------------------------

_LAYERS_HANDLERS: dict[str, str] = {
    "nav_doc": "_search_layers_nav_docs",
    "nav_cluster": "_search_layers_nav_clusters",
    "navigation_tree": "_search_layers_navigation_tree",
    "chunk": "_search_layers_chunks",
    "all": "_search_layers_all",
}


async def search_dataset_layers(
    dataset_id: str,
    tenant_id: str,
    query: str,
    mode: str,
    *,
    top_k: int | None = None,
    doc_scope: list[str] | None = None,
) -> tuple[bool, dict]:
    """跨数据集不同知识层的统一搜索。

    参数：
        模式：其中之一``"chunk"``, ``"nav_doc"``, `ZXQK EEP00410004ZXQ`, ``"navigation_tree"``, `ZXQKEEP00 410008ZXQ`.
            - chunk: raw document chunks (via the main retrieval pipeline)
            - nav_doc: navigation tree document leaves
            - nav_cluster: navigation tree cluster nodes
            - navigation_tree: tree-structured BFS beam descent
            - all: union of all modes, deduplicated by doc_id with best score
        doc_scope: Optional set of documents to restrict the search to.  None or
            empty means all documents of the dataset.  Forwarded to every mode:
            nav modes filter the compiled nav rows by doc, ``chunk`` restricts
            the retriever via `ZXQKEEP0041001 2ZXQ`.  Applied query-time so scoped rows
            are never dropped by the ``top_k`` truncation.

        Items are shaped as ``{"doc_id": str, "score": float}``。"""
    from rag.advanced_rag.knowlege_compile.dataset_nav import search_dataset_nav

    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, {"error": "no authorization", "code": RetCode.PERMISSION_ERROR}
    if mode not in _LAYERS_HANDLERS:
        return False, {"error": f"unknown mode: {mode}, expected one of {list(_LAYERS_HANDLERS.keys())}", "code": RetCode.ARGUMENT_ERROR}
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    try:
        from api.db.joint_services.tenant_model_service import get_tenant_default_model_by_type
        from api.db.services.llm_service import LLMBundle

        if kb.embd_id:
            embd_model_config = resolve_model_config(kb.tenant_id, LLMType.EMBEDDING, kb.embd_id)
        else:
            embd_model_config = get_tenant_default_model_by_type(kb.tenant_id, LLMType.EMBEDDING)
        embd_mdl = LLMBundle(kb.tenant_id, embd_model_config)
    except Exception as e:
        logging.warning(
            "search_dataset_layers：无法为租户=%s创建LLMBundle（EMBEDDING）：%s： %s",
            kb.tenant_id,
            type(e).__name__,
            e,
        )
        logging.exception("LLMBundle(EMBEDDING)故障的完整回溯")
        embd_mdl = None

    logging.debug(
        "search_dataset_layers：调度作用域模式=%s，数据集=%s，scoped_docs=%d",
        mode,
        dataset_id,
        len([d for d in (doc_scope or []) if str(d).strip()]),
    )
    if mode == "nav_doc":
        return await _search_layers_nav_docs(tenant_id, dataset_id, query, top_k, embd_mdl, search_dataset_nav, doc_scope=doc_scope)
    elif mode == "nav_cluster":
        return await _search_layers_nav_clusters(tenant_id, dataset_id, query, top_k, embd_mdl, search_dataset_nav, doc_scope=doc_scope)
    elif mode == "navigation_tree":
        return await _search_layers_navigation_tree(tenant_id, dataset_id, query, top_k, embd_mdl, doc_scope=doc_scope)
    elif mode == "chunk":
        return await _search_layers_chunks(tenant_id, dataset_id, query, top_k, embd_mdl, kb, doc_scope=doc_scope)
    elif mode == "all":
        return await _search_layers_all(tenant_id, dataset_id, query, top_k, embd_mdl, kb, search_dataset_nav, doc_scope=doc_scope)
    else:
        return False, {"error": f"unknown mode: {mode}", "code": RetCode.ARGUMENT_ERROR}


async def _search_layers_nav_docs(tenant_id, dataset_id, query, top_k, embd_mdl, search_fn, *, doc_scope=None):
    items = await _nav_search_result(
        tenant_id,
        dataset_id,
        query,
        top_k,
        embd_mdl,
        search_fn,
        type_kwd="nav_doc",
        doc_scope=doc_scope,
    )
    return True, {"mode": "nav_doc", "total": len(items), "items": items}


async def _search_layers_nav_clusters(tenant_id, dataset_id, query, top_k, embd_mdl, search_fn, *, doc_scope=None):
    items = await _nav_search_result(
        tenant_id,
        dataset_id,
        query,
        top_k,
        embd_mdl,
        search_fn,
        type_kwd="nav_cluster",
        doc_scope=doc_scope,
    )
    return True, {"mode": "nav_cluster", "total": len(items), "items": items}


async def _search_layers_navigation_tree(tenant_id, dataset_id, query, top_k, embd_mdl, *, doc_scope=None):
    from rag.advanced_rag.knowlege_compile.dataset_nav import search_nav_tree_descent

    items = await search_nav_tree_descent(
        tenant_id,
        dataset_id,
        query,
        embd_mdl,
        top_k=top_k,
        doc_scope=doc_scope,
    )
    return True, {"mode": "navigation_tree", "total": len(items), "items": items}


async def _nav_search_result(tenant_id, dataset_id, query, top_k, embd_mdl, search_fn, **kwargs):
    doc_scope = kwargs.pop("doc_scope", None)
    results = await search_fn(
        tenant_id,
        dataset_id,
        query,
        embd_mdl=embd_mdl,
        top_k=top_k,
        doc_scope=doc_scope,
        **kwargs,
    )
    scope_set = {str(d).strip() for d in doc_scope if str(d).strip()} if doc_scope else None
    items: list[dict] = []
    for r in results:
        raw_doc_id = r.get("doc_id")
        if isinstance(raw_doc_id, str) and raw_doc_id.strip():
            doc_id = raw_doc_id.strip()
        else:
            # Cluster 行：选择范围内的第一个 doc_id（如果范围
            # 已设置），因此我们绝不会将超出范围的 doc_id 泄漏到结果中。
            doc_ids = r.get("doc_ids") or []
            if scope_set:
                doc_id = next((str(d).strip() for d in doc_ids if str(d).strip() in scope_set), "")
            else:
                doc_id = str(doc_ids[0]).strip() if doc_ids else ""
        if not doc_id:
            continue
        item = {
            "doc_id": doc_id,
            "score": round(float(r.get("score", 0.0)), 4),
        }
        # 保留完整的导航节点信息，以便调用者无需
        # 第二次 ES 往返。  簇具有 doc_id="" 但带有名称。
        if r.get("name") or r.get("description"):
            item["_nav"] = r
        items.append(item)
    return items


async def _search_layers_chunks(tenant_id, dataset_id, query, top_k, embd_mdl, kb, *, doc_scope=None):
    from common import settings

    tenant_ids = [tenant_id]

    kwargs = {}
    if top_k is not None:
        kwargs["knn_top_k"] = top_k
    if doc_scope:
        kwargs["doc_ids"] = [str(d) for d in doc_scope if str(d).strip()]

    fetch_k = max(top_k, 10) * 3 if top_k is not None else 1024
    try:
        ranks = await settings.retriever.retrieval(
            query,
            embd_mdl,
            tenant_ids,
            [dataset_id],
            1,
            fetch_k,
            0.0,
            0.3,
            **kwargs,
        )
    except Exception:
        return False, {"error": "chunk retrieval failed", "code": RetCode.SERVER_ERROR}

    doc_scores: dict[str, float] = {}
    for c in ranks.get("chunks", []):
        doc_id = (c.get("doc_id") or "").strip()
        score = float(c.get("similarity") or c.get("score") or 0.0)
        if doc_id and score > doc_scores.get(doc_id, -1.0):
            doc_scores[doc_id] = score

    items = sorted(
        ({"doc_id": d, "score": round(s, 4)} for d, s in doc_scores.items()),
        key=lambda x: x["score"],
        reverse=True,
    )
    if top_k is not None and top_k > 0:
        items = items[:top_k]

    return True, {"mode": "chunk", "total": len(items), "items": items}


async def _search_layers_all(tenant_id, dataset_id, query, top_k, embd_mdl, kb, search_fn, *, doc_scope=None):
    """运行所有模式并返回 doc_ids 的并集，每个文档的得分最高。"""
    import asyncio as _asyncio

    result_lists = await _asyncio.gather(
        _search_layers_nav_docs(tenant_id, dataset_id, query, top_k, embd_mdl, search_fn, doc_scope=doc_scope),
        _search_layers_nav_clusters(tenant_id, dataset_id, query, top_k, embd_mdl, search_fn, doc_scope=doc_scope),
        _search_layers_navigation_tree(tenant_id, dataset_id, query, top_k, embd_mdl, doc_scope=doc_scope),
        _search_layers_chunks(tenant_id, dataset_id, query, top_k, embd_mdl, kb, doc_scope=doc_scope),
        return_exceptions=True,
    )

    doc_scores: dict[str, dict] = {}
    for result in result_lists:
        if isinstance(result, Exception):
            continue
        ok, data = result
        if not ok:
            continue
        for item in data.get("items", []):
            doc_id = item.get("doc_id", "")
            score = float(item.get("score", 0.0))
            # 集群无doc_id；按名称键，这样它们就可以在重复数据删除后幸存下来。
            key = doc_id or item.get("_nav", {}).get("name", "")
            if key and score > doc_scores.get(key, {}).get("score", -1.0):
                doc_scores[key] = item

    items = sorted(
        doc_scores.values(),
        key=lambda x: x["score"],
        reverse=True,
    )
    if top_k is not None and top_k > 0:
        items = items[:top_k]

    return True, {"mode": "all", "total": len(items), "items": items}


async def _enrich_nav_items(dataset_id: str, tenant_id: str, items: list[dict]) -> list[dict]:
    """丰富``{doc_id, score}`` items with full nav node info.

    Items that already carry a ``_nav`` payload (from nav_doc/nav_cluster/
    navigation_tree search) are shaped in-process.  Items without it (e.g.
    chunk hits) are batch-fetched from ES by ``doc_id``。

    对于每个匹配的文档，其父集群也会被解析并添加到前面
    到结果集（集群分数 = 最大子分数），以便前端可以
    在树中突出显示该文档及其包含的簇。"""
    from common.doc_store.doc_store_base import OrderByExpr

    if not items:
        return items

    _, kb = KnowledgebaseService.get_by_id(dataset_id)
    pack = _compiled_index_or_none(kb.tenant_id, dataset_id) if kb else None
    if pack is None:
        return items
    index_nm, _ = pack

    # 第 1 阶段：塑造已有 _nav 的项目；收集 doc_ids 以获得仅块命中。
    doc_items: list[dict] = []
    missing_doc_ids: list[str] = []
    for it in items:
        nav = it.get("_nav")
        if nav:
            is_cluster = nav.get("type") == "nav_cluster"
            item = {
                "name": nav.get("name") or "",
                "description": nav.get("description") or "",
                "keywords": nav.get("keywords") or [],
                "entities": nav.get("entities") or [],
                "graph_content": nav.get("graph_content") or "",
                "doc_count": nav.get("doc_count") or (0 if is_cluster else 1),
                "type": "cluster" if is_cluster else "doc",
                "doc_id": nav.get("doc_id"),
                "has_children": is_cluster,
                "score": it.get("score", 0.0),
            }
            # 捕获 parent_kwd 的文档，以便我们可以获取父集群。
            if not is_cluster:
                pn = nav.get("parent_kwd") or []
                if isinstance(pn, (list, tuple, set)):
                    pn = next(iter(pn), "")
                if pn:
                    item["_parent_kwd"] = str(pn)
            doc_items.append(item)
        else:
            doc_id = it.get("doc_id", "")
            if doc_id:
                missing_doc_ids.append(doc_id)
            doc_items.append(it)

    # 第 2 阶段：批量获取 nav_doc 行以进行仅块命中（需要 parent_kwd）。
    all_doc_ids = missing_doc_ids + [d["doc_id"] for d in doc_items if d.get("type") == "doc" and d.get("doc_id") and not d.get("_parent_kwd")]
    nav_rows_by_doc_id: dict[str, dict] = {}
    if all_doc_ids:
        try:
            res = settings.docStoreConn.search(
                select_fields=_NAV_FIELDS,
                highlight_fields=[],
                condition={"compile_kwd": [_NAV_COMPILE_KWD], "doc_id": all_doc_ids},
                match_expressions=[],
                order_by=OrderByExpr(),
                offset=0,
                limit=len(all_doc_ids),
                index_names=index_nm,
                knowledgebase_ids=[dataset_id],
            )
            field_map = settings.docStoreConn.get_fields(res, _NAV_FIELDS)
        except Exception:
            logging.exception("_enrich_nav_items：kb = %s 的 docStore 搜索失败", dataset_id)
            field_map = None

        for row in (field_map or {}).values():
            did = row.get("doc_id") or row.get("name") or ""
            if isinstance(did, (list, tuple, set)):
                did = next(iter(did), "")
            if did:
                nav_rows_by_doc_id[str(did)] = row

    # 使用导航信息 + parent_kwd 丰富仅块项目。
    for i, it in enumerate(doc_items):
        if it.get("type"):
            continue
        doc_id = it.get("doc_id", "")
        row = nav_rows_by_doc_id.get(doc_id)
        if row:
            nav_item = _nav_item(row)
            nav_item["score"] = it.get("score", 0.0)
            nav_item["_parent_kwd"] = _first_str(row.get("parent_kwd"))
            doc_items[i] = nav_item

    # 第 3 阶段：获取 _nav 文档项的 parent_kwd（search_dataset_nav 删除它）。
    nav_doc_names = [d["name"] for d in doc_items if d.get("type") == "doc" and not d.get("_parent_kwd") and d.get("name")]
    if nav_doc_names:
        name_field = "name.keyword" if settings.DOC_ENGINE.lower() in {"elasticsearch", "opensearch"} else "name"
        try:
            res = settings.docStoreConn.search(
                select_fields=["name", "parent_kwd"],
                highlight_fields=[],
                condition={"compile_kwd": [_NAV_COMPILE_KWD], "type_kwd": ["nav_doc"], name_field: nav_doc_names},
                match_expressions=[],
                order_by=OrderByExpr(),
                offset=0,
                limit=len(nav_doc_names),
                index_names=index_nm,
                knowledgebase_ids=[dataset_id],
            )
            field_map = settings.docStoreConn.get_fields(res, ["name", "parent_kwd"])
        except Exception:
            field_map = None

        parent_by_name: dict[str, str] = {}
        for row in (field_map or {}).values():
            nm = _first_str(row.get("name"))
            pn = _first_str(row.get("parent_kwd"))
            if nm and pn:
                parent_by_name[nm] = pn

        for it in doc_items:
            if it.get("type") == "doc" and not it.get("_parent_kwd"):
                it["_parent_kwd"] = parent_by_name.get(it.get("name", ""), "")

    # 第 4 阶段：批量获取父簇行。
    parent_names = {it["_parent_kwd"] for it in doc_items if it.get("_parent_kwd")}
    cluster_by_name: dict[str, dict] = {}
    if parent_names:
        name_field = "name.keyword" if settings.DOC_ENGINE.lower() in {"elasticsearch", "opensearch"} else "name"
        try:
            res = settings.docStoreConn.search(
                select_fields=_NAV_FIELDS,
                highlight_fields=[],
                condition={"compile_kwd": [_NAV_COMPILE_KWD], "type_kwd": ["nav_cluster"], name_field: list(parent_names)},
                match_expressions=[],
                order_by=OrderByExpr(),
                offset=0,
                limit=len(parent_names),
                index_names=index_nm,
                knowledgebase_ids=[dataset_id],
            )
            field_map = settings.docStoreConn.get_fields(res, _NAV_FIELDS)
        except Exception:
            field_map = None

        for row in (field_map or {}).values():
            nm = _first_str(row.get("name"))
            if nm:
                cluster_by_name[nm] = _nav_item(row)

    # 第 5 阶段：构建结果 - 集群（最大子分数）然后文档。
    cluster_scores: dict[str, float] = {}
    doc_parent: dict[int, str] = {}
    for i, it in enumerate(doc_items):
        pn = it.pop("_parent_kwd", "")
        doc_parent[i] = pn
        if pn and pn in cluster_by_name:
            cluster_scores[pn] = max(cluster_scores.get(pn, 0.0), it.get("score", 0.0))

    clusters = [{**nav_item, "score": round(cluster_scores.get(name, 0.0), 4)} for name, nav_item in cluster_by_name.items()]
    cluster_names = {c["name"] for c in clusters}

    # 父簇也在结果中的匹配文档已经存在
    # 由该簇 's tree — don't 表示，将其表面作为第二个，
    # 独立根。  这使得搜索能够同时命中集群及其
    # 子文档下降到单个 nav_cluster 树而不是两个根。
    standalone_docs = [it for i, it in enumerate(doc_items) if it.get("type") != "cluster" and doc_parent.get(i) not in cluster_names]
    return clusters + standalone_docs


async def update_wiki_page(
    dataset_id: str,
    tenant_id: str,
    page_type: str,
    slug: str,
    content_md: str,
    *,
    user_id: str | None = None,
    title: str | None = None,
    comments: str | None = None,
):
    """从画布双击对话框中就地编辑工件页面。

    身体必须包含``content_md`` — the (possibly edited) page markdown.
    We run it through ``_wiki_transform_links`` so any newly typed
    ``[[slug]]`` references upgrade to clickable artifact URLs (and pre-rendered
    links pass through unchanged — the transform is idempotent on already-
    rendered markdown). `ZXQK EEP00010006ZXQ` is re-derived from the new rendered text.
    ``outlinks_kwd`` is rebuilt from the link-transform pass.

    Per the v1 contract, only the page row is updated. The canvas
    ``wiki_page_graph`` / `ZXQKEEP00 010012ZXQ` / ``wiki_relation``
    rows stay stale until the next full artifact compile.

    Side effect: when the rendered post-save markdown differs from the
    prior stored content, one ``wiki_commit`` row is recorded
    (git-style audit). No-op saves are silently skipped — empty diff,
    no row.

    Returns `ZXQKEEP0001001 8ZXQ` mirroring ``get_wiki_page``, or
    ``(True, None)`` when the row is missing, or
    ``(False, message)``授权失败时。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _wiki_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, None
    index_nm, _ = pack

    from api.db.services.file_commit_service import FileCommitService
    from rag.advanced_rag.knowlege_compile.wiki import (
        _wiki_extract_summary,
        _wiki_transform_links,
    )

    full_slug = f"{page_type}/{slug}" if "/" not in slug else slug

    # 捕获预编辑渲染内容+行id。两者都来自
    # 相同的搜索：行id是返回的dict key
    # docStoreConn.get_fields。我们特别需要 id 因为
    # 通用非id更新路径（ESConnection.update慢分支）路由
    # 通过无痛脚本擦除换行符/单引号/
    # 反斜杠从字符串值中转义 — 这会崩溃每个
    # 段落中保存的markdown 为一行。将行 id 传入
    # ``condition`` 选择保留的快速部分更新分支
    # JSON 逐字值。
    from common.doc_store.doc_store_base import OrderByExpr

    row_id: str | None = None
    content_before = ""
    try:
        res = settings.docStoreConn.search(
            select_fields=["id", "md_with_weight", "content_with_weight"],
            highlight_fields=[],
            condition={
                "compile_kwd": [WIKI_PAGE_COMPILE_KWD],
                "page_type_kwd": [page_type],
                "slug_kwd": [full_slug],
            },
            match_expressions=[],
            order_by=OrderByExpr(),
            offset=0,
            limit=1,
            index_names=index_nm,
            knowledgebase_ids=[dataset_id],
        )
        field_map = settings.docStoreConn.get_fields(
            res,
            ["id", "md_with_weight", "content_with_weight"],
        )
        if field_map:
            row_id, row = next(iter(field_map.items()))
            content_before = row.get("md_with_weight") or row.get("content_with_weight") or ""
    except Exception:
        logging.exception(
            "update_wiki_page：kb = %s slug = %s 查找失败",
            dataset_id,
            full_slug,
        )
    if not row_id:
        return True, None

    content_md = content_md or ""
    rendered, outlinks = _wiki_transform_links(content_md, dataset_id)
    summary = _wiki_extract_summary(rendered) or ""

    try:
        # id 键控条件强制部分更新快速路径 — 否
        # 换行擦洗。请参阅查找上方的评论
        # 完整推理。
        ok = settings.docStoreConn.update(
            {"id": row_id},
            {
                "md_with_weight": rendered,
                "content_with_weight": rendered,
                "summary_with_weight": summary,
                "outlinks_kwd": list(outlinks),
            },
            index_nm,
            dataset_id,
        )
    except Exception:
        logging.exception(
            "update_wiki_page：kb=%s slug=%s 的 docStore 更新失败",
            dataset_id,
            full_slug,
        )
        return True, None

    if not ok:
        return True, None

    refresh_idx = getattr(settings.docStoreConn, "refresh_idx", None)
    if callable(refresh_idx):
        try:
            await thread_pool_exec(refresh_idx, index_nm)
        except Exception:
            logging.exception(
                "update_wiki_page：kb = %s slug = %s 的索引刷新失败",
                dataset_id,
                full_slug,
            )

    # 在每次实际更改时记录 file_commit 行。 ``record_page_edit``
    # 对于空差异保存返回 None，我们默默地接受。
    try:
        FileCommitService.record_page_edit(
            tenant_id=tenant_id,
            kb_id=dataset_id,
            page_type=page_type,
            slug=full_slug,
            content_before=content_before,
            content_after=rendered,
            title=title,
            comments=comments,
            user_id=user_id,
        )
    except Exception:
        logging.exception(
            "update_wiki_page: file_commit 记录失败，kb=%s slug=%s",
            dataset_id,
            full_slug,
        )

    # 重新读取该行，以便对话框获得规范的更新后状态。
    return await get_wiki_page(dataset_id, tenant_id, page_type, slug)


# ``list_wiki_commits`` / ``get_wiki_commit`` 退役 — 两个
# ``/datasets/<id>/artifacts/.../commits`` REST 端点现在通过
# 通用文件提交路由（“`/datasets/<id>/commits`”，带有
# 可选“`?slug=`”过滤器），支持
# ：方法：`FileCommitService.list_page_commits` 和
# ：甲基：`FileCommitService.get_page_commit_detail`。


# 工件管道写入的所有行类型。依附列出
# 顺序，以便早期删除的部分失败不会留下状态
# 下游阶段将默默地重用。 ``wiki_page_graph``
# 是从精炼页面衍生出来的物化画布图——
# 数据集 Artifact 选项卡的图形视图准确读取这一行。
_WIKI_COMPILE_KWDS = (
    "wiki_map_extract",
    "wiki_map_state",
    "wiki_map_state_meta",
    "wiki_reduce_result",
    "wiki_compilation_plan",
    "wiki_page_draft",
    "wiki_page",
    "wiki_page_topic",
    "wiki_canonical_entity",
    "wiki_plan_group",
    "wiki_doc_page_source",
    "wiki_mode_meta",
    "wiki_entity",
    "wiki_relation",
    "wiki_page_graph",
)

# 增量图形加载器的可调参数。参见“`get_wiki_graph`”。
_WIKI_GRAPH_ENTITY_KWD = "wiki_entity"
_WIKI_GRAPH_RELATION_KWD = "wiki_relation"
_WIKI_GRAPH_ENTITY_PAGE_SIZE = 32
_WIKI_GRAPH_MAX_LOADING_ENTITY = 512


def _wiki_entity_payload(row: dict) -> dict | None:
    """项目一``wiki_entity`` ES row onto the canvas entity shape.

    The row stores the canvas payload pre-built as JSON in
    ``content_with_weight``；我们解析它并覆盖列
    作者独立设置（weight_int，source_chunk_ids）所以
    无论任何情况，前端都会获取权威数字
    JSON-vs-柱漂移。"""
    raw = row.get("content_with_weight") or ""
    payload: dict = {}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            pass
    slug = payload.get("slug") or _scalar(row.get("slug_kwd"))
    if not isinstance(slug, str) or not slug:
        return None
    out = {
        "slug": slug,
        "name": payload.get("name") or slug,
        "aliases": list(payload.get("aliases") or []),
        "description": payload.get("description") or "",
        "type": payload.get("type") or "concept",
        "weight": int(row.get("weight_int") or payload.get("weight") or 0),
    }
    source_chunk_ids = row.get("source_chunk_ids") or []
    if isinstance(source_chunk_ids, list):
        out["source_chunk_ids"] = [c for c in source_chunk_ids if isinstance(c, str) and c]
    return out


def _wiki_relation_payload(row: dict) -> dict | None:
    raw = row.get("content_with_weight") or ""
    payload: dict = {}
    if isinstance(raw, str) and raw.strip():
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                payload = parsed
        except Exception:
            pass
    src = payload.get("from") or row.get("from_kwd")
    tgt = payload.get("to") or row.get("to_kwd")
    if not isinstance(src, str) or not src or not isinstance(tgt, str) or not tgt:
        return None
    return {"from": src, "to": tgt}


async def _wiki_search_entity_page(
    index_nm,
    dataset_id: str,
    offset: int,
    limit: int,
    keywords: str = "",
):
    """wiki_entity 行的一页。

    没有``keywords``: ordered by ``weight_int DESC`` (heaviest nodes first).
    With `ZXQKEEP00 340005ZXQ`: BM25 full-text match over ``content_ltks`` (slug +
    summary), ordered by relevance. ``wiki_entity``行仅是 BM25（无
    嵌入向量），所以这是一个词法搜索，而不是密集的 KNN。"""
    from common.doc_store.doc_store_base import OrderByExpr

    select_fields = [
        "id",
        "slug_kwd",
        "weight_int",
        "source_chunk_ids",
        "content_with_weight",
    ]

    keywords = (keywords or "").strip()
    match_expressions: list = []
    order_by = OrderByExpr()
    if keywords:
        try:
            match_text, _ = settings.retriever.qryr.question(keywords, min_match=0.1)
            match_expressions = [match_text]
        except Exception:
            logging.exception("get_wiki_graph：无法为 kb=%s 构建关键字查询", dataset_id)
            match_expressions = []
    if not match_expressions:
        # 没有关键字（或查询构建失败）→ 最重的优先。当一个
        # 文本匹配显示商店按 BM25 分数排名。
        try:
            order_by.desc("weight_int")
        except Exception:
            order_by = OrderByExpr()

    res = await thread_pool_exec(
        settings.docStoreConn.search,
        select_fields,
        [],
        {"compile_kwd": [_WIKI_GRAPH_ENTITY_KWD]},
        match_expressions,
        order_by,
        offset,
        limit,
        index_nm,
        [dataset_id],
    )
    return settings.docStoreConn.get_fields(res, select_fields)


async def _wiki_search_entities_by_slugs(
    index_nm,
    dataset_id: str,
    slugs: list[str],
):
    """获取实体行``slug_kwd`` is in ``slugs``. Unordered.

    Like :func:`_wiki_search_关系_来自`, we avoid pushing ``slug_kwd`` (a
    *_kwd analysed field) into the search filter — a `slug_kwd： [..]` 约 20
    条目触发 TOO_MANY_CONNECTIONS。拉取所有实体行一次并过滤
    记忆中。"""
    if not slugs:
        return {}

    from common.doc_store.doc_store_base import OrderByExpr

    select_fields = [
        "id",
        "slug_kwd",
        "weight_int",
        "source_chunk_ids",
        "content_with_weight",
    ]
    wanted = set(slugs)
    results = {}
    offset, page_size = 0, 1000
    while True:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            select_fields,
            [],
            {"compile_kwd": [_WIKI_GRAPH_ENTITY_KWD]},
            [],
            OrderByExpr(),
            offset,
            page_size,
            index_nm,
            [dataset_id],
        )
        rows = settings.docStoreConn.get_fields(res, select_fields)
        if not rows:
            break
        for row in rows.values():
            slug = row.get("slug_kwd")
            if isinstance(slug, list):
                slug = slug[0] if slug else ""
            if slug in wanted:
                results[row.get("id", len(results))] = row
        if len(rows) < page_size:
            break
        offset += page_size
    return results


async def _wiki_search_relations_from(
    index_nm,
    dataset_id: str,
    from_slugs: list[str],
):
    """获取关系行，其``from_kwd`` is in ``from_slugs``.

    IMPORTANT: we do NOT push `ZXQKEEP00 390004ZXQ` into the search filter. ``from_kwd``
    is a *_kwd (whitespace-# analysed) field, and the generic search path turns
    `from_kwd：每个值的 [v1, v2, ...]` into one ``filter_fulltext`` 子句。一个
    仅约 20 个 slugs 批次就已经超过了 Infinity 的每个查询连接
    预算和表面为 TOO_MANY_CONNECTIONS（增量写入器发出
    很多关系，所以sub_slugs很容易超过20）。相反，我们拉 ALL 关系
    ONE 廉价查询中数据集的行（关系短且少）和
    内存中的过滤器。"""
    if not from_slugs:
        return {}

    from common.doc_store.doc_store_base import OrderByExpr

    select_fields = ["id", "from_kwd", "to_kwd", "content_with_weight"]
    wanted = set(from_slugs)
    # 单个查询，无需巨大的 from_kwd IN 过滤器。翻页结果为
    # 情况下数据集有超过 10000 个关系。
    results = {}
    offset, page_size = 0, 1000
    while True:
        res = await thread_pool_exec(
            settings.docStoreConn.search,
            select_fields,
            [],
            {"compile_kwd": [_WIKI_GRAPH_RELATION_KWD]},
            [],
            OrderByExpr(),
            offset,
            page_size,
            index_nm,
            [dataset_id],
        )
        rows = settings.docStoreConn.get_fields(res, select_fields)
        if not rows:
            break
        for row in rows.values():
            frm = row.get("from_kwd")
            if isinstance(frm, list):
                frm = frm[0] if frm else ""
            if frm in wanted:
                results[row.get("id", len(results))] = row
        if len(rows) < page_size:
            break
        offset += page_size
    return results


async def get_wiki_graph(
    dataset_id: str,
    tenant_id: str,
    node: str | None = None,
    keywords: str | None = None,
    top_n: int | None = None,
):
    """从每行数据增量加载画布图有效负载。

    ``top_n`` overrides the entity budget (default ``_WIKI_GRAPH_MAX_LOADING_ENTITY``).
    ``keywords`Z XQKEEP00420005ZXQ`node`` is given) seeds the
    graph from the best BM25 matches on ``wiki_entity`` rows instead of the
    heaviest-weighted ones. Only entities referenced by a relation are returned.

    Two modes:

    * **Overview** (`ZXQK EEP00420010ZXQ` is None) — paginate ``wiki_entity`` rows
      ordered by ``weight_int DESC`ZXQKEE P00420015ZXQ`_WIKI_GRAPH_ENTITY_PAGE_SIZE``. For each page, append entities
      to a running set while the **cumulative** weight stays within
      ``_WIKI_GRAPH_MAX_LOADING_ENTITY``. Pull `ZXQKEEP00 420020ZXQ`
      rows whose ``from_kwd`` is in the just-added entities; pull the
      ``to`ZXQKEEP0042 0025ZXQ`node`` is a slug) — load the centre entity (always
      included), pull every ``wiki_relation`` with `ZXQKEEP0042003 0ZXQ`,
      then pull the ``to`` entities. Capped at
      ``_WIKI_GRAPH_MAX_LOADING_ENTITY`ZXQKEEP00420035Z XQ`(True, {"entities": [...], "relations": [...]})`` shaped
    exactly as the frontend ``ForceGraph`` adapter consumes, or
    ``(False, message)``授权失败时。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    empty = {"entities": [], "relations": []}

    pack = _wiki_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, empty
    index_nm, _ = pack

    keywords = (keywords or "").strip()
    # 实体预算：调用者可重写，限制在合理的范围内，因此是一个错误的参数
    # 既不能禁用上限，也不能炸毁响应。
    if top_n is not None:
        try:
            cap = max(1, min(int(top_n), 1024))
        except (TypeError, ValueError):
            cap = _WIKI_GRAPH_MAX_LOADING_ENTITY
    else:
        cap = _WIKI_GRAPH_MAX_LOADING_ENTITY
    page_size = _WIKI_GRAPH_ENTITY_PAGE_SIZE

    # ``entities`` 保留第一次出现的顺序，使画布优先绘制权重最高的节点；单击模式下则优先
    # 绘制中心节点。以 slug 为键的字典还能低成本去重，例如同一节点既是某条关系的目标，
    # 后续又以高权重实体的身份出现时，只保留一份实体记录。
    entities: dict[str, dict] = {}
    relations: list[dict] = []
    relation_keys: set[tuple[str, str]] = set()

    def _add_entity(payload: dict) -> bool:
        slug = payload.get("slug")
        if not isinstance(slug, str) or not slug or slug in entities:
            return False
        entities[slug] = payload
        return True

    def _add_relation(payload: dict) -> None:
        key = (payload["from"], payload["to"])
        if key in relation_keys:
            return
        relation_keys.add(key)
        relations.append(payload)

    # ---- 流程 B — 单击以“`node`”为中心的展开。 ----------------
    if isinstance(node, str) and node.strip():
        center_slug = node.strip()
        try:
            field_map = await _wiki_search_entities_by_slugs(
                index_nm,
                dataset_id,
                [center_slug],
            )
        except Exception:
            logging.exception(
                "get_wiki_graph：中心查找失败 kb=%s 节点=%s",
                dataset_id,
                center_slug,
            )
            return True, empty

        for row in (field_map or {}).values():
            payload = _wiki_entity_payload(row)
            if payload:
                _add_entity(payload)
                break

        if center_slug not in entities:
            # 调用者指向一个不存在的slug；返回空
            # 而不是令人困惑的部分图。
            return True, empty

        # 从中心向外的边缘，由 MAX_LOADING_ENTITY 覆盖。
        try:
            rel_map = await _wiki_search_relations_from(
                index_nm,
                dataset_id,
                [center_slug],
            )
        except Exception:
            logging.exception(
                "get_wiki_graph：关系查找失败 kb=%s 节点=%s",
                dataset_id,
                center_slug,
            )
            return True, {"entities": list(entities.values()), "relations": []}

        to_slugs: list[str] = []
        for row in (rel_map or {}).values():
            payload = _wiki_relation_payload(row)
            if payload is None:
                continue
            if payload["from"] != center_slug:
                continue
            # 中心节点上限：一旦超过，就停止接受更多关系
            # 目标设定将使我们超出实体预算。
            if payload["to"] not in entities and len(entities) + len(to_slugs) >= cap:
                continue
            _add_relation(payload)
            if payload["to"] != center_slug and payload["to"] not in entities:
                if payload["to"] not in to_slugs:
                    to_slugs.append(payload["to"])

        if to_slugs:
            try:
                to_map = await _wiki_search_entities_by_slugs(
                    index_nm,
                    dataset_id,
                    to_slugs,
                )
            except Exception:
                logging.exception(
                    "get_wiki_graph：邻居查找失败 kb=%s 节点=%s",
                    dataset_id,
                    center_slug,
                )
                to_map = {}
            for row in (to_map or {}).values():
                payload = _wiki_entity_payload(row)
                if payload and len(entities) < cap * 2:
                    _add_entity(payload)

        return True, {
            "entities": list(entities.values()),
            "relations": relations,
        }

    # ---- 流程 A — 概览、最高权重分页与累积预算。 ---
    cumulative_weight = 0
    page = 1
    while len(entities) < cap:
        offset = (page - 1) * page_size
        try:
            field_map = await _wiki_search_entity_page(
                index_nm,
                dataset_id,
                offset,
                page_size,
                keywords=keywords,
            )
        except Exception:
            logging.exception(
                "get_wiki_graph：实体页面获取失败kb = %s页面= %d",
                dataset_id,
                page,
            )
            break
        if not field_map:
            break

        # 保留来自 ES 的 weight_int DESC 订单。字典的迭代
        # get_fields生产的
        # 保持插入顺序； ES 已退回
        # 已排序，因此我们可以信赖它。
        page_rows = list(field_map.values())

        e_sub: list[dict] = []
        for row in page_rows:
            payload = _wiki_entity_payload(row)
            if payload is None:
                continue
            if payload["slug"] in entities:
                continue
            w = max(0, int(payload.get("weight") or 0))
            # 步骤 2：整个流程中的累积（根据规范）。
            # 当添加此条目会超出预算时停止。
            # 如果页面上的第一个实体都无法容纳，我们就退出
            # 外循环如下；这保留了“权重最小的优先”
            # 排除”语义。
            # 如果 cumulative_weight + w > cap 和 len(实体) + len(e_sub) > 0:
            # 断线
            cumulative_weight += w
            e_sub.append(payload)
            if len(entities) + len(e_sub) >= cap:
                break

        if not e_sub:
            break

        for payload in e_sub:
            _add_entity(payload)

        # 步骤 3：源自 E_sub 的关系。
        sub_slugs = [p["slug"] for p in e_sub]
        try:
            rel_map = await _wiki_search_relations_from(
                index_nm,
                dataset_id,
                sub_slugs,
            )
        except Exception:
            logging.exception(
                "get_wiki_graph：关系页面获取失败 kb=%s",
                dataset_id,
            )
            rel_map = {}

        missing_to: list[str] = []
        for row in (rel_map or {}).values():
            payload = _wiki_relation_payload(row)
            if payload is None:
                continue
            _add_relation(payload)
            if payload["to"] not in entities and payload["to"] not in missing_to:
                missing_to.append(payload["to"])

        # 步骤 4：水合目标（它们计入上限）。
        if missing_to:
            try:
                to_map = await _wiki_search_entities_by_slugs(
                    index_nm,
                    dataset_id,
                    missing_to,
                )
            except Exception:
                logging.exception(
                    "get_wiki_graph：目标水合物失败 kb=%s",
                    dataset_id,
                )
                to_map = {}
            for row in (to_map or {}).values():
                if len(entities) >= cap:
                    break
                payload = _wiki_entity_payload(row)
                if payload:
                    _add_entity(payload)

        # 步骤 5：仅当上限允许另一次迭代时才向前翻页。
        if len(entities) >= cap or len(page_rows) < page_size:
            break
        page += 1

    return True, {
        "entities": list(entities.values()),
        "relations": relations,
    }


async def clear_wiki(dataset_id: str, tenant_id: str):
    """从 ES 中擦除此 KB 的每个与工件相关的行。

    在身份验证失败时触及所有工件“`compile_kwd`` row types the artifact pipeline writes
    (MAP extracts, REDUCE results, PLAN output, drafts, pages, topics, and graph
    rows). After this completes the next "Artifact" run starts from a clean
    slate: no resume cache to short-circuit MAP, no prior pages to reconcile
    against in PLAN.

    Returns ``(True, {"deleted": {kwd: count_or_True}})`` on success or
    ``(False, str)`”。"""
    if not KnowledgebaseService.accessible(dataset_id, tenant_id):
        return False, "no authorization"
    _, kb = KnowledgebaseService.get_by_id(dataset_id)

    pack = _wiki_index_or_none(kb.tenant_id, dataset_id)
    if pack is None:
        return True, {"deleted": {}}
    index_nm, _ = pack

    # 在删除之前修复由之前的 doc-page-source upsert 损坏的行
    # 那个桶。由“`doc_id`”更新，可以标记普通源
    # 块为“`wiki_doc_page_source`”。这些行保留块内容；
    # 正品跟踪行没有。仅删除坏标记可以保留
    # 源块，同时允许在下面清除真实的跟踪行。
    try:
        from common.doc_store.doc_store_base import OrderByExpr

        fields = ["id", "content_with_weight"]
        offset = 0
        page_size = 1000
        damaged_row_ids: list[str] = []
        while True:
            res = await thread_pool_exec(
                settings.docStoreConn.search,
                fields,
                [],
                {"compile_kwd": ["wiki_doc_page_source"]},
                [],
                OrderByExpr(),
                offset,
                page_size,
                index_nm,
                [dataset_id],
            )
            rows = settings.docStoreConn.get_fields(res, fields) or {}
            for row_id, row in rows.items():
                if row.get("content_with_weight"):
                    damaged_row_ids.append(row_id)
            if len(rows) < page_size:
                break
            offset += page_size
        for row_id in damaged_row_ids:
            await thread_pool_exec(
                settings.docStoreConn.update,
                {"id": row_id},
                {"remove": "compile_kwd"},
                index_nm,
                dataset_id,
            )
        if damaged_row_ids:
            logging.warning(
                "clear_wiki：已修复 %d 源块错误标记为 wiki_doc_page_source kb=%s",
                len(damaged_row_ids),
                dataset_id,
            )
    except Exception:
        logging.exception("clear_wiki：无法修复错误标记的源块 kb=%s", dataset_id)
        return False, "Failed to repair legacy Wiki state before clearing"

    deleted: dict[str, object] = {}
    for kwd in _WIKI_COMPILE_KWDS:
        try:
            res = settings.docStoreConn.delete(
                {"compile_kwd": kwd},
                index_nm,
                dataset_id,
            )
            # 不同的后端返回不同的形状（int count、dict、
            # 布尔）。显示我们得到的任何内容，以便调用者可以记录它。
            deleted[kwd] = res if res is not None else True
        except Exception:
            logging.exception(
                "clear_wiki：kwd = %s kb = %s 删除失败",
                kwd,
                dataset_id,
            )
            deleted[kwd] = False

    from api.db.services.file_commit_service import FileCommitService

    if all(result is not False for result in deleted.values()):
        try:
            deleted["file_commit_history"] = FileCommitService.delete_all_page_history(dataset_id)
        except Exception:
            logging.exception(
                "clear_wiki：无法删除 kb=%s 的页面版本历史记录",
                dataset_id,
            )
            deleted["file_commit_history"] = False
    else:
        deleted["file_commit_history"] = False

    return True, {"deleted": deleted}
