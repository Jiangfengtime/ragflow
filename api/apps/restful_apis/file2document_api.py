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
#  limitations under the License
#

import asyncio
import logging
from pathlib import Path

from quart import request

from api.common.check_team_permission import check_file_team_permission, check_kb_team_permission
from api.db.services import duplicate_name
from api.db.services.file2document_service import File2DocumentService
from api.db.services.file_service import FileService

from api.apps import login_required, current_user
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.utils.api_utils import get_data_error_result, get_json_result, get_request_json, server_error_response, validate_request
from common.constants import RetCode
from common.misc_utils import get_uuid
from api.db import FileType
from api.db.services.document_service import DocumentService

logger = logging.getLogger(__name__)


def _convert_files(file_ids, kb_ids, user_id, mode):
    """在线程池中把工作区 File 映射为一个或多个知识库 Document。

    ``add`` 只补充缺失的知识库关联；``replace`` 先删除该 File 的全部旧 Document
    和 File2Document，再按 ``kb_ids`` 重建。这里不会触发文档解析，创建出的 Document
    仍需调用 ``POST /documents/ingest`` 才会产生 Task 和 Chunk。
    """
    replace_existing = mode == "replace"
    created_count = 0
    removed_count = 0
    logger.info(
        "文件关联知识库已开始 用户ID=%s 模式=%s 文件数=%d 知识库数=%d",
        user_id,
        mode,
        len(file_ids),
        len(kb_ids),
    )
    for id in file_ids:
        e, file = FileService.get_by_id(id)
        if not e:
            continue

        existing_links = File2DocumentService.get_by_file_id(id)
        existing_kb_ids = set()
        if replace_existing:
            for inform in existing_links:
                doc_id = inform.document_id
                e, doc = DocumentService.get_by_id(doc_id)
                if e and doc:
                    tenant_id = DocumentService.get_tenant_id(doc_id)
                    if not tenant_id:
                        raise RuntimeError("Tenant not found!")
                    if not DocumentService.remove_document(doc, tenant_id):
                        raise RuntimeError("Database error (Document removal)!")
                File2DocumentService.delete_by_document_id(doc_id)
                removed_count += 1
            if existing_links:
                File2DocumentService.delete_by_file_id(id)
        else:
            for inform in existing_links:
                e, doc = DocumentService.get_by_id(inform.document_id)
                if e and doc:
                    existing_kb_ids.add(doc.kb_id)

        for kb_id in kb_ids:
            if kb_id in existing_kb_ids:
                continue
            e, kb = KnowledgebaseService.get_by_id(kb_id)
            if not e:
                continue
            filename = duplicate_name(DocumentService.query, name=file.name, kb_id=kb.id)
            doc = DocumentService.insert(
                {
                    "id": get_uuid(),
                    "kb_id": kb.id,
                    "parser_id": FileService.get_parser(file.type, filename, kb.parser_id),
                    "pipeline_id": kb.pipeline_id,
                    "parser_config": kb.parser_config,
                    "created_by": user_id,
                    "type": file.type,
                    "name": filename,
                    "suffix": Path(filename).suffix.lstrip("."),
                    "location": file.location,
                    "size": file.size,
                }
            )
            File2DocumentService.insert(
                {
                    "id": get_uuid(),
                    "file_id": id,
                    "document_id": doc.id,
                }
            )
            created_count += 1
            logger.debug(
                "文件与知识库关联已创建 user_id=%s file_id=%s kb_id=%s doc_id=%s 模式=%s",
                user_id,
                id,
                kb_id,
                doc.id,
                mode,
            )
    logger.info(
        "文件关联知识库完成 用户ID=%s 模式=%s 新建关联数=%d 删除关联数=%d",
        user_id,
        mode,
        created_count,
        removed_count,
    )


@manager.route("/files/link-to-datasets", methods=["POST"])  # noqa: F821
@login_required
@validate_request("file_ids", "kb_ids")
async def convert():
    """校验权限并异步调度 File -> Document 关系转换。"""
    req = await get_request_json()
    kb_ids = req["kb_ids"]
    file_ids = req["file_ids"]
    mode = (request.args.get("mode", "replace") or "replace").lower()
    if mode not in {"replace", "add"}:
        return get_json_result(code=RetCode.ARGUMENT_ERROR, message="mode must be 'add' or 'replace'")

    try:
        files = FileService.get_by_ids(file_ids)
        files_set = {file.id: file for file in files}

        # 在开始任何工作之前验证所有文件是否存在
        for file_id in file_ids:
            if not files_set.get(file_id):
                logger.warning(
                    "文件关联知识库校验失败 用户ID=%s 资源类型=文件 资源ID=%s 原因=文件不存在 文件ID列表=%s 知识库ID列表=%s",
                    current_user.id,
                    file_id,
                    file_ids,
                    kb_ids,
                )
                return get_data_error_result(message="File not found!")

        # 在调度后台工作之前验证所有 kb_ids 是否存在
        kb_map = {}
        for kb_id in kb_ids:
            e, kb = KnowledgebaseService.get_by_id(kb_id)
            if not e:
                logger.warning(
                    "文件关联知识库校验失败 用户ID=%s 资源类型=知识库 资源ID=%s 原因=知识库不存在 文件ID列表=%s 知识库ID列表=%s",
                    current_user.id,
                    kb_id,
                    file_ids,
                    kb_ids,
                )
                return get_data_error_result(message="Can't find this dataset!")
            kb_map[kb_id] = kb

        # 将文件夹展开到最里面的文件 IDs
        all_file_ids = []
        for file_id in file_ids:
            file = files_set[file_id]
            if file.type == FileType.FOLDER.value:
                all_file_ids.extend(FileService.get_all_innermost_file_ids(file_id, []))
            else:
                all_file_ids.append(file_id)

        user_id = current_user.id
        for file_id in all_file_ids:
            e, file = FileService.get_by_id(file_id)
            if not e or not file:
                logger.warning(
                    "展开目录后文件校验失败 user_id=%s 资源类型=文件 resource_id=%s 原因=文件不存在 file_ids=%s kb_ids=%s",
                    user_id,
                    file_id,
                    file_ids,
                    kb_ids,
                )
                return get_data_error_result(message="File not found!")
            if not check_file_team_permission(file, user_id):
                logger.warning(
                    "文件关联知识库权限校验失败 用户ID=%s 资源类型=文件 资源ID=%s 结果=拒绝 文件ID列表=%s 知识库ID列表=%s",
                    user_id,
                    file_id,
                    file_ids,
                    kb_ids,
                )
                return get_data_error_result(message="no authorization")

        for kb_id, kb in kb_map.items():
            if not check_kb_team_permission(kb, user_id):
                logger.warning(
                    "文件关联知识库权限校验失败 用户ID=%s 资源类型=知识库 资源ID=%s 结果=拒绝 文件ID列表=%s 知识库ID列表=%s",
                    user_id,
                    kb_id,
                    file_ids,
                    kb_ids,
                )
                return get_data_error_result(message="no authorization")

        # 关系转换可能递归处理整个目录并删除旧索引，属于阻塞型数据库/存储操作；
        # 在线程池中 fire-and-forget 后立即响应。此处 data=true 表示“已调度”，
        # 真正完成要观察“文件关联知识库完成”日志。
        loop = asyncio.get_running_loop()
        future = loop.run_in_executor(None, _convert_files, all_file_ids, kb_ids, user_id, mode)
        future.add_done_callback(lambda f: logging.error("文件关联知识库后台任务失败: %s", f.exception()) if f.exception() else None)
        logger.info(
            "文件关联知识库后台任务已调度 用户ID=%s 资源ID=批次 文件ID列表=%s 知识库ID列表=%s",
            user_id,
            all_file_ids,
            kb_ids,
        )
        return get_json_result(data=True)
    except Exception as e:
        return server_error_response(e)
