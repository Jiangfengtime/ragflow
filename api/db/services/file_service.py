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
import asyncio
import base64
import logging
import re
import sys
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import ClassVar

logger = logging.getLogger(__name__)

import xxhash
from peewee import fn

from api.db import KNOWLEDGEBASE_FOLDER_NAME, SKILLS_FOLDER_NAME, FileType
from api.db.db_models import DB, Document, File, File2Document, Knowledgebase, Task
from api.db.services import duplicate_name
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from api.db.services.file2document_service import File2DocumentService
from api.db.services.knowledgebase_service import KnowledgebaseService
from api.db.services.task_service import TaskService
from api.utils.file_utils import filename_type, read_potential_broken_pdf, sanitize_path, thumbnail_img
from common import settings
from common.constants import MAXIMUM_PAGE_NUMBER, FileSource, ParserType, TaskStatus
from common.misc_utils import get_uuid
from common.ssrf_guard import assert_url_is_safe
from rag.llm.cv_model import GptV4


class FileService(CommonService):
    # Service 用于管理文件操作和存储的类
    model = File

    @classmethod
    @DB.connection_context()
    def get_by_pf_id(cls, tenant_id, pf_id, page_number, items_per_page, orderby, desc, keywords, exclude_skills):
        # 通过父文件夹 ID 获取文件，具有分页和过滤功能
        # 参数：
        # tenant_id：租户的ID
        # pf_id：父文件夹 ID
        # page_number：分页的页码
        # items_per_page：每页的项目数
        # orderby：排序依据字段
        # desc：表示降序的布尔值
        # 关键字： Search 关键字
        # exclude_skills：是否排除pf_id直属下的技能文件夹
        # 返回：
        # (file_list、total_count) 的元组
        if keywords:
            # 关键字搜索覆盖pf_id so文件下的整个子树并且
            # 也可以找到嵌套在子文件夹中的
            # 文件夹。
            subtree_ids = cls.get_subtree_ids(tenant_id, pf_id)
            files = cls.model.select().where(
                (cls.model.tenant_id == tenant_id), (cls.model.parent_id.in_(subtree_ids)), (fn.LOWER(cls.model.name).contains(keywords.lower())), ~(cls.model.id == pf_id)
            )
        else:
            files = cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == pf_id), ~(cls.model.id == pf_id))
        if exclude_skills:
            files = files.where(~((cls.model.parent_id == pf_id) & (cls.model.name == SKILLS_FOLDER_NAME)))
        count = files.count()
        if desc:
            files = files.order_by(cls.model.getter_by(orderby).desc())
        else:
            files = files.order_by(cls.model.getter_by(orderby).asc())

        files = files.paginate(page_number, items_per_page)

        res_files = list(files.dicts())
        # 通过文件 ID 进行重复数据删除，作为针对任何剩余重复行的安全网
        # （e.g。由竞争条件创建的重复 'skills' 或 '.knowledgebase' 文件夹）。
        seen_ids = set()
        unique_files = []
        for file in res_files:
            if file["id"] not in seen_ids:
                seen_ids.add(file["id"])
                unique_files.append(file)
        res_files = unique_files
        for file in res_files:
            if file["type"] == FileType.FOLDER.value:
                file["size"] = cls.get_folder_size(file["id"])
                file["kbs_info"] = []
                children = list(
                    cls.model.select()
                    .where(
                        (cls.model.tenant_id == tenant_id),
                        (cls.model.parent_id == file["id"]),
                        ~(cls.model.id == file["id"]),
                    )
                    .dicts()
                )
                file["has_child_folder"] = any(value["type"] == FileType.FOLDER.value for value in children)
                continue
            kbs_info = cls.get_kb_id_by_file_id(file["id"])
            file["kbs_info"] = kbs_info

        return res_files, count

    @classmethod
    @DB.connection_context()
    def get_subtree_ids(cls, tenant_id, pf_id):
        # 返回 pf_id 本身以及嵌套在其下的所有条目的 IDs
        # （文件夹和文件），用于确定递归关键字搜索的范围。
        rows = list(cls.model.select(cls.model.id, cls.model.parent_id).where(cls.model.tenant_id == tenant_id).dicts())
        children = {}
        for row in rows:
            children.setdefault(row["parent_id"], []).append(row["id"])

        ids = [pf_id]
        in_tree = {pf_id}
        queue = deque([pf_id])
        while queue:
            current = queue.popleft()
            for child in children.get(current, []):
                if child in in_tree:
                    continue
                in_tree.add(child)
                ids.append(child)
                queue.append(child)
        return ids

    @classmethod
    @DB.connection_context()
    def get_kb_id_by_file_id(cls, file_id):
        # 获取与文件关联的数据集 IDs
        # 参数：
        # file_id：File ID
        # 返回：
        # 包含数据集 IDs 和名称的字典列表
        kbs = (
            cls.model.select(*[Knowledgebase.id, Knowledgebase.name, File2Document.document_id])
            .join(File2Document, on=(File2Document.file_id == file_id))
            .join(Document, on=(File2Document.document_id == Document.id))
            .join(Knowledgebase, on=(Knowledgebase.id == Document.kb_id))
            .where(cls.model.id == file_id)
        )
        if not kbs:
            return []
        kbs_info_list = []
        for kb in list(kbs.dicts()):
            kbs_info_list.append({"kb_id": kb["id"], "kb_name": kb["name"], "document_id": kb["document_id"]})
        return kbs_info_list

    @classmethod
    @DB.connection_context()
    def get_by_pf_id_name(cls, id, name):
        # 通过父文件夹 ID 和名称获取文件
        # 参数：
        # id：父文件夹 ID
        # 名称： File 名称
        # 返回：
        # File 对象，如果未找到则为 None
        file = cls.model.select().where((cls.model.parent_id == id) & (cls.model.name == name))
        if file.count():
            e, file = cls.get_by_id(file[0].id)
            if not e:
                raise RuntimeError("Database error (File retrieval)!")
            return file
        return None

    @classmethod
    @DB.connection_context()
    def get_id_list_by_id(cls, id, name, count, res):
        # 通过遍历文件夹结构递归获取文件列表 IDs
        # 参数：
        # id：起始文件夹 ID
        # name：要遍历的文件夹名称列表
        # count：当前遍历深度
        # res：存储结果的列表
        # 返回：
        # 文件列表 IDs
        if count < len(name):
            file = cls.get_by_pf_id_name(id, name[count])
            if file:
                res.append(file.id)
                return cls.get_id_list_by_id(file.id, name, count + 1, res)
            else:
                return res
        else:
            return res

    @classmethod
    @DB.connection_context()
    def get_all_innermost_file_ids(cls, folder_id, result_ids):
        # 获取最深层文件夹所有文件的IDs
        # 参数：
        # folder_id：启动文件夹 ID
        # result_ids：存储结果的列表
        # 返回：
        # 文件列表 IDs
        subfiles = cls.model.select().where((cls.model.parent_id == folder_id) & (cls.model.id != folder_id))
        for subfile in subfiles:
            if subfile.type == FileType.FOLDER.value:
                cls.get_all_innermost_file_ids(subfile.id, result_ids)
            else:
                result_ids.append(subfile.id)
        return result_ids

    @classmethod
    @DB.connection_context()
    def get_all_file_ids_by_tenant_id(cls, tenant_id):
        fields = [cls.model.id]
        files = cls.model.select(*fields).where(cls.model.tenant_id == tenant_id)
        files = files.order_by(cls.model.create_time.asc())
        offset, limit = 0, 100
        res = []
        while True:
            file_batch = files.offset(offset).limit(limit)
            _temp = list(file_batch.dicts())
            if not _temp:
                break
            res.extend(_temp)
            offset += limit
        return res

    @classmethod
    @DB.connection_context()
    def create_folder(cls, file, parent_id, name, count, tenant_id, created_by):
        # 递归创建文件夹结构
        # 参数：
        # file：当前文件对象
        # parent_id：父文件夹 ID
        # name：要创建的文件夹名称列表
        # 计数：当前创作深度
        # tenant_id：租户 ID
        # created_by：由用户 ID 创建
        # 返回：
        # 创建文件对象
        if count > len(name) - 2:
            return file
        else:
            file = cls.insert(
                {"id": get_uuid(), "parent_id": parent_id, "tenant_id": tenant_id, "created_by": created_by, "name": name[count], "location": "", "size": 0, "type": FileType.FOLDER.value}
            )
            return cls.create_folder(file, file.id, name, count + 1, tenant_id, created_by)

    @classmethod
    @DB.connection_context()
    def is_parent_folder_exist(cls, parent_id):
        # 检查父文件夹是否存在
        # 参数：
        # parent_id：父文件夹 ID
        # 返回：
        # 布尔值，指示文件夹是否存在
        parent_files = cls.model.select().where(cls.model.id == parent_id)
        if parent_files.count():
            return True
        cls.delete_folder_by_pf_id(parent_id)
        return False

    @classmethod
    @DB.connection_context()
    def get_root_folder(cls, tenant_id):
        # 获取或创建租户根文件夹
        # 参数：
        # tenant_id：租户 ID
        # 返回：
        # 根文件夹字典
        for file in cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == cls.model.id)):
            return file.to_dict()

        file_id = get_uuid()
        file = {
            "id": file_id,
            "parent_id": file_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": "/",
            "type": FileType.FOLDER.value,
            "size": 0,
            "location": "",
        }
        cls.save(**file)
        return file

    @classmethod
    @DB.connection_context()
    def get_kb_folder(cls, tenant_id):
        # 获取租户的数据集文件夹
        # 参数：
        # tenant_id：租户 ID
        # 返回：
        # 知识库文件夹字典
        root_folder = cls.get_root_folder(tenant_id)
        root_id = root_folder["id"]
        kb_folder = cls.model.select().where((cls.model.tenant_id == tenant_id), (cls.model.parent_id == root_id), (cls.model.name == KNOWLEDGEBASE_FOLDER_NAME)).first()
        if not kb_folder:
            kb_folder = cls.new_a_file_from_kb(tenant_id, KNOWLEDGEBASE_FOLDER_NAME, root_id)
            return kb_folder
        return kb_folder.to_dict()

    @classmethod
    @DB.connection_context()
    def new_a_file_from_kb(cls, tenant_id, name, parent_id, ty=FileType.FOLDER.value, size=0, location=""):
        # 从数据集中创建一个新文件，或返回现有文件。
        # 包括重复数据删除以处理并发情况下的竞争条件
        # 请求可能已创建重复条目。
        # 参数：
        # tenant_id：租户 ID
        # 名称： File 名称
        # parent_id：父文件夹 ID
        # 型：File型
        # 尺寸： File 尺寸
        # 位置： File 位置
        # 返回：
        # 创建或现有文件字典
        existing = list(cls.model.select().where((cls.model.tenant_id == tenant_id) & (cls.model.parent_id == parent_id) & (cls.model.name == name)).order_by(cls.model.create_time.asc()))
        if existing:
            if len(existing) > 1:
                logger.warning(
                    "发现 %d 个名为“%s”的重复条目，父目录为 %s，仅保留第一个",
                    len(existing),
                    name,
                    parent_id,
                )
                keep_id = existing[0].id
                with DB.atomic():
                    for dup in existing[1:]:
                        File.update(parent_id=keep_id).where(File.parent_id == dup.id).execute()
                        cls.delete_by_id(dup.id)
            return existing[0].to_dict()
        file = {
            "id": get_uuid(),
            "parent_id": parent_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": name,
            "type": ty,
            "size": size,
            "location": location,
            "source_type": FileSource.KNOWLEDGEBASE,
        }
        cls.save(**file)
        return file

    @classmethod
    @DB.connection_context()
    def init_skills_folder(cls, root_id, tenant_id):
        # 如果技能文件夹不存在则初始化。
        # 删除可能已创建的重复条目
        # 通过并发竞争条件 (TOCTOU)。
        # 参数：
        # root_id：根文件夹 ID
        # tenant_id：租户 ID
        existing = list(cls.model.select().where((cls.model.name == SKILLS_FOLDER_NAME) & (cls.model.parent_id == root_id) & (cls.model.tenant_id == tenant_id)).order_by(cls.model.create_time.asc()))
        if existing:
            if len(existing) > 1:
                logger.warning(
                    "发现 %d 个重复的“%s”文件夹，根目录为 %s，仅保留第一个",
                    len(existing),
                    SKILLS_FOLDER_NAME,
                    root_id,
                )
                keep_id = existing[0].id
                with DB.atomic():
                    for dup in existing[1:]:
                        cls.model.update(parent_id=keep_id).where(cls.model.parent_id == dup.id).execute()
                        cls.delete_by_id(dup.id)
            return
        file_id = get_uuid()
        file = {
            "id": file_id,
            "parent_id": root_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": SKILLS_FOLDER_NAME,
            "type": FileType.FOLDER.value,
            "size": 0,
            "location": "",
        }
        cls.save(**file)

    @classmethod
    @DB.connection_context()
    def init_knowledgebase_docs(cls, root_id, tenant_id):
        # 初始化数据集文档。
        # 删除可能已创建的重复条目
        # 通过并发竞争条件 (TOCTOU)。
        # 参数：
        # root_id：根文件夹 ID
        # tenant_id：租户 ID
        existing = list(
            cls.model.select().where((cls.model.name == KNOWLEDGEBASE_FOLDER_NAME) & (cls.model.parent_id == root_id) & (cls.model.tenant_id == tenant_id)).order_by(cls.model.create_time.asc())
        )
        if existing:
            if len(existing) > 1:
                logger.warning(
                    "发现 %d 个重复的“%s”文件夹，根目录为 %s，仅保留第一个",
                    len(existing),
                    KNOWLEDGEBASE_FOLDER_NAME,
                    root_id,
                )
                keep_id = existing[0].id
                with DB.atomic():
                    for dup in existing[1:]:
                        cls.model.update(parent_id=keep_id).where(cls.model.parent_id == dup.id).execute()
                        cls.delete_by_id(dup.id)
            return
        folder = cls.new_a_file_from_kb(tenant_id, KNOWLEDGEBASE_FOLDER_NAME, root_id)

        for kb in Knowledgebase.select(*[Knowledgebase.id, Knowledgebase.name]).where(Knowledgebase.tenant_id == tenant_id):
            kb_folder = cls.new_a_file_from_kb(tenant_id, kb.name, folder["id"])
            for doc in DocumentService.query(kb_id=kb.id):
                FileService.add_file_from_kb(doc.to_dict(), kb_folder["id"], tenant_id)

    @classmethod
    @DB.connection_context()
    def get_parent_folder(cls, file_id):
        # 获取文件的父文件夹
        # 参数：
        # file_id：File ID
        # 返回：
        # 父文件夹对象
        file = cls.model.select().where(cls.model.id == file_id)
        if file.count():
            e, file = cls.get_by_id(file[0].parent_id)
            if not e:
                raise RuntimeError("Database error (File retrieval)!")
        else:
            raise RuntimeError("Database error (File doesn't exist)!")
        return file

    @classmethod
    @DB.connection_context()
    def get_all_parent_folders(cls, start_id):
        # 获取路径中的所有父文件夹
        # 参数：
        # start_id：启动文件ID
        # 返回：
        # 父文件夹对象列表
        parent_folders = []
        current_id = start_id
        while current_id:
            e, file = cls.get_by_id(current_id)
            if e and file.parent_id != file.id:
                parent_folders.append(file)
                current_id = file.parent_id
            else:
                parent_folders.append(file)
                break
        return parent_folders

    @classmethod
    @DB.connection_context()
    def insert(cls, file):
        # 插入新文件记录
        # 参数：
        # 文件：File 数据字典
        # 返回：
        # 创建文件对象
        if not cls.save(**file):
            raise RuntimeError("Database error (File)!")
        return File(**file)

    @classmethod
    @DB.connection_context()
    def delete(cls, file):
        return cls.delete_by_id(file.id)

    @classmethod
    @DB.connection_context()
    def delete_by_pf_id(cls, folder_id):
        return cls.model.delete().where(cls.model.parent_id == folder_id).execute()

    @classmethod
    @DB.connection_context()
    def delete_folder_by_pf_id(cls, user_id, folder_id):
        try:
            files = cls.model.select().where((cls.model.tenant_id == user_id) & (cls.model.parent_id == folder_id))
            for file in files:
                cls.delete_folder_by_pf_id(user_id, file.id)
            return (cls.model.delete().where((cls.model.tenant_id == user_id) & (cls.model.id == folder_id)).execute(),)
        except Exception:
            logger.exception("按父目录ID删除文件夹失败")
            raise RuntimeError("Database error (File retrieval)!")

    @classmethod
    @DB.connection_context()
    def get_file_count(cls, tenant_id):
        files = cls.model.select(cls.model.id).where(cls.model.tenant_id == tenant_id)
        return len(files)

    @classmethod
    @DB.connection_context()
    def get_folder_size(cls, folder_id):
        size = 0

        def dfs(parent_id):
            nonlocal size
            for f in cls.model.select(*[cls.model.id, cls.model.size, cls.model.type]).where(cls.model.parent_id == parent_id, cls.model.id != parent_id):
                size += f.size
                if f.type == FileType.FOLDER.value:
                    dfs(f.id)

        dfs(folder_id)
        return size

    @classmethod
    @DB.connection_context()
    def add_file_from_kb(cls, doc, kb_folder_id, tenant_id):
        for _ in File2DocumentService.get_by_document_id(doc["id"]):
            return
        file = {
            "id": get_uuid(),
            "parent_id": kb_folder_id,
            "tenant_id": tenant_id,
            "created_by": tenant_id,
            "name": doc["name"],
            "type": doc["type"],
            "size": doc["size"],
            "location": doc["location"],
            "source_type": FileSource.KNOWLEDGEBASE,
        }
        # 文件管理器中的文件节点
        cls.save(**file)
        # file 与 document 的关联关系
        File2DocumentService.save(id=get_uuid(), file_id=file["id"], document_id=doc["id"])

    @classmethod
    @DB.connection_context()
    def move_file(cls, file_ids, folder_id):
        try:
            cls.filter_update((cls.model.id << file_ids,), {"parent_id": folder_id})
        except Exception:
            logger.exception("移动文件失败")
            raise RuntimeError("Database error (File move)!")

    @classmethod
    def _discard_orphaned_document(cls, doc) -> bool:
        """删除因已删除的知识库及其碎片而搁浅的文档。

        连接器同步从外部文档派生文档 ID，因此一行
        以这种方式搁浅，不断回答“`get_by_id`` and blocks that document
        from ever being ingested again -- while being invisible to the user,
        because the knowledge base it names is gone. Returns whether it was
        removed.

        Mirrors the teardown ``delete_docs`”执行，减去块工作：
        这些块随着数据集删除时删除的索引而去，并且
        文档的租户不再可以通过其知识库进行解析，
        所以没有剩下的索引需要寻址。存储和行清理是
        尽力而为——重点是畅通摄入，因此碎片无法
        达到一定程度后，不得再次发生碰撞。"""
        if KnowledgebaseService.get_or_none(id=doc.kb_id) is not None:
            return False

        logger.warning("丢弃孤立文档 %s：其 知识库ID=%s 不再存在。", doc.id, doc.kb_id)
        try:
            bucket, location = File2DocumentService.get_storage_address(doc_id=doc.id)
            TaskService.filter_delete([Task.doc_id == doc.id])
            f2d = File2DocumentService.get_by_document_id(doc.id)
            deleted_file_count = 0
            if f2d:
                deleted_file_count = cls.filter_delete([File.source_type == FileSource.KNOWLEDGEBASE, File.id == f2d[0].file_id])
            File2DocumentService.delete_by_document_id(doc.id)
            if deleted_file_count > 0:
                settings.STORAGE_IMPL.rm(bucket, location)
        except Exception:
            logger.exception("无法完全清理孤立文档 %s；无论如何删除该行", doc.id)

        DocumentService.delete_by_id(doc.id)
        return True

    # 文件上传
    @classmethod
    @DB.connection_context()
    def upload_document(self, kb, file_objs, user_id, src="local", parent_path: str | None = None, parser_config_override: dict | None = None):
        """保存上传文件及其数据库关系，但不创建解析任务。

        链路顺序：
        1. 准备用户文件根目录和知识库目录；
        2. 为每个上传文件生成稳定的 ``doc_id``；
        3. 将完整原文件保存到对象存储，键为 ``kb.id + location``；
        4. 如能生成缩略图，则另存为 ``kb.id + thumbnail_location``；
        5. 将 Document 元数据写入 MySQL；
        6. 由 ``add_file_from_kb`` 创建文件树及 File2Document 关联。

        Document 通过 ``kb_id/location/thumbnail`` 定位对象存储内容；
        File2Document 负责把文件管理模块中的 file_id 映射到 document_id。
        解析任务由后续 ``POST /documents/ingest`` 创建，而不是由本方法创建。

        参数：
            kb: Knowledgebase ORM 对象；``kb.id`` 同时是原文件/缩略图的对象存储 bucket。
            file_objs: Quart ``FileStorage`` 列表；每项提供 filename/read()，可选携带稳定 id。
            user_id: 当前上传用户/租户，用于文件树和 Document.created_by。
            src: 文档来源标记，例如 local、web 或连接器来源。
            parent_path: 可选的对象 key 前缀；会先做路径清洗，不能越出知识库目录。
            parser_config_override: 仅允许上传表格时覆盖的解析配置，合并后写入 Document。
        """
        root_folder = self.get_root_folder(user_id)
        pf_id = root_folder["id"]
        self.init_knowledgebase_docs(pf_id, user_id)
        kb_root_folder = self.get_kb_folder(user_id)
        kb_folder = self.new_a_file_from_kb(kb.tenant_id, kb.name, kb_root_folder["id"])

        safe_parent_path = sanitize_path(parent_path)

        # 将 parser_config_override 与 KB parser_config 合并（如果提供）
        base_parser_config = kb.parser_config or {}
        if parser_config_override and isinstance(parser_config_override, dict):
            merged_parser_config = {**base_parser_config, **parser_config_override}
        else:
            merged_parser_config = base_parser_config

        err, files = [], []
        for file in file_objs:
            # doc_id 是整条 RAG 链路的业务主键：MySQL document.id、task.doc_id、
            # ES Chunk.doc_id 以及最终检索结果中的 doc_id 都使用它进行关联。
            doc_id = file.id if hasattr(file, "id") else get_uuid()
            e, doc = DocumentService.get_by_id(doc_id)
            if e and str(doc.kb_id) != str(kb.id):
                if not self._discard_orphaned_document(doc):
                    logger.warning(
                        "检测到 %s 的现有文档 ID 冲突：属于 知识库ID=%s，传入 知识库ID=%s。跳过更新以避免交叉 KB 覆盖。",
                        doc_id,
                        doc.kb_id,
                        kb.id,
                    )
                    user_msg = f"Existing document id collision with knowledge base '{doc.kb_id}'; skipping update."
                    err.append(file.filename + ": " + user_msg)
                    continue
                # 滞留排没了；作为新文档摄取。
                e, doc = False, None
            if e:
                try:
                    blob = file.read()
                    # 连接器提供的指纹 (e.g.xxhash128(S3 ETag))
                    # 优先：对于连接器来源的文档，绕过
                    # 路径使用指纹为 content_hash，因此恢复
                    # 到 xxhash128(blob) 这里会击败它。
                    incoming_fp = getattr(file, "fingerprint", None)
                    new_hash = incoming_fp or xxhash.xxh128(blob).hexdigest()
                    old_hash = doc.content_hash or ""

                    # 同 doc_id 再次上传时覆盖原对象；只有 content_hash 变化才把该文档加入
                    # files 返回值，避免内容未变化时触发不必要的后续解析。
                    settings.STORAGE_IMPL.put(kb.id, doc.location, blob, kb.tenant_id)
                    doc.size = len(blob)
                    doc.content_hash = new_hash
                    doc = doc.to_dict()
                    # 创建 Document 元数据
                    DocumentService.update_by_id(doc["id"], doc)
                    if new_hash != old_hash:
                        files.append((doc, blob))
                except Exception as exc:
                    logger.exception("更新文档失败 %s", doc_id)
                    err.append(file.filename + ": " + str(exc))
                continue
            try:
                DocumentService.check_doc_health(kb.tenant_id, file.filename)
                filename = duplicate_name(DocumentService.query, name=file.filename, kb_id=kb.id)  # 处理重复名称, 如果同一知识库中已经存在相同文件名，系统会生成一个不冲突的新名称
                # 判断文件类型
                filetype = filename_type(filename)
                if filetype == FileType.OTHER.value:
                    raise RuntimeError("This type of file has not been supported yet!")

                location = filename if not safe_parent_path else f"{safe_parent_path}/{filename}"
                while settings.STORAGE_IMPL.obj_exist(kb.id, location):
                    location += "_"

                blob = file.read()
                if filetype == FileType.PDF.value:
                    # PDF文档解析
                    blob = read_potential_broken_pdf(blob)
                # 对象存储中保存完整原文件，而不是 Chunk 文本。
                # bucket=kb.id，object key=location；MySQL document.location 保存 object key。
                settings.STORAGE_IMPL.put(kb.id, location, blob)

                # 生成缩略图
                img = thumbnail_img(filename, blob)
                thumbnail_location = ""
                # 如果生成了缩略图, 也将缩略图存储到MinIO
                if img is not None:
                    thumbnail_location = f"thumbnail_{doc_id}.png"
                    # 缩略图与原文件位于同一个知识库 bucket，document.thumbnail 保存其 object key。
                    settings.STORAGE_IMPL.put(kb.id, thumbnail_location, img)

                incoming_fp = getattr(file, "fingerprint", None)

                # 创建document记录
                doc = {
                    "id": doc_id,
                    "kb_id": kb.id,
                    # 判断解析器
                    "parser_id": self.get_parser(filetype, filename, kb.parser_id),
                    "pipeline_id": kb.pipeline_id,
                    # Chunk 大小、重叠率、页码范围等。
                    "parser_config": merged_parser_config,
                    "created_by": user_id,
                    "type": filetype,
                    "name": filename,
                    "source_type": src,
                    "suffix": Path(filename).suffix.lstrip("."),
                    # MinIO 中的对象名称
                    "location": location,
                    "size": len(blob),
                    "thumbnail": thumbnail_location,
                    # 用于判断文件内容是否发生变化
                    "content_hash": incoming_fp or xxhash.xxh128(blob).hexdigest(),
                }

                # 先写 Document 元数据，再建立文件管理记录和 File2Document 关系。
                # 关联依赖显式 ID/字段而非文件名：Document.id=doc_id，Document.kb_id/location
                # 指向对象存储，File2Document.file_id/document_id 连接文件树与知识库文档。
                DocumentService.insert(doc)
                FileService.add_file_from_kb(doc, kb_folder["id"], kb.tenant_id)
                logger.info(
                    "文档存储记录已创建 文档ID=%s 知识库ID=%s 存储位置=%s 文件类型=%s 文件大小=%d 是否有缩略图=%s",
                    doc_id,
                    kb.id,
                    location,
                    filetype,
                    len(blob),
                    bool(thumbnail_location),
                )
                files.append((doc, blob))
            except Exception as e:  # noqa: BLE001 - collect per-file errors and keep processing the rest
                err.append(file.filename + ": " + str(e))

        return err, files

    @classmethod
    @DB.connection_context()
    def list_all_files_by_parent_id(cls, parent_id):
        try:
            files = cls.model.select().where((cls.model.parent_id == parent_id) & (cls.model.id != parent_id))
            return list(files)
        except Exception:
            logger.exception("list_by_parent_id 失败")
            raise RuntimeError("Database error (list_by_parent_id)!")

    @staticmethod
    def parse_docs(file_objs, user_id):
        with ThreadPoolExecutor(max_workers=12) as exe:
            threads = []
            for file in file_objs:
                threads.append(exe.submit(FileService.parse, file.filename, file.read(), False))

            res = []
            for th in threads:
                res.append(th.result())

        return "\n\n".join(res)

    @staticmethod
    def parse(filename, blob, img_base64=True, tenant_id=None, layout_recognize=None):
        from api.apps import current_user
        from rag.app import audio, email, naive, picture, presentation

        def dummy(prog=None, msg=""):
            pass

        FACTORY = {ParserType.PRESENTATION.value: presentation, ParserType.PICTURE.value: picture, ParserType.AUDIO.value: audio, ParserType.EMAIL.value: email}
        parser_config = {"chunk_token_num": 16096, "delimiter": "\n!?;。；！？", "layout_recognize": layout_recognize or "Plain Text"}
        kwargs = {"lang": "English", "callback": dummy, "parser_config": parser_config, "from_page": 0, "to_page": MAXIMUM_PAGE_NUMBER, "tenant_id": current_user.id if current_user else tenant_id}
        file_type = filename_type(filename)
        if img_base64 and file_type == FileType.VISUAL.value:
            return GptV4.image2base64(blob)
        cks = FACTORY.get(FileService.get_parser(filename_type(filename), filename, ""), naive).chunk(filename, blob, **kwargs)
        return f"\n -----------------\nFile: {filename}\nContent as following: \n" + "\n".join([ck["content_with_weight"] for ck in cks])

    @staticmethod
    def get_parser(doc_type, filename, default):
        if doc_type == FileType.VISUAL:
            return ParserType.PICTURE.value
        if doc_type == FileType.AURAL:
            return ParserType.AUDIO.value
        if re.search(r"\.(ppt|pptx|pages)$", filename):
            return ParserType.PRESENTATION.value
        if re.search(r"\.(msg|eml)$", filename):
            return ParserType.EMAIL.value
        return default

    @staticmethod
    def get_blob(user_id, location):
        bname = f"{user_id}-downloads"
        return settings.STORAGE_IMPL.get(bname, location)

    @staticmethod
    def put_blob(user_id, location, blob):
        bname = f"{user_id}-downloads"
        return settings.STORAGE_IMPL.put(bname, location, blob)

    @classmethod
    @DB.connection_context()
    def delete_docs(cls, doc_ids, tenant_id):
        root_folder = FileService.get_root_folder(tenant_id)
        pf_id = root_folder["id"]
        FileService.init_knowledgebase_docs(pf_id, tenant_id)
        errors = ""
        kb_table_num_map = {}
        for doc_id in doc_ids:
            try:
                e, doc = DocumentService.get_by_id(doc_id)
                if not e:
                    raise RuntimeError("document not found")
                tenant_id = DocumentService.get_tenant_id(doc_id)
                if not tenant_id:
                    raise RuntimeError("Tenant not found!")

                b, n = File2DocumentService.get_storage_address(doc_id=doc_id)

                TaskService.filter_delete([Task.doc_id == doc_id])
                if not DocumentService.remove_document(doc, tenant_id):
                    raise RuntimeError("Database error (Document removal)!")

                f2d = File2DocumentService.get_by_document_id(doc_id)
                deleted_file_count = 0
                if f2d:
                    deleted_file_count = FileService.filter_delete([File.source_type == FileSource.KNOWLEDGEBASE, File.id == f2d[0].file_id])
                File2DocumentService.delete_by_document_id(doc_id)
                if deleted_file_count > 0:
                    settings.STORAGE_IMPL.rm(b, n)

                doc_parser = doc.parser_id
                if doc_parser == ParserType.TABLE:
                    kb_id = doc.kb_id
                    if kb_id not in kb_table_num_map:
                        counts = DocumentService.count_by_kb_id(kb_id=kb_id, keywords="", run_status=[TaskStatus.DONE], types=[])
                        kb_table_num_map[kb_id] = counts
                    kb_table_num_map[kb_id] -= 1
                    if kb_table_num_map[kb_id] <= 0:
                        KnowledgebaseService.delete_field_map(kb_id)
            except Exception as e:  # noqa: BLE001 - aggregate per-document errors and continue deleting the rest
                errors += str(e)

        return errors

    _ALLOWED_SCHEMES: ClassVar[set[str]] = {"http", "https"}

    @staticmethod
    def _validate_url_for_crawl(url: str) -> tuple[str, str]:
        """如果 URL 无法安全爬行（SSRF 防护装置），请升起 ValueError。

        委托给 :func:`common.ssrf_guard.assert_url_is_safe`，其中
        验证方案、主机名和每个 DNS 解析的地址，以及
        返回 DNS 固定的“`(hostname, resolved_ip)`”。

        仅方案和主机（以及端口（如果存在））被转发到
        保护，以便 *url* 中的凭据或查询参数永远不会
        写入日志。"""
        from urllib.parse import urlparse

        parsed = urlparse(url)
        port_suffix = f":{parsed.port}" if parsed.port else ""
        redacted = f"{parsed.scheme}://{parsed.hostname}{port_suffix}"
        return assert_url_is_safe(redacted, allowed_schemes=FileService._ALLOWED_SCHEMES)

    @staticmethod
    def upload_info(user_id, file, url: str | None = None):
        def structured(filename, filetype, blob, content_type):
            nonlocal user_id
            if filetype == FileType.PDF.value:
                blob = read_potential_broken_pdf(blob)

            location = get_uuid()
            FileService.put_blob(user_id, location, blob)

            return {
                "id": location,
                "name": filename,
                "size": sys.getsizeof(blob),
                "extension": filename.split(".")[-1].lower(),
                "mime_type": content_type,
                "created_by": user_id,
                "created_at": time.time(),
                "preview_url": None,
            }

        if url:
            from urllib.parse import urljoin as _urljoin

            import requests as _requests

            from api.utils.web_utils import BROWSER_FETCH_TIMEOUT, browser_fetch_slot

            _MAX_CRAWL_REDIRECTS = 10

            with browser_fetch_slot():
                # 预解析完整重定向链，以便 AsyncWebCrawler 永远不会
                # 遵循服务器发送的重定向到未经验证的（可能
                # 内部）主机。每个跃点在被跟踪之前都会经过 SSRF 检查；
                # 已验证的（主机名、ip）对通过 Chromium 固定
                # --host-resolver-rules 因此浏览器无法重新解析其中任何一个
                # 通过新鲜的DNS 查询。
                current_url = url
                current_hostname, current_ip = FileService._validate_url_for_crawl(current_url)
                # 为我们在链中遇到的每个主机名累积 MAP 规则。
                host_pins: dict[str, str] = {current_hostname: current_ip}

                for _ in range(_MAX_CRAWL_REDIRECTS):
                    try:
                        _resp = _requests.get(
                            current_url,
                            timeout=10,
                            allow_redirects=False,
                        )
                    except _requests.RequestException as _exc:
                        raise ValueError(f"Failed to fetch {current_url!r}: {_exc}") from _exc

                    if _resp.status_code not in (301, 302, 303, 307, 308):
                        break

                    _location = _resp.headers.get("Location")
                    if not _location:
                        break

                    _next_url = _urljoin(current_url, _location)
                    _next_hostname, _next_ip = FileService._validate_url_for_crawl(_next_url)
                    host_pins[_next_hostname] = _next_ip
                    current_url = _next_url
                else:
                    raise ValueError(f"Exceeded {_MAX_CRAWL_REDIRECTS} redirects fetching {url!r}")

                # 构建覆盖每个经过验证的主机名的单个 MAP 规则字符串
                # 重定向链中的
                # 。 Chromium 对每个都使用固定的 IP，
                # 完全跳过 DNS 并消除重新绑定窗口。
                _map_rules = ",".join(f"MAP {h} {ip}" for h, ip in host_pins.items())

                from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CrawlResult, DefaultMarkdownGenerator, PruningContentFilter

                filename = re.sub(r"\?.*", "", url.split("/")[-1])

                async def adownload():
                    browser_config = BrowserConfig(
                        headless=True,
                        verbose=False,
                        extra_args=[f"--host-resolver-rules={_map_rules}"],
                    )
                    async with AsyncWebCrawler(config=browser_config) as crawler:
                        crawler_config = CrawlerRunConfig(markdown_generator=DefaultMarkdownGenerator(content_filter=PruningContentFilter()), pdf=True, screenshot=False)
                        # 使用最终解析的 URL，以便浏览器从
                        # 重定向目的地而不是重新跟踪链。
                        result: CrawlResult = await asyncio.wait_for(crawler.arun(url=current_url, config=crawler_config), timeout=BROWSER_FETCH_TIMEOUT)
                        return result

                page = asyncio.run(adownload())
                if page.pdf:
                    if filename.split(".")[-1].lower() != "pdf":
                        filename += ".pdf"
                    return structured(filename, "pdf", page.pdf, page.response_headers["content-type"])

                return structured(filename, "html", str(page.markdown).encode("utf-8"), page.response_headers["content-type"])

        DocumentService.check_doc_health(user_id, file.filename)
        return structured(file.filename, filename_type(file.filename), file.read(), file.content_type)

    @staticmethod
    def get_files(files: None | list[dict], raw: bool = False, layout_recognize: str | None = None) -> list[str] | tuple[list[str], list[dict]]:
        if not files:
            return []

        def image_to_base64(file):
            return "data:{};base64,{}".format(file["mime_type"], base64.b64encode(FileService.get_blob(file["created_by"], file["id"])).decode("utf-8"))

        with ThreadPoolExecutor(max_workers=5) as exe:
            threads = []
            imgs = []
            for file in files:
                if file["mime_type"].find("image") >= 0:
                    if raw:
                        imgs.append(FileService.get_blob(file["created_by"], file["id"]))
                    else:
                        threads.append(exe.submit(image_to_base64, file))
                    continue
                threads.append(exe.submit(FileService.parse, file["name"], FileService.get_blob(file["created_by"], file["id"]), True, file["created_by"], layout_recognize))

            if raw:
                return [th.result() for th in threads], imgs
            else:
                return [th.result() for th in threads]
