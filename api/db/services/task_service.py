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
import os
import random
import xxhash
from datetime import datetime

from api.db.db_utils import bulk_insert_into_db
from deepdoc.parser import PdfParser
from peewee import JOIN
from api.db.db_models import DB, File2Document, File
from api.db import FileType
from api.db.db_models import Task, Document, Knowledgebase, Tenant
from api.db.joint_services.tenant_model_service import get_composite_model_name_by_id
from api.db.services.common_service import CommonService
from api.db.services.document_service import DocumentService
from common.misc_utils import get_uuid
from common.time_utils import current_timestamp, get_format_time
from common.constants import StatusEnum, TaskStatus, MAXIMUM_PAGE_NUMBER, MAXIMUM_TASK_PAGE_NUMBER
from deepdoc.parser.excel_parser import RAGFlowExcelParser
from rag.utils.redis_conn import REDIS_CONN
from common import settings
from rag.nlp import search

CANVAS_DEBUG_DOC_ID = "dataflow_x"
GRAPH_RAPTOR_FAKE_DOC_ID = "graph_raptor_x"
TASK_MAX_LOG_LENGTH = int(os.environ.get("TASK_MAX_LOG_LENGTH", 3000))  # TEXT MAX 是 64 KiB 字节！
DOC_CHUNKING_COUNTER_TTL_SECONDS = 7 * 24 * 3600


def _doc_chunking_pending_key(doc_id: str) -> str:
    return f"doc:chunking_pending:{doc_id}"


def _doc_chunking_aborted_key(doc_id: str) -> str:
    return f"doc:chunking_aborted:{doc_id}"


def _doc_chunking_done_key(task_id: str) -> str:
    return f"doc:chunking_done:{task_id}"


def seed_doc_chunking_counter(doc_id: str, pending_count: int) -> bool:
    if not doc_id or pending_count <= 0:
        return False
    try:
        REDIS_CONN.delete(_doc_chunking_aborted_key(doc_id))
        return REDIS_CONN.set(
            _doc_chunking_pending_key(doc_id),
            str(pending_count),
            exp=DOC_CHUNKING_COUNTER_TTL_SECONDS,
        )
    except Exception:
        logging.exception("无法为文档 %s 播种分块计数器", doc_id)
        return False


def clear_doc_chunking_counter(doc_id: str) -> None:
    if not doc_id:
        return
    try:
        REDIS_CONN.delete(_doc_chunking_pending_key(doc_id))
    except Exception:
        logging.exception("无法清除文档 %s 的分块计数器", doc_id)


# todo 这个方法的功能
def abort_doc_chunking_counter(doc_id: str) -> None:
    if not doc_id:
        return
    try:
        REDIS_CONN.delete(_doc_chunking_pending_key(doc_id))
        REDIS_CONN.set(
            _doc_chunking_aborted_key(doc_id),
            "1",
            exp=DOC_CHUNKING_COUNTER_TTL_SECONDS,
        )
    except Exception:
        logging.exception("无法中止文档 %s 的分块计数器", doc_id)


def is_doc_chunking_aborted(doc_id: str) -> bool:
    if not doc_id:
        return False
    try:
        return bool(REDIS_CONN.get(_doc_chunking_aborted_key(doc_id)))
    except Exception:
        logging.exception("无法读取文档 %s 的分块中止标记", doc_id)
        return False


def credit_doc_chunking_task(doc_id: str, task_id: str) -> int | None:
    """完成一项标准分块任务。

    返回此任务记入时的递减后挂起计数
    第一次。当该任务已经完成时返回正值
    已记入，因此调用者将重试视为非最后重试。"""
    if not doc_id or not task_id:
        return None
    try:
        first_credit = REDIS_CONN.set_if_absent(
            _doc_chunking_done_key(task_id),
            "1",
            exp=DOC_CHUNKING_COUNTER_TTL_SECONDS,
        )
        if not first_credit:
            return 1
        pending_key = _doc_chunking_pending_key(doc_id)
        if REDIS_CONN.get(pending_key) is None:
            return -1
        return REDIS_CONN.decrby(pending_key, 1)
    except Exception:
        logging.exception("无法为文档 %s 分配分块任务 %s", task_id, doc_id)
        return None


def trim_header_by_lines(text: str, max_length) -> str:
    # 将标题文本修剪到最大长度，同时保留换行符
    # 参数：
    # text：输入要修剪的文本
    # max_length：最大允许长度
    # 返回：
    # 修剪后的文本
    len_text = len(text)
    if len_text <= max_length:
        return text
    for i in range(len_text):
        if text[i] == "\n" and len_text - i <= max_length:
            return text[i + 1 :]
    return text


class TaskService(CommonService):
    """Service 用于管理文档处理任务的类。

    该类扩展了CommonService，为文档提供专门的功能
    处理任务管理，包括任务创建、进度跟踪和块
    管理。它处理各种文档类型（PDF、Excel 等）并管理它们的
    处理生命周期。

    该类实现了一个具有重试机制和进度的健壮任务队列系统
    跟踪，支持同步和异步任务执行。

    属性：
        model：数据库操作的Task模型类。"""

    model = Task

    @classmethod
    @DB.connection_context()
    def get_task(cls, task_id, doc_ids=[]):
        """通过任务 ID 检索详细任务信息。

        该方法获取全面的任务详细信息，包括相关文档、
        数据集和租户信息。它还处理任务重试逻辑和
        进度更新。

        参数：
            task_id (str)：要检索的任务的唯一标识符。

        返回：
            字典：Task 详细字典包含所有任务信息和相关元数据。
                 如果未找到任务或已超出重试限制，则返回 None。"""
        doc_id = cls.model.doc_id
        if doc_id == CANVAS_DEBUG_DOC_ID and doc_ids:
            doc_id = doc_ids[0]

        fields = [
            cls.model.id,
            cls.model.doc_id,
            cls.model.from_page,
            cls.model.to_page,
            cls.model.task_type,
            cls.model.retry_count,
            Document.kb_id,
            Document.parser_id,
            Document.parser_config,
            Document.name,
            Document.type,
            Document.location,
            Document.size,
            Knowledgebase.tenant_id,
            Knowledgebase.language,
            Knowledgebase.embd_id,
            Knowledgebase.tenant_embd_id,
            Knowledgebase.pagerank,
            Knowledgebase.parser_config.alias("kb_parser_config"),
            Tenant.img2txt_id,
            Tenant.asr_id,
            Tenant.llm_id,
            Tenant.tenant_llm_id,
            cls.model.update_time,
        ]
        docs = (
            cls.model.select(*fields)
            .join(Document, on=(doc_id == Document.id))
            .join(Knowledgebase, on=(Document.kb_id == Knowledgebase.id))
            .join(Tenant, on=(Knowledgebase.tenant_id == Tenant.id))
            .where(cls.model.id == task_id)
        )
        docs = list(docs.dicts())
        if not docs:
            return None
        doc = docs[0]

        msg = f"\n{datetime.now().strftime('%H:%M:%S')} Task has been received."
        prog = random.random() / 10.0
        if doc["retry_count"] >= 3:
            msg = "\nERROR: Task is abandoned after 3 times attempts."
            prog = -1

        cls.model.update(
            progress_msg=cls.model.progress_msg + msg,
            progress=prog,
            retry_count=doc["retry_count"] + 1,
        ).where(cls.model.id == doc["id"]).execute()

        if docs[0]["retry_count"] >= 3:
            abort_doc_chunking_counter(docs[0]["doc_id"])
            DocumentService.update_by_id(docs[0]["doc_id"], {"progress": -1, "run": TaskStatus.FAIL.value, "update_time": current_timestamp(), "update_date": get_format_time()})
            return None

        return doc

    @classmethod
    @DB.connection_context()
    def get_tasks(cls, doc_id: str):
        """检索与文档关联的所有任务。

        此方法获取给定文档的所有处理任务，按页面排序
        数量和创建时间。它包括任务进度和块信息。

        参数：
            doc_id (str)：文档的唯一标识符。

        返回：
            list[dict]：包含任务详细信息的任务字典列表。
                       如果未找到任务，则返回 None。"""
        fields = [
            cls.model.id,
            cls.model.from_page,
            cls.model.progress,
            cls.model.digest,
            cls.model.chunk_ids,
        ]
        tasks = cls.model.select(*fields).order_by(cls.model.from_page.asc(), cls.model.create_time.desc()).where(cls.model.doc_id == doc_id)
        tasks = list(tasks.dicts())
        if not tasks:
            return None
        return tasks

    @classmethod
    @DB.connection_context()
    def get_tasks_progress_by_doc_ids(cls, doc_ids: list[str]):
        """检索与特定文档关联的所有任务。

        此方法获取给定文档 ID 的所有处理任务，按顺序排列
        创建时间。它包括任务进度和块信息。

        参数：
            doc_ids (str)：文档的唯一标识符。

        返回：
            list[dict]：包含任务详细信息的任务字典列表。
                       如果未找到任务，则返回 None。"""
        fields = [cls.model.id, cls.model.doc_id, cls.model.from_page, cls.model.progress, cls.model.progress_msg, cls.model.digest, cls.model.chunk_ids, cls.model.create_time]
        tasks = cls.model.select(*fields).order_by(cls.model.create_time.desc()).where(cls.model.doc_id.in_(doc_ids))
        tasks = list(tasks.dicts())
        if not tasks:
            return None
        return tasks

    @classmethod
    @DB.connection_context()
    def update_chunk_ids(cls, id: str, chunk_ids: str):
        """更新与任务关联的块 IDs。

        该方法更新任务的chunk_ids字段，该字段存储了任务的IDs
        以空格分隔的字符串格式处理文档块。

        参数：
            id (str)：任务的唯一标识符。
            chunk_ids (str)：以空格分隔的块标识符字符串。"""
        cls.model.update(chunk_ids=chunk_ids).where(cls.model.id == id).execute()

    @classmethod
    @DB.connection_context()
    def get_ongoing_doc_name(cls):
        """获取当前正在处理的文档的名称。

        此方法检索有关处于处理状态的文档的信息，
        包括它们的位置和相关的 IDs。它使用数据库锁定来确保
        访问任务信息时的线程安全。

        返回：
            list[tuple]：元组列表，每个元组包含（parent_id/kb_id，位置）
                        用于当前正在处理的文档。如果返回空列表
                        没有文件正在处理。"""
        with DB.lock("get_task", -1):
            docs = (
                cls.model.select(*[Document.id, Document.kb_id, Document.location, File.parent_id])
                .join(Document, on=(cls.model.doc_id == Document.id))
                .join(
                    File2Document,
                    on=(File2Document.document_id == Document.id),
                    join_type=JOIN.LEFT_OUTER,
                )
                .join(
                    File,
                    on=(File2Document.file_id == File.id),
                    join_type=JOIN.LEFT_OUTER,
                )
                .where(
                    Document.status == StatusEnum.VALID.value,
                    Document.run == TaskStatus.RUNNING.value,
                    ~(Document.type == FileType.VIRTUAL.value),
                    cls.model.progress < 1,
                    cls.model.create_time >= current_timestamp() - 1000 * 600,
                )
            )
            docs = list(docs.dicts())
            if not docs:
                return []

            return list(
                set(
                    [
                        (
                            d["parent_id"] if d["parent_id"] else d["kb_id"],
                            d["location"],
                        )
                        for d in docs
                    ]
                )
            )

    @classmethod
    @DB.connection_context()
    def do_cancel(cls, id):
        """根据任务的文档状态检查是否应取消任务。

        此方法通过检查任务来确定是否应取消任务
        关联文档的运行状态和进度。任务应该被取消
        如果其文档被标记为取消或有负面进展。

        参数：
            id (str)：要检查的任务的唯一标识符。

        返回：
            bool：如果应取消任务则为 True，否则为 False。"""
        task = cls.model.get_by_id(id)
        _, doc = DocumentService.get_by_id(task.doc_id)
        return doc.run == TaskStatus.CANCEL.value or doc.progress < 0

    @classmethod
    @DB.connection_context()
    def update_progress(cls, id, info):
        """更新任务的进度信息。

        此方法会更新任务的进度消息和完成百分比。
        它处理特定于平台的行为（macOS 与其他平台）并使用数据库锁定
        必要时保证线程安全。

        更新规则：
            - progress_msg：始终将新消息附加到现有消息，并将结果修剪为最多 3000 行。
            - 进度：当 (a) 新进度 >= 1（允许从 -1 恢复）时更新，或者
                        (b) 当前进度 != -1 AND （新进度比现有进度大 -1 OR）。

        参数：
            id (str)：要更新的任务的唯一标识符。
            info (dict)：包含进度信息的字典，其键为：
                        - progress_msg（str，可选）：要附加的进度消息
                        - 进度（float，可选）：进度百分比（0.0 到 1.0）"""
        try:
            task = cls.model.get_by_id(id)
        except cls.model.DoesNotExist:
            logging.info("跳过已删除任务的进度更新 %s", id)
            return
        if not task:
            logging.warning("Update_progress 错误：找不到任务")
            return

        if os.environ.get("MACOS"):
            if info["progress_msg"]:
                progress_msg = trim_header_by_lines((task.progress_msg or "") + "\n" + info["progress_msg"], TASK_MAX_LOG_LENGTH)
                cls.model.update(progress_msg=progress_msg).where(cls.model.id == id).execute()
            if "progress" in info:
                prog = info["progress"]
                cls.model.update(progress=prog).where((cls.model.id == id) & ((prog >= 1) | ((cls.model.progress != -1) & ((prog == -1) | (prog > cls.model.progress))))).execute()
        else:
            with DB.lock("update_progress", -1):
                if info["progress_msg"]:
                    progress_msg = trim_header_by_lines((task.progress_msg or "") + "\n" + info["progress_msg"], TASK_MAX_LOG_LENGTH)
                    cls.model.update(progress_msg=progress_msg).where(cls.model.id == id).execute()
                if "progress" in info:
                    prog = info["progress"]
                    cls.model.update(progress=prog).where((cls.model.id == id) & ((prog >= 1) | ((cls.model.progress != -1) & ((prog == -1) | (prog > cls.model.progress))))).execute()

        begin_at = task.begin_at
        if begin_at is not None:
            process_duration = (datetime.now() - begin_at).total_seconds()
            cls.model.update(process_duration=process_duration).where(cls.model.id == id).execute()
        if info.get("progress") == -1:
            doc_info = {"progress": -1, "run": TaskStatus.FAIL.value, "update_time": current_timestamp(), "update_date": get_format_time()}
            if info.get("progress_msg"):
                doc_info["progress_msg"] = trim_header_by_lines((task.progress_msg or "") + "\n" + info["progress_msg"], TASK_MAX_LOG_LENGTH)
            DocumentService.model.update(doc_info).where(
                (DocumentService.model.id == task.doc_id) & ((DocumentService.model.run.is_null(True)) | (DocumentService.model.run != TaskStatus.CANCEL.value))
            ).execute()

    @classmethod
    @DB.connection_context()
    def delete_by_doc_ids(cls, doc_ids):
        """删除与文档关联的任务。"""
        return cls.model.delete().where(cls.model.doc_id.in_(doc_ids)).execute()


def queue_tasks(doc: dict, bucket: str, name: str, priority: int):
    """创建文档处理任务并对其进行排队。

    此函数根据文档的类型和配置创建文档的处理任务。
    它以不同的方式处理不同的文档类型（PDF、Excel等）并管理任务
    分块和配置。它还通过检查来实现任务重用优化
    对于之前完成的任务。

    参数：
        doc (dict)：Document 字典，包含元数据和配置。
        Bucket (str)：存储文档的存储桶名称。
        name (str): File 文档的名称。
        priority（int，可选）：任务排队的优先级（默认为0）。

    注意：
        - 对于 PDF 文档，根据配置在每个页面范围创建任务
        - 对于 Excel 文档，任务是按行范围创建的
        - 计算 Task 摘要以进行优化和重用
        - 以前的任务块如果可用的话可以重用"""

    # 【链路二：创建并投递解析 Task】
    # 一个 document 不一定只对应一个 Task：PDF 通常按页范围拆分，表格按行范围拆分。
    # Task 表保存可恢复、可追踪的持久化状态；Redis Stream 只负责把任务通知给 Worker。
    # Redis 消息主要携带 task.id/doc_id/page range，Worker 收到后会再从 MySQL 补全
    # 文档、知识库、模型和 parser_config 等运行上下文。
    def new_task():
        return {
            "id": get_uuid(),
            "doc_id": doc["id"],
            "progress": 0.0,
            "from_page": 0,
            "to_page": MAXIMUM_TASK_PAGE_NUMBER,
            "begin_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }

    parse_task_array = []

    if doc["type"] == FileType.PDF.value:
        # 此处读取原文件仅用于计算 PDF 总页数，从而决定如何拆 Task；并未开始内容解析。
        file_bin = settings.STORAGE_IMPL.get(bucket, name)
        # 获取文件页数
        pages = PdfParser.total_page_number(doc["name"], file_bin)
        if pages is None:
            pages = 0
        # 每个 Task 只负责 [from_page, to_page) 范围，默认每 12 页一个任务。
        page_size = doc["parser_config"].get("task_page_size") or 12
        if doc["parser_id"] == "paper":
            page_size = doc["parser_config"].get("task_page_size") or 22

        # 将 MinerU 解析拆分为基于页面的任务会重复将整个 PDF 上传到 MinerU API 服务器，增加网络带宽使用量，但不会提高解析速度。 MinerU API 服务器还会存储这些文件的重复副本，从而浪费磁盘空间。
        is_mineru = False
        layout_recognizer = doc["parser_config"].get("layout_recognize", "")  # 'DeepDOC'
        # 如果 layout_recognizer 是一个 32 位字符串，就暂时认为它可能是 tenant_model 表中的模型 ID。
        if isinstance(layout_recognizer, str) and len(layout_recognizer) == 32:
            try:
                # 如果存的是模型 ID，单从 ID 看不出它是不是 MinerU，所以需要查询数据库：
                layout_recognizer = get_composite_model_name_by_id(layout_recognizer)
                if layout_recognizer.lower().endswith("@mineru"):
                    is_mineru = True
            except LookupError:
                pass
        if is_mineru:
            logging.info("Document %s 选择了 MinerU 不分割任务模式，页面大小为 %s", doc["id"], MAXIMUM_TASK_PAGE_NUMBER)
        # 如果是mineru, 则不拆分
        if doc["parser_id"] in ["one", "knowledge_graph"] or doc["parser_config"].get("toc_extraction", False) or is_mineru:
            page_size = MAXIMUM_TASK_PAGE_NUMBER
        # parser_config.pages 是面向用户的 1-based 闭区间；Task 保存的是供 Python 切片使用的
        # 0-based [from_page, to_page) 半开区间，解析器只处理自己负责的页面。
        page_ranges = doc["parser_config"].get("pages") or [(1, MAXIMUM_PAGE_NUMBER)]
        for s, e in page_ranges:
            s -= 1
            s = max(0, s)
            e = min(e - 1, pages)
            for p in range(s, e, page_size):
                task = new_task()
                task["from_page"] = p
                task["to_page"] = min(p + page_size, e)
                parse_task_array.append(task)

    elif doc["parser_id"] == "table":
        # 表格解析把行号复用在 from_page/to_page 字段中，每 3000 行一个 Task。
        # 字段名虽然叫 page，但对 table parser 表示行范围。
        file_bin = settings.STORAGE_IMPL.get(bucket, name)
        rn = RAGFlowExcelParser.row_number(doc["name"], file_bin)
        for i in range(0, rn, 3000):
            task = new_task()
            task["from_page"] = i
            task["to_page"] = min(i + 3000, rn)
            parse_task_array.append(task)
    else:
        parse_task_array.append(new_task())

    # 根据parser_id确定后缀（与SAAS版本第444行一致）
    suffix = "common" if doc["parser_id"] != "resume" else "resume"

    # digest 由解析配置、doc_id 和页范围共同决定，用来判断重新解析时能否复用旧 Task 的 Chunk。
    chunking_config = DocumentService.get_chunking_config(doc["id"])
    for task in parse_task_array:
        hasher = xxhash.xxh64()
        for field in sorted(chunking_config.keys()):
            if field == "parser_config":
                for k in ["raptor", "graphrag"]:
                    if k in chunking_config[field]:
                        del chunking_config[field][k]
            hasher.update(str(chunking_config[field]).encode("utf-8"))
        for field in ["doc_id", "from_page", "to_page"]:
            hasher.update(str(task.get(field, "")).encode("utf-8"))
        task_digest = hasher.hexdigest()
        task["digest"] = task_digest
        task["progress"] = 0.0
        task["priority"] = priority

    prev_tasks = TaskService.get_tasks(doc["id"])
    ck_num = 0
    if prev_tasks:
        # 重跑时先尝试按 digest+页范围复用已完成 Task 的 Chunk；不能复用的旧 Chunk
        # 会从 doc store 删除，再以本次新任务重新生成，避免同一文档残留新旧版本。
        for task in parse_task_array:
            ck_num += reuse_prev_task_chunks(task, prev_tasks, chunking_config)
        TaskService.filter_delete([Task.doc_id == doc["id"]])
        pre_chunk_ids = []
        for pre_task in prev_tasks:
            if pre_task["chunk_ids"]:
                pre_chunk_ids.extend(pre_task["chunk_ids"].split())
        if pre_chunk_ids:
            settings.docStoreConn.delete({"id": pre_chunk_ids}, search.index_name(chunking_config["tenant_id"]), chunking_config["kb_id"])
    DocumentService.update_by_id(doc["id"], {"chunk_num": ck_num})

    # 先写 MySQL，再发 Redis。这样 Worker 即使立即取得消息，也能根据 task.id
    # 查询到完整任务；document.id 与 task.doc_id 建立一对多关系。
    bulk_insert_into_db(Task, parse_task_array, True)
    DocumentService.begin2parse(doc["id"])  # 将 document 更新为“排队/解析中”。

    # 已复用的 Task progress=1，不需要再次进入 Redis；只有未完成任务才投递给 Worker。
    unfinished_task_array = [task for task in parse_task_array if task["progress"] < 1.0]
    chunking_n = sum(1 for task in unfinished_task_array if not task.get("task_type"))
    if chunking_n > 0:
        # 同一文档可能被拆成多个 Task。Redis 计数器记录还有多少分页任务未结束，
        # 只有最后一个 Task 才负责执行文档级收尾流程。
        assert seed_doc_chunking_counter(doc["id"], chunking_n), "Can't access Redis. Please check the Redis' status."
    logging.info(
        "文档任务已创建 文档ID=%s 知识库ID=%s 解析器ID=%s 任务总数=%d 待处理数=%d 优先级=%s 队列后缀=%s",
        doc["id"],
        doc.get("kb_id"),
        doc.get("parser_id"),
        len(parse_task_array),
        len(unfinished_task_array),
        priority,
        suffix,
    )
    try:
        for unfinished_task in unfinished_task_array:
            # XADD 到 Redis Stream。投递成功不代表解析完成，只代表任务可以被消费者组领取。
            assert REDIS_CONN.queue_product(settings.get_svr_queue_name(priority, suffix), message=unfinished_task), "Can't access Redis. Please check the Redis' status."
        logging.info(
            "文档任务已进入队列 文档ID=%s 队列=%s 任务ID列表=%s",
            doc["id"],
            settings.get_svr_queue_name(priority, suffix),
            [task["id"] for task in unfinished_task_array],
        )
    except Exception:
        abort_doc_chunking_counter(doc["id"])
        raise


def reuse_prev_task_chunks(task: dict, prev_tasks: list[dict], chunking_config: dict):
    """尝试重用以前任务中的块以进行优化。

    此函数检查之前完成的任务中的块是否可以重用
    当前任务，可以显着提高处理效率。它匹配
    基于页面范围和配置摘要的任务。

    参数：
        任务 (dict)：可能重用块的当前任务字典。
        prev_tasks (list[dict])：要检查重用的先前任务字典列表。
        chunking_config (dict)：用于块处理的配置字典。

    返回：
        int：成功重用的块数。如果没有可以重用的块，则返回 0。

    注意：
        仅在以下情况下才可以重复使用块：
        - 先前的任务存在且具有匹配的页面范围和配置摘要
        - 上一个任务已成功完成（进度= 1.0）
        - 上一个任务具有有效块 IDs"""
    idx = 0
    while idx < len(prev_tasks):
        prev_task = prev_tasks[idx]
        if prev_task.get("from_page", 0) == task.get("from_page", 0) and prev_task.get("digest", 0) == task.get("digest", ""):
            break
        idx += 1

    if idx >= len(prev_tasks):
        return 0
    prev_task = prev_tasks[idx]
    if prev_task["progress"] < 1.0 or not prev_task["chunk_ids"]:
        return 0
    task["chunk_ids"] = prev_task["chunk_ids"]
    task["progress"] = 1.0
    if "from_page" in task and "to_page" in task and int(task["to_page"]) - int(task["from_page"]) >= 10**6:
        task["progress_msg"] = f"Page({int(task['from_page']) + 1}~{int(task['to_page'])}): "
    else:
        task["progress_msg"] = ""
    task["progress_msg"] = " ".join([datetime.now().strftime("%H:%M:%S"), task["progress_msg"], "Reused previous task's chunks."])
    prev_task["chunk_ids"] = ""

    return len(task["chunk_ids"].split())


def cancel_all_task_of(doc_id):
    abort_doc_chunking_counter(doc_id)
    for t in TaskService.query(doc_id=doc_id):
        try:
            REDIS_CONN.set(f"{t.id}-cancel", "x")
        except Exception as e:
            logging.exception(e)


def has_canceled(task_id):
    try:
        if REDIS_CONN.get(f"{task_id}-cancel"):
            logging.info(f"Task: {task_id} has been canceled")
            return True
    except Exception as e:
        logging.exception(e)
    return False


def queue_dataflow(tenant_id: str, flow_id: str, task_id: str, doc_id: str = CANVAS_DEBUG_DOC_ID, file: dict = None, priority: int = 0, rerun: bool = False) -> tuple[bool, str]:
    task = dict(
        id=task_id,
        doc_id=doc_id,
        from_page=0,
        to_page=MAXIMUM_TASK_PAGE_NUMBER,
        task_type="dataflow" if not rerun else "dataflow_rerun",
        priority=priority,
        begin_at=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    )
    if doc_id not in [CANVAS_DEBUG_DOC_ID, GRAPH_RAPTOR_FAKE_DOC_ID]:
        TaskService.model.delete().where(TaskService.model.doc_id == doc_id).execute()
        DocumentService.begin2parse(doc_id)
    bulk_insert_into_db(model=Task, data_source=[task], replace_on_conflict=True)

    task["kb_id"] = DocumentService.get_knowledgebase_id(doc_id)
    task["tenant_id"] = tenant_id
    task["dataflow_id"] = flow_id
    task["file"] = file

    if not REDIS_CONN.queue_product(settings.get_svr_queue_name(priority, "common"), message=task):
        return False, "Can't access Redis. Please check the Redis' status."

    return True, ""
