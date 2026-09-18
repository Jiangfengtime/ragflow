#
#  Copyright 2025 The InfiniFlow Authors. All Rights Reserved.
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
from datetime import datetime
from typing import List

from common import settings
from common.time_utils import current_timestamp, timestamp_to_date, format_iso_8601_to_ymd_hms
from common.constants import MemoryType, LLMType
from common.doc_store.doc_store_base import FusionExpr
from common.misc_utils import get_uuid
from api.db.db_utils import bulk_insert_into_db
from api.db.db_models import Task
from api.db.services.task_service import TaskService
from api.db.services.memory_service import MemoryService
from api.db.services.llm_service import LLMBundle
from api.db.joint_services.tenant_model_service import resolve_model_config, get_model_config_by_id
from api.utils.memory_utils import get_memory_type_human
from memory.services.messages import MessageService
from memory.services.query import MsgTextQuery, get_vector
from memory.utils.prompt_util import PromptAssembler
from memory.utils.msg_util import get_json_result_from_llm_response
from rag.utils.redis_conn import REDIS_CONN


async def save_to_memory(memory_id: str, message_dict: dict):
    """：参数memory_id：
    ：参数 message_dict：{
        "user_id"：str，
        "agent_id"：str，
        "session_id"：str，
        "user_input"：str，
        "agent_response"：str
    }"""
    memory = MemoryService.get_by_memory_id(memory_id)
    if not memory:
        return False, f"Memory '{memory_id}' not found."

    tenant_id = memory.tenant_id
    extracted_content = (
        await extract_by_llm(
            tenant_id,
            memory.tenant_llm_id,
            {"temperature": memory.temperature},
            get_memory_type_human(memory.memory_type),
            message_dict.get("user_input", ""),
            message_dict.get("agent_response", ""),
            system_prompt=memory.system_prompt,
            user_prompt=memory.user_prompt,
            llm_id=memory.llm_id,
        )
        if memory.memory_type != MemoryType.RAW.value
        else []
    )  # 如果只有RAW，则无需解压
    raw_message_id = REDIS_CONN.generate_auto_increment_id(namespace="memory")
    message_list = [
        {
            "message_id": raw_message_id,
            "message_type": MemoryType.RAW.name.lower(),
            "source_id": 0,
            "memory_id": memory_id,
            "user_id": message_dict.get("user_id", ""),
            "agent_id": message_dict["agent_id"],
            "session_id": message_dict["session_id"],
            "content": f"User Input: {message_dict.get('user_input')}\nAgent Response: {message_dict.get('agent_response')}",
            "valid_at": timestamp_to_date(current_timestamp()),
            "invalid_at": None,
            "forget_at": None,
            "status": True,
        },
        *[
            {
                "message_id": REDIS_CONN.generate_auto_increment_id(namespace="memory"),
                "message_type": content["message_type"],
                "source_id": raw_message_id,
                "memory_id": memory_id,
                "user_id": message_dict.get("user_id", ""),
                "agent_id": message_dict["agent_id"],
                "session_id": message_dict["session_id"],
                "content": content["content"],
                "valid_at": content["valid_at"],
                "invalid_at": content["invalid_at"] if content["invalid_at"] else None,
                "forget_at": None,
                "status": True,
            }
            for content in extracted_content
        ],
    ]
    return await embed_and_save(memory, message_list)


async def save_extracted_to_memory_only(memory_id: str, message_dict, source_message_id: int, task_id: str = None):
    """任务执行器阶段：从一条已落库的原始消息提取长期记忆并写入消息索引。

    ``queue_save_to_memory_task`` 已经同步保存 RAW 消息，本方法只生成 semantic、
    episodic、procedural 等派生消息。``source_message_id`` 把派生消息关联回原始对话，
    ``task_id`` 用于把 LLM、Embedding 和存储进度写回 MySQL Task。
    """
    memory = MemoryService.get_by_memory_id(memory_id)
    if not memory:
        msg = f"Memory '{memory_id}' not found."
        if task_id:
            TaskService.update_progress(task_id, {"progress": -1, "progress_msg": timestamp_to_date(current_timestamp()) + " " + msg})
        return False, msg

    if memory.memory_type == MemoryType.RAW.value:
        msg = f"Memory '{memory_id}' don't need to extract."
        if task_id:
            TaskService.update_progress(task_id, {"progress": 1.0, "progress_msg": timestamp_to_date(current_timestamp()) + " " + msg})
        return True, msg

    tenant_id = memory.tenant_id
    extracted_content = await extract_by_llm(
        tenant_id,
        memory.tenant_llm_id,
        {"temperature": memory.temperature},
        get_memory_type_human(memory.memory_type),
        message_dict.get("user_input", ""),
        message_dict.get("agent_response", ""),
        system_prompt=memory.system_prompt,
        user_prompt=memory.user_prompt,
        task_id=task_id,
        llm_id=memory.llm_id,
    )
    message_list = [
        {
            "message_id": REDIS_CONN.generate_auto_increment_id(namespace="memory"),
            "message_type": content["message_type"],
            "source_id": source_message_id,
            "memory_id": memory_id,
            "user_id": message_dict.get("user_id", ""),
            "agent_id": message_dict["agent_id"],
            "session_id": message_dict["session_id"],
            "content": content["content"],
            "valid_at": content["valid_at"],
            "invalid_at": content["invalid_at"] if content["invalid_at"] else None,
            "forget_at": None,
            "status": True,
        }
        for content in extracted_content
    ]
    if not message_list:
        msg = "No memory extracted from raw message."
        if task_id:
            TaskService.update_progress(task_id, {"progress": 1.0, "progress_msg": timestamp_to_date(current_timestamp()) + " " + msg})
        return True, msg

    if task_id:
        TaskService.update_progress(task_id, {"progress": 0.5, "progress_msg": timestamp_to_date(current_timestamp()) + " " + f"Extracted {len(message_list)} messages from raw dialogue."})
    return await embed_and_save(memory, message_list, task_id)


async def extract_by_llm(
    tenant_id: str,
    tenant_llm_id: str | None,
    extract_conf: dict,
    memory_type: List[str],
    user_input: str,
    agent_response: str,
    system_prompt: str = "",
    user_prompt: str = "",
    task_id: str = None,
    llm_id: str = "",
) -> List[dict]:
    # MemoryType 决定系统 Prompt 要求模型抽取哪些类别；这里一次调用 LLM 后把
    # JSON 结果展平为多条消息。日志不能记录 conversation_content，避免泄露对话正文。
    if not system_prompt:
        system_prompt = PromptAssembler.assemble_system_prompt({"memory_type": memory_type})
    conversation_content = f"User Input: {user_input}\nAgent Response: {agent_response}"
    conversation_time = timestamp_to_date(current_timestamp())
    user_prompts = []
    if user_prompt:
        user_prompts.append({"role": "user", "content": user_prompt})
        user_prompts.append({"role": "user", "content": f"Conversation: {conversation_content}\nConversation Time: {conversation_time}\nCurrent Time: {conversation_time}"})
    else:
        user_prompts.append({"role": "user", "content": PromptAssembler.assemble_user_prompt(conversation_content, conversation_time, conversation_time)})
    if tenant_llm_id:
        try:
            llm_config = get_model_config_by_id(tenant_id, LLMType.CHAT, tenant_llm_id)
        except LookupError:
            llm_config = resolve_model_config(tenant_id, LLMType.CHAT, llm_id)
    else:
        llm_config = resolve_model_config(tenant_id, LLMType.CHAT, llm_id)
    with LLMBundle(tenant_id, llm_config) as llm:
        if task_id:
            TaskService.update_progress(task_id, {"progress": 0.15, "progress_msg": timestamp_to_date(current_timestamp()) + " " + "Prepared prompts and LLM."})
        res = await llm.async_chat(system_prompt, user_prompts, extract_conf)
        res_json = get_json_result_from_llm_response(res)
        if task_id:
            TaskService.update_progress(task_id, {"progress": 0.35, "progress_msg": timestamp_to_date(current_timestamp()) + " " + "Get extracted result from LLM."})
        return [
            {
                "content": extracted_content["content"],
                "valid_at": format_iso_8601_to_ymd_hms(extracted_content.get("valid_at", ""), fallback=conversation_time),
                "invalid_at": format_iso_8601_to_ymd_hms(extracted_content["invalid_at"], fallback="") if extracted_content.get("invalid_at") else "",
                "message_type": message_type,
            }
            for message_type, extracted_content_list in res_json.items()
            for extracted_content in extracted_content_list
        ]


async def embed_and_save(memory, message_list: list[dict], task_id: str = None):
    """为记忆消息生成向量，执行容量淘汰，然后写入独立消息索引。

    Memory 不复用知识库的 ``ragflow_<tenant>`` Chunk 索引。``MessageService`` 按
    ``tenant_id + memory_id`` 定位消息索引；同一条消息同时保存全文检索字段和
    ``content_embed``，因此后续 ``query_message`` 可以做 BM25/向量混合召回。
    """
    logging.info(
        "记忆消息向量化已开始 任务ID=%s 租户ID=%s 记忆库ID=%s 消息数=%d",
        task_id or "-",
        memory.tenant_id,
        memory.id,
        len(message_list),
    )
    if memory.tenant_embd_id:
        try:
            embd_model_config = get_model_config_by_id(memory.tenant_id, LLMType.EMBEDDING, memory.tenant_embd_id)
        except LookupError:
            embd_model_config = resolve_model_config(memory.tenant_id, LLMType.EMBEDDING, memory.embd_id)
    else:
        embd_model_config = resolve_model_config(memory.tenant_id, LLMType.EMBEDDING, memory.embd_id)
    with LLMBundle(memory.tenant_id, embd_model_config) as embedding_model:
        if task_id:
            TaskService.update_progress(task_id, {"progress": 0.65, "progress_msg": timestamp_to_date(current_timestamp()) + " " + "Prepared embedding model."})
        vector_list, _ = embedding_model.encode([msg["content"] for msg in message_list])
        for idx, msg in enumerate(message_list):
            msg["content_embed"] = vector_list[idx]
        if task_id:
            TaskService.update_progress(task_id, {"progress": 0.85, "progress_msg": timestamp_to_date(current_timestamp()) + " " + "Embedded extracted content."})
        vector_dimension = len(vector_list[0])
        if not MessageService.has_index(memory.tenant_id, memory.id):
            created = MessageService.create_index(memory.tenant_id, memory.id, vector_size=vector_dimension)
            if not created:
                error_msg = "Failed to create message index."
                if task_id:
                    TaskService.update_progress(task_id, {"progress": -1, "progress_msg": timestamp_to_date(current_timestamp()) + " " + error_msg})
                return False, error_msg
            logging.info(
                "记忆消息索引已创建 租户ID=%s 记忆库ID=%s 向量维度=%d",
                memory.tenant_id,
                memory.id,
                vector_dimension,
            )

        new_msg_size = sum([MessageService.calculate_message_size(m) for m in message_list])
        current_memory_size = get_memory_size_cache(memory.id, memory.tenant_id)
        if new_msg_size + current_memory_size > memory.memory_size:
            size_to_delete = current_memory_size + new_msg_size - memory.memory_size
            if memory.forgetting_policy == "FIFO":
                message_ids_to_delete, delete_size = MessageService.pick_messages_to_delete_by_fifo(memory.id, memory.tenant_id, size_to_delete)
                # 容量限制按估算字节数执行；FIFO 先删最老消息，再插入本批消息。
                MessageService.delete_message({"message_id": message_ids_to_delete}, memory.tenant_id, memory.id)
                decrease_memory_size_cache(memory.id, delete_size)
                logging.info(
                    "记忆容量淘汰已执行 租户ID=%s 记忆库ID=%s 淘汰消息数=%d 淘汰大小=%d 需要释放大小=%d",
                    memory.tenant_id,
                    memory.id,
                    len(message_ids_to_delete),
                    delete_size,
                    size_to_delete,
                )
            else:
                error_msg = "Failed to insert message into memory. Memory size reached limit and cannot decide which to delete."
                if task_id:
                    TaskService.update_progress(task_id, {"progress": -1, "progress_msg": timestamp_to_date(current_timestamp()) + " " + error_msg})
                return False, error_msg
        fail_cases = MessageService.insert_message(message_list, memory.tenant_id, memory.id)
        if fail_cases:
            error_msg = "Failed to insert message into memory. Details: " + "; ".join(fail_cases)
            if task_id:
                TaskService.update_progress(task_id, {"progress": -1, "progress_msg": timestamp_to_date(current_timestamp()) + " " + error_msg})
            return False, error_msg

        if task_id:
            TaskService.update_progress(task_id, {"progress": 0.95, "progress_msg": timestamp_to_date(current_timestamp()) + " " + "Saved messages to storage."})
        increase_memory_size_cache(memory.id, new_msg_size)
        logging.info(
            "记忆消息向量化完成 任务ID=%s 租户ID=%s 记忆库ID=%s 消息数=%d 向量维度=%d 写入大小=%d",
            task_id or "-",
            memory.tenant_id,
            memory.id,
            len(message_list),
            vector_dimension,
            new_msg_size,
        )
        return True, "Message saved successfully."


def query_message(filter_dict: dict, params: dict):
    """：参数 filter_dict：{
        "memory_id"：列表[str]，
        "agent_id"：可选
        "session_id"：可选
        "user_id"：可选
    }
    ：参数参数：{
        "query"：问题str，
        "similarity_threshold"：浮动，
        "keywords_similarity_weight"：浮动，
        "top_n"：整数
    }"""
    # 召回与知识库检索思路相同，但数据源是 Memory 独立消息索引：
    # MsgTextQuery 生成全文条件，get_vector 生成稠密向量条件，FusionExpr 用
    # keywords_similarity_weight 把两路得分做 weighted_sum。
    memory_ids = filter_dict["memory_id"]
    memory_list = MemoryService.get_by_ids(memory_ids)
    if not memory_list:
        return []

    condition_dict = {k: v for k, v in filter_dict.items() if v}
    uids = [memory.tenant_id for memory in memory_list]

    question = params["query"]
    question = question.strip()
    memory = memory_list[0]
    embd_model_config = resolve_model_config(memory.tenant_id, LLMType.EMBEDDING, memory.embd_id)
    embd_model = LLMBundle(memory.tenant_id, embd_model_config)
    match_dense = get_vector(question, embd_model, similarity=params["similarity_threshold"])
    match_text, _ = MsgTextQuery().question(question, min_match=params["similarity_threshold"])
    keywords_similarity_weight = params.get("keywords_similarity_weight", 0.7)
    fusion_expr = FusionExpr("weighted_sum", params["top_n"], {"weights": ",".join([str(1 - keywords_similarity_weight), str(keywords_similarity_weight)])})

    logging.info(
        "记忆检索已开始 租户ID列表=%s 记忆库ID列表=%s 查询长度=%d 返回上限=%d 相似度阈值=%s 关键词权重=%s 过滤字段=%s",
        uids,
        memory_ids,
        len(question),
        params["top_n"],
        params["similarity_threshold"],
        keywords_similarity_weight,
        sorted(condition_dict.keys()),
    )
    return MessageService.search_message(memory_ids, condition_dict, uids, [match_text, match_dense, fusion_expr], params["top_n"])


def init_message_id_sequence():
    message_id_redis_key = "id_generator:memory"
    if REDIS_CONN.exist(message_id_redis_key):
        current_max_id = REDIS_CONN.get(message_id_redis_key)
        logging.info(f"No need to init message_id sequence, current max id is {current_max_id}.")
    else:
        max_id = 1
        exist_memory_list = MemoryService.get_all_memory()
        if not exist_memory_list:
            REDIS_CONN.set(message_id_redis_key, max_id)
        else:
            max_id = MessageService.get_max_message_id(uid_list=[m.tenant_id for m in exist_memory_list], memory_ids=[m.id for m in exist_memory_list])
            REDIS_CONN.set(message_id_redis_key, max_id)
        logging.info(f"Init message_id sequence done, current max id is {max_id}.")


def get_memory_size_cache(memory_id: str, uid: str):
    redis_key = f"memory_{memory_id}"
    if REDIS_CONN.exist(redis_key):
        return int(REDIS_CONN.get(redis_key))
    else:
        memory_size_map = MessageService.calculate_memory_size([memory_id], [uid])
        memory_size = memory_size_map.get(memory_id, 0)
        set_memory_size_cache(memory_id, memory_size)
        return memory_size


def set_memory_size_cache(memory_id: str, size: int):
    redis_key = f"memory_{memory_id}"
    return REDIS_CONN.set(redis_key, size)


def increase_memory_size_cache(memory_id: str, size: int):
    redis_key = f"memory_{memory_id}"
    return REDIS_CONN.incrby(redis_key, size)


def decrease_memory_size_cache(memory_id: str, size: int):
    redis_key = f"memory_{memory_id}"
    return REDIS_CONN.decrby(redis_key, size)


def init_memory_size_cache():
    memory_list = MemoryService.get_all_memory()
    if not memory_list:
        logging.info("未找到内存，无需初始化内存大小。")
    else:
        for m in memory_list:
            get_memory_size_cache(m.id, m.tenant_id)
        logging.info("Memory 大小缓存初始化完成。")


def fix_missing_tokenized_memory():
    if settings.DOC_ENGINE != "elasticsearch":
        logging.info("不使用elasticsearch作为文档引擎，无需修复丢失的标记化内存。")
        return
    memory_list = MemoryService.get_all_memory()
    if not memory_list:
        logging.info("未找到内存，无需修复丢失的标记化内存。")
    else:
        for m in memory_list:
            message_list = MessageService.get_missing_field_messages(m.id, m.tenant_id, "tokenized_content_ltks")
            for msg in message_list:
                # 重新写入正文，以触发分词字段刷新。
                MessageService.update_message({"message_id": msg["message_id"], "memory_id": m.id}, {"content": msg["content"]}, m.tenant_id, m.id)
            if message_list:
                logging.info(f"Fixed {len(message_list)} messages missing tokenized field in memory: {m.name}.")
        logging.info("修复丢失的标记化内存已完成。")


def judge_system_prompt_is_default(system_prompt: str, memory_type: int | list[str]):
    memory_type_list = memory_type if isinstance(memory_type, list) else get_memory_type_human(memory_type)
    return PromptAssembler.is_default_system_prompt(system_prompt, {"memory_type": memory_type_list})


async def queue_save_to_memory_task(memory_ids: list[str], message_dict: dict):
    """同步保存原始对话，再为每个记忆库创建异步抽取任务。

    顺序是：RAW 消息 Embedding/写消息索引 -> Task 写 MySQL -> task_type=memory
    消息写 Redis Stream。API 返回成功只表示这些步骤完成，派生记忆仍由
    Task Executor 的 ``handle_save_to_memory_task`` 异步生成。

    :param memory_ids:
    :param message_dict: {
        "user_id": str,
        "agent_id": str,
        "session_id": str,
        "user_input": str,
        "agent_response": str
    }
    """

    def new_task(_memory_id: str, _source_id: int):
        return {"id": get_uuid(), "doc_id": _memory_id, "task_type": "memory", "progress": 0.0, "begin_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "digest": str(_source_id)}

    not_found_memory = []
    failed_memory = []
    for memory_id in memory_ids:
        memory = MemoryService.get_by_memory_id(memory_id)
        if not memory:
            not_found_memory.append(memory_id)
            continue

        raw_message_id = REDIS_CONN.generate_auto_increment_id(namespace="memory")
        raw_message = {
            "message_id": raw_message_id,
            "message_type": MemoryType.RAW.name.lower(),
            "source_id": 0,
            "memory_id": memory_id,
            "user_id": message_dict.get("user_id", ""),
            "agent_id": message_dict["agent_id"],
            "session_id": message_dict["session_id"],
            "content": f"User Input: {message_dict.get('user_input')}\nAgent Response: {message_dict.get('agent_response')}",
            "valid_at": timestamp_to_date(current_timestamp()),
            "invalid_at": None,
            "forget_at": None,
            "status": True,
        }
        res, msg = await embed_and_save(memory, [raw_message])
        if not res:
            failed_memory.append({"memory_id": memory_id, "fail_msg": msg})
            continue

        task = new_task(memory_id, raw_message_id)
        # Task.doc_id 在 memory 任务中复用为 memory_id，digest 保存 RAW message_id；
        # 任务执行器依靠 task_type="memory" 分流，不会进入普通文档切片流程。
        bulk_insert_into_db(Task, [task], replace_on_conflict=True)
        task_message = {
            "id": task["id"],
            "task_id": task["id"],
            "task_type": task["task_type"],
            "memory_id": memory_id,
            "tenant_id": memory.tenant_id,
            "source_id": raw_message_id,
            "message_dict": message_dict,
        }
        if not REDIS_CONN.queue_product(settings.get_svr_queue_name(priority=0), message=task_message):
            failed_memory.append({"memory_id": memory_id, "fail_msg": "Can't access Redis."})
            continue
        logging.info(
            "记忆抽取任务已进入队列 任务ID=%s 租户ID=%s 记忆库ID=%s 原始消息ID=%s 队列=%s",
            task["id"],
            memory.tenant_id,
            memory_id,
            raw_message_id,
            settings.get_svr_queue_name(priority=0),
        )

    error_msg = ""
    if not_found_memory:
        error_msg = f"Memory {not_found_memory} not found."
    if failed_memory:
        error_msg += "".join([f"Memory {fm['memory_id']} failed. Detail: {fm['fail_msg']}" for fm in failed_memory])

    if error_msg:
        return False, error_msg

    return True, "All add to task."


async def handle_save_to_memory_task(task_param: dict):
    """任务执行器消费 ``task_type=memory`` 消息的最终入口。

    本方法更新 MySQL Task 状态并调用 LLM 抽取、Embedding 与消息索引写入；
    Redis 消息的 ACK 仍由外围 TaskManager/Executor 在本方法返回后处理。

    :param task_param: {
        "id": task_id
        "memory_id": id
        "source_id": id
        "message_dict": {
            "user_id": str,
            "agent_id": str,
            "session_id": str,
            "user_input": str,
            "agent_response": str
        }
    }
    """
    _, task = TaskService.get_by_id(task_param["id"])
    if not task:
        return False, f"Task {task_param['id']} is not found."
    if task.progress == -1:
        return False, f"Task {task_param['id']} is already failed."
    now_time = current_timestamp()
    TaskService.update_by_id(task_param["id"], {"begin_at": timestamp_to_date(now_time)})

    memory_id = task_param["memory_id"]
    source_id = task_param["source_id"]
    message_dict = task_param["message_dict"]
    logging.info(
        "记忆抽取任务已开始 任务ID=%s 租户ID=%s 记忆库ID=%s 原始消息ID=%s",
        task.id,
        task_param.get("tenant_id", "-"),
        memory_id,
        source_id,
    )
    success, msg = await save_extracted_to_memory_only(memory_id, message_dict, source_id, task.id)
    if success:
        TaskService.update_progress(task.id, {"progress": 1.0, "progress_msg": timestamp_to_date(current_timestamp()) + " " + msg})
        logging.info("记忆抽取任务完成 任务ID=%s 记忆库ID=%s 原始消息ID=%s", task.id, memory_id, source_id)
        return True, msg

    logging.error("记忆抽取任务失败 任务ID=%s 记忆库ID=%s 原始消息ID=%s 错误=%s", task.id, memory_id, source_id, msg)
    TaskService.update_progress(task.id, {"progress": -1, "progress_msg": timestamp_to_date(current_timestamp()) + " " + msg})
    return False, msg
