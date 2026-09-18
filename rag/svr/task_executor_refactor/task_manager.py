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

"""Task管理模块。

提供 [`TaskManager`](rag/svr/task_executor_refactor/task_manager.py:50) 作为入口点
用于执行文档处理任务，支持生产和试运行（比较）模式。"""

import logging
from typing import Any, Optional

from rag.svr.task_executor_refactor.comparator import ContextComparator
from rag.svr.task_executor_refactor.task_context import TaskCallbacks, TaskDict, TaskLimiters
from rag.svr.task_executor_refactor.dataflow_service import BillingHook
from rag.svr.task_executor_refactor.recording_context import (
    BaseRecordingContext,
    RecordingContext,
    _NULL_RECORDING_CONTEXT,
    set_recording_context,
    recording_context_manager,
)
from rag.svr.task_executor_refactor.task_context import TaskContext
from rag.svr.task_executor_refactor.task_handler import TaskHandler
from rag.svr.task_executor_refactor.write_operation_interceptor import (
    WriteOperationInterceptor,
)


class TaskManager:
    """执行文档处理任务的入口点。

    此类提供了以下方法：
    - 生产任务执行（run_refactored_task）
    - 试运行任务执行并进行比较（dry_run_task）

    用途：
        经理 = TaskManager()
        等待 manager.run_refactored_task(任务, chat_limiter, ...)
        # 或
        等待 manager.dry_run_task(任务, recording_ctx1, ...)"""

    @classmethod
    async def run_refactored_task(
        cls,
        task: dict,
        chat_limiter: Any,  # 限制同时调用 Chat/LLM 的数量。
        minio_limiter: Any,  # 限制同时读取 MinIO 的数量。
        chunk_limiter: Any,  # 限制同时解析、切分文档的数量。
        embed_limiter: Any,  # 限制同时调用 Embedding 模型的数量
        kg_limiter: Any,  # 限制同时执行知识图谱任务的数量
        set_progress: Any,  # 更新 MySQL 中任务进度和进度信息
        has_canceled: Any,  # 检查任务是否已被用户取消
        billing_hook: Optional[BillingHook] = None,  # 可选的计费回调，默认不传
    ) -> None:
        """在生产模式下运行文档处理任务。

        参数：
            任务：Task配置字典。
            chat_limiter：聊天操作的速率限制器。
            minio_limiter：MinIO 操作的速率限制器。
            chunk_limiter：分块操作的速率限制器。
            embed_limiter：嵌入操作的速率限制器。
            kg_limiter：知识图操作的速率限制器。
            set_progress：进度回调函数。
            has_canceled：检查任务是否被取消的函数。
            billing_hook：管道 success/error 回调的可选计费挂钩。"""
        # 该方法是新版执行器的装配入口，本身不解析文件：它把 Redis/MySQL 得到的 task、
        # 五类并发限流器、进度更新和取消检查统一包装成 TaskContext，再交给 TaskHandler。
        with recording_context_manager(_NULL_RECORDING_CONTEXT):
            # 在生产中使用 NullRecordingContext 以避免内存分配
            set_recording_context(_NULL_RECORDING_CONTEXT)  # 创建无记录模式, 正常生产模式不保存新旧流程对比数据，减少内存消耗

            # 使用所有执行资源创建TaskContext
            task_context = TaskContext(  # 把零散参数封装成 TaskContext
                task=task,
                limiters=TaskLimiters(
                    chat=chat_limiter,
                    minio=minio_limiter,
                    chunk=chunk_limiter,
                    embed=embed_limiter,
                    kg=kg_limiter,
                ),
                callbacks=TaskCallbacks(
                    progress=set_progress,
                    has_canceled=has_canceled,
                ),
                recording_context=_NULL_RECORDING_CONTEXT,
            )

            # TaskHandler 通过 ctx 读取任务字段并使用绑定过 task_id/page range 的 progress_cb，
            # 因而下游服务无需反复传递大量独立参数。
            # 从这里开始，后续组件通过同一个 ctx 取得 doc_id、页范围、模型配置、限流器、
            # 取消函数和进度回调，避免各阶段反复传递大量独立参数。
            handler = TaskHandler(ctx=task_context, billing_hook=billing_hook)
            await handler.handle_task()  # 真正的业务处理入口。

    @classmethod
    async def dry_run_task(
        cls,
        task: TaskDict,
        recording_ctx1: BaseRecordingContext,
        chat_limiter: Any,
        minio_limiter: Any,
        chunk_limiter: Any,
        embed_limiter: Any,
        kg_limiter: Any,
        set_progress: Any,
        has_canceled: Any,
    ) -> None:
        """以空运行模式运行文档处理任务进行比较。

        这使用记录操作的写操作拦截器来执行任务
        所有写入操作，然后将结果与生产运行进行比较。

        参数：
            任务：Task配置字典。
            recording_ctx1：来自生产执行的RecordingContext。
            chat_limiter：聊天操作的速率限制器。
            minio_limiter：MinIO 操作的速率限制器。
            chunk_limiter：分块操作的速率限制器。
            embed_limiter：嵌入操作的速率限制器。
            kg_limiter：知识图操作的速率限制器。
            set_progress：进度回调函数。
            has_canceled：检查任务是否被取消的函数。"""
        interceptor = WriteOperationInterceptor(recording_ctx1.get_all_func_return_values())
        recording_ctx2 = RecordingContext()

        with recording_context_manager(recording_ctx2):
            set_recording_context(recording_ctx2)

            # 使用所有执行资源创建TaskContext
            task_context = TaskContext(
                task=task,
                limiters=TaskLimiters(
                    chat=chat_limiter,
                    minio=minio_limiter,
                    chunk=chunk_limiter,
                    embed=embed_limiter,
                    kg=kg_limiter,
                ),
                callbacks=TaskCallbacks(
                    progress=set_progress,
                    has_canceled=has_canceled,
                ),
                write_interceptor=interceptor,
                recording_context=recording_ctx2,
            )

            # 使用 TaskHandler 执行
            handler = TaskHandler(ctx=task_context)
            await handler.handle_task()

            # 比较结果
            comp: ContextComparator = ContextComparator()
            comp_result = comp.compare(task_context.id, recording_ctx1, recording_ctx2)
            logging.info(f"-------{task_context.name}, compare result:{comp_result.to_markdown()}")
            if interceptor.remaining_values_count() > 0 or comp_result.mismatched_keys > 0:
                logging.info(
                    f"------task:{task_context.id} {task_context.name} differs, "
                    f"interceptor.remaining_values_count():{interceptor.remaining_values_count()}, "
                    f"mismatched_keys:{comp_result.mismatched_keys}"
                )
                if interceptor.remaining_values_count() > 0:
                    logging.info(f"------task:{task_context.id}, remaining values:{interceptor.remaining_values()}")
                if comp_result.mismatched_keys > 0:
                    logging.info(f"-------compare result:{comp_result.details}")
            else:
                logging.info(f"------task:{task_context.id} {task_context.name} same result for prod and dry run ")
