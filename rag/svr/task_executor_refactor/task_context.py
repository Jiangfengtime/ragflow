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

"""任务上下文模块。

``TaskContext`` 是原始任务字典的类型化包装，为任务执行器常用字段提供统一属性访问。
本模块同时定义：原始任务结构 ``TaskDict``、并发限制器 ``TaskLimiters``、回调集合
``TaskCallbacks``，以及组合这些组件的主入口 ``TaskContext``。

使用示例：

    from rag.svr.task_executor_refactor.task_context import TaskContext, TaskLimiters, TaskCallbacks

    ctx = TaskContext(
        task=task_dict,
        limiters=TaskLimiters(
            chat=chat_limiter,
            minio=minio_limiter,
            chunk=chunk_limiter,
            embed=embed_limiter,
            kg=kg_limiter,
        ),
        callbacks=TaskCallbacks(
            progress=progress_callback,
            has_canceled=has_canceled_func,
        ),
        write_interceptor=write_interceptor,
        recording_context=recording_context,
    )

    # 直接读取任务属性
    task_id = ctx.id
    tenant_id = ctx.tenant_id
    kb_id = ctx.kb_id
"""

import asyncio
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Required, TypedDict

from rag.svr.task_executor_refactor.recording_context import BaseRecordingContext
from rag.svr.task_executor_refactor.write_operation_interceptor import WriteOperationInterceptor


# ============================================================================
# 类型定义
# ============================================================================


class TaskDict(TypedDict, total=False):
    """TypedDict 定义原始任务字典的结构。

    除了必填的“id”和“tenant_id”之外，所有字段都是可选的。"""

    id: Required[str]
    """Task 标识符（必需）。"""

    tenant_id: Required[str]
    """租户标识符（必填）。"""

    kb_id: str
    """知识库/数据集标识符。"""

    doc_id: str
    """Document 标识符。"""

    doc_ids: List[str]
    """文档标识符列表（适用于 RAPTOR/GraphRAG 等批处理任务）。"""

    name: str
    """Document 名称。"""

    location: str
    """文档在对象存储中的位置或路径。"""

    size: int
    """Document 文件大小（以字节为单位）。"""

    parser_id: str
    """解析器标识符（e.g.、'naive'、'table'、'paper'）。"""

    parser_config: Dict[str, Any]
    """Document 级解析器配置。"""

    kb_parser_config: Dict[str, Any]

    """知识库级别解析器配置。"""

    language: str
    """Document语言（e.g.、'en'、'zh'）。"""

    llm_id: str
    """LLM 型号标识符。"""

    tenant_llm_id: str | None
    """LLM 的租户模型 ID（tenant_model 表中的 ID）。"""

    embd_id: str
    """Embedding 型号标识符。"""

    tenant_embd_id: str | None
    """用于嵌入的租户模型 ID（tenant_model 表中的 id）。"""

    from_page: int
    """起始页/行，0-based 且包含；table parser 中表示起始行。"""

    to_page: int
    """结束页/行，0-based 且不包含；通常与 from_page 组成 [from_page, to_page)。"""

    task_type: str
    """任务分流类型；空字符串通常是普通文档解析，dataflow/raptor/graphrag 等进入专用分支。"""

    dataflow_id: str
    """Dataflow/pipeline 标识符。"""

    pagerank: int
    """PageRank 文档评分值。"""

    file: Any
    """File 用于数据流处理的对象。"""

    memory_id: str
    """Memory 内存任务标识符。"""

    source_id: str
    """内存任务的源标识符。"""

    message_dict: Dict[str, Any]
    """内存任务的消息字典。"""


# ============================================================================
# 数据类
# ============================================================================


@dataclass
class TaskLimiters:
    """封装任务执行的所有速率限制器。

    每个限制器都是一个asyncio.Semaphore，用于控制并发
    用于不同类型的操作。"""

    chat: asyncio.Semaphore = None
    """用于聊天模型速率限制的异步信号量。"""

    minio: asyncio.Semaphore = None
    """用于 MinIO 速率限制的异步信号量。"""

    chunk: asyncio.Semaphore = None
    """用于块构建速率限制的异步信号量。"""

    embed: asyncio.Semaphore = None
    """用于嵌入速率限制的异步信号量。"""

    kg: asyncio.Semaphore = None
    """用于知识图速率限制的异步信号量。"""


def _noop_progress(**kwargs: Any) -> None:
    """无操作进度回调。"""
    pass


def _not_canceled(task_id: str) -> bool:
    """默认取消检查 - 始终返回 False。"""
    return False


@dataclass
class TaskCallbacks:
    """封装任务执行的所有回调函数。"""

    progress: Callable = field(default_factory=lambda: _noop_progress)
    """进度更新回调函数（原始，需要 task_id、from_page、to_page）。"""

    has_canceled: Callable = field(default_factory=lambda: _not_canceled)
    """检查任务是否被取消的函数。"""


# ============================================================================
# 主类
# ============================================================================


class TaskContext:
    """围绕任务字典的类型包装器，提供方便的属性访问器。

    该类使用组合来封装：
    1.原始任务字典（TaskDict）
    2. 执行限制器（TaskLimiters）
    3.回调函数（TaskCallbacks）
    4.可选的写操作拦截器
    5. 中间结果的可选记录上下文

    这些属性提供了一个用于访问任务属性的干净接口
    无需在整个过程中使用带有字符串键的字典访问
    代码库。"""

    # 可选任务字段的默认值
    _DEFAULTS: Dict[str, Any] = {
        "kb_id": "",
        "doc_id": "",
        "doc_ids": [],
        "name": "",
        "location": "",
        "size": 0,
        "parser_id": "",
        "parser_config": {},
        "kb_parser_config": {},
        "language": "Chinese",
        "llm_id": "",
        "tenant_llm_id": "",
        "embd_id": "",
        "tenant_embd_id": "",
        "from_page": 0,
        "to_page": -1,
        "task_type": "",
        "dataflow_id": "",
        "pagerank": 0,
        "memory_id": "",
        "source_id": "",
        "message_dict": {},
    }

    def __init__(
        self,
        task: TaskDict,
        limiters: TaskLimiters,
        callbacks: TaskCallbacks,
        write_interceptor: WriteOperationInterceptor = None,
        recording_context: BaseRecordingContext = None,
    ):
        """初始化TaskContext。

        参数：
            任务：包含所有任务属性的原始任务字典。
            限制器：TaskLimiters 数据类包含所有速率限制器。
            回调：包含所有回调函数的 TaskCallbacks 数据类。
            write_interceptor：写操作的可选拦截器。
            recording_context：中间结果可选 BaseRecordingContext
                捕获。必须通过构造函数注入。

        加薪：
            ValueError：如果任务中缺少必填字段（“id”、“tenant_id”）。"""
        # 验证必填字段
        if "id" not in task:
            raise ValueError("Task must contain 'id'")
        if "tenant_id" not in task:
            raise ValueError("Task must contain 'tenant_id'")

        self._task = task
        self.limiters = limiters
        self.callbacks = callbacks
        self._write_interceptor = write_interceptor
        self._recording_context = recording_context

        # 准备进度回调并将其设置在上下文中
        progress_cb = partial(
            callbacks.progress,
            self.id,
            self.from_page,
            self.to_page,
        )
        self._progress_cb = progress_cb

    # =========================================================================
    # 核心任务身份属性
    # =========================================================================

    @property
    def id(self) -> str:
        """Task 标识符。"""
        return self._task["id"]

    @property
    def tenant_id(self) -> str:
        """租户标识符。"""
        return self._task["tenant_id"]

    @property
    def kb_id(self) -> str:
        """知识库/数据集标识符。"""
        return self._task.get("kb_id", self._DEFAULTS["kb_id"])

    @property
    def doc_id(self) -> str:
        """Document 标识符。"""
        return self._task.get("doc_id", self._DEFAULTS["doc_id"])

    @property
    def doc_ids(self) -> List[str]:
        """文档标识符列表（适用于 RAPTOR/GraphRAG 等批处理任务）。"""
        return self._task.get("doc_ids", list(self._DEFAULTS["doc_ids"]))

    # =========================================================================
    # Document 元数据属性
    # =========================================================================

    @property
    def name(self) -> str:
        """Document 名称。"""
        return self._task.get("name", self._DEFAULTS["name"])

    @property
    def location(self) -> str:
        """文档在对象存储中的位置或路径。"""
        return self._task.get("location", self._DEFAULTS["location"])

    @property
    def size(self) -> int:
        """Document 文件大小（以字节为单位）。"""
        return self._task.get("size", self._DEFAULTS["size"])

    # =========================================================================
    # 解析器配置属性
    # =========================================================================

    @property
    def parser_id(self) -> str:
        """解析器标识符（e.g.、'naive'、'table'、'paper'）。"""
        return self._task.get("parser_id", self._DEFAULTS["parser_id"])

    @property
    def parser_config(self) -> Dict[str, Any]:
        """Document 级解析器配置。"""
        return self._task.get("parser_config", {})

    @property
    def kb_parser_config(self) -> Dict[str, Any]:
        """知识库级别解析器配置。"""
        return self._task.get("kb_parser_config", {})

    # =========================================================================
    # 语言和模型属性
    # =========================================================================

    @property
    def language(self) -> str:
        """Document 语言（e.g., 'en', 'zh'）。"""
        return self._task.get("language", self._DEFAULTS["language"])

    @property
    def llm_id(self) -> str:
        """LLM 型号标识符。"""
        return self._task.get("llm_id", self._DEFAULTS["llm_id"])

    @property
    def tenant_llm_id(self) -> str | None:
        """LLM 的租户模型 ID（tenant_model 表中的 ID）。"""
        return self._task.get("tenant_llm_id", self._DEFAULTS["tenant_llm_id"]) or None

    @property
    def embd_id(self) -> str:
        """Embedding 型号标识符。"""
        return self._task.get("embd_id", self._DEFAULTS["embd_id"])

    @property
    def tenant_embd_id(self) -> str | None:
        """用于嵌入的租户模型 ID（tenant_model 表中的 id）。"""
        return self._task.get("tenant_embd_id", self._DEFAULTS["tenant_embd_id"]) or None

    # =========================================================================
    # 页面范围属性
    # =========================================================================

    @property
    def from_page(self) -> int:
        """处理的起始页码（从 0 开始）。"""
        return self._task.get("from_page", self._DEFAULTS["from_page"])

    @property
    def to_page(self) -> int:
        """处理的结束页码（-1 表示所有页）。"""
        return self._task.get("to_page", self._DEFAULTS["to_page"])

    # =========================================================================
    # Task 类型和路由属性
    # =========================================================================

    @property
    def task_type(self) -> str:
        """Task 类型（e.g.、“数据流”、“猛禽”、“graphrag”、“内存”）。"""
        return self._task.get("task_type", self._DEFAULTS["task_type"])

    @property
    def dataflow_id(self) -> str:
        """Dataflow/pipeline 标识符。"""
        return self._task.get("dataflow_id", self._DEFAULTS["dataflow_id"])

    # =========================================================================
    # 其他属性
    # =========================================================================

    @property
    def pagerank(self) -> int:
        """PageRank 文档评分值。"""
        return self._task.get("pagerank", self._DEFAULTS["pagerank"])

    @property
    def file(self) -> Optional[Any]:
        """File 用于数据流处理的对象。"""
        return self._task.get("file")

    # =========================================================================
    # Memory 任务特定属性
    # =========================================================================

    @property
    def memory_id(self) -> str:
        """Memory 内存任务标识符。"""
        return self._task.get("memory_id", self._DEFAULTS["memory_id"])

    @property
    def source_id(self) -> str:
        """内存任务的源标识符。"""
        return self._task.get("source_id", self._DEFAULTS["source_id"])

    @property
    def message_dict(self) -> Dict[str, Any]:
        """内存任务的消息字典。"""
        return self._task.get("message_dict", {})

    # =========================================================================
    # 原始任务字典访问
    # =========================================================================

    @property
    def raw_task(self) -> Dict[str, Any]:
        """返回原始任务字典。"""
        return self._task

    def get(self, key: str, default: Any = None) -> Any:
        """从任务字典中获取一个默认值。

        参数：
            key：查找键。
            default：如果未找到密钥，则使用默认值。

        返回：
            与键关联的值，如果未找到，则为默认值。"""
        return self._task.get(key, default)

    # =========================================================================
    # 限制器属性（TaskLimiters 数据类的代理）
    # =========================================================================

    @property
    def chat_limiter(self) -> asyncio.Semaphore:
        """用于聊天模型速率限制的异步信号量。"""
        return self.limiters.chat or asyncio.Semaphore(1)

    @property
    def minio_limiter(self) -> asyncio.Semaphore:
        """用于 MinIO 速率限制的异步信号量。"""
        return self.limiters.minio or asyncio.Semaphore(1)

    @property
    def chunk_limiter(self) -> asyncio.Semaphore:
        """用于块构建速率限制的异步信号量。"""
        return self.limiters.chunk or asyncio.Semaphore(1)

    @property
    def embed_limiter(self) -> asyncio.Semaphore:
        """用于嵌入速率限制的异步信号量。"""
        return self.limiters.embed or asyncio.Semaphore(1)

    @property
    def kg_limiter(self) -> asyncio.Semaphore:
        """用于知识图速率限制的异步信号量。"""
        return self.limiters.kg or asyncio.Semaphore(1)

    # =========================================================================
    # 上下文和拦截器属性
    # =========================================================================

    @property
    def recording_context(self) -> BaseRecordingContext:
        """BaseRecordingContext 用于此任务。

        必须通过构造函数注入。如果访问则引发 RuntimeError
        在初始化之前或如果未提供上下文。"""
        if self._recording_context is None:
            raise RuntimeError("recording_context accessed but not injected into TaskContext")
        return self._recording_context

    @property
    def write_interceptor(self) -> WriteOperationInterceptor:
        """比较模式的写操作拦截器。"""
        return self._write_interceptor

    # =========================================================================
    # 回调属性（TaskCallbacks 数据类的代理）
    # =========================================================================

    @property
    def has_canceled_func(self) -> Callable:
        """检查任务是否被取消的函数。"""
        return self.callbacks.has_canceled

    # =========================================================================
    # 预绑定进度回调
    # =========================================================================

    @property
    def progress_cb(self) -> Callable:
        """预绑定进度回调（task_id、from_page、to_page 已绑定）。

        在服务中使用此属性来更新进度。
        如果未设置 progress_cb，则回退到 progress_callback。"""
        return self._progress_cb
