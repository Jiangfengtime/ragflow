---
sidebar_position: 1
title: 第一册：运行时、配置与数据模型
sidebar_label: 运行时、配置与数据模型
---

# 第一册：运行时、配置与数据模型

本册解决三个基础问题：RAGFlow 启动后究竟有哪些进程，配置如何变成运行时对象，以及一份业务数据分别保存在哪里。

## 1. 一次启动建立了什么

Python 源码开发模式至少包含：

| 角色 | 入口 | 是否处理 HTTP | 主要依赖 |
|---|---|---|---|
| API | `api/ragflow_server.py` | 是 | MySQL、Redis、MinIO、Doc Store |
| Task Executor | `rag/svr/task_executor.py` | 否 | Redis、MySQL、MinIO、Doc Store、Embedding |
| Web | `web/src/main.tsx` | 浏览器端 | API |
| 基础设施 | Docker Compose | 各自协议 | 本地数据卷 |

API 和 Worker 分离意味着：HTTP 请求可以快速创建任务，而解析 PDF、OCR、Embedding 等重操作由后台进程完成。API 正常不代表摄取链路正常。

## 2. API 启动代码逐段理解

打开 `api/ragflow_server.py`，按下面顺序阅读。

### 2.1 导入期

模块导入期间已经可能发生配置读取、日志初始化和全局对象声明。调试“服务尚未打印启动日志就退出”时，要考虑导入异常，而不只看 `main`。

### 2.2 `settings.init_settings()`

`common/settings.py` 将文本配置变成运行对象，核心结果包括：

- `DB_TYPE` 与元数据库配置；
- `STORAGE_IMPL` 对象存储实现；
- `docStoreConn` 检索存储连接；
- `retriever` 普通检索器；
- `kg_retriever` 知识图谱检索器；
- 服务监听地址、模型与运行开关。

这一步失败通常是配置、依赖库或基础设施连接问题。

### 2.3 数据库初始化

`init_web_db()` 建立 Peewee 数据库连接，`init_web_data()` 初始化运行所需的基础数据。数据库连接成功与表结构适配当前代码版本是两个不同条件，升级后要特别关注迁移。

### 2.4 插件和后台任务

`GlobalPluginManager.load_plugins()` 加载插件。API 还会启动进度更新、聊天渠道等后台协程，因此它不仅是同步请求处理器。

### 2.5 Quart 监听

最后由 `app.run(host=settings.HOST_IP, port=settings.HOST_PORT)` 启动。信号处理器负责退出时关闭资源。

## 3. 路由为什么看起来是“散的”

`api/apps/__init__.py` 负责创建 Quart App、CORS、Schema、Session 和鉴权。路由不是全部集中在一个文件，而是动态发现：

```text
api/apps/*_app.py
api/apps/sdk/*
api/apps/restful_apis/*.py
```

寻找接口推荐使用：

```bash
rg -n 'route\(|@.*(get|post|put|delete)|chat/completions' api/apps
```

接口通常存在两套风格：历史 Web API 与 `/api/v1` REST API。分析前先从浏览器 Network 或调用方确认真实 URL。

## 4. 配置加载模型

`common/config_utils.py::read_config()` 先加载：

```text
conf/service_conf.yaml
```

再用：

```text
conf/local.service_conf.yaml
```

覆盖顶层块。这里使用的是浅层更新。例如：

```yaml
mysql:
  host: localhost
  port: 3307
```

会把整个 `mysql` 顶层值替换成当前对象，而不是保证与默认块逐字段递归合并。本地文件应包含该块实际需要的字段。

### 配置排查方法

1. 确认代码实际读取的是哪个配置文件；
2. 查本地覆盖是否替换了完整顶层块；
3. 区分容器网络地址和宿主机地址；
4. 查端口映射，不要只看容器内部端口；
5. 在服务启动日志中核对最终脱敏配置。

## 5. 四种持久化角色

### 5.1 关系数据库

保存结构化业务事实：

- 谁创建了知识库；
- 文档属于哪个租户；
- 文档是否已解析；
- 使用哪个解析器和模型；
- 会话、消息和 Agent 定义是什么。

它不适合保存和检索大量向量，也不负责保存完整 PDF 二进制。

### 5.2 对象存储

`settings.STORAGE_IMPL` 屏蔽 MinIO、S3、OSS、Azure、GCS、OpenDAL 等实现差异。对象地址通常由业务记录保存或经 `File2DocumentService` 解析。

对象存储中的内容包括原始文件、切片图片、缩略图、Agent 附件等。数据库记录仍在但对象丢失时，会出现页面可见、重新解析或下载失败的“幽灵文档”。

### 5.3 Doc Store

`DocStoreConnection` 表示可全文与向量检索的存储。默认常见实现是 Elasticsearch，也支持 Infinity、OpenSearch 等。

它保存切片级记录，不是 `Document` 表的镜像。一个 `Document` 对应零到多个 chunk。

### 5.4 Redis/NATS

Python 摄取主链路使用 Redis 队列；Go ingestor 配置可以使用 NATS。消息系统承担任务分发和协调，而不是最终业务真相。

## 6. 数据模型关系

### 6.1 多租户主线

```text
User ── UserTenant ── Tenant
                         ├── Knowledgebase
                         ├── Dialog
                         ├── Model configuration
                         ├── Canvas
                         └── API Token
```

Tenant 是模型配置、索引命名和数据权限的重要边界。看到 `tenant_id` 时不能轻易省略或用当前用户 ID 替代。

### 6.2 文件与文档不是同一对象

```text
File ── File2Document ── Document ── Task
                              │
                              └── chunks in Doc Store
```

- `File` 服务于文件管理器的树形结构；
- `Document` 服务于知识库摄取和检索；
- `File2Document` 连接两种视角；
- `Task` 记录一次或一段解析工作；
- chunk 位于检索存储，不在 Peewee 模型列表中。

### 6.3 对话模型

```text
Dialog
  ├── kb_ids
  ├── llm_id
  ├── prompt_config
  ├── similarity_threshold
  ├── vector_similarity_weight
  └── top_n / top_k
       │
       └── Conversation ── messages/reference
```

Dialog 是可复用助手配置，Conversation 是一次持续会话。修改 Dialog 不应被理解为直接改写所有历史消息。

## 7. Service 层的作用

`api/db/services/common_service.py::CommonService` 提供通用查询和持久化能力，各业务 Service 在此之上表达规则。

阅读 Service 时区分：

- ORM 查询：怎样读写表；
- 业务不变量：哪些状态允许转换；
- 跨存储副作用：是否同时操作对象存储、队列和 Doc Store；
- 事务边界：失败时哪些操作能够回滚。

不要在 Route 层复制 Service 的校验逻辑。Route 应处理协议、认证和参数，Service 才是业务行为的所有者。

## 8. 实验：追踪一个文档的四份身份

选择一个小文件完成以下记录：

| 阶段 | 要记录的值 |
|---|---|
| 上传响应 | `dataset_id`、`document_id` |
| 数据库 | Document 状态、location、parser_id |
| 文件关联 | File 与 File2Document |
| 队列 | task id、priority、页范围 |
| 对象存储 | bucket、object name、size |
| Doc Store | chunk id、doc_id、kb_id、向量字段 |

完成后画出它们的关联。这个实验是理解后续所有源码的基础。

## 9. 本册检查点

1. API 正常、Worker 停止时哪些操作仍然成功？
2. 为什么容器内的 `localhost:9000` 和 Mac 上的 `localhost:9010` 可能指向不同服务？
3. `Document`、`File` 和 chunk 各自的所有者是谁？
4. 为什么本地配置缺少默认块中的字段可能导致意外错误？
5. 哪些数据可以由原始文件重建，哪些不能？
