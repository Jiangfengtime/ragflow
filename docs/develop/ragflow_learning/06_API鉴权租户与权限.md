---
sidebar_position: 6
title: 第六册：API、鉴权、租户与权限
sidebar_label: 06. API、鉴权、租户与权限
---

# 第六册：API、鉴权、租户与权限

本册解释请求进入 RAGFlow 后，如何识别用户、确定租户、校验资源权限，以及前端、内部 REST API 和公开 API 之间的边界。

## 1. Quart 路由从哪里来

Python API 入口是 `api/ragflow_server.py`，Quart 应用和认证基础设施主要在 `api/apps/__init__.py`。业务接口分散在：

- `api/apps/*_app.py`：页面业务接口；
- `api/apps/restful_apis/*.py`：REST 风格接口，也是当前 Python SDK 请求的服务端入口；
- `api/apps/services/*.py`：多个路由复用的业务流程；
- `api/db/services/*.py`：数据库访问和业务状态更新。

查找接口时建议搜索路由字符串：

```bash
rg 'route\(' api/apps
rg 'documents/ingest|datasets/search' api/apps
```

不要只看文件名推断 URL。Blueprint 的前缀和应用注册逻辑可能在上层统一添加。

当前 SDK 客户端位于 `sdk/python/ragflow_sdk/`，工作区没有 `api/apps/sdk/` 目录；注册器中的通用扫描模式不代表该目录实际存在。SDK 与 REST 的调用契约见[第二十册](./20_SDK连接器与质量验证.md)。

`register_page()` 对 `restful_apis` 模块使用 `/api/v1`，非 REST 模块使用 `/v1/<page_name>`。以下团队路由的完整路径也都包含 `/api/v1`。定位一条请求时，应同时记录前缀、模块内路径、HTTP 方法和认证类型。

## 2. 一次请求经过的通用层次

```text
HTTP 请求
  → Nginx/开发代理
  → Quart 路由匹配
  → login_required
  → validate_request / get_request_json
  → tenant/resource permission check
  → API Service / DB Service
  → Storage / Doc Store / Redis / LLM
  → get_json_result
```

典型 JSON 响应：

```json
{
  "code": 0,
  "message": "success",
  "data": {}
}
```

`code` 是 RAGFlow 业务码；HTTP 状态码是传输层状态。`get_json_result()`、`get_result()` 负责响应体，`build_error_result()` 才会把部分内部 `RetCode` 映射为 400、403 或 500。不同接口采用的辅助函数不同，因此业务失败可能仍返回 HTTP 200，调用方必须读取响应体的 `code` 和 `message`。

## 3. `login_required` 做了什么

`api/apps/__init__.py:login_required` 是路由级认证装饰器。它调用 `_load_user()`，成功后把用户放入请求上下文，再执行真正的路由函数；认证失败则返回错误或抛出未授权异常。

认证范围由路由传入的 `auth_types` 决定；默认是 `JWT` 和 `API`，`BETA` 需显式启用：

| 类型 | `_load_user()` 的实际解析 |
|---|---|
| `JWT` | 用 `URLSafeTimedSerializer` 解签 Authorization 值，再按 `User.access_token` 查询有效用户 |
| `API` | 按 `APIToken.token` 查询，再按其 `tenant_id` 加载有效用户 |
| `BETA` | 按 `APIToken.beta` 查询，用于 bot 等专用路由 |
| 无 Authorization | 仅在允许 `JWT` 时尝试 Session 的 `_user_id`，并复核用户与 access_token 是否有效 |

这里的 `JWT` 是源码认证类型名，实际实现是 itsdangerous 签名令牌。Authorization 可带大小写不敏感的 `Bearer ` 前缀；已有 Header 但认证失败时，不会继续尝试无 Header 的 Session 分支。排障时应核对具体解析分支，不能只看浏览器是否有 Cookie。

认证回答的是“调用者是谁”，权限校验回答的是“这个调用者能否访问目标资源”。不能因为一个接口有 `@login_required`，就认为它已经完成知识库、文档或租户权限校验。

## 4. `current_user` 与 `tenant_id`

`current_user` 是请求上下文代理。代码中经常出现：

```python
tenant_id = current_user.id
```

这反映了 RAGFlow 的一个重要约定：用户自己的默认工作空间通常以该用户 ID 作为租户 ID。团队协作时，一个用户还可以通过 `UserTenant` 加入其他租户。

`add_tenant_id_to_kwargs` 会把 `current_user.id` 注入路由参数：

```python
kwargs["tenant_id"] = current_user.id
```

因此，看到函数参数叫 `tenant_id` 时，必须继续确认它来自：

1. 当前认证用户；
2. URL 路径；
3. 请求体；
4. 数据库关联查询。

来源不同，信任级别和权限校验要求也不同。

## 5. 用户、租户与成员关系

核心关系如下：

```text
User
  └─< UserTenant >─ Tenant
                    ├─ Knowledgebase
                    ├─ TenantModelProvider → Instance → Model
                    ├─ Dialog / Canvas
                    └─ API Token
```

`UserTenant` 字段包括：

| 字段 | 含义 |
|---|---|
| `user_id` | 成员用户 |
| `tenant_id` | 所属租户/团队 |
| `role` | 成员角色 |
| `invited_by` | 邀请人 |
| `status` | 关系是否有效 |

`UserTenantService.query(user_id=...)` 返回列表，是因为用户可以同时属于自己的租户和多个团队租户。这不是“按主键查询单条”的接口。

## 6. 团队邀请流程

团队接口位于 `api/apps/restful_apis/tenant_api.py`：

```text
团队所有者输入已注册邮箱
  → POST /api/v1/tenants/<tenant_id>/users
  → 校验 current_user.id == tenant_id
  → 查找目标 User
  → 创建 UserTenant(role=INVITE)
  → 异步发送邀请邮件
  → 被邀请人 PATCH /api/v1/tenants/<tenant_id>
  → role 更新为 NORMAL
```

关键点：

- 当前页面只支持邀请已经注册的用户；
- 邀请创建的是关系记录，不是复制一份用户；
- 接受邀请后，用户查询可访问租户时会得到多条记录；
- 移除成员是删除相应 `UserTenant` 关系，不是删除用户账户。

关系写入先于后台邮件任务完成，邮件发送失败不意味着关系已回滚。调试“收到邀请失败”时要分别检查关系、角色和 SMTP 任务日志。

## 7. 知识库权限

`Knowledgebase` 中与权限相关的核心字段：

| 字段 | 含义 |
|---|---|
| `tenant_id` | 知识库归属租户 |
| `created_by` | 创建者 |
| `permission` | `me` 或 `team` |

一个知识库是否可见，不能只比较 `kb.tenant_id == current_user.id`。团队知识库需要结合当前用户的 `UserTenant` 关系及 `permission` 判断。

建议阅读检索入口处的权限代码：它通常会先查当前用户可访问的租户列表，再过滤请求中的 `kb_ids`。这也是 `validate_dataset_embedding_models` 之前必须先验证数据集归属的原因之一。

## 8. 文档权限为什么通常通过知识库判断

`Document` 保存 `kb_id`，并不直接保存 `tenant_id`。因此常见权限链为：

```text
doc_id
  → Document.kb_id
  → Knowledgebase.tenant_id / permission
  → UserTenant
  → current_user
```

上传、解析、删除和下载都应先验证知识库访问权。只按客户端传入的 `doc_id` 查到 Document，不等于已经授权。

## 9. API Token 的数据边界

`APIToken` 的复合主键是 `(tenant_id, token)`，还可绑定 `dialog_id` 和 `source`。认证代码先按 token 查 `APIToken`，再加载对应租户用户。

模型供应商密钥不是 `APIToken`：

- `APIToken.token`：调用 RAGFlow API 的凭据；
- `TenantModelInstance.api_key`：当前供应商实例调用模型的凭据；
- 浏览器 Session/JWT：登录态凭据。

三者不要混用，也不要在日志、截图、提交记录或学习文档中写入真实值。

## 10. 模型配置与租户隔离

当前供应商设置与模型运行解析应先阅读 provider/instance/model 结构及 `api/db/joint_services/tenant_model_service.py`：

| 模型 | 作用 |
|---|---|
| `LLMFactories` | 供应商目录与能力标签 |
| `LLM` | 可选模型目录，如名称、类型、最大 Token |
| `TenantModelProvider` | 租户供应商记录，同租户内 provider_name 唯一 |
| `TenantModelInstance` | 供应商实例、实例名、API Key 和扩展配置 |
| `TenantModel` | 实例下的模型名与能力 bit flags |
| `Tenant` | 默认模型名称及 `tenant_llm_id/tenant_embd_id` 等模型记录 ID |
| `Knowledgebase` | 数据集 Embedding 的 `embd_id/tenant_embd_id` |
| `Dialog` | Chat、Rerank 名称及对应 `tenant_*_id` |

模型解析链为：业务模型引用 → `get_model_config_by_id()` / `resolve_model_config()` → Provider 与租户权限 → Instance 凭据及 `extra.base_url` → Model 能力、状态和扩展参数 → `LLMBundle` → 供应商适配器。ID 解析会验证调用方拥有供应商或已加入供应商所属租户，并检查模型能力位；这层权限不能替代对知识库、助手等业务资源的授权。

`TenantLLM` 表和 `TenantLLMService` 中仍存在直接查询方法，但 `LLMBundle` 继承该类的运行包装器，并不能据类名判断当前凭据来自哪张表。应沿调用者传入的 `model_config` 反查 joint service。模型的 `extra` 与实例的 `extra` 也不同：实例保存 endpoint/region，模型保存 `max_tokens/is_tools` 等能力参数。

`TenantLLM.api_key` 和 `TenantModelInstance.api_key` 都是敏感字段，ORM 将其建模为普通文本/字符字段。实例 `api_key` 还可能保存供应商专用的 JSON 凭据，不能假设始终是一个简单字符串。当前 `list_provider_instances()` 不返回 Key，但 `show_provider_instance()` 会把原始 `api_key` 返回给已认证、归属校验通过的实例编辑请求；页面密码控件的隐藏不代表接口或数据库已脱敏。

调试时只检查是否为空、长度或自行生成的掩码，不应打印、截图或提交实例详情响应。数据库备份也应按秘密数据管理；新增无需凭据的接口应显式挑选返回字段。

## 11. `validate_dataset_embedding_models`

多知识库检索需要验证 Embedding 配置兼容性。因为查询只生成一份向量，而不同 Embedding 模型可能具有不同的语义空间和向量维度；同一查询向量不能直接与不兼容的 Chunk 向量混算。

`api/db/services/knowledgebase_service.py::validate_dataset_embedding_models(kbs)` 接收已经加载的知识库列表，实际校验两件事：所有数据集都配置 Embedding 或都不配置；配置时，将 `tenant_embd_id`/原始模型 ID 尽力解析为名称，再比较去掉实例和供应商后缀的基础模型名。不满足条件时返回错误字符串，成功时返回 `None`。

这个函数不执行资源授权，不校验供应商凭据，也不调用模型探测向量维度。不同 endpoint 对同名模型的实现若不一致，基础名检查通过仍可能在编码或查询阶段失败；维度与向量字段需继续在 Embedding 输出和 Doc Store mapping 中核对。

## 12. 前端如何调用 API

前端请求通常按以下层次组织：

```text
页面/组件
  → React Hook
  → service 封装
  → request 工具
  → HTTP
  → code/message/data 解包
```

定位一个按钮的请求：

1. 用页面文案或组件名找到组件；
2. 找 `onClick`、mutation 或 Hook；
3. 找 Hook 引用的 service 方法；
4. 确认 URL、method、body；
5. 再回到 Python 路由搜索 URL 片段。

例如“上传后解析”不是上传接口内部的隐式逻辑，而是前端上传成功后根据 `parseOnCreation` 再调用 `/api/v1/documents/ingest`。

## 13. 参数校验、业务校验和权限校验

三者应分开理解：

| 类型 | 示例 | 常见位置 |
|---|---|---|
| 参数校验 | 缺少 `doc_ids`、类型不对 | `validate_request`、路由开头 |
| 业务校验 | 文档正在解析、Embedding 不兼容 | API Service、DB Service |
| 权限校验 | 用户不属于租户、无权访问知识库 | 路由或业务服务入口 |

新增接口时，至少要逐项检查：认证、租户来源、资源归属、参数限制、敏感字段、错误码和审计日志。

## 14. 常见安全误区

- 不要信任请求体中的 `tenant_id`；应从认证上下文或已授权资源反查。
- 不要把 `login_required` 当作资源授权。
- 不要在异常日志中序列化整个请求体；其中可能有 API Key、Prompt 或文档内容。
- 不要把真实 `local.service_conf.yaml` 提交到 Git。
- 不要用文件名作为 Document、MinIO 对象和 Chunk 的唯一关联依据。
- 普通业务接口应避免返回模型 API Key、内部连接串或堆栈；实例编辑接口当前涉及凭据返回，应检查其认证、归属校验和调用方，避免扩散完整响应。

## 15. 调试清单

请求返回 401/403 时按顺序检查：

1. 浏览器是否携带 Cookie 或 Authorization Header；
2. `_load_user()` 选择了哪种认证分支；
3. `current_user.id` 是什么；
4. 目标 `kb_id/doc_id` 反查出的租户是谁；
5. `UserTenant` 是否有有效关系、角色是否正确；
6. `Knowledgebase.permission` 是否允许团队访问；
7. 返回的是 HTTP 状态错误还是业务 `RetCode`。

## 16. 本册检查点

- 能解释 `current_user.id` 为什么经常作为 `tenant_id`。
- 能解释 `UserTenantService.query()` 为什么返回列表。
- 能区分认证、租户关系和资源权限。
- 能区分 API Token、模型 API Key 和浏览器登录态。
- 能从一个前端按钮定位到后端路由和权限检查。
- 能说明为什么多知识库检索必须校验 Embedding 模型。
