---
sidebar_position: 6
title: 第六册：API、鉴权、租户与权限
sidebar_label: 06. API、鉴权、租户与权限
---

# 第六册：API、鉴权、租户与权限

本册解释请求进入 RAGFlow 后，如何识别用户、确定租户、校验资源权限，以及前端、内部 REST API 和公开 API 之间的边界。

## 1. Quart 路由从哪里来

Python API 入口是 `api/ragflow_server.py`，Quart 应用和认证基础设施主要在 `api/apps/__init__.py`。业务接口分散在：

- `api/apps/*_app.py`：较早的页面业务接口；
- `api/apps/restful_apis/*.py`：REST 风格接口；
- `api/apps/services/*.py`：多个路由复用的业务流程；
- `api/db/services/*.py`：数据库访问和业务状态更新。

查找接口时建议搜索路由字符串：

```bash
rg 'route\(' api/apps
rg 'documents/ingest|datasets/search' api/apps
```

不要只看文件名推断 URL。Blueprint 的前缀和应用注册逻辑可能在上层统一添加。

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

`code` 是 RAGFlow 业务码；HTTP 状态码是传输层状态。`api/utils/api_utils.py` 会把部分内部 `RetCode` 映射为 400、403 或 500，但调用方仍应读取响应体中的 `code` 和 `message`。

## 3. `login_required` 做了什么

`api/apps/__init__.py:login_required` 是路由级认证装饰器。它调用 `_load_user()`，成功后把用户放入请求上下文，再执行真正的路由函数；认证失败则返回错误或抛出未授权异常。

当前代码支持多种认证来源，具体启用范围由路由传入的 `auth_types` 决定。常见来源包括：

- 浏览器登录会话/JWT；
- `Authorization: Bearer ...` API Token；
- 部分兼容或公开 API 使用的专用认证类型。

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
                    ├─ TenantLLM
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
  → POST /tenants/<tenant_id>/users
  → 校验 current_user.id == tenant_id
  → 查找目标 User
  → 创建 UserTenant(role=INVITE)
  → 异步发送邀请邮件
  → 被邀请人 PATCH /tenants/<tenant_id>
  → role 更新为 NORMAL
```

关键点：

- 当前页面只支持邀请已经注册的用户；
- 邀请创建的是关系记录，不是复制一份用户；
- 接受邀请后，用户查询可访问租户时会得到多条记录；
- 移除成员是删除相应 `UserTenant` 关系，不是删除用户账户。

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
- `TenantLLM.api_key`：RAGFlow 调用模型供应商的凭据；
- 浏览器 Session/JWT：登录态凭据。

三者不要混用，也不要在日志、截图、提交记录或学习文档中写入真实值。

## 10. 模型配置与租户隔离

当前仓库同时保留旧模型配置表和新的 provider/instance/model 结构。新接口优先通过新结构解析模型，部分旧 Service 仍使用 `TenantLLM`：

| 模型 | 作用 |
|---|---|
| `LLMFactories` | 供应商目录与能力标签 |
| `LLM` | 可选模型目录，如名称、类型、最大 Token |
| `TenantLLM` | 旧结构中的租户供应商、模型、API Key 和 API Base |
| `TenantModelProvider` | 新结构中的租户供应商记录 |
| `TenantModelInstance` | 供应商实例、实例名、API Key 和扩展配置 |
| `TenantModel` | 实例下的模型名与能力 bit flags |
| `Tenant` | 默认 chat/embedding/rerank 等模型指向 |
| `Knowledgebase` | 数据集实际使用的 Embedding 模型 |
| `Dialog` | Chat 与 Rerank 等问答配置 |

`TenantLLM.api_key` 和 `TenantModelInstance.api_key` 都是敏感字段。当前 ORM 将其建模为普通文本/字符字段，页面上的 `masked_key` 只是脱敏展示，不代表数据库字段经过了不可逆加密。调试时只允许检查是否为空、长度或掩码，不应直接查询、打印或提交完整值。数据库备份也应按秘密数据管理。

## 11. `validate_dataset_embedding_models`

多知识库检索需要验证 Embedding 配置兼容性。因为查询只生成一份向量，而不同 Embedding 模型可能具有不同的语义空间和向量维度；同一查询向量不能直接与不兼容的 Chunk 向量混算。

该验证通常负责：

1. 读取所有目标知识库；
2. 确认其 Embedding 模型配置可共同查询；
3. 解析租户模型实例；
4. 在不兼容时提前返回明确错误，而不是等 ES 查询时报维度错误。

因此它既是参数校验，也是检索正确性保护。

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
- 不要向前端返回 `TenantLLM.api_key`、内部连接串或堆栈。

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
