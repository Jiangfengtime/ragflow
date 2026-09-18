---
sidebar_position: 4
title: 第四册：Agent Canvas 与前端
sidebar_label: Agent Canvas 与前端
---

# 第四册：Agent Canvas 与前端

本册从两个方向解释交互层：前端如何把用户操作变成 API 请求，Agent 后端如何把画布 DSL 变成事件驱动的工作流。

## 1. 前端启动

`web/src/main.tsx` 在首次渲染前等待：

1. `initLanguage()` 初始化界面语言；
2. `fetchBackendLanguage()` 请求 `/api/v1/language`；
3. React 挂载 `App`。

后端探测先于渲染，避免先按 Python 页面渲染再切换 Go 变体。

## 2. App Provider

`web/src/app.tsx` 组合：

- `QueryClientProvider`：服务器状态缓存；
- `ThemeProvider`：主题；
- `TooltipProvider`；
- 全局 Toaster；
- `RouterProvider`。

页面数据应优先进入 React Query，而非在多个组件中手工复制 loading、error 和 cache 状态。

## 3. 路由

`web/src/routes.tsx` 使用 `createBrowserRouter`，页面组件懒加载。主要业务页面包括：

- Datasets 与 Document Chunk；
- Chats 与 Searches；
- Agents 与 Templates；
- Memories、Files、Skills；
- User Settings；
- Admin。

追踪页面推荐顺序：

```text
route
  -> page
  -> component
  -> hook
  -> service
  -> request utility
  -> Vite proxy
  -> backend route
```

## 4. 请求层

新代码应使用 `web/src/utils/next-request.ts`。`web/src/utils/request.ts` 已有废弃标记，不应继续扩大旧抽象。

请求层通常负责：

- Authorization Header；
- camelCase/snake_case 转换；
- tenant 参数；
- 401 清理与登录跳转；
- 错误通知；
- blob/stream 等特殊响应。

Service 负责表达 API 方法，Hook 负责把它接入 React 生命周期和缓存。

## 5. 三种后端方案

`web/vite.config.ts` 的 `API_PROXY_SCHEME`：

```text
python -> Python API 9380 / Admin 9381
go     -> Go API 9384 / Admin 9383
hybrid -> 按 URL 精确分流
```

混合模式不是自动负载均衡，而是一组明确的路由正则。接口迁移后需要同步更新这里，否则页面可能仍调用旧实现。

Go 响应会添加 `X-API-Source: go`，可用浏览器 Network 快速确认实际后端。

## 6. 流式聊天

聊天不能完全复用普通 JSON 请求封装，相关代码还包括：

- `web/src/services/chat-completion-stream.ts`；
- `web/src/hooks/use-send-message.ts`；
- 直接 `fetch` 读取流的逻辑。

前端需要处理增量 answer、thinking、最终 reference、中断、重试和会话状态。调试“服务端已返回但页面卡住”时，应检查事件终止条件和流解析缓冲区。

## 7. Agent DSL 心智模型

Agent Canvas 是有状态图：

```text
节点 = Component instance
边   = upstream/downstream
数据 = inputs/outputs/globals/history
位置 = path
状态 = task_id + Redis cancel/log + persisted DSL
输出 = async events
```

DSL 同时服务于前端编辑和后端执行，字段契约比某一个类的内部实现更重要。

## 8. `Graph.load()`

`agent/canvas.py::Graph` 初始化时：

1. 解析 JSON DSL；
2. 规范化 DSL schema；
3. 遍历 components；
4. 根据 `component_name + Param` 创建参数类；
5. 用 DSL 参数更新并校验；
6. 根据 `component_name` 创建运行组件；
7. 恢复 path。

`component_class()` 会动态搜索：

```text
agent.component
agent.tools
rag.flow
```

类名是 DSL 与 Python 运行时之间的重要协议。

## 9. 变量系统

Canvas 包含系统变量：

- `sys.query`；
- `sys.user_id`；
- `sys.conversation_turns`；
- `sys.files`；
- `sys.history`；
- `sys.date`。

还可有 `env.*` 和组件输出引用。变量解析需要处理模板替换、嵌套字段、类型保持和不存在值。

常见错误：前端保存的是显示名，后端引用需要节点 ID；上游输出 schema 改变但下游模板没有同步。

## 10. `Canvas.run()`

每次运行会：

1. 初始化 token usage 和 tracing context；
2. 更新日期、message id 和用户输入；
3. 清理上一次运行遗留的组件输入输出；
4. 处理 webhook payload 和文件；
5. 更新系统变量与对话轮次；
6. 判断新运行还是 UserFillUp 恢复；
7. 检查取消状态；
8. 产出 `workflow_started`；
9. 按 path 和图关系执行组件；
10. 产出节点、消息、错误和结束事件；
11. 清理 ContextVar，防止状态泄漏到下一次运行。

它使用异步生成器，是因为 Agent 运行可能持续较久，需要把中间事件及时交给客户端。

## 11. 组件基类

`agent/component/base.py` 分为：

- `ComponentParamBase`：参数默认值、更新、校验、序列化；
- `ComponentBase`：输入输出、变量解析、执行和错误处理能力。

参数类与执行类分开，使保存 DSL 时不需要序列化运行时连接、线程池或模型客户端。

## 12. 组件分类

| 类别 | 示例 | 重点 |
|---|---|---|
| 起止与消息 | Begin、Message | 输入 schema、用户交互 |
| 模型 | LLM、AgentWithTools | 模型配置、流式输出、工具循环 |
| 控制流 | Switch、Loop、Iteration、ExitLoop | 分支、恢复、终止条件 |
| 数据 | VariableAssigner、Aggregator、List/String/Data Operations | 类型和引用 |
| 外部调用 | Invoke、Browser、Tools | 凭证、超时、副作用 |
| 文档 | DocsGenerator、Retrieval 等 | 文件与知识上下文 |

## 13. 错误、重试和取消

组件参数支持最大重试、错误后延迟、异常处理方式、默认值和跳转。Canvas 通过 Redis 中的取消标记检查任务取消。

设计工具组件时必须明确：

- 调用是否幂等；
- 重试是否会重复产生外部副作用；
- 超时后远端任务是否仍执行；
- 输出是否可以安全记录；
- API Key 如何脱敏。

## 14. 前后端新增组件清单

新增一个 Agent 节点通常需要：

1. 后端 Param 类；
2. 后端 Component 类；
3. 输入输出 schema 与变量暴露；
4. 前端节点类型和图标；
5. 前端表单；
6. DSL 序列化/反序列化；
7. 运行事件展示；
8. 单元测试；
9. 必要时的后端能力路由或工具注册。

只增加 Python 类，画布通常无法创建和配置它。

## 15. 实验：追踪最小 Agent

创建 Begin → LLM → Message：

1. 浏览器保存时复制 DSL；
2. 标记三个组件 ID 和边；
3. 找到 LLM 的输入引用；
4. 在 `Graph.load()` 观察类实例化；
5. 在 `Canvas.run()` 观察 path；
6. 记录每个流式 event；
7. 添加 Switch，比较 DSL 和执行路径变化；
8. 中途取消，观察 Redis 标记和结束事件。

## 16. 本册检查点

1. 页面接口调用如何经过 Hook、Service 和代理？
2. hybrid 模式为何可能让 Python 断点始终不触发？
3. DSL 中的 component name 怎样变成类实例？
4. path、history、globals 分别保存什么状态？
5. 工具重试为什么可能产生重复副作用？
