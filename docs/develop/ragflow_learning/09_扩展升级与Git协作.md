---
sidebar_position: 9
title: 第九册：扩展、升级与 Git 协作
sidebar_label: 09. 扩展、升级与 Git 协作
---

# 第九册：扩展、升级与 Git 协作

本册回答两个问题：怎样在正确的抽象层修改 RAGFlow，以及怎样持续吸收上游版本而不丢失自己的学习注释和本机配置。

## 1. 修改前先确定“行为所有者”

| 需求 | 优先修改位置 | 不应首先修改 |
|---|---|---|
| 新增 API | `api/apps/` 路由与 service | 前端写死特殊返回 |
| 修改业务数据 | `api/db/services/` | 路由里散写 Peewee SQL |
| 新解析策略 | `rag/app/`、Chunk Service | MinIO 实现 |
| 新底层格式解析 | `deepdoc/` 或 `internal/parser/` | API 路由 |
| 新 Doc Store | `common/doc_store/` 抽象实现 | `Dealer` 里写后端专用调用 |
| 新对象存储 | `rag/utils/` 存储实现与 settings 装配 | FileService 到处加分支 |
| 新模型供应商 | 模型工厂/适配器、租户模型接口 | 检索器里直接调供应商 SDK |
| 改检索融合 | `rag/nlp/search.py` | ES mapping 中硬编码业务权重 |
| 新 Agent 组件 | `agent/component/` 与前端节点注册 | Canvas 核心写组件特例 |
| 新前端请求 | Hook + service + request 层 | 组件中重复 fetch |

原则是让新实现遵守现有抽象，而不是在相邻层建立第二条兼容路径。

## 2. 新增一个 API 的完整清单

1. 在正确 Blueprint 中注册路由和 HTTP 方法。
2. 使用 `login_required` 或该 API 类型要求的认证方式。
3. 从认证上下文或已授权资源确定 tenant，不信任客户端自报。
4. 用 `validate_request` 或显式代码校验参数。
5. 在 Service 层实现业务逻辑和数据库边界。
6. 使用 `get_json_result` 等统一响应格式。
7. 不返回 ORM 对象、API Key、连接串或内部异常栈。
8. 增加成功、无权限、无资源、非法参数和异常测试。
9. 前端在 service 层声明请求，由 Hook 处理状态与缓存失效。
10. 记录必要 ID 和耗时，不记录秘密和完整正文。

新增接口后用以下方式确认真实路由：

```bash
rg 'route\(' api/apps/path/to/file.py
rg '新接口片段' web/src api
```

## 3. 新增一种文档解析策略

### 3.1 高层 parser 与底层文件 parser

- `rag/app/` 负责面向业务的切片策略，如 paper、manual、table、naive；
- `deepdoc/` 负责 PDF/OCR/版面/表格等底层解析；
- `ParserType` 是 Document/Knowledgebase 保存的高层策略枚举；
- `rag/app/naive.py:PARSERS` 等映射负责具体文件类型选择。

不要把“新增一种文件后缀”和“新增一种业务切片策略”混为一谈。

### 3.2 实现步骤

1. 明确输入格式、输出 Chunk schema 和是否需要 OCR。
2. 若是新高层策略，扩展 `ParserType`。
3. 实现 `rag/app/<parser>.py` 的入口约定。
4. 注册到 Task Executor/Chunk Service 使用的 factory。
5. 在上传阶段补充后缀到 parser 的推断。
6. 明确 PDF/表格是否需要 Task 拆分。
7. 保留页码、bbox、标题层级和图片等 provenance。
8. 测试空文件、损坏文件、超大文件、取消和重试。
9. 比较 Chunk 数、token 分布和真实检索质量。

### 3.3 Chunk 输出最低要求

至少关注：

- 稳定 `id`；
- `doc_id`、`kb_id`；
- 原始内容与加权检索内容；
- 页码/位置；
- token 数；
- 图片或对象引用；
- metadata；
- Embedding 字段。

只让 parser “不报错”不代表可用于 RAG，必须验证引用能回到原页、表格语义没有被破坏、检索能够命中。

## 4. 新增模型供应商

需要同时考虑：

1. 供应商目录与可用能力类型；
2. 租户实例字段和 API Key；
3. Chat/Embedding/Rerank 等适配器接口；
4. 模型名、API Base、自定义 Header；
5. 同步/异步和流式返回；
6. 超时、限流、重试与错误归一化；
7. token 用量；
8. 前端配置表单和验证接口；
9. API Key 脱敏；
10. Embedding 维度及数据重建影响。

Embedding 模型一旦用于已有知识库，切换模型不能只改配置。旧 Chunk 向量属于旧语义空间，通常需要重新解析或至少重新 Embedding 并重建索引。

## 5. 新增 Doc Store

抽象入口是 `common/doc_store/doc_store_base.py:DocStoreConnection`。现有 Elasticsearch、Infinity、OceanBase/GaussDB 等实现说明 Doc Store 不等同于“只存向量”。实现需覆盖项目真正使用的能力：

- 索引创建、存在性与删除；
- 批量 insert/update/delete/get；
- 文本匹配与高亮；
- 向量相似度；
- filter/order/pagination/aggregation；
- refresh 与一致性语义；
- metadata index；
- 错误返回约定。

验证不能只跑向量 top-k，还要跑全文、混合、过滤、重排候选、删除和重新解析。

## 6. 新增对象存储

业务层依赖 `settings.STORAGE_IMPL`，常用契约包括：

- `put(bucket, object, bytes)`；
- `get(bucket, object)`；
- `rm(bucket, object)`；
- `obj_exist(bucket, object)`；
- bucket 操作。

实现时重点处理：

- bucket/object 命名合法性；
- 大文件与超时；
- 同名覆盖语义；
- 租户隔离；
- 凭据与 endpoint；
- 删除幂等；
- 断点/重试后是否会残留部分对象。

不要让业务代码依赖 MinIO 专用 SDK 对象，否则抽象会失效。

## 7. 修改检索算法

把修改分为四层：

| 层 | 典型代码 | 验证重点 |
|---|---|---|
| 查询构造 | `Dealer.search` | filter、BM25、KNN 是否正确下推 |
| 候选召回 | Doc Store `search` | recall、候选数量、延迟 |
| 融合排序 | `Dealer.retrieval` | term/vector 权重、阈值、稳定排序 |
| 模型重排 | `rerank_by_model` | rerank 输入、截断、成本、降级 |

修改前建立固定评测集，至少记录：

- 命中率/Recall@K；
- MRR 或 nDCG；
- 无答案问题的误召回；
- 首阶段和 rerank 延迟；
- Embedding/Rerank 费用；
- 引用正确性。

只凭一两个问题“看起来更好”不足以证明算法改动有效。

## 8. 新增 Agent 组件

完整改动面通常包括：

1. 后端组件类和输入/输出定义；
2. DSL 序列化字段；
3. 变量解析与类型约束；
4. `Canvas`/Graph 注册；
5. 前端节点、图标、表单和默认值；
6. 流式事件和错误展示；
7. 取消、超时与重试；
8. 单元测试与最小 Canvas 运行测试。

组件不得直接假设某个页面状态；运行时只应依赖 DSL、上下文和明确注入的服务。

## 9. 测试选择

### Python

先运行最窄测试：

```bash
uv run pytest path/to/test_file.py -q
ruff check path/to/changed_file.py
ruff format --check path/to/changed_file.py
```

### 前端

```bash
cd web
npm run lint
npm run type-check
npm run test
```

### Go

使用仓库脚本装配 CGO 和原生库：

```bash
bash build.sh --test ./path/to/package/...
bash build.sh --test-integration ./path/to/package/...
```

真实 MySQL/MinIO/ES/LLM 测试必须放入 integration/e2e/manual tier，不得让默认 unit tier 依赖外部服务。

## 10. Git 远端模型

推荐同时保留：

```text
origin    上游 infiniflow/ragflow
personal  自己的 GitHub 仓库
```

检查：

```bash
git remote -v
git branch -vv
git status --short --branch
```

当前学习分支示例为 `codex/ragflow-learning`。推送自己的远端：

```bash
git push -u personal codex/ragflow-learning
```

不要把 `conf/local.service_conf.yaml`、真实 `.env`、API Key 或数据库备份提交到仓库。

## 11. 提交策略

尽量让提交按职责拆分：

```text
docs: add retrieval learning guide
fix: recover pending ingestion tasks
feat: add parser for ...
test: cover ...
```

提交前：

```bash
git status --short
git diff --check
git diff --stat
git diff
```

重点确认：

- 没有秘密；
- 没有无关格式化；
- 没有临时 Debug 输出；
- 文档路径和符号仍存在；
- 测试与实现处于同一提交或清晰的连续提交中。

## 12. 从上游吸收更新

长期维护分支建议：

```bash
git fetch origin --tags
git switch codex/ragflow-learning
git rebase origin/main
```

如果团队习惯保留合并节点，也可以 merge。选择标准不是命令偏好，而是分支是否已共享、是否允许改写历史。

- 尚未共享的个人分支：rebase 通常更清晰；
- 已被多人使用的分支：优先 merge 或先协调；
- rebase 后推送自己的远端可能需要 `--force-with-lease`，绝不使用裸 `--force`。

冲突处理时以新版本实际代码为准，重新验证注释，不要机械保留旧函数名和旧行号。

## 13. 版本升级前的备份

升级前至少保护：

1. MySQL 全量备份；
2. MinIO 数据或数据卷快照；
3. `conf/local.service_conf.yaml`；
4. `docker/docker-compose.local.yml`；
5. `web/.env.development.local`；
6. 自己的 Git 提交和分支；
7. 必要时 ES 数据，尤其是重建成本高时。

MySQL 和 MinIO 是最关键的恢复基础。ES 理论上可重建，但可能产生大量时间与模型费用。

## 14. 升级到后续版本的标准流程

### 14.1 阅读变更

确认：

- Python/Node/Go 最低版本；
- Compose 服务版本；
- 配置字段变化；
- 数据库迁移；
- 索引 mapping 变化；
- 是否要求重新解析或重新 Embedding；
- 已废弃接口和前端环境变量。

### 14.2 停止写入

停止 API、Worker 和前端，确保没有运行中的解析或迁移。容器可按升级说明停止。

### 14.3 备份并记录基线

记录当前：

```text
Git commit/tag
数据库版本与表数量
知识库/文档/Chunk 数量
容器镜像版本
关键健康检查结果
一个可重复的上传与查询样例
```

### 14.4 获取新代码

如果只是验证上游 tag，可创建临时分支：

```bash
git fetch origin --tags
git switch -c codex/upgrade-v0.27.2 v0.27.2
```

若要把升级合并进长期学习分支，则在学习分支 rebase/merge 相应上游提交，并逐项解决冲突。

### 14.5 更新依赖

```bash
uv sync --python 3.13 --all-extras
uv run python3 ragflow_deps/download_deps.py

cd web
npm install
cd ..
```

Go 原生依赖和构建以新版本 `build.sh` 为准。

### 14.6 更新基础服务

```bash
docker compose \
  -f docker/docker-compose-base.yml \
  -f docker/docker-compose.local.yml \
  pull

docker compose \
  -f docker/docker-compose-base.yml \
  -f docker/docker-compose.local.yml \
  up -d --wait
```

不要添加 `-v`，否则会删除命名卷。升级前仍必须有独立备份。

### 14.7 执行迁移

当前代码在数据库初始化中包含 schema 检查和迁移逻辑，仓库也可能提供独立迁移脚本。必须以目标版本发布说明和目标版本代码为准。迁移前先在备份副本或测试环境演练。

### 14.8 冒烟验证

```text
API health/version
  → 登录
  → 模型配置可读取
  → 创建测试知识库
  → 上传小文件
  → 创建 Task
  → Worker 完成解析
  → ES 有文本和向量
  → BM25/向量检索
  → Chat 流式回答与引用
  → 删除测试文档并确认清理
```

只确认页面打开不足以宣布升级成功。

## 15. 升级冲突的判断顺序

1. 上游是否已经实现了本地补丁的目标？若是，删除本地重复实现。
2. 上游是否改变了拥有该行为的抽象？把本地需求迁到新抽象。
3. 本地注释是否仍描述真实代码？过期就重写或删除。
4. 本地配置是否可以通过 override 文件表达？避免改上游模板。
5. 测试是否仍保护当前有效行为，而不是旧兼容路径？

不要为了“减少冲突”永久保留新旧两条实现，这会让后续升级更困难。

## 16. 回滚策略

升级前就要定义回滚：

- 代码：回到旧 tag/commit；
- Python/Node 依赖：按旧 lockfile 重装；
- 容器：恢复旧镜像 tag；
- MySQL：恢复升级前备份；
- MinIO：恢复对象数据；
- ES：恢复快照或从旧数据重新构建。

数据库迁移后仅回退代码可能无法启动，因此回滚必须包含数据库方案。

## 17. 文档维护规则

升级后检查本学习文档中的：

- 版本号；
- 启动命令；
- 入口文件；
- 类名和函数名；
- 状态枚举；
- 端口；
- 存储后端；
- API URL；
- 日志标记。

文档尽量引用稳定符号而非固定行号。用 `rg` 验证引用：

```bash
rg 'class TaskHandler|def queue_tasks|class Dealer' api rag
```

## 18. 本册检查点

- 能把新功能放到真正拥有该行为的层。
- 能列出新增 API、parser、模型或 Doc Store 的完整改动面。
- 能为检索修改建立可重复评测，而不是只看主观样例。
- 能维护 origin/personal 双远端和长期学习分支。
- 能完成带备份、迁移、冒烟测试和回滚方案的版本升级。
- 能在上游重构后删除重复兼容路径并更新文档。
