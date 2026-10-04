---
sidebar_position: 4
title: macOS 源码开发与版本升级
sidebar_label: macOS 源码开发与版本升级
slug: /macos_source_development_zh
sidebar_custom_props: {
  categoryIcon: LucideLaptop
}
---

# macOS 源码开发与版本升级

本文记录在 Apple Silicon Mac 上以源码开发模式运行 RAGFlow 的方法，包含存储组件职责、日常启停、健康检查、端口冲突处理和版本升级流程。

本文对应的运行方式是：

- Colima 提供 Docker 运行环境；
- Docker 运行 MySQL、MinIO、Redis 和 Elasticsearch；
- macOS 本机运行 Python API、任务执行器和 Vite 前端；
- 版本流程以 `v0.27.1` 为示例基线，实际 checkout、配置和服务状态须现场确认。

源码扩展、迁移和 Git 协作的完整说明见[第九册](./ragflow_learning/09_扩展升级与Git协作.md)。本文的命令面向这一套本机 Python 开发方式；使用其他 Doc Store、数据库或 Go 服务时，需按实际后端调整。

## 组件职责

| 组件 | 主要职责 |
|---|---|
| MySQL | 保存用户、知识库、文档记录、任务状态和模型配置等结构化元数据 |
| MinIO | 保存原始文件、图片、缩略图和 Agent 附件等二进制对象 |
| Elasticsearch | 保存文档切片、全文索引、向量及其他检索数据 |
| Redis | 提供缓存、分布式锁和异步任务队列 |

### MinIO 在 RAGFlow 中的作用

MinIO 是兼容 Amazon S3 API 的对象存储。用户上传的 PDF、Word、Excel、图片等文件不会直接保存在 MySQL 中，而是保存到 MinIO；MySQL 只记录文档名称、对象位置、解析状态等元数据。

典型的文档处理流程如下：

```text
上传文件
  -> API 将原始文件写入 MinIO
  -> MySQL 创建文档及解析任务记录
  -> Redis 将任务分发给 task executor
  -> task executor 从 MinIO 读取原始文件
  -> 解析、切片、向量化
  -> 检索数据写入 Elasticsearch
  -> 图片等解析产物写回 MinIO
```

因此，删除 MinIO 数据卷可能造成“数据库中仍有文档记录，但原始文件无法读取”。

以下是本机端口配置的示例，不表示当前服务已经启动：

| 项目 | 地址或名称 |
|---|---|
| MinIO API | `http://localhost:9010` |
| MinIO 控制台 | `http://localhost:9011` |

数据卷名称受 Compose project name 和 override 配置影响，不能仅根据目录名推断。通过当前项目的 `docker compose ... ps -q minio` 获取容器 ID，再用 `docker inspect <实际容器ID> --format '{{range .Mounts}}{{println .Type .Name .Destination}}{{end}}'` 核对挂载；不要因为看到旧卷就直接删除它。

## 本机配置

如果本机已有 MySQL 占用 `3306`，或其他服务占用 `9000/9001`，可使用以下开发端口。是否冲突应先用 `lsof` 检查，不能将示例描述当作当前机器状态。

| 服务 | 容器端口 | macOS 端口 |
|---|---:|---:|
| MySQL | 3306 | 3307 |
| MinIO API | 9000 | 9010 |
| MinIO Console | 9001 | 9011 |
| Elasticsearch | 9200 | 1200 |
| Redis | 6379 | 6379 |

示例端口和开发数据卷可通过 `docker/docker-compose.local.yml` 覆盖：

```yaml
services:
  mysql:
    ports: !override
      - '3307:3306'
  minio:
    ports: !override
      - '9010:9000'
      - '9011:9001'
    volumes:
      - minio_dev_data:/data

volumes:
  minio_dev_data:
```

源码进程通过 `conf/local.service_conf.yaml` 连接这些端口。该文件会覆盖 `conf/service_conf.yaml` 中同名的顶层配置项：

```yaml
mysql:
  name: 'rag_flow'
  user: '<实际数据库用户>'
  password: '<实际数据库密码>'
  host: 'localhost'
  port: 3307
  max_connections: 900
  stale_timeout: 300
  max_allowed_packet: 1073741824
minio:
  user: '<实际对象存储访问键>'
  password: '<实际对象存储密钥>'
  host: 'localhost:9010'
  bucket: ''
  prefix_path: ''
```

以上占位符必须替换为本机实际配置，不是可直接使用的凭据。`common/config_utils.py::read_config()` 使用顶层 `dict.update`，例如整个本地 `mysql` 块会替换默认 `mysql` 块，并非逐字段深合并；因此要保留该服务需要的完整配置。真实配置和密钥不应提交到 Git。

前端的 `web/.env.development.local` 使用 Python API 代理：

```dotenv
API_PROXY_SCHEME='python'
```

## 首次安装

### 配置 Colima

本项目的 Elasticsearch 和解析服务内存占用较高。建议为 Colima 分配至少 4 CPU 和 12 GB 内存：

```bash
colima stop
colima start --cpu 4 --memory 12 --disk 100
```

Mac 总内存不足时，应相应降低并发和各容器内存，而不是让 Elasticsearch 持续因 OOM 重启。退出码 `137` 和 `OOMKilled=true` 通常表示容器被内存限制杀死。

### 安装本机依赖

```bash
brew install pkg-config jemalloc
uv sync --python 3.13 --all-extras
```

安装文本处理资源：

```bash
source .venv/bin/activate
export NLTK_DATA="$(pwd)/ragflow_deps/nltk_data"

python3 -c "import nltk; [nltk.download(name, download_dir='$NLTK_DATA', raise_on_error=True) for name in ('wordnet', 'punkt', 'punkt_tab')]"
```

安装前端依赖：

```bash
cd web
npm install
cd ..
```

`unixODBC` 缺失只影响可选的 ODBC/SQL Server 工具。如需使用相关连接器，可以执行：

```bash
brew install unixodbc
```

## 日常启动

仓库提供脚本启动和 IDE Debug 两种入口，下面保留手工启动命令，便于逐阶段定位问题。

### 脚本启动

在项目根目录执行：

```bash
bash start_local.sh
```

脚本使用项目 `.venv/bin/python3`，设置 `PYTHONPATH` 和 `NLTK_DATA`，必要时以 4 CPU、12 GB 内存启动 Colima，然后启动 Compose 基础组件及后台 API、`mac_local_0` common Worker、Vite。日志写入 `logs/local/`，PID 文件写入 `.run/local/`。它还会检查 Colima 是否能解析 Docker Registry；检查失败时会改写 VM 的 `/etc/resolv.conf`。已有 DNS/代理配置时，先检查脚本的这段行为是否适合本机环境。

脚本要求依赖、前端包和本机 override 已准备好，不执行 `uv sync`、`npm install` 或模型数据迁移。它等待 Compose 健康状态及固定地址 `9380/healthz`、`9222`，但 PID 存活不能证明 Worker 已完成模型初始化、能消费任务。

### PyCharm Debug

先确认 Colima/Docker 已启动，再执行：

```bash
bash start_debug.sh start
```

该脚本停止 `.run/local/` 中记录的 API/Worker，检查 9380 监听者：能识别为本仓库绝对路径的 API 时尝试停止它及对应父进程，其他进程则报告端口冲突并退出。随后只启动 Compose 和 Vite，API/Worker 由 PyCharm 启动。它不会启动 Colima、修复 DNS 或等待前端 URL 就绪，也不会识别所有手工运行的 Worker。

仓库已有 `.idea/runConfigurations/` 中的两个配置：

| 配置 | 入口/参数 |
|---|---|
| `RAGFlow API Debug` | `api/ragflow_server.py` |
| `RAGFlow Task Executor Debug` | `rag/svr/task_executor.py -i debug_0 -t common` |

在 PyCharm 中确认解释器实际指向项目 `.venv/bin/python3`，工作目录为项目根，环境变量包含 `PYTHONPATH=<项目根>`、`NLTK_DATA=<项目根>/ragflow_deps/nltk_data`。XML 中的 `SDK_NAME="uv (ragflow)"` 只是 SDK 名称，换机器后需重新确认解释器绑定。普通启动脚本和 IDE 不应同时启动同一个后端；只调试 API 时也需要有一个正常 Worker 执行解析。

### 手工启动

手工方式需要三个前台终端，分别运行 API、任务执行器和前端。下述示例中的项目路径应替换为本机实际 checkout。

### 1. 启动 Colima

```bash
colima start
```

确认资源配置：

```bash
colima status
docker info --format '{{.NCPU}} CPUs, {{.MemTotal}} bytes'
```

### 2. 启动基础组件

在项目根目录运行：

```bash
docker compose \
  -f docker/docker-compose-base.yml \
  -f docker/docker-compose.local.yml \
  up -d
```

检查状态：

```bash
docker compose \
  -f docker/docker-compose-base.yml \
  -f docker/docker-compose.local.yml \
  ps
```

检查实际启用的 MySQL、MinIO、Redis 和 Elasticsearch 状态；是否显示 `healthy` 取决于对应 Compose 服务是否定义 healthcheck。

### 3. 启动 API

在第一个终端中运行：

```bash
cd /Users/jiangfengtime/WorkSpace/ragflow

source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export NLTK_DATA="$(pwd)/ragflow_deps/nltk_data"

python3 api/ragflow_server.py
```

API 地址为 `http://localhost:9380`。

### 4. 启动任务执行器

在第二个终端中运行：

```bash
cd /Users/jiangfengtime/WorkSpace/ragflow

source .venv/bin/activate
export PYTHONPATH="$(pwd)"
export NLTK_DATA="$(pwd)/ragflow_deps/nltk_data"

python3 rag/svr/task_executor.py -i mac_local_0 -t common
```

当前源码初始化完成时记录以下形式的日志：

```text
RAGFlow ingestion is ready after ...s initialization.
```

不启动任务执行器时，页面和 API 仍可访问，但上传后的文档解析、切片和入库任务不会执行。

### 5. 启动前端

在第三个终端中运行：

```bash
cd /Users/jiangfengtime/WorkSpace/ragflow/web
npm run dev
```

浏览器访问：

```text
http://localhost:9222
```

当前 Vite 默认端口是 9222，但 `strictPort=false`：端口被占用时可能自动换端口。以实际 Vite 日志为准；两个启动脚本使用固定 9222 地址，不能用该端口已有页面的 HTTP 成功代替确认本次前端进程。

### 6. 健康检查

```bash
curl http://localhost:9380/api/v1/system/version
curl http://localhost:9380/api/v1/system/healthz
```

正常的健康检查结果如下：

```json
{
  "db": "ok",
  "doc_engine": "ok",
  "redis": "ok",
  "status": "ok",
  "storage": "ok"
}
```

该结构来自 `api/utils/health_utils.py::run_health_checks()`。任何组件探测失败会返回相应 `nok`、总体 `status="nok"`，可能附加 `_meta` 错误信息，`/system/healthz` 的 HTTP 状态为 500；全部成功才是 HTTP 200。该接口检查关系库、Redis、Doc Store 和对象存储，不检查 Worker 消费能力或整条上传/问答流程。

也可以通过前端代理验证前后端连通性：

```bash
curl http://localhost:9222/api/v1/system/version
```

## 日常停止

手工前台运行时，在 API、任务执行器和前端终端中分别按 `Ctrl+C`。脚本启动的后台进程可使用共享 PID 管理入口：

```bash
bash start_debug.sh stop
```

它只发送停止信号并移除所记录的 PID 文件，不停止 Docker；它也不保证终止 npm 的全部子进程或未记录的手工/IDE 后端。IDE Debug 应在 PyCharm 中停止，并用端口和进程检查确认退出。`start_local.sh` 当前没有 `stop` 参数。

停止基础容器：

```bash
docker compose \
  -f docker/docker-compose-base.yml \
  -f docker/docker-compose.local.yml \
  down
```

停止 Colima：

```bash
colima stop
```

:::danger 不要随意删除数据卷

除非明确准备清空全部本地数据，否则不要执行：

```bash
docker compose down -v
```

`-v` 会删除 Compose 管理的 MySQL、MinIO、Redis 和 Elasticsearch 数据卷。

:::

## 从 v0.27.1 升级到后续版本

以下只假设目标标签是 `v0.27.2`，不表示它已经发布或存在。先核验正式发布说明、远端来源和真实标签，再选择固定版本。以下占位路径、数据库和卷名称也必须按实际环境替换。

### 1. 阅读发布说明

确认新版本是否包含：

- 数据库结构迁移；
- Elasticsearch 索引变更；
- Docker 基础组件版本变更；
- Python 或 Node.js 版本变更；
- 配置文件新增或删除的字段；
- 不可逆的数据格式变更。

### 2. 停止源码进程

停止前端、API 和任务执行器，避免升级期间继续产生写入。

### 3. 备份 MySQL

先记录当前 commit/tag、容器镜像、数据库和文档数量，以及一个可重复的上传/查询样例。将备份保存到仓库之外的受保护目录；下面的凭据文件必须事先准备并在容器内可读，不要把密码写进命令行。

```bash
RAGFLOW_BACKUP_DIR='/绝对路径/升级前备份目录'
mkdir -p "$RAGFLOW_BACKUP_DIR"

docker compose \
  -f docker/docker-compose-base.yml \
  -f docker/docker-compose.local.yml \
  exec -T mysql \
  mysqldump \
  --defaults-extra-file=/容器内路径/受保护备份凭据.cnf \
  --single-transaction \
  --routines \
  --triggers \
  --databases '<实际数据库名>' \
  > "$RAGFLOW_BACKUP_DIR/mysql.sql"
```

MySQL 的逻辑备份需要具备相应权限的用户；若实际使用 PostgreSQL/GaussDB，应采用该数据库支持的备份工具。检查命令退出码、备份大小，并在隔离副本验证可恢复性。

### 4. 备份 MinIO

保留全部原始文件及解析图片、附件等对象，而不只备份上传文档目录。在线备份采用对象存储原生工具；直接归档数据卷前，停止使用目标卷的容器并确认没有写入。先解析实际项目、服务和挂载，不使用写死的容器名：

```bash
RAGFLOW_MINIO_CONTAINER=$(docker compose \
  -f docker/docker-compose-base.yml \
  -f docker/docker-compose.local.yml \
  ps -q minio)
docker inspect "$RAGFLOW_MINIO_CONTAINER" \
  --format '{{range .Mounts}}{{println .Type .Name .Destination}}{{end}}'
```

确认 MinIO `/data` 对应的真实命名卷或绑定目录，再用只读挂载归档到备份目录。当前仓库的 `docker/migration.sh` 支持按 Compose project name 备份四个固定卷后缀：

```text
mysql_data
minio_data
esdata01
redis_data
```

脚本要求相关容器已停止，不能自动覆盖本机 override 的 `minio_dev_data` 或其他后端的卷；restore 会改写目标卷。执行前核对源码、项目名及全部实际挂载，不要把固定四卷当成完整备份范围。

同时备份实际 Doc Store 的全部相关业务数据，包括 Chunk、独立文档元数据、Memory 消息、GraphRAG、Wiki/Skill 和其他编译产物。普通 Chunk 可由原文件重建，但人工元数据和长期记忆等不能保证恢复；重新解析、Embedding 和高级构建也可能产生显著费用。Elasticsearch/OpenSearch 使用其快照机制，其他后端按实际支持的导出/备份方式处理。Redis 还可能保存自动生成的系统签名密钥等运行状态，要记录来源和恢复方案。

### 5. 保存本机适配

需要单独保护的本机适配通常包括：

```text
conf/local.service_conf.yaml
docker/docker-compose.local.yml
web/.env.development.local
实际使用的 .env / 证书 / 密钥文件
start_local.sh / start_debug.sh 的本机修改
```

Git 备份与服务数据备份分开处理。检查已提交代码、学习注释和本地修改：

```bash
git status --short
git diff --stat
```

将已经审阅、无秘密的源码改动提交到自己的分支，或另存可恢复的补丁；未跟踪/被忽略的配置必须单独备份。普通 stash 不包含 ignored 文件，也不能用它代替数据库或对象备份。真实密钥和备份文件不进入 Git。

### 6. 切换到新版本

```bash
git remote -v
git branch -vv
git status --short --branch
```

`origin`、`upstream`、`personal` 都只是远端名称，先确认哪个 URL 是要获取的正式上游、哪个是自己的仓库，不能按名字推断。当前工作树妥善保存后，再执行：

```bash
RAGFLOW_RELEASE_REMOTE='<已核验的上游远端名>'
RAGFLOW_TARGET_TAG='v0.27.2'
git fetch "$RAGFLOW_RELEASE_REMOTE" --tags
git show-ref --verify "refs/tags/$RAGFLOW_TARGET_TAG"
git switch -c codex/upgrade-check "$RAGFLOW_TARGET_TAG"
```

上面的 tag 仍是待验证示例：不存在就停止切换，换成已确认发布的标签。若要把更新纳入长期学习分支，应按分支共享情况选择 merge/rebase；本机适配逐项对比后重新应用。检查新版本是否已经实现对应本地补丁，过时补丁和说明及时移除。

### 7. 更新依赖

更新 Python 依赖：

```bash
uv sync --python 3.13 --all-extras
```

如果发布说明要求刷新解析资源：

```bash
uv run python3 ragflow_deps/download_deps.py --china-mirrors
```

更新前端依赖：

```bash
cd web
npm install
cd ..
```

### 8. 更新基础容器

```bash
docker compose \
  -f docker/docker-compose-base.yml \
  -f docker/docker-compose.local.yml \
  pull

docker compose \
  -f docker/docker-compose-base.yml \
  -f docker/docker-compose.local.yml \
  up -d
```

不使用 `-v` 时不会因这组命令主动删除已有命名卷，但 project name、volume 声明或挂载改变仍可能让新容器连接到不同数据位置。启动前比较目标版本配置并确认旧数据挂载保持正确。

### 9. 执行数据库初始化和迁移

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

python3 -c \
  "from api.db.db_models import init_database_tables; init_database_tables()"
```

`init_database_tables()` 处理 Peewee 建表/列迁移；它不等同于模型供应商数据迁移。当前 API 启动会调用前者，`start_local.sh` 不执行后者。先查看目标版本的迁移阶段，再使用已准备好的完整连接配置做 dry-run：

```bash
uv run python tools/scripts/mysql_migration.py --list-stages
uv run python tools/scripts/mysql_migration.py \
  --config /受保护路径/完整MySQL迁移配置.yaml \
  --stages tenant_model_provider,tenant_model_instance,tenant_model,model_id_config
```

这条阶段命令未加 `--execute`。确认目标连接和演练结果后，当前完整 MySQL 迁移入口是：

```bash
PY=.venv/bin/python3 bash tools/scripts/run_migrations.sh \
  /受保护路径/完整MySQL迁移配置.yaml
```

该脚本第一个位置参数就是配置文件路径；当前依次执行 `v0.26.0` 建表/配置阶段及 `v0.27.1` 数据填充、模型类型合并、模型 ID 更新阶段，成功后写数据库版本标记。`mysql_migration.py` 只读取所传的单个 YAML 中 `database` 或 `mysql` 块，不自动合并 `conf/service_conf.yaml` 与本地 override，也不复用应用的密码解密过程；字段缺失或读取失败会使用默认连接值。因此本地 YAML 只有少量覆盖字段时不能直接作为迁移配置，必须先形成包含目标 host/port/user/password/name 的完整块并核对目标数据库。

:::warning 迁移脚本以新版本为准

如果新版本发布说明提供了不同的迁移命令，应优先遵循新版本说明。数据库迁移可能不可逆，因此必须先完成备份。

:::

### 10. 启动并验证新版本

按照“日常启动”一节重新启动 API、任务执行器和前端，然后执行：

```bash
curl http://localhost:9380/api/v1/system/version
curl http://localhost:9380/api/v1/system/healthz
```

确认版本号已经更新，并且所有组件均为 `ok`。随后上传一个小型测试文档，确认以下完整链路可用：

```text
上传 -> MinIO -> Redis 任务 -> task executor -> Elasticsearch -> 检索
```

继续验证模型实例配置、Chat 流式回答与引用、人工文档元数据和已有高级产物，最后删除测试文档并确认清理。升级前要制定回滚方案，覆盖旧代码/lockfile、旧镜像、关系库、对象存储、Doc Store 和必要 Redis 状态；发生不可逆迁移后，只退回旧代码不足以恢复服务。

## 常见问题

### Elasticsearch 持续重启并显示退出码 137

检查：

```bash
RAGFLOW_ES_CONTAINER=$(docker compose \
  -f docker/docker-compose-base.yml \
  -f docker/docker-compose.local.yml \
  ps -a -q es01)
docker inspect "$RAGFLOW_ES_CONTAINER" \
  --format 'OOMKilled={{.State.OOMKilled}} ExitCode={{.State.ExitCode}} RestartCount={{.RestartCount}}'
```

先用 `docker compose ... ps -a` 确认实际搜索服务名称，上面 `es01` 仅适用于该服务名称存在的 Compose 配置。

如果 `OOMKilled=true`，应增加 Colima 内存或降低服务内存配置。

### API 可以启动，但 MySQL 报密码错误

先确认 `3306` 是否被本机 MySQL 占用：

```bash
lsof -nP -iTCP:3306 -sTCP:LISTEN
```

如果采用本文的端口覆盖，开发配置应连接 Docker MySQL 的 `3307`；其他部署以实际端口和凭据为准。

### MinIO 返回 InvalidAccessKeyId

确认请求是否误发到另一个占用 `9000` 的服务，并核对实际 MinIO 端口：

```bash
lsof -nP -iTCP:9000 -sTCP:LISTEN
docker compose \
  -f docker/docker-compose-base.yml \
  -f docker/docker-compose.local.yml \
  port minio 9000
```

如果采用本文的端口覆盖，本机源码配置连接 `localhost:9010`；同时检查访问键和密钥是否匹配该对象存储实例。

### 前端 API 请求访问 9384 并失败

检查 `web/.env.development.local`：

```dotenv
API_PROXY_SCHEME='python'
```

修改后重启 Vite。

### 页面可访问，但上传文档一直不解析

确认任务执行器正在运行，并检查：

```text
logs/local/task_executor.log
```

API 服务本身不会代替 task executor 消费文档解析任务。

## 源码复核入口

| 行为 | 当前实现 |
|---|---|
| 后台启动、Colima/DNS 和就绪检查 | `start_local.sh` |
| IDE 前置环境、9380 冲突与 PID 停止 | `start_debug.sh` |
| PyCharm 入口与环境 | `.idea/runConfigurations/RAGFlow_API_Debug.xml`、`RAGFlow_Task_Executor_Debug.xml` |
| 本地配置合并 | `common/config_utils.py::read_config` |
| 健康输出和 HTTP 状态 | `api/utils/health_utils.py::run_health_checks`、`api/apps/restful_apis/system_api.py::healthz` |
| Vite 端口与代理 | `web/vite.config.ts` |
| 建表与迁移 | `api/db/db_models.py::init_database_tables`、`tools/scripts/run_migrations.sh`、`mysql_migration.py::MigrationConfig.from_config_file` |
| 四卷归档/恢复的实际范围 | `docker/migration.sh` |
