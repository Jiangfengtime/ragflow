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
- 当前示例版本为 `v0.27.1`。

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

当前本机开发环境使用：

| 项目 | 地址或名称 |
|---|---|
| MinIO API | `http://localhost:9010` |
| MinIO 控制台 | `http://localhost:9011` |
| 当前数据卷 | `docker_minio_dev_data` |
| 保留的旧数据卷 | `docker_minio_data` |

## 本机配置

本机已有 MySQL 占用 `3306`，Docker Desktop 中的服务占用 `9000/9001`。为避免影响已有服务，RAGFlow 使用以下端口：

| 服务 | 容器端口 | macOS 端口 |
|---|---:|---:|
| MySQL | 3306 | 3307 |
| MinIO API | 9000 | 9010 |
| MinIO Console | 9001 | 9011 |
| Elasticsearch | 9200 | 1200 |
| Redis | 6379 | 6379 |

端口和开发数据卷由 `docker/docker-compose.local.yml` 覆盖：

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
  user: 'root'
  password: 'infini_rag_flow'
  host: 'localhost'
  port: 3307
  max_connections: 900
  stale_timeout: 300
  max_allowed_packet: 1073741824
minio:
  user: 'rag_flow'
  password: 'infini_rag_flow'
  host: 'localhost:9010'
  bucket: ''
  prefix_path: ''
```

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

启动过程需要三个前台终端，分别运行 API、任务执行器和前端。

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

MySQL、MinIO、Redis 和 Elasticsearch 应为 `healthy`。

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

看到以下日志表示任务执行器已经就绪：

```text
RAGFlow ingestion is ready
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

也可以通过前端代理验证前后端连通性：

```bash
curl http://localhost:9222/api/v1/system/version
```

## 日常停止

在 API、任务执行器和前端终端中分别按 `Ctrl+C`。

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

以下以升级到 `v0.27.2` 为例。应升级到正式发布的固定标签，而不是直接将已有数据长期运行在 `main` 分支上。

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

```bash
cd /Users/jiangfengtime/WorkSpace/ragflow
mkdir -p ../ragflow-backup-v0.27.1

docker exec docker-mysql-1 \
  mysqldump \
  -uroot \
  -pinfini_rag_flow \
  --single-transaction \
  --routines \
  --triggers \
  rag_flow \
  > ../ragflow-backup-v0.27.1/rag_flow.sql
```

### 4. 备份 MinIO

```bash
docker run --rm \
  --entrypoint sh \
  --volumes-from docker-minio-1 \
  -v "$PWD/../ragflow-backup-v0.27.1:/backup" \
  pgsty/silo:RELEASE.2026-08-06T00-00-00Z \
  -c 'tar czf /backup/minio-data.tar.gz -C /data .'
```

主要数据卷包括：

```text
docker_mysql_data
docker_minio_dev_data
docker_esdata01
docker_redis_data
```

MySQL 和 MinIO 必须重点备份。Elasticsearch 可以从原始文档重新构建，但重新解析和向量化可能耗时且产生模型调用费用。

### 5. 保存本机适配

当前本机适配文件包括：

```text
conf/local.service_conf.yaml
docker/docker-compose.local.yml
web/.env.development.local
```

如果工作区还有对受版本控制文件的修改，可以统一暂存：

```bash
git status --short
git stash push -u -m "local macOS config before v0.27.2"
```

### 6. 切换到新版本

```bash
git fetch origin --tags
git tag --list 'v0.27.*'
git checkout v0.27.2
git stash pop
```

如果 `git stash pop` 产生冲突，应逐项对比新版本配置，而不是直接覆盖上游代码。特别要检查新版本是否已经原生支持 macOS Bash 和本机端口覆盖。

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

不使用 `-v` 时，重新创建容器不会删除已有命名数据卷。

### 9. 执行数据库初始化和迁移

```bash
source .venv/bin/activate
export PYTHONPATH="$(pwd)"

python3 -c \
  "from api.db.db_models import init_database_tables; init_database_tables()"
```

运行新版本自带的迁移脚本，并显式传入本机配置，以连接 `3307` 而不是本机已有 MySQL 使用的 `3306`：

```bash
PY=.venv/bin/python \
  tools/scripts/run_migrations.sh \
  conf/local.service_conf.yaml
```

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

## 常见问题

### Elasticsearch 持续重启并显示退出码 137

检查：

```bash
docker inspect docker-es01-1 \
  --format 'OOMKilled={{.State.OOMKilled}} ExitCode={{.State.ExitCode}} RestartCount={{.RestartCount}}'
```

如果 `OOMKilled=true`，应增加 Colima 内存或降低服务内存配置。

### API 可以启动，但 MySQL 报密码错误

先确认 `3306` 是否被本机 MySQL 占用：

```bash
lsof -nP -iTCP:3306 -sTCP:LISTEN
```

本机开发配置应连接 Docker MySQL 的 `3307`。

### MinIO 返回 InvalidAccessKeyId

确认请求是否误发到 Docker Desktop 占用的 `9000`：

```bash
lsof -nP -iTCP:9000 -sTCP:LISTEN
docker port docker-minio-1
```

本机源码配置应连接 `localhost:9010`。

### 前端 API 请求访问 9384 并失败

检查 `web/.env.development.local`：

```dotenv
API_PROXY_SCHEME='python'
```

修改后重启 Vite。

### 页面可访问，但上传文档一直不解析

确认任务执行器正在运行，并检查：

```text
logs/task_executor_common_mac_local_0.log
```

API 服务本身不会代替 task executor 消费文档解析任务。
