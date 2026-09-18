#!/usr/bin/env bash

# 本地 Debug 启动脚本：
# 1. 启动 MySQL、MinIO、Redis、Elasticsearch；
# 2. 启动前端；
# 3. 不启动 API 和 Task Executor，后端由 PyCharm Debug 配置启动。

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/.run/local"
LOG_DIR="${PROJECT_DIR}/logs/local"
COMPOSE_ARGS=(-f "${PROJECT_DIR}/docker/docker-compose-base.yml" -f "${PROJECT_DIR}/docker/docker-compose.local.yml")

mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"

pid_is_running() {
  local pid_file="$1"
  [[ -f "${pid_file}" ]] || return 1
  local pid
  pid="$(cat "${pid_file}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

stop_process() {
  local name="$1"
  local pid_file="${RUNTIME_DIR}/${name}.pid"
  if pid_is_running "${pid_file}"; then
    local pid
    pid="$(cat "${pid_file}")"
    kill "${pid}" 2>/dev/null || true
    echo "已停止 ${name}，PID: ${pid}"
  fi
  rm -f "${pid_file}"
}

stop_existing_api() {
  local pid parent_pid command parent_command
  local api_script="${PROJECT_DIR}/api/ragflow_server.py"
  local pids

  pids="$(lsof -nP -tiTCP:9380 -sTCP:LISTEN 2>/dev/null || true)"
  [[ -n "${pids}" ]] || return 0

  for pid in ${pids}; do
    command="$(ps -p "${pid}" -o command= 2>/dev/null || true)"
    if [[ "${command}" != *"${api_script}"* ]]; then
      echo "端口 9380 被其他进程占用（PID: ${pid}），请先检查：${command}" >&2
      return 1
    fi

    parent_pid="$(ps -p "${pid}" -o ppid= 2>/dev/null | tr -d ' ' || true)"
    parent_command="$(ps -p "${parent_pid}" -o command= 2>/dev/null || true)"
    echo "正在停止占用 9380 的旧 RAGFlow API（PID: ${pid}）"
    kill "${pid}" 2>/dev/null || true
    if [[ "${parent_pid}" =~ ^[0-9]+$ ]] && (( parent_pid > 1 )) && [[ "${parent_command}" == *"${api_script}"* ]]; then
      kill "${parent_pid}" 2>/dev/null || true
    fi
  done

  for (( attempt = 0; attempt < 20; attempt++ )); do
    [[ -z "$(lsof -nP -tiTCP:9380 -sTCP:LISTEN 2>/dev/null || true)" ]] && return 0
    sleep 0.5
  done
  echo "端口 9380 仍被占用，请检查后重试。" >&2
  return 1
}

start_frontend() {
  local pid_file="${RUNTIME_DIR}/frontend.pid"
  local log_file="${LOG_DIR}/frontend.log"

  if pid_is_running "${pid_file}"; then
    echo "前端已运行，PID: $(cat "${pid_file}")"
    return
  fi

  (
    cd "${PROJECT_DIR}/web"
    nohup npm run dev >>"${log_file}" 2>&1 </dev/null &
    echo $! >"${pid_file}"
  )
  echo "前端已启动，日志: ${log_file}"
}

case "${1:-start}" in
  start)
    # 避免脚本启动的后端与 PyCharm Debug 冲突。
    stop_process api
    stop_process task_executor
    stop_existing_api

    docker compose "${COMPOSE_ARGS[@]}" up -d --wait --wait-timeout 180
    start_frontend

    echo
    echo "Debug 环境已准备完成："
    echo "  前端: http://localhost:9222"
    echo "  API:  由 PyCharm 的 RAGFlow API Debug 启动"
    echo "  Worker: 由 PyCharm 的 RAGFlow Task Executor Debug 启动"
    ;;
  stop)
    stop_process api
    stop_process task_executor
    stop_process frontend
    echo "本地 Debug 进程已停止，Docker 依赖服务保持运行。"
    ;;
  *)
    echo "用法: $0 [start|stop]"
    exit 1
    ;;
esac
