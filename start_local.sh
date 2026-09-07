#!/usr/bin/env bash

set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNTIME_DIR="${PROJECT_DIR}/.run/local"
LOG_DIR="${PROJECT_DIR}/logs/local"
PYTHON_BIN="${PROJECT_DIR}/.venv/bin/python3"
COMPOSE_ARGS=(-f "${PROJECT_DIR}/docker/docker-compose-base.yml" -f "${PROJECT_DIR}/docker/docker-compose.local.yml")

mkdir -p "${RUNTIME_DIR}" "${LOG_DIR}"

export PYTHONPATH="${PROJECT_DIR}"
export NLTK_DATA="${PROJECT_DIR}/ragflow_deps/nltk_data"

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "缺少命令: $1"
    exit 1
  fi
}

pid_is_running() {
  local pid_file="$1"
  local pid

  [[ -f "${pid_file}" ]] || return 1
  pid="$(cat "${pid_file}")"
  [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null
}

start_backend_process() {
  local name="$1"
  shift

  local pid_file="${RUNTIME_DIR}/${name}.pid"
  local log_file="${LOG_DIR}/${name}.log"

  if pid_is_running "${pid_file}"; then
    echo "${name} 已运行，PID: $(cat "${pid_file}")"
    return
  fi

  rm -f "${pid_file}"
  (
    cd "${PROJECT_DIR}"
    nohup "$@" >>"${log_file}" 2>&1 </dev/null &
    echo $! >"${pid_file}"
  )

  sleep 1
  if ! pid_is_running "${pid_file}"; then
    echo "${name} 启动失败，请查看 ${log_file}"
    exit 1
  fi

  echo "${name} 已启动，PID: $(cat "${pid_file}")，日志: ${log_file}"
}

start_frontend() {
  local pid_file="${RUNTIME_DIR}/frontend.pid"
  local log_file="${LOG_DIR}/frontend.log"

  if pid_is_running "${pid_file}"; then
    echo "frontend 已运行，PID: $(cat "${pid_file}")"
    return
  fi

  rm -f "${pid_file}"
  (
    cd "${PROJECT_DIR}/web"
    nohup npm run dev >>"${log_file}" 2>&1 </dev/null &
    echo $! >"${pid_file}"
  )

  sleep 1
  if ! pid_is_running "${pid_file}"; then
    echo "frontend 启动失败，请查看 ${log_file}"
    exit 1
  fi

  echo "frontend 已启动，PID: $(cat "${pid_file}")，日志: ${log_file}"
}

wait_for_url() {
  local name="$1"
  local url="$2"
  local attempts="${3:-60}"
  local count=1

  while (( count <= attempts )); do
    if curl -fsS "${url}" >/dev/null 2>&1; then
      echo "${name} 已就绪: ${url}"
      return 0
    fi
    sleep 2
    count=$((count + 1))
  done

  echo "${name} 尚未就绪，请检查 ${LOG_DIR} 中的日志"
  return 1
}

require_command colima
require_command docker
require_command npm
require_command curl

if [[ ! -x "${PYTHON_BIN}" ]]; then
  echo "Python 虚拟环境不存在，请先运行: uv sync --python 3.13 --all-extras"
  exit 1
fi

if ! colima status >/dev/null 2>&1; then
  echo "正在启动 Colima..."
  colima start --cpu 4 --memory 12 --disk 100 --dns 1.1.1.1 --dns 8.8.8.8
else
  echo "Colima 已运行"
fi

# Some Colima VM images leave /etc/resolv.conf pointing at a missing
# systemd-resolved file. Repair that exact condition before Docker pulls.
if ! colima ssh -- getent hosts registry-1.docker.io >/dev/null 2>&1; then
  echo "正在修复 Colima DNS..."
  colima ssh -- sudo rm -f /etc/resolv.conf
  colima ssh -- sudo install -m 644 "${PROJECT_DIR}/docker/colima-resolv.conf" /etc/resolv.conf
fi

if ! colima ssh -- getent hosts registry-1.docker.io >/dev/null 2>&1; then
  echo "Colima DNS 仍不可用，请检查代理或网络设置"
  exit 1
fi

echo "正在启动 MySQL、MinIO、Redis 和 Elasticsearch..."
docker compose "${COMPOSE_ARGS[@]}" up -d --wait --wait-timeout 180

start_backend_process api "${PYTHON_BIN}" api/ragflow_server.py
start_backend_process task_executor "${PYTHON_BIN}" rag/svr/task_executor.py -i mac_local_0 -t common
start_frontend

wait_for_url "API" "http://127.0.0.1:9380/api/v1/system/healthz" 90
wait_for_url "前端" "http://127.0.0.1:9222" 60

echo
echo "RAGFlow 启动完成"
echo "前端: http://localhost:9222"
echo "API:  http://localhost:9380"
echo "日志: ${LOG_DIR}"
