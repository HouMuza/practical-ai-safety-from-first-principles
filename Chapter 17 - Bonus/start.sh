#!/usr/bin/env bash
set -Eeuo pipefail

CHAPTER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BACKEND_DIR="$CHAPTER_DIR/safety-evaluation-pipeline"
FRONTEND_DIR="$CHAPTER_DIR/reporting-dashboard"
BACKEND_PORT="${SAFETY_API_PORT:-8788}"
FRONTEND_PORT="${DASHBOARD_PORT:-3000}"

for command_name in python3 npm; do
  if ! command -v "$command_name" >/dev/null 2>&1; then
    echo "Missing required command: $command_name" >&2
    exit 1
  fi
done

if [[ ! -d "$FRONTEND_DIR/node_modules" ]]; then
  echo "Frontend dependencies are missing. Run: cd reporting-dashboard && npm install" >&2
  exit 1
fi

backend_pid=""
frontend_pid=""

cleanup() {
  trap - INT TERM EXIT
  [[ -n "$backend_pid" ]] && kill "$backend_pid" 2>/dev/null || true
  [[ -n "$frontend_pid" ]] && kill "$frontend_pid" 2>/dev/null || true
  wait "$backend_pid" "$frontend_pid" 2>/dev/null || true
}
trap cleanup INT TERM EXIT

echo "Starting safety-evaluation backend on http://localhost:$BACKEND_PORT"
(
  cd "$BACKEND_DIR"
  PYTHONPATH=src SAFETY_API_PORT="$BACKEND_PORT" python3 -m safety_eval.api
) &
backend_pid=$!

echo "Starting reporting dashboard on http://localhost:$FRONTEND_PORT"
(
  cd "$FRONTEND_DIR"
  npm run dev -- --host 127.0.0.1 --port "$FRONTEND_PORT"
) &
frontend_pid=$!

echo "Press Ctrl+C to stop both services."
while kill -0 "$backend_pid" 2>/dev/null && kill -0 "$frontend_pid" 2>/dev/null; do
  sleep 1
done

echo "One service stopped; shutting down the local platform." >&2
exit 1
