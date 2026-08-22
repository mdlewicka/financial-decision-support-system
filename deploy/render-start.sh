#!/bin/sh
set -eu

export FINANCIAL_DECISION_SUPPORT_API_URL="${FINANCIAL_DECISION_SUPPORT_API_URL:-http://127.0.0.1:8000}"

python -m uvicorn backend.src.main:app   --host 127.0.0.1   --port 8000 &

backend_pid=$!

for attempt in $(seq 1 60); do
  if python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/v1/system/readiness', timeout=2)" >/dev/null 2>&1
  then
    echo "Backend gotowy."
    break
  fi

  if [ "$attempt" -eq 60 ]; then
    echo "Backend nie uruchomił się w wyznaczonym czasie."
    kill "$backend_pid" 2>/dev/null || true
    exit 1
  fi

  sleep 1
done

python -m streamlit run frontend/app.py   --server.address=0.0.0.0   --server.port="${PORT:-10000}"   --server.headless=true   --browser.gatherUsageStats=false &

frontend_pid=$!

cleanup() {
  kill "$frontend_pid" "$backend_pid" 2>/dev/null || true
  wait "$frontend_pid" "$backend_pid" 2>/dev/null || true
}

trap cleanup EXIT INT TERM

wait "$frontend_pid"
