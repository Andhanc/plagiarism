#!/bin/sh
set -e
HOST="${QDRANT_HOST:-localhost}"
PORT="${QDRANT_PORT:-6333}"
export QDRANT_URL="${QDRANT_URL:-http://${HOST}:${PORT}}"

echo "Waiting for Qdrant at ${QDRANT_URL} ..."
n=0
while [ "$n" -lt 90 ]; do
  if curl -sf "${QDRANT_URL}/readyz" >/dev/null 2>&1; then
    echo "Qdrant is ready."
    break
  fi
  n=$((n + 1))
  sleep 2
done
if [ "$n" -ge 90 ]; then
  echo "Timeout waiting for Qdrant"
  exit 1
fi

python init_qdrant_collection.py
exec uvicorn api_server:app --host 0.0.0.0 --port 8765
