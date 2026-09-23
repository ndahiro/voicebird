#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
COMPOSE_FILE="${COMPOSE_FILE:-docker-compose.yml}"
SERVICE="${MODEL_PRELOAD_SERVICE:-asr-service}"
MODEL_VOLUME_NAME="voicebird_model_cache"

cd "$ROOT_DIR"

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not in PATH."
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "Error: docker compose v2 is required."
  exit 1
fi

echo "Building service image: ${SERVICE}"
docker compose -f "$COMPOSE_FILE" build "$SERVICE"

echo "Preloading models into Docker volume: ${MODEL_VOLUME_NAME}"
echo "This may take a while the first time."
docker compose -f "$COMPOSE_FILE" run --rm --no-deps "$SERVICE" python setup_models.py

echo ""
echo "Model preload complete."
echo "The cache is stored in Docker volume: ${MODEL_VOLUME_NAME}"
echo "It will persist across 'docker compose down' and container rebuilds."
