#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
PACKAGE_NAME="${1:-MAD-Avatar-RS}"
OUTPUT_PATH="${2:-${ROOT_DIR}/${PACKAGE_NAME}.tar.gz}"

TMP_DIR="$(mktemp -d)"
STAGE_DIR="${TMP_DIR}/${PACKAGE_NAME}"

mkdir -p "${STAGE_DIR}"

rsync -a \
  --exclude '.git' \
  --exclude '.codex' \
  --exclude '__pycache__' \
  --exclude '*.pyc' \
  --exclude '.DS_Store' \
  --exclude 'build' \
  --exclude '*.egg-info' \
  --exclude 'output' \
  --exclude 'data' \
  --exclude '*.tar.gz' \
  "${ROOT_DIR}/" "${STAGE_DIR}/"

tar -czf "${OUTPUT_PATH}" -C "${TMP_DIR}" "${PACKAGE_NAME}"
rm -rf "${TMP_DIR}"

echo "Packed release to ${OUTPUT_PATH}"
