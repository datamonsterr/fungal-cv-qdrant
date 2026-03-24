#!/usr/bin/env bash
set -euo pipefail

print_help() {
  cat <<'EOF'
Sync this project to a cloud VM using rsync over SSH.

Usage:
  ./sync_project_to_cloud.sh --host <HOST_OR_IP> [options]

Required:
  --host <HOST_OR_IP>        VM hostname or IP

Options:
  --user <USER>              SSH user (default: root)
  --port <PORT>              SSH port (default: 22)
  --key <KEY_PATH>           SSH private key path (default: ~/.ssh/id_ed25519)
  --remote-dir <PATH>        Destination folder on VM (default: /workspace/fungal-cv-qdrant)
  --include-dataset          Include Dataset/ (can be very large)
  --include-results          Include results/, report/, qdrant_storage/
  --delete-remote-extra      Delete files on remote that do not exist locally
  --dry-run                  Show what would be copied without transferring
  --help                     Show this help message

Examples:
  ./sync_project_to_cloud.sh --host 203.0.113.10 --user ubuntu --key ~/.ssh/fungal
  ./sync_project_to_cloud.sh --host 203.0.113.10 --user ubuntu --include-dataset
EOF
}

require_cmd() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "Error: command not found: $1" >&2
    exit 1
  fi
}

HOST=""
SSH_USER="root"
SSH_PORT="22"
SSH_KEY="${HOME}/.ssh/id_ed25519"
REMOTE_DIR="/workspace/fungal-cv-qdrant"
INCLUDE_DATASET="false"
INCLUDE_RESULTS="false"
DELETE_REMOTE_EXTRA="false"
DRY_RUN="false"

while [ "$#" -gt 0 ]; do
  case "$1" in
    --host)
      HOST="${2:-}"
      shift 2
      ;;
    --user)
      SSH_USER="${2:-}"
      shift 2
      ;;
    --port)
      SSH_PORT="${2:-}"
      shift 2
      ;;
    --key)
      SSH_KEY="${2:-}"
      shift 2
      ;;
    --remote-dir)
      REMOTE_DIR="${2:-}"
      shift 2
      ;;
    --include-dataset)
      INCLUDE_DATASET="true"
      shift
      ;;
    --include-results)
      INCLUDE_RESULTS="true"
      shift
      ;;
    --delete-remote-extra)
      DELETE_REMOTE_EXTRA="true"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
      ;;
    --help|-h)
      print_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      print_help
      exit 1
      ;;
  esac
done

if [ -z "$HOST" ]; then
  echo "Error: --host is required" >&2
  print_help
  exit 1
fi

if [ ! -f "$SSH_KEY" ]; then
  echo "Error: SSH key not found: $SSH_KEY" >&2
  exit 1
fi

require_cmd rsync
require_cmd ssh

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$SCRIPT_DIR"

SSH_OPTS=(
  -p "$SSH_PORT"
  -i "$SSH_KEY"
  -o StrictHostKeyChecking=accept-new
)

RSYNC_OPTS=(
  -avz
  --no-owner
  --no-group
  --human-readable
  --progress
)

if [ "$DRY_RUN" = "true" ]; then
  RSYNC_OPTS+=(--dry-run)
fi

if [ "$DELETE_REMOTE_EXTRA" = "true" ]; then
  RSYNC_OPTS+=(--delete)
fi

# Always ignore VCS and Python cache artifacts.
EXCLUDES=(
  ".git/"
  ".venv/"
  "__pycache__/"
  "*.pyc"
  "*.pyo"
  ".mypy_cache/"
  ".pytest_cache/"
)

# Large generated artifacts are excluded unless explicitly requested.
if [ "$INCLUDE_DATASET" != "true" ]; then
  EXCLUDES+=("Dataset/")
fi

if [ "$INCLUDE_RESULTS" != "true" ]; then
  EXCLUDES+=("results/" "report/" "qdrant_storage/")
fi

for pattern in "${EXCLUDES[@]}"; do
  RSYNC_OPTS+=(--exclude "$pattern")
done

REMOTE_TARGET="${SSH_USER}@${HOST}:${REMOTE_DIR}/"

echo "Preparing remote directory: ${REMOTE_DIR}"
ssh "${SSH_OPTS[@]}" "${SSH_USER}@${HOST}" "mkdir -p '${REMOTE_DIR}'"

echo "Syncing project from ${PROJECT_ROOT} to ${REMOTE_TARGET}"
rsync "${RSYNC_OPTS[@]}" -e "ssh ${SSH_OPTS[*]}" "${PROJECT_ROOT}/" "${REMOTE_TARGET}"

echo "Done."
