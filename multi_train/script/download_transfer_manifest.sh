#!/usr/bin/env bash
set -euo pipefail

# Run this script on your local machine to pull the selected handoff files
# from the remote server using rsync and the manifest file.

REMOTE_USER="${REMOTE_USER:-zhoujiawei}"
REMOTE_HOST="${REMOTE_HOST:-YOUR_SERVER_HOST}"
REMOTE_ROOT="${REMOTE_ROOT:-/mnt/shared-storage-gpfs2/safewt2-share/jiawei/fusion/multi-reft}"
LOCAL_ROOT="${LOCAL_ROOT:-./multi-reft_handoff}"
MANIFEST_PATH="${MANIFEST_PATH:-multi_train/TRANSFER_MANIFEST.txt}"

mkdir -p "${LOCAL_ROOT}"

echo "Remote: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}"
echo "Local : ${LOCAL_ROOT}"
echo "Manifest: ${MANIFEST_PATH}"

rsync -av \
  --files-from="${MANIFEST_PATH}" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/" \
  "${LOCAL_ROOT}/"

echo "Done. Files downloaded to: ${LOCAL_ROOT}"
