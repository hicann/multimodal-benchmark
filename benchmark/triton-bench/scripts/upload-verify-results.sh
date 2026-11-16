#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# ============================================================================
# upload-verify-results.sh —— 仅上传机器上已有的 verify-records/（不重跑评测）
#
# 适用场景：ascend-ops-verify.sh 跑完后因中断/异常退出，没走到上传步，
# 但 verify-records/ascend-ops-<时间戳>.jsonl 已经生成。直接补传，省去重跑 220 用例。
#
# 用法（在真机上，需已在仓库目录下）：
#   export WORKDIR=/home/<user>/triton-eval
#   cd "$WORKDIR"
#   bash scripts/upload-verify-results.sh
#   export TRITON_EVAL_PUSH_URL=origin  # 使用 SSH 或 Git credential helper 管理认证
# ============================================================================
set -euo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/git-auth.sh"

REPO_URL="${TRITON_EVAL_PUSH_URL:-origin}"
WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${WORKDIR}"
require_safe_git_remote "${TRITON_EVAL_PUSH_URL:-origin}" || exit 1

if ! ls verify-records/ascend-ops-*.jsonl >/dev/null 2>&1; then
  echo "没有可上传结果；请先检查持久目录 verify-records/。"
  exit 1
fi

# 关键：直接对显式 URL 推送，绕开 .git/config 中可能残留的 remote.origin.pushurl
#（之前若设过 SSH 地址会被持久化，导致 unset 环境变量后仍走 SSH 报 Permission denied）。
PUSH_URL="${TRITON_EVAL_PUSH_URL:-${REPO_URL}}"
git fetch --all --prune >/dev/null 2>&1 || true

git config user.email "verify@local" 2>/dev/null
git config user.name "ascend-ops-verify" 2>/dev/null

STAMP="$(date +%Y%m%d-%H%M%S)"
BRANCH="verify-records/${STAMP}"
echo "[upload-verify-results] 上传已有结果 -> ${BRANCH}"

# 先把已有结果提交（若无新变更，commit 会失败但不影响后续 push）
git add -A verify-records 2>/dev/null
git commit -q -m "verify: upload ${STAMP} (from existing results)" 2>/dev/null || \
  echo "[upload-verify-results] 无新增提交（结果可能已提交过），仍推送当前 HEAD"

sanitize_log() { sed -E 's#(https?://)[^/@[:space:]]+@#\1***@#g'; }

if git push -v "${PUSH_URL}" "HEAD:refs/heads/${BRANCH}" >/tmp/ascend-upload.log 2>&1; then
  tail -4 /tmp/ascend-upload.log | sanitize_log
  echo "[upload-verify-results] 已上传 -> ${BRANCH}"
else
  tail -8 /tmp/ascend-upload.log | sanitize_log
  echo "[upload-verify-results][ERROR] 上传失败：检查推送写权限（可 export TRITON_EVAL_PUSH_URL=<可写地址>）"
  exit 1
fi
