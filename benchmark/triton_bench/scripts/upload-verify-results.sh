#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
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
# 若推送报权限错误，可显式指定可写地址（HTTPS 令牌最稳；SSH 需先信任 gitcode.com host key）：
#   export TRITON_EVAL_PUSH_URL=https://<你的PAT>@gitcode.com/hkhkhkzxk/triton-eval.git
# ============================================================================
set -euo pipefail

REPO_URL="https://oauth2:CWSsMZKs3jEN8g5Ysv7HZrBD@gitcode.com/hkhkhkzxk/triton-eval.git"
WORKDIR="${WORKDIR:-/home/$(whoami)/triton-eval}"
cd "${WORKDIR}"

if ! ls verify-records/ascend-ops-*.jsonl >/dev/null 2>&1; then
  # 结果常被上一轮自举的 `git reset --hard origin/main && git clean -fdx` 抹掉，
  # 但提交对象还在 reflog 里。这里自动从最近一次 verify 提交恢复，省得手工抄哈希。
  LAST_VERIFY="$(git reflog --format='%H %gs' 2>/dev/null \
                 | grep -m1 'commit: verify' | awk '{print $1}' || true)"
  if [ -n "${LAST_VERIFY}" ]; then
    echo "[upload-verify-results] 工作区无结果，尝试从 ${LAST_VERIFY:0:7} 恢复 verify-records/"
    git checkout "${LAST_VERIFY}" -- verify-records/ || true
  fi
fi

if ! ls verify-records/ascend-ops-*.jsonl >/dev/null 2>&1; then
  echo "[upload-verify-results] 没找到 verify-records/ascend-ops-*.jsonl，请先跑主脚本生成结果"
  echo "[upload-verify-results] 若之前跑过，可手工从 reflog 恢复："
  echo "  git reflog --oneline | grep 'commit: verify'"
  echo "  git checkout <上面输出的哈希> -- verify-records/"
  exit 1
fi

# 推送地址：默认用脚本内 REPO_URL（HTTPS 令牌，已实测可推）；TRITON_EVAL_PUSH_URL 可覆盖。
# 关键：直接对显式 URL 推送，绕开 .git/config 中可能残留的 remote.origin.pushurl
#（之前若设过 SSH 地址会被持久化，导致 unset 环境变量后仍走 SSH 报 Permission denied）。
PUSH_URL="${TRITON_EVAL_PUSH_URL:-${REPO_URL}}"
git remote set-url --delete --push origin 2>/dev/null || true
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

# 推送日志含带令牌的 URL，输出前统一脱敏，避免令牌被 echo 到终端/其他归档里
sanitize_log() { sed -E 's#(https?://)[^/@[:space:]]+@#\1***@#g'; }

if git push -v "${PUSH_URL}" "HEAD:refs/heads/${BRANCH}" >/tmp/ascend-upload.log 2>&1; then
  tail -4 /tmp/ascend-upload.log | sanitize_log
  echo "[upload-verify-results] 已上传 -> ${BRANCH}"
else
  tail -8 /tmp/ascend-upload.log | sanitize_log
  echo "[upload-verify-results][ERROR] 上传失败：检查推送写权限（可 export TRITON_EVAL_PUSH_URL=<可写地址>）"
  exit 1
fi
