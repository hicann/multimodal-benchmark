#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ============================================================================
# upload-heartbeat.sh —— 评测运行期间的定期增量上传
#
# 为什么需要它（这是退出钩子补不了的洞）：
#   EXIT / TERM / INT trap 都依赖进程「正常走完退出流程」。而 SIGKILL 下进程被
#   内核直接终止，任何 trap 都不会执行。真机环境每两小时定时清空，220 个用例很
#   可能没跑完就被杀，此时已产出的结果会全部丢失，只能重跑再猜。
#   心跳在运行期间定期把当前进度推上去，最坏只丢最后一个间隔的进度。
#
# 与主脚本的分工：
#   * 本脚本只做「增量上传」，不判断成败、不产报告；
#   * 上传状态由标记文件（${UPLOAD_MARK}）协调，两边都不会重复推同一批结果；
#   * 自带轮数上限，主脚本猝死后也能自行收敛，不会变成永久孤儿。
#
# 用法（一般由 ascend-ops-verify.sh 自动拉起，不需要手工跑）：
#   WORKDIR=<仓库目录> RUN_STAMP=<时间戳> UPLOAD_MARK=<标记文件路径> \
#   TRITON_EVAL_PUSH_URL=<可写地址> bash scripts/upload-heartbeat.sh [间隔秒] [最多轮数]
# ============================================================================
set -uo pipefail

INTERVAL="${1:-300}"
MAX_ROUNDS="${2:-36}"
WORKDIR="${WORKDIR:-/home/$(whoami)/triton-eval}"
RUN_STAMP="${RUN_STAMP:?需要 RUN_STAMP（本轮时间戳）}"
UPLOAD_MARK="${UPLOAD_MARK:?需要 UPLOAD_MARK（上传标记文件路径）}"
PUSH_URL="${TRITON_EVAL_PUSH_URL:?需要 TRITON_EVAL_PUSH_URL（可写仓库地址）}"
BRANCH="verify-records/${RUN_STAMP}"

cd "${WORKDIR}" || exit 0

i=0
while [ "${i}" -lt "${MAX_ROUNDS}" ]; do
  sleep "${INTERVAL}"
  i=$((i + 1))

  # 主脚本已成功收尾（含它自己的 5/5 上传），心跳就没必要再动了
  [ -f "${UPLOAD_MARK}" ] && break

  # 只在确有产出时才推：结果 jsonl / 诊断 / 运行日志，三者有其一
  if ! ls verify-records/ascend-ops-*.jsonl >/dev/null 2>&1 \
     && ! ls verify-records/diagnose-*.md >/dev/null 2>&1 \
     && ! ls verify-records/run-*.log >/dev/null 2>&1; then
    continue
  fi

  # 并发时 .git/index.lock 会让其中一方失败，忽略即可，下一轮再来
  git add -A verify-records 2>/dev/null || continue
  git commit -q -m "verify: heartbeat ${RUN_STAMP} #${i}" 2>/dev/null || continue
  if git push "${PUSH_URL}" "HEAD:refs/heads/${BRANCH}" >/dev/null 2>&1; then
    echo "[heartbeat] 第 ${i} 轮增量上传完成 -> ${BRANCH}"
  fi
done
