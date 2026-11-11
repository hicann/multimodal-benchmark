#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ============================================================================
# ascend-ops-verify.sh —— 真机运行脚本（仅跑 triton-ascend-ops 的 11 个映射算子）
#
# 设计要点（针对「真机环境每两小时自动清空」）：
#   1) 每次运行**先拉取仓库**（git clone / reset --hard），不依赖任何本地残留；
#   2) 运行 env-bootstrap.sh 重建 venv / 锁单卡 / 屏蔽用户级 triton；
#   3) 只跑 v0.6 语料的 11 个算子（--operators 显式列出），**不碰 v0.5 的 25 个**；
#   4) 结果写入 verify-records/，便于归档与比对。
#
# 用法（在真机上）：
#   export WORKDIR=/home/<user>/triton-eval          # 落盘目录（环境清空后重建）
#   bash scripts/ascend-ops-verify.sh [smoke|standard|official]
# 说明：
#   * 克隆/上传均走 REPO_URL（令牌已硬编码在下方，无需设置任何环境变量）；
#   * 结果推送到 gitcode 的 verify-records/<时间戳> 分支，便于回溯。
# ============================================================================
set -euo pipefail

REPO_URL="https://oauth2:CWSsMZKs3jEN8g5Ysv7HZrBD@gitcode.com/hkhkhkzxk/triton-eval.git"
WORKDIR="${WORKDIR:-/home/$(whoami)/triton-eval}"
PROFILE="${1:-smoke}"

ASCEND_OPS="VectorAdd,VectorCmp,DiscreteMemoryAccess,InsertSlice,ExtractSlice,\
AssignReqToTokenPool,DecodeGroupedAttention,CausalConv1d,GatherScatter,\
BinnedGatherScatter,PaddedGatherScatter"

SELF="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/$(basename "${BASH_SOURCE[0]}")"

# ══════════════════════════════════════════════════════════════════════════════
# 兜底上传：无论成功、失败、Ctrl-C、被 SIGTERM 杀掉、SSH 断连，现场都要传上去
#
# 目标：真机每跑一轮，助手侧都能直接 `git fetch` 拉取复盘，不依赖人工复制粘贴。
#
# 实测结论（据此设计，别凭直觉改）：
#   * bash 在收到未捕获的 SIGTERM / SIGHUP / SIGINT 时**仍会**执行 EXIT trap，
#     所以「多捕获几个信号」并不是原来上传失败的原因；
#   * 但 SIGKILL 下内核直接终止进程，**任何** trap 都不执行 —— 这才是兜不住的
#     场景。真机每两小时清空、220 用例很可能跑不完，因此必须靠运行期心跳
#     增量上传（见下方 start_heartbeat），把「全丢」降级成「只丢一个间隔」；
#   * trap 注册必须在**自举之前**：自举的 clone / reset / clean 失败同样要留证据；
#   * 日志必须落盘进 verify-records 一起上传 —— 终端输出环境一清空就没了，
#     落盘日志才是复盘的唯一可靠依据。
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${WORKDIR}/verify-records/run-${RUN_STAMP}.log"
# 上传成功标记。刻意放在仓库外：verify-records 里的未跟踪文件可能被自举的
# clean -fdx 清掉，而心跳进程与主脚本要靠它共享状态。
UPLOAD_MARK="${HOME}/.ascend-verify-uploaded-${RUN_STAMP}"
rm -f "${UPLOAD_MARK}"
HEARTBEAT_PID=""

# 阶段标记：每进入一个新阶段就追加一行，中断时能一眼看出卡在哪一步
mark_stage() {
  printf -- '- %s  %s\n' "$(date +%H:%M:%S)" "$1" \
    >> "${WORKDIR}/verify-records/PROGRESS.md" 2>/dev/null || true
}

# 推送日志里会带含令牌的仓库 URL（https://oauth2:<TOKEN>@gitcode.com/...），
# 而现在所有输出都会落盘并随结果上传——不脱敏就等于把令牌提交进仓库。
sanitize_log() { sed -E 's#(https?://)[^/@[:space:]]+@#\1***@#g'; }

# 心跳：评测跑得久，中途被强杀（SIGKILL）时 EXIT/TERM trap 都不会执行，
# 几小时的进度会全丢。运行期间定期增量上传，最坏只丢一个间隔。
start_heartbeat() {
  export RUN_STAMP
  export UPLOAD_MARK
  export TRITON_EVAL_PUSH_URL="${TRITON_EVAL_PUSH_URL:-${REPO_URL}}"
  # setsid 让它脱离本进程组：环境清空若按进程组杀，心跳仍可能存活并补传。
  # 自带轮数上限兜底，主脚本猝死也不会变成永久孤儿。
  if command -v setsid >/dev/null 2>&1; then
    setsid bash scripts/upload-heartbeat.sh "${HEARTBEAT_INTERVAL:-120}" 36 \
      >> "${LOG_FILE}" 2>&1 &
  else
    bash scripts/upload-heartbeat.sh "${HEARTBEAT_INTERVAL:-120}" 36 \
      >> "${LOG_FILE}" 2>&1 &
  fi
  HEARTBEAT_PID=$!
}

stop_heartbeat() {
  if [ -n "${HEARTBEAT_PID}" ]; then
    kill "${HEARTBEAT_PID}" 2>/dev/null || true
    HEARTBEAT_PID=""
  fi
}

upload_results() {
  cd "${WORKDIR}" 2>/dev/null || return 0
  # 成功推过就直接返回。这里必须用**文件**而非 shell 变量：心跳进程是 setsid
  # 脱离出去的子 shell，与主脚本不共享变量，只有文件能作为两边共同的状态标记。
  if [ -f "${UPLOAD_MARK}" ]; then
    return 0
  fi
  # 上传前先停心跳，避免两边同时操作 git 索引
  stop_heartbeat
  [ -d verify-records ] || return 0
  # tee 是独立进程，脚本被信号终止时它可能还没把缓冲刷进文件，日志会缺尾巴
  # （而尾部往往正是报错现场）。这里给足刷盘时间再上传。
  sleep 2
  # 有结果 jsonl / 诊断文件 / 本轮日志，三者有其一就上传。
  # 即便 0 用例，日志里也有环境校验与 run 的完整输出，足以定位问题。
  if ! ls verify-records/ascend-ops-*.jsonl >/dev/null 2>&1 \
     && ! ls verify-records/diagnose-*.md >/dev/null 2>&1 \
     && [ ! -s "${LOG_FILE}" ]; then
    return 0
  fi

  git config user.email "verify@local" 2>/dev/null
  git config user.name "ascend-ops-verify" 2>/dev/null
  # 与 upload-verify-results.sh 同一套推法：先清掉可能残留的 pushurl，再对显式
  # URL 推送。origin.pushurl 持久化在 .git/config 里，残留过 SSH 地址就会走 SSH
  # 报 Permission denied——表现为「自动上传失败、手工补传却成功」。
  local PUSH_URL="${TRITON_EVAL_PUSH_URL:-${REPO_URL}}"
  git remote set-url --delete --push origin 2>/dev/null || true
  # 统一用本轮 RUN_STAMP：心跳与 5/5 步必须推同一个分支。若各自取当前时间，
  # 一轮会推出两个分支，且标记文件的防重判断会因分支不同而被绕过。
  local br="verify-records/${RUN_STAMP}"

  echo "[ascend-ops-verify][upload] 上传现场到 gitcode (${br} 分支)"
  git add -A verify-records 2>/dev/null || true

  local attempt=1
  while [ "${attempt}" -le 3 ]; do
    # 无新变更时 commit 会失败，不影响随后重试 push（提交可能已在上一轮完成）
    git commit -q -m "verify: ${RUN_STAMP} (profile=${PROFILE:-unknown})" 2>/dev/null || true
    if git push -v "${PUSH_URL}" "HEAD:refs/heads/${br}" >/tmp/ascend-upload.log 2>&1; then
      tail -4 /tmp/ascend-upload.log | sanitize_log
      echo "[ascend-ops-verify][upload] 已上传 -> ${br}"
      touch "${UPLOAD_MARK}"
      return 0
    fi
    echo "[ascend-ops-verify][upload][WARN] 第 ${attempt}/3 次推送失败，5s 后重试"
    tail -4 /tmp/ascend-upload.log | sanitize_log
    attempt=$((attempt + 1))
    sleep 5
  done
  echo "[ascend-ops-verify][upload][ERROR] 3 次推送均失败：检查写权限或网络"
  return 1
}

# EXIT：正常结束 / set -e 退出 / 显式 exit 都会走
# TERM/HUP/INT/QUIT：显式捕获后 exit，让 EXIT trap 完成收尾
trap upload_results EXIT
trap 'upload_results; exit 143' TERM HUP
trap 'upload_results; exit 130' INT QUIT
mark_stage "启动 (profile=${PROFILE}, stamp=${RUN_STAMP})"

# 自举：确保运行的是 origin/main 最新版。
# 本脚本会 git reset 自身——若当前运行的不是最新版，先更新磁盘再重启本脚本一次，
# 否则会出现「执行的是旧版、磁盘却是新版」的修复延后（每轮都卡在上一个版本的 bug）。
if [ -z "${_ASCEND_VERIFY_REEXEC:-}" ]; then
  if [ ! -d "${WORKDIR}/.git" ]; then
    git clone "${REPO_URL}" "${WORKDIR}"
  fi
  cd "${WORKDIR}"
  git fetch --all --prune
  # 确保 origin 使用脚本内指定的仓库地址（令牌可能已更新）
  git remote set-url origin "${REPO_URL}" 2>/dev/null || git remote add origin "${REPO_URL}" 2>/dev/null || true
  if [ "$(git rev-parse HEAD 2>/dev/null)" != "$(git rev-parse origin/main 2>/dev/null)" ]; then
    # 抢救上一轮「已提交但上传失败」的 verify-records：
    # 本机 HEAD != origin/main，多半就是本地还挂着没推上去的结果提交，
    # 直接 reset --hard 会让它变悬空、clean -fdx 再删掉未跟踪文件，220 用例白跑。
    #   1) 把当前 HEAD 存成备份引用（即便 reflog 过期也能找回）；
    #   2) 把 verify-records 整目录拷到仓库外，clean -fdx 管辖不到。
    _STAMP="$(date +%Y%m%d-%H%M%S)"
    git update-ref "refs/verify-backup/${_STAMP}" "$(git rev-parse HEAD)" 2>/dev/null || true
    if [ -d verify-records ]; then
      mkdir -p "${HOME}/verify-records-backup"
      cp -a verify-records "${HOME}/verify-records-backup/verify-records-${_STAMP}" 2>/dev/null || true
      echo "[ascend-ops-verify] 已备份未上传结果 -> ~/verify-records-backup/verify-records-${_STAMP}"
    fi
    git reset --hard origin/main
    git clean -fdx
    export _ASCEND_VERIFY_REEXEC=1
    echo "[ascend-ops-verify] 已更新到最新版，重启脚本以使用新代码"
    exec "${BASH}" "${SELF}" "$@"
  fi
fi
cd "${WORKDIR}"

# 从此刻起，所有输出（含后续各阶段与 python 的 stderr）同时落盘到
# verify-records/run-<时间戳>.log，随结果一起上传。
# 注意：必须放在自举**之后**——自举里的 `git clean -fdx` 会删掉仓库内未跟踪文件。
mkdir -p verify-records
# stdbuf 强制 tee 行缓冲：进程被 SIGTERM 杀掉时，块缓冲的尾部（往往正是报错现场）会丢
if command -v stdbuf >/dev/null 2>&1; then
  exec > >(stdbuf -oL -eL tee -a "${LOG_FILE}") 2>&1
else
  exec > >(tee -a "${LOG_FILE}") 2>&1
fi
echo "[ascend-ops-verify] 本轮时间戳 ${RUN_STAMP}，日志落盘 ${LOG_FILE}"

mark_stage "自举完成，进入环境自举"
echo "[ascend-ops-verify] 2/4 环境自举"
# source（非 bash）：env-bootstrap 内含 venv 激活与单卡锁定，须在调用方 shell 生效
source scripts/env-bootstrap.sh
# triton_eval 是本地包、未预装。在激活的 venv 内做可编辑安装（--no-deps 不联网、
# --no-build-isolation 不另建构建环境），装到 venv site-packages（永远在 sys.path 上），
# 这样 `python -m triton_eval` 可靠导入，不依赖 PYTHONPATH。失败则回退到 PYTHONPATH。
python -m pip install -e . --no-deps --no-build-isolation 2>&1 | tail -5 \
  || echo "[env] pip install -e 失败，回退到 PYTHONPATH"
export PYTHONPATH="${WORKDIR}/src:${PYTHONPATH:-}"

# ───────────────────────── 校验环境依赖 ─────────────────────────
# 清空环境后先逐项确认关键依赖，缺什么一眼可见，避免在 220 用例跑到一半才抛
# ModuleNotFoundError。硬依赖（triton / torch_npu）缺失则快速失败并提示；
# CANN 扩展缺失仅告警——其余 6 个算子仍可评测。
echo "[ascend-ops-verify] 2.2/4 校验环境依赖"
mark_stage "2.2/4 校验环境依赖"
PYTHONPATH="${WORKDIR}/src" python - <<'PY'
import sys
def ok(m):  print(f"  [OK]   {m}")
def bad(m): print(f"  [FAIL] {m}")
def warn(m):print(f"  [WARN] {m}")

import subprocess
try:
    head = subprocess.check_output(["git","rev-parse","HEAD"]).decode().strip()[:12]
    ok(f"repo HEAD = {head}")
except Exception as e:
    warn(f"无法读取 git HEAD: {e}")

# 硬依赖：triton
try:
    import triton
    ok(f"triton {triton.__version__} @ {triton.__file__}")
except Exception as e:
    bad(f"triton 不可用: {e}"); sys.exit(1)

# 硬依赖：torch_npu
try:
    import torch_npu  # noqa
    ok("torch_npu 可用")
except Exception as e:
    bad(f"torch_npu 不可用: {e}"); sys.exit(1)

# NPU 可见性
try:
    import torch
    ok(f"可见 NPU 数 = {torch.npu.device_count()}")
except Exception as e:
    warn(f"无法探测 NPU: {e}")

# 软依赖：CANN 语言扩展（仅 best_practice 5 算子需要）
try:
    import triton.language.extra.cann  # noqa
    ok("triton.language.extra.cann 可用（5 个 best_practice 算子可加载）")
except Exception:
    warn("triton.language.extra.cann 不可用：")
    warn("  DecodeGroupedAttention / CausalConv1d / GatherScatter /")
    warn("  BinnedGatherScatter / PaddedGatherScatter 将在加载阶段失败。")
    warn("  重建含 triton-ascend 的环境：bash env/setup-venv.sh")
    warn("  或 export TRITON_EVAL_VENV=<含 triton-ascend 的 venv 路径>")
PY
echo

echo "[ascend-ops-verify] 2.5/4 生成 v0.7 修订语料（CPU 参考门禁）"
mark_stage "2.5/4 生成 v0.7 修订语料"
python scripts/build_corpus_v06_ascend_ops.py
[ -f corpus/v0.7-ascend-contract/manifest.json ] \
  || { echo "[ascend-ops-verify][ERROR] 语料清单未生成，后续 run 必然 0 用例"; exit 1; }

# ─── 预检：进评测前确认环境真的就绪 ───
# 真机每两小时清空，一轮里最贵、最不可控的开销是「重装编译器」：联网安装实测
# 20 分钟以上，还可能卡死到超时。这里把关键状态打清楚，便于事后判断本轮是
# 跑在就绪环境上、还是把窗口耗在了装环境上（曾出现跑到一半被清空、无结果）。
mark_stage "预检：确认环境就绪"
echo "[ascend-ops-verify] 2.8/4 预检环境状态"
python - <<'PY' 2>/dev/null || true
import glob, os, sys

def line(k, v): print(f"  {k:<10} = {v}")

try:
    import triton
    line("triton", f"{triton.__version__} @ {triton.__file__}")
except Exception as e:
    line("triton", f"不可用 ({e})")

try:
    import triton.language.extra.cann  # noqa
    line("CANN 扩展", "可用")
except Exception as e:
    line("CANN 扩展", f"不可用 ({type(e).__name__}: {e})")

wd = os.environ.get("TRITON_EVAL_WHEEL_DIR", "/opt/atomgit/wheels")
w = sorted(glob.glob(os.path.join(wd, "triton_ascend-*.whl")))
line("wheel 备份", w[0] if w else f"无（{wd} 下没有，下次环境重置将需联网安装）")

line("venv", sys.prefix)
PY

echo "[ascend-ops-verify] 3/5 运行 triton_eval —— 仅 11 个 ascend 算子 (profile=${PROFILE})"
mark_stage "3/5 运行 triton_eval (profile=${PROFILE})"
STAMP="${RUN_STAMP}"
RUNS_DIR="verify-records/ascend-ops-${STAMP}"
mkdir -p verify-records

# 评测可能跑满整个环境窗口，一旦被强杀（SIGKILL）退出钩子就来不及执行。
# 心跳在运行期间定期把已产出的结果推上去，把「全丢」降级成「只丢一个间隔」。
start_heartbeat

# run 子命令无 --output：结果落在 runs/run-<N>/results.jsonl 并写入本地 SQLite
# 保留 run 退出码，继续生成报告并上传后再返回。
VERIFY_RC=0
PYTHONPATH="${WORKDIR}/src" python -m triton_eval run \
  --profile "${PROFILE}" \
  --corpus corpus/v0.7-ascend-contract \
  --submission examples/ascend-ops-submission \
  --runs "${RUNS_DIR}" \
  --operators "${ASCEND_OPS}" || VERIFY_RC=$?

# 复制原始结果 jsonl 留存（供上传回溯）
LATEST_RUN="$(ls -dt "${RUNS_DIR}"/run-* 2>/dev/null | head -1 || true)"
if [ -n "${LATEST_RUN}" ] && [ -f "${LATEST_RUN}/results.jsonl" ]; then
  cp "${LATEST_RUN}/results.jsonl" "verify-records/ascend-ops-${STAMP}.jsonl"
  echo "[ascend-ops-verify] 原始结果 -> verify-records/ascend-ops-${STAMP}.jsonl"
fi

# ─── 结果自检：0 行必须显式失败 ───
# 空 jsonl 照样会被 cp、提交、上传，看起来「跑完了」，实际一个用例都没测。
# 这里拦下来，删掉空壳，把诊断落盘（trap 会把它一起传上来）。
RESULT_LINES=0
if [ -f "verify-records/ascend-ops-${STAMP}.jsonl" ]; then
  RESULT_LINES="$(wc -l < "verify-records/ascend-ops-${STAMP}.jsonl" | tr -d ' ')"
fi
echo "[ascend-ops-verify] 结果行数 = ${RESULT_LINES}"
if [ "${RESULT_LINES}" -eq 0 ]; then
  rm -f "verify-records/ascend-ops-${STAMP}.jsonl"
  DIAG="verify-records/diagnose-${STAMP}.md"
  {
    echo "# 诊断：本轮 0 用例产出"
    echo
    echo "- 时间戳：${STAMP}"
    echo "- profile：${PROFILE}"
    echo "- 算子：${ASCEND_OPS}"
    echo "- 最新 run 目录：${LATEST_RUN:-（无）}"
    echo
    echo "## runs 目录列表"
    ls -dt "${RUNS_DIR}"/run-* 2>/dev/null | head -5 || echo "（本轮无 run 目录）"
    echo
    echo "## events.jsonl 尾部"
    tail -40 "${LATEST_RUN}/events.jsonl" 2>/dev/null || echo "（无 events.jsonl）"
    echo
    echo "## integrity.json"
    cat "${LATEST_RUN}/integrity.json" 2>/dev/null || echo "（无 integrity.json）"
  } > "${DIAG}"
  echo "[ascend-ops-verify][ERROR] 本轮 0 个用例产出结果，诊断已落盘：${DIAG}"
  echo "[ascend-ops-verify][ERROR] 排查顺序：1) 2.2/4 环境校验是否 WARN 缺 triton-ascend；"
  echo "                           2) 2.5/4 语料清单是否生成成功；3) 上面 events.jsonl 的报错。"
  exit 1
fi

python scripts/check_run_completeness.py \
  --manifest corpus/v0.7-ascend-contract/manifest.json \
  --results "verify-records/ascend-ops-${STAMP}.jsonl" \
  --require-pass \
  > "verify-records/completeness-${STAMP}.json" || VERIFY_RC=1
cat "verify-records/completeness-${STAMP}.json"

echo "[ascend-ops-verify] 4/5 生成报告 -> verify-records/report-run-<id>.md/.html"
mark_stage "4/5 生成报告"
PYTHONPATH="${WORKDIR}/src" python -m triton_eval score --run latest || VERIFY_RC=1
PYTHONPATH="${WORKDIR}/src" python -m triton_eval report --run latest --out verify-records/ || VERIFY_RC=1
echo "--- 报告内容（节选）---"
cat verify-records/report-run-*.md 2>/dev/null | tail -40

# 5/5 上传结果到 gitcode：推到 verify-records/<时间戳> 分支，便于助手侧拉取查看
#（无需复制粘贴）。统一调用 upload_results —— 它是唯一上传入口，靠标记文件
# 防重复，所以后面 EXIT trap 兜底时不会再多推一个分支。
echo "[ascend-ops-verify] 5/5 上传结果到 gitcode (verify-records/${STAMP} 分支)"
mark_stage "5/5 上传结果"
upload_results || echo "[ascend-ops-verify][ERROR] 上传未成功，详见上方日志"
exit "${VERIFY_RC}"
