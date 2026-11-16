#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# 在当前持久工作目录运行，不自动克隆、重置、清理或推送仓库。
set -euo pipefail

WORKDIR="${WORKDIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
PROFILE="${1:-smoke}"
ASCEND_OPS="VectorAdd,VectorCmp,DiscreteMemoryAccess,InsertSlice,ExtractSlice,AssignReqToTokenPool,DecodeGroupedAttention,CausalConv1d,GatherScatter,BinnedGatherScatter,PaddedGatherScatter"
cd "${WORKDIR}"
RUN_STAMP="$(date +%Y%m%d-%H%M%S)"
LOG_FILE="${WORKDIR}/verify-records/run-${RUN_STAMP}.log"
mark_stage() {
  printf -- '- %s  %s\n' "$(date +%H:%M:%S)" "$1" >> "${WORKDIR}/verify-records/PROGRESS.md"
}

# 从此刻起，所有输出（含后续各阶段与 python 的 stderr）同时落盘到
# verify-records/run-<时间戳>.log，随结果一起上传。
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
# 20 分钟以上，还可能卡死到超时。这里把关键状态打清楚，便于事后判断本轮是
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

echo "[ascend-ops-verify] 结果已保存在 verify-records/；如需发布，请手动运行上传脚本。"
exit "${VERIFY_RC}"
