#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 本地编辑 → 同步到远程昇腾主机 → 跑指定档位。一条命令拿到反馈。
#
# 用法（在**本地开发机**执行，不是远程）：
#     export TRITON_EVAL_HOST=user@host
#     export TRITON_EVAL_REMOTE_DIR=/home/developer/triton-eval   # 可选
#     bash env/sync-and-run.sh                    # 冒烟档位
#     bash env/sync-and-run.sh standard           # 标准档位
#     bash env/sync-and-run.sh smoke --limit 3    # 只跑 3 个用例
#     bash env/sync-and-run.sh --setup-only       # 只部署不运行
#
# 两个关键点：
#   1. --exclude runs/  远程的测量结果不能被本地的空目录覆盖掉
#   2. --setup          同步后必须在远程 venv 里安装本体，否则
#                       `python -m triton_eval` 会报 No module named triton_eval

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
HOST="${TRITON_EVAL_HOST:-}"
REMOTE_DIR="${TRITON_EVAL_REMOTE_DIR:-/home/developer/triton-eval}"
VENV_DIR="${TRITON_EVAL_VENV:-$HOME/venv/te-compiler}"

SETUP_ONLY=0
PROFILE="smoke"
EXTRA_ARGS=""

# 解析参数：第一个非 -- 开头的参数是档位，其余透传给 run
args=()
while [ $# -gt 0 ]; do
    case "$1" in
        --setup-only) SETUP_ONLY=1; shift ;;
        --profile) PROFILE="${2:-smoke}"; shift 2 ;;
        --*) EXTRA_ARGS="${EXTRA_ARGS} $1"; shift ;;
        *) PROFILE="$1"; shift ;;
    esac
done

if [ -z "${HOST}" ]; then
    cat >&2 <<'EOF'
[sync] 错误：未设置 TRITON_EVAL_HOST。

请先导出远程主机地址，例如：
    export TRITON_EVAL_HOST=developer@your-cannlab-host

如果 CANNLab 不提供 SSH（只有 Web IDE），请改用打包方式：
    bash env/pack.sh
然后把生成的 tar 包通过 Web IDE 上传，在远程解压后运行
    source env/env.sh && pip install -e .
EOF
    exit 1
fi

echo "[sync] 本地   ${PROJECT_ROOT}"
echo "[sync] 远程   ${HOST}:${REMOTE_DIR}（排除 runs/）"

rsync -avz --delete \
    --exclude 'runs/' \
    --exclude '.git/' \
    --exclude '__pycache__/' \
    --exclude '*.db' \
    --exclude '.venv/' \
    --exclude 'venv/' \
    -e ssh "${PROJECT_ROOT}/" "${HOST}:${REMOTE_DIR}/"

echo "[setup] 在远程 venv 中安装评测工具链本体"
ssh "${HOST}" "cd ${REMOTE_DIR} && \
    source env/env.sh && \
    source ${VENV_DIR}/bin/activate && \
    export PYTHONNOUSERSITE=1 && \
    pip install -q -e . && \
    python -c 'import triton_eval; print(\"  已安装 triton_eval\", triton_eval.__version__)'"

if [ "${SETUP_ONLY}" -eq 1 ]; then
    echo "[sync] 已按 --setup-only 跳过运行"
    exit 0
fi

echo "[run] 远程执行：--profile ${PROFILE} ${EXTRA_ARGS}"
ssh "${HOST}" "cd ${REMOTE_DIR} && \
    source env/env.sh && \
    source ${VENV_DIR}/bin/activate && \
    export PYTHONNOUSERSITE=1 && \
    python -m triton_eval run --profile ${PROFILE} ${EXTRA_ARGS}"
