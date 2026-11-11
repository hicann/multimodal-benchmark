#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 建立被测编译器的独立虚拟环境。
#
# 用法（在昇腾执行机上，且已 source env/env.sh）：
#     bash env/setup-venv.sh
#
# 设计要点：
#   --system-site-packages  继承系统的 torch / torch_npu（体积大且由平台提供）
#   PYTHONNOUSERSITE=1      屏蔽 ~/.local 下可能残留的社区 triton
#   venv 内只装被测编译器   使其安装状态可被精确快照（pip freeze 入环境指纹）

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$HOME/venv/te-compiler}"
REQ_COMPILER="${PROJECT_ROOT}/env/requirements-compiler.txt"
REQ_EVAL="${PROJECT_ROOT}/env/requirements-eval.txt"

echo "[setup-venv] 项目根 = ${PROJECT_ROOT}"
echo "[setup-venv] 目标环境 = ${VENV_DIR}"

# 1. 建环境
if [ ! -d "${VENV_DIR}" ]; then
    python3 -m venv --system-site-packages "${VENV_DIR}"
    echo "[setup-venv] 已创建 venv"
else
    echo "[setup-venv] venv 已存在，跳过创建"
fi

# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

# 2. 建好后先确认环境是干净的：不应存在任何 triton
echo "[setup-venv] 检查建好的 venv 中是否残留 triton ..."
if python -c "import triton" 2>/dev/null; then
    echo "[setup-venv] 错误：venv 中已存在 triton，环境不干净。" >&2
    python -c "import triton; print(triton.__file__)" >&2
    echo "[setup-venv] 请删除 ${VENV_DIR} 后重试，或检查 PYTHONNOUSERSITE / PYTHONPATH。" >&2
    # 自相矛盾点：本脚本用 --system-site-packages 建 venv，必然看得到系统
    # site-packages；而平台预装容器往往自带社区版 triton —— 于是这条自检在
    # 「系统已经有 triton」的机器上必定失败。此时正确做法不是新建环境，而是
    # 走 env-bootstrap.sh 的「继承式 venv + 覆盖安装 triton-ascend」路径。
    echo "[setup-venv] 若上面路径位于系统 site-packages 而非 ${VENV_DIR}：" >&2
    echo "[setup-venv]   说明系统已装社区版 triton，请改用 env-bootstrap.sh，" >&2
    echo "[setup-venv]   由它建继承式 venv 并覆盖安装 triton-ascend。" >&2
    exit 1
fi
echo "[setup-venv] 干净：无残留 triton"

# 3. 装被测编译器（官方修复命令顺序：先装 triton-ascend，不再单独装 triton）
# --timeout/--retries：pip 默认无限重试，网络不通时会静默空转十几分钟且无任何
# 输出（实测 14 分钟仅 12 秒 CPU、缓存零增长），加了才会快速失败并留下信号。
echo "[setup-venv] 安装被测编译器 ..."
pip install --timeout 30 --retries 2 --upgrade pip
pip install --no-deps --timeout 30 --retries 2 -r "${REQ_COMPILER}"

# 4. 装评测工具链（不含 triton）
echo "[setup-venv] 安装评测工具链 ..."
pip install --timeout 30 --retries 2 -r "${REQ_EVAL}"

# 5. 自检
echo "[setup-venv] 自检 ..."
python - <<'PY'
import sys
import triton

print("  python           =", sys.executable)
print("  triton.__version__ =", triton.__version__)
print("  triton.__file__    =", triton.__file__)

venv_prefix = sys.prefix
if not triton.__file__.startswith(venv_prefix):
    raise SystemExit(
        f"错误：triton 未从当前 venv 加载（{triton.__file__} 不在 {venv_prefix} 下）"
    )
print("  加载路径位于当前 venv：OK")
PY

echo "[setup-venv] 完成。使用前请执行："
echo "    source env/env.sh && source ${VENV_DIR}/bin/activate"
