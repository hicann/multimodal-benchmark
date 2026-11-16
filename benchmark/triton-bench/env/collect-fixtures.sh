#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 在昇腾执行机上采集测试夹具（fixture）。
#
# 用法（已 source env/env.sh 与被测编译器 venv）：
#     bash env/collect-fixtures.sh
#
# 这些原始输出进 Git 后，L0 层的解析逻辑就能在没有 NPU 的机器上做单元测试。
# 环境回收后，这些夹具是「还能继续开发」的保险。
#
# 注意：计时样本采集委托给 scripts/collect_bench_fixture.py。
# 不能在这里用 heredoc 内联写 @triton.jit —— 从 stdin 加载的 kernel
# 没有源文件，inspect.getsourcelines 会失败。

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
FIXTURES="${PROJECT_ROOT}/tests/fixtures"
mkdir -p "${FIXTURES}"

echo "[fixtures] 采集 npu-smi info ..."
npu-smi info > "${FIXTURES}/npu_smi_info.txt"

echo "[fixtures] 采集 npu-smi version ..."
npu-smi version > "${FIXTURES}/npu_smi_version.txt" 2>&1 || \
    echo "(npu-smi version 不支持，写入占位)" > "${FIXTURES}/npu_smi_version.txt"

echo "[fixtures] 采集 CANN 版本 ..."
echo "${ASCEND_HOME_PATH:-<unset>}" > "${FIXTURES}/cann_home.txt"

echo "[fixtures] 采集 pip freeze 快照 ..."
pip freeze > "${FIXTURES}/pip_freeze.txt"

echo "[fixtures] 采集 do_bench 原始样本 ..."
python "${PROJECT_ROOT}/scripts/collect_bench_fixture.py"

echo "[fixtures] 完成，文件位于 ${FIXTURES}"
ls -la "${FIXTURES}"
