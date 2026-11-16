#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 评测执行环境入口。
#
# 用法（必须 source，不能直接执行）：
#     source env/env.sh
#
# 职责：
#   1. 注入 CANN 环境变量
#   2. 锁定执行到单张 NPU 卡
#   3. 屏蔽用户级 site-packages，避免残留社区 triton 覆盖被测编译器

# 非 root 安装时 CANN 位于 $HOME/Ascend；root 安装时为 /usr/local/Ascend
if [ -f "${ASCEND_TOOLKIT_HOME:-$HOME/Ascend/ascend-toolkit}/set_env.sh" ]; then
    ASCEND_TOOLKIT_HOME="${ASCEND_TOOLKIT_HOME:-$HOME/Ascend/ascend-toolkit}"
elif [ -f "/usr/local/Ascend/ascend-toolkit/set_env.sh" ]; then
    ASCEND_TOOLKIT_HOME="/usr/local/Ascend/ascend-toolkit"
else
    echo "[env.sh] 错误：未找到 ascend-toolkit/set_env.sh。" >&2
    echo "[env.sh] 请设置 ASCEND_TOOLKIT_HOME，或先安装 CANN。" >&2
    return 1 2>/dev/null || exit 1
fi

# shellcheck disable=SC1091
source "${ASCEND_TOOLKIT_HOME}/set_env.sh"

# 锁定单卡：多卡混跑会让同一用例前后落在不同芯片上，复跑偏差直接爆掉
export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0}"

# 屏蔽 ~/.local/lib/python*/site-packages
# triton 与 triton-ascend 安装目录同名，用户级的社区 triton 会静默覆盖被测对象
export PYTHONNOUSERSITE=1

echo "[env.sh] toolkit      = ${ASCEND_TOOLKIT_HOME}"
echo "[env.sh] ASCEND_HOME  = ${ASCEND_HOME_PATH:-<unset>}"
echo "[env.sh] visible npu  = ${ASCEND_VISIBLE_DEVICES}"
echo "[env.sh] PYTHONNOUSERSITE = ${PYTHONNOUSERSITE}"
