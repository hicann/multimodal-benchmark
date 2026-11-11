#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# ============================================================================
# save-compiler-wheel.sh —— 把被测编译器（triton-ascend）的 wheel 备份到持久目录
#
# 为什么需要：真机每两小时清空一次，且到华为镜像的网络并不可靠。曾经出现过
# pip 联网安装空转 14 分钟仍未完成（累计仅 12 秒 CPU、缓存零增长）的情况，
# 评测被彻底卡在环境自举这一步。把 wheel 备份到不会被清空的目录后，
# env-bootstrap 会优先离线安装，秒级完成且不依赖网络。
#
# 备份目录默认 /opt/atomgit/wheels（在实测机型上 /opt 跨轮保留，$HOME 会被清）。
# 可用 TRITON_EVAL_WHEEL_DIR 覆盖。
#
# 用法：
#   bash scripts/save-compiler-wheel.sh
# 一般由 env-bootstrap.sh 在编译器安装成功后自动调用，无需手工执行。
# ============================================================================
set -uo pipefail

WHEEL_DIR="${TRITON_EVAL_WHEEL_DIR:-/opt/atomgit/wheels}"
mkdir -p "${WHEEL_DIR}" 2>/dev/null || {
    echo "[save-wheel] 无法创建 ${WHEEL_DIR}，跳过备份"
    exit 0
}

# 已经有备份就不重复下载
if ls "${WHEEL_DIR}"/triton_ascend-*.whl >/dev/null 2>&1; then
    echo "[save-wheel] 备份已存在，跳过：$(ls "${WHEEL_DIR}"/triton_ascend-*.whl)"
    exit 0
fi

# 先捡 pip 已解压/缓存的 wheel（本轮刚装过就有了），零网络开销
found=""
for cand in /tmp/pip-unpack-*/triton_ascend-*.whl; do
    [ -f "${cand}" ] && { found="${cand}"; break; }
done

if [ -n "${found}" ]; then
    cp -v "${found}" "${WHEEL_DIR}/"
    echo "[save-wheel] 已从本地缓存备份 -> ${WHEEL_DIR}"
    exit 0
fi

# 否则下载一份（需要网络，失败不影响主流程）
echo "[save-wheel] 本地无缓存，尝试下载 wheel 到 ${WHEEL_DIR} ..."
pip download --no-deps --timeout 30 --retries 2 \
    -d "${WHEEL_DIR}" -r env/requirements-compiler.txt \
    && echo "[save-wheel] 备份完成" \
    || echo "[save-wheel][WARN] 下载失败，暂无离线备份（不影响本轮评测）"
