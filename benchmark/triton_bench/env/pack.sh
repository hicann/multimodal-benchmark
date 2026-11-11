#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 打包项目，用于在没有 SSH 的情况下通过 Web IDE / VSCode 上传到远程主机。
#
# 用法（在本地开发机执行）：
#     bash env/pack.sh
# 产出：~/triton-eval.tar.gz
#
# 远程侧操作：
#     mkdir -p ~/triton-eval && tar xzf triton-eval.tar.gz -C ~/triton-eval
#     cd ~/triton-eval && source env/env.sh
#     source ~/venv/te-compiler/bin/activate && export PYTHONNOUSERSITE=1
#     pip install -e .

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="${HOME}/triton-eval.tar.gz"

echo "[pack] 源  ${PROJECT_ROOT}"
echo "[pack] 目标 ${OUT}（排除 runs/ 与 .git）"

tar czf "${OUT}" \
    -C "$(dirname "${PROJECT_ROOT}")" \
    --exclude 'runs' \
    --exclude '.git' \
    --exclude '__pycache__' \
    --exclude '*.db' \
    --exclude '.venv' \
    --exclude 'venv' \
    "$(basename "${PROJECT_ROOT}")"

ls -lh "${OUT}"
echo "[pack] 完成。上传到远程后解压即可。"
