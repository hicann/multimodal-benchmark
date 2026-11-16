#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
if [ "$#" -ne 0 ]; then
  echo "This entry runs only Neg, Matmul, MaskedSoftmax and CrossEntropyLoss." >&2
  exit 2
fi
export TRITON_EVAL_CAPTURE_PASSES=1
export MLIR_ENABLE_DUMP=1
bash "$TASK_ROOT/scripts/coverage-suite-verify.sh" --operators Neg,Matmul,MaskedSoftmax,CrossEntropyLoss
