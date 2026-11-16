#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -eo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
# Reject an unready static corpus before bootstrap, device probes or publishing.
TASK_PREFLIGHT_PYTHON="${TRITON_EVAL_PREFLIGHT_PYTHON:-${TRITON_EVAL_VENV:-$HOME/venv/te-compiler}/bin/python}"
if [ ! -x "$TASK_PREFLIGHT_PYTHON" ]; then
  TASK_PREFLIGHT_PYTHON="$(command -v python3)"
fi
"$TASK_PREFLIGHT_PYTHON" scripts/check_coverage_readiness.py \
  --corpus "$TASK_ROOT/corpus/v0.16-backend-pattern-certification-candidates" \
  --submission "$TASK_ROOT/examples/coverage-v16-ascend-submission"
bash scripts/ascend-suite-verify.sh --profile smoke "$@" \
  --corpus "$TASK_ROOT/corpus/v0.16-backend-pattern-certification-candidates" \
  --submission "$TASK_ROOT/examples/coverage-v16-ascend-submission" \
  --isolate-reference-gates
