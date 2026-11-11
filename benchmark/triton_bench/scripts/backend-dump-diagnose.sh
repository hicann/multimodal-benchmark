#!/bin/bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -euo pipefail
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
if [ "$#" -ne 0 ]; then
  echo "This entry replays only the archived v15 Neg and Matmul compiler inputs." >&2
  exit 2
fi
SOURCE_RECORD="20261026-120532-223940"
RESULTS="${TRITON_EVAL_BACKEND_SOURCE:-$TASK_ROOT/runs/ascend-suite/$SOURCE_RECORD/evidence}"
if [ ! -f "$RESULTS/coverage-validation.json" ]; then
  if [ -n "${TRITON_EVAL_BACKEND_SOURCE:-}" ]; then
    echo "The explicitly selected evidence directory is missing: $RESULTS" >&2
    exit 1
  fi
  git fetch origin '+refs/heads/npu-results:refs/remotes/origin/npu-results'
  SOURCE_REF="$(git rev-parse refs/remotes/origin/npu-results)"
  if ! git cat-file -e "$SOURCE_REF:records/$SOURCE_RECORD/evidence/coverage-validation.json"; then
    echo "The v15 source record has expired from the rolling branch. Set TRITON_EVAL_BACKEND_SOURCE to its archived evidence directory; do not rerun the full suite." >&2
    exit 1
  fi
  mkdir -p "$TASK_ROOT/runs"
  SOURCE_DIR="$(mktemp -d "$TASK_ROOT/runs/backend-diagnostic-source.XXXXXX")"
  git archive "$SOURCE_REF" "records/$SOURCE_RECORD" | tar -x -C "$SOURCE_DIR"
  RESULTS="$SOURCE_DIR/records/$SOURCE_RECORD/evidence"
fi
source scripts/env-bootstrap.sh
python scripts/diagnose_backend_dump.py --results "$RESULTS" --publish
