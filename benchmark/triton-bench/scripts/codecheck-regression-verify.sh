#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

set -eo pipefail
TASK_SUITE="${1:-failed-last}"
if [ "$#" -gt 1 ] || { [ "$TASK_SUITE" != failed-last ] && [ "$TASK_SUITE" != full ]; }; then
    echo 'Usage: bash scripts/codecheck-regression-verify.sh [failed-last|full] (isolated NPU host only)' >&2
    exit 2
fi
TASK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$TASK_ROOT"
TASK_PYTHON="$(command -v python3)"
TASK_STAMP="codecheck-$($TASK_PYTHON -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))')"
TASK_RECORD="$TASK_ROOT/runs/codecheck-regression/records/$TASK_STAMP"
TASK_WORK="$TASK_ROOT/runs/codecheck-regression/work/$TASK_STAMP"
TASK_BOOTSTRAP="$TASK_ROOT/runs/codecheck-regression/$TASK_STAMP.log"
mkdir -p "$(dirname "$TASK_RECORD")"

finish() {
    TASK_RUN_RC=$?
    trap - EXIT INT TERM HUP
    mkdir -p "$TASK_RECORD"
    cp "$TASK_BOOTSTRAP" "$TASK_RECORD/terminal.log"
    "$TASK_PYTHON" - "$TASK_RECORD" "$TASK_RUN_RC" "$TASK_SUITE" <<'STATUS'
import json
import subprocess
import sys
from pathlib import Path
record = Path(sys.argv[1])
path = record / "execution.json"
status = json.loads(path.read_text()) if path.exists() else dict(
    kind="codecheck-regression",
    scope="SUBMISSION_E2E_FAILED_LAST" if sys.argv[3] == "failed-last" else "SUBMISSION_E2E", steps=[],
    code_commit=subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip())
status.update(state="finished", returncode=int(sys.argv[2]))
path.write_text(json.dumps(status, indent=2) + "\n")
STATUS
    if ! "$TASK_PYTHON" "$TASK_ROOT/scripts/publish_ascend_results.py" "$TASK_RECORD"; then
        printf '[quality] Upload failed; retry without rerunning tests: '
        printf '%q ' "$TASK_PYTHON" "$TASK_ROOT/scripts/publish_ascend_results.py" "$TASK_RECORD"
        printf '\n'
        exit 2
    fi
    echo "[quality] Test exit code: $TASK_RUN_RC; results published to npu-results"
    exit "$TASK_RUN_RC"
}
trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP
set +e
(
    set -e
    export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0}"
    source scripts/env-bootstrap.sh
    export PYTHONPATH="$TASK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1
    export NO_PROXY="127.0.0.1,localhost${NO_PROXY:+,$NO_PROXY}"
    export no_proxy="$NO_PROXY"
    python scripts/verify_codecheck_regression.py --record "$TASK_RECORD" --work "$TASK_WORK" \
        --acknowledge-isolation --suite "$TASK_SUITE"
) 2>&1 | tee "$TASK_BOOTSTRAP"
TASK_PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${TASK_PIPE_RC[0]}" -ne 0 ]; then
    exit "${TASK_PIPE_RC[0]}"
fi
exit "${TASK_PIPE_RC[1]}"
