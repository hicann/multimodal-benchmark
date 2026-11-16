#!/bin/bash
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
TASK_PYTHON="$(command -v python3)"
TASK_STAMP="$($TASK_PYTHON -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))')"
TASK_RECORD="$TASK_ROOT/runs/ascend-suite/$TASK_STAMP"
mkdir -p "$TASK_RECORD"

"$TASK_PYTHON" - "$TASK_RECORD" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

root = Path(sys.argv[1])
commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
(root / "execution.json").write_text(json.dumps({"code_commit": commit, "state": "starting"}) + "\n")
PY

finish() {
    TASK_RUN_RC=$?
    trap - EXIT INT TERM HUP
    "$TASK_PYTHON" - "$TASK_RECORD/execution.json" "$TASK_RUN_RC" <<'PY'
import json
import sys
from pathlib import Path

path = Path(sys.argv[1])
status = json.loads(path.read_text())
status.update(state="finished", returncode=int(sys.argv[2]))
path.write_text(json.dumps(status, indent=2) + "\n")
PY
    if ! "$TASK_PYTHON" scripts/publish_ascend_results.py "$TASK_RECORD"; then
        echo "[suite] 上传失败；补传：python3 scripts/publish_ascend_results.py '$TASK_RECORD'"
        exit 2
    fi
    echo "[suite] 评测退出码：${TASK_RUN_RC}；结果已上传，可直接通知助手查看 GitCode"
    exit "$TASK_RUN_RC"
}

trap finish EXIT
trap 'exit 130' INT
trap 'exit 143' TERM HUP

# Wait for tee to finish before publishing, including when environment startup
# calls exit. A regular pipeline also works where /dev/fd is unavailable.
set +e
(
    set -e
    export ASCEND_VISIBLE_DEVICES="${ASCEND_VISIBLE_DEVICES:-0}"
    source scripts/env-bootstrap.sh
    export PYTHONPATH="$TASK_ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
    export PYTHONUNBUFFERED=1
    python -c 'import sys; print("[suite] Python:", sys.executable)'
    python scripts/run_ascend_suite.py "$@" --out "$TASK_RECORD/evidence"
) 2>&1 | tee -a "$TASK_RECORD/terminal.log"
TASK_PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${TASK_PIPE_RC[0]}" -ne 0 ]; then
    exit "${TASK_PIPE_RC[0]}"
fi
exit "${TASK_PIPE_RC[1]}"
