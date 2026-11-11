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
TASK_MODE="${1:-independent}"
if [ "$#" -gt 1 ] || { [ "$TASK_MODE" != normal ] && [ "$TASK_MODE" != independent ]; }; then
    echo '用法：bash scripts/submission-simulation-verify.sh [independent|normal]；仅用于受控隔离真机。' >&2
    exit 2
fi
TASK_PYTHON="$(command -v python3)"
TASK_STAMP="$("$TASK_PYTHON" -c 'from datetime import datetime, timezone; print(datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S-%f"))')"
TASK_RECORD="$TASK_ROOT/runs/submission-simulation/records/$TASK_STAMP"
TASK_WORK="$TASK_ROOT/runs/submission-simulation/work/$TASK_STAMP"
mkdir -p "$TASK_RECORD"

"$TASK_PYTHON" - "$TASK_RECORD" "$TASK_MODE" <<'PY'
import json
import subprocess
import sys
from pathlib import Path

commit = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
(Path(sys.argv[1]) / "execution.json").write_text(json.dumps(dict(
    kind="submission-simulation", mode=sys.argv[2], code_commit=commit, state="starting")) + "\n")
PY

finish() {
    TASK_RUN_RC=$?
    trap - EXIT INT TERM HUP
    "$TASK_PYTHON" - "$TASK_RECORD" "$TASK_RUN_RC" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
path = root / "execution.json"
status = json.loads(path.read_text())
status.update(state="finished", returncode=int(sys.argv[2]),
              worker_paused=(root / "worker-paused.txt").exists())
path.write_text(json.dumps(status, indent=2) + "\n")
PY
    if ! "$TASK_PYTHON" "$TASK_ROOT/scripts/publish_ascend_results.py" "$TASK_RECORD"; then
        printf '[simulation] 上传失败；补传：'
        printf '%q ' "$TASK_PYTHON" "$TASK_ROOT/scripts/publish_ascend_results.py" "$TASK_RECORD"
        printf '\n'
        exit 2
    fi
    echo "[simulation] 退出码：${TASK_RUN_RC}；结果已上传 GitCode npu-results"
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
    python scripts/submission_simulate.py verify \
        --record "$TASK_RECORD" --work "$TASK_WORK" --mode "$TASK_MODE" --acknowledge-isolation
) 2>&1 | tee -a "$TASK_RECORD/terminal.log"
TASK_PIPE_RC=("${PIPESTATUS[@]}")
set -e
if [ "${TASK_PIPE_RC[0]}" -ne 0 ]; then
    exit "${TASK_PIPE_RC[0]}"
fi
exit "${TASK_PIPE_RC[1]}"
