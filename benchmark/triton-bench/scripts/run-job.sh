#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# Fixed remote entry point. Configuration is supplied by the local operator.
set -eo pipefail
TASK_DIRECTORY=$1
TASK_PYTHON=$2
TASK_CANN_ENV=$3
cd "$TASK_DIRECTORY"
source "$TASK_CANN_ENV"
set -u
export PYTHONPATH="$TASK_DIRECTORY/src:$TASK_DIRECTORY/scripts"
export PYTHONUNBUFFERED=1
export ASCEND_VISIBLE_DEVICES=0
unset TRITON_EVAL_CHILD_CMD TRITON_EVAL_CASE_TIMEOUT_S TRITON_EVAL_IDLE_MARGIN_S TRITON_EVAL_GRACE_S
exec "$TASK_PYTHON" -m triton_eval.remote_runner --directory "$TASK_DIRECTORY"
