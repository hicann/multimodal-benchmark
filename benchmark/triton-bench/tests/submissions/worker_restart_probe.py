# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""A real worker process terminated by the restart test before device execution."""

from __future__ import annotations

import json
import signal
import sys
from pathlib import Path

import pytest

from test_submission_service_worker import intercept_engine_processes
from triton_eval.submission_service import EvaluationService


def main(arguments):
    root, corpus, identifier, marker_name = arguments
    service = EvaluationService(Path(root), Path(corpus), execute=True)
    marker = Path(marker_name)

    def before_device(current, phase, command):
        if current['id'] != identifier or phase != 'run':
            raise AssertionError('worker intercepted an unexpected task or phase')
        if current['status'] != 'RUNNING':
            raise AssertionError('worker must be RUNNING before device execution')
        temporary = marker.with_suffix('.tmp')
        temporary.write_text(json.dumps(current))
        temporary.replace(marker)
        signal.pause()
        raise AssertionError('worker unexpectedly resumed after termination')

    with pytest.MonkeyPatch.context() as patch:
        intercept_engine_processes(patch, service, before_device)
        service.work_once()


if __name__ == '__main__':
    main(sys.argv[1:])
