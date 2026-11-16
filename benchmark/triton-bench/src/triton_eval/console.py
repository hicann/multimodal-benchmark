# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Logging-backed command output with stable stdout/stderr semantics."""

from __future__ import annotations

import logging
import sys


class _ConsoleHandler(logging.StreamHandler):
    """A broken result/output stream is an error, never a successful emission."""

    def handleError(self, record):
        error = sys.exc_info()[1]
        if error is not None:
            raise error


def emit(*values: object, sep: str = " ", end: str = "\n", file=None, flush: bool = False) -> None:
    """Emit CLI data independently of the embedding application's root logger.

    Each invocation binds its current stream so redirected output, subprocess
    protocols and concurrent callers do not share handlers or change root state.
    StreamHandler flushes every record, including when ``flush`` is false.
    """
    stream = sys.stdout if file is None else file
    handler = _ConsoleHandler(stream)
    handler.setFormatter(logging.Formatter("%(message)s"))
    handler.terminator = end
    logger = logging.Logger("triton_eval.console", level=logging.INFO)
    logger.addHandler(handler)
    logger.info("%s", sep.join(str(value) for value in values))
    if flush:
        handler.flush()
