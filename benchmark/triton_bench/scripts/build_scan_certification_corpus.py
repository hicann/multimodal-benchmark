#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Freeze a new evaluator release without changing any v17 case or kernel."""
import sys
from pathlib import Path
import logging

import json

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


from triton_eval.pattern_generation import SCAN_REPAIR_VERSION
from triton_eval.pattern_release import SCAN_CORPUS, SCAN_SUBMISSION, build
from triton_eval.console import emit


def main():
    if len(sys.argv) != 1 or sys.version_info < (3, 11):
        raise SystemExit('Use Python 3.11+ with no arguments; the CPU reference gate cannot be skipped.')
    report = build(SCAN_CORPUS, SCAN_SUBMISSION, version=SCAN_REPAIR_VERSION)
    emit(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
