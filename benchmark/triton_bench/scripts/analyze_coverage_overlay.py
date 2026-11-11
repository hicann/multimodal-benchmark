#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Overlay repaired operator evidence without rewriting historical frozen reports."""
import json
import sys
from pathlib import Path
import logging

import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


from triton_eval.coverage_overlay import overlay_results
from triton_eval.console import emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--baseline', type=Path, required=True)
    parser.add_argument('--repair', type=Path, required=True)
    parser.add_argument('--baseline-submission', type=Path, required=True)
    parser.add_argument('--repair-submission', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--reanalyze', action='store_true')
    arguments = parser.parse_args()
    report = overlay_results(arguments.baseline, arguments.repair,
                             arguments.baseline_submission, arguments.repair_submission,
                             reanalyze=arguments.reanalyze)
    arguments.out.parent.mkdir(parents=True, exist_ok=True)
    arguments.out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    emit(json.dumps({key: value for key, value in report.items() if key != 'feature_origins'},
                            ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
