#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Recompute compilation and execution coverage only from real NPU artifacts."""
import json
import sys
from pathlib import Path
import logging

import argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from triton_eval.coverage_evidence import audit_suite
from triton_eval.console import emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'corpus/v0.16-backend-pattern-certification-candidates')
    parser.add_argument('--results', type=Path, required=True)
    parser.add_argument('--submission', type=Path, default=ROOT / 'examples/coverage-v16-ascend-submission')
    parser.add_argument('--operator-regression', action='store_true')
    args = parser.parse_args()
    report = audit_suite(args.corpus, args.results, args.submission,
                         operator_regression=args.operator_regression)
    (args.results / 'coverage-validation.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    omitted = {'evidence_sha256', 'observed_ir_operations'}
    summary = {key: value for key, value in report.items() if key not in omitted}
    emit(json.dumps(summary, ensure_ascii=False, indent=2))
    return int(not (report['selected_operator_validation_passed'] if args.operator_regression
                    else report['verified_npu_coverage_target_met']))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
