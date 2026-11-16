#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check all submission entry contracts without compilation or device execution."""
import json
import sys
from pathlib import Path
import logging

import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from triton_eval.corpusmod import load_manifest
from triton_eval.submission_preflight import submission_readiness
from triton_eval.console import emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'corpus/v0.16-backend-pattern-certification-candidates')
    parser.add_argument('--submission', type=Path, default=ROOT / 'examples/coverage-v16-ascend-submission')
    parser.add_argument(
        '--load',
        action='store_true',
        help='also import actual modules; requires target Triton environment')
    parser.add_argument('--out', type=Path)
    args = parser.parse_args()
    manifest = load_manifest(args.corpus / 'manifest.json')
    report = submission_readiness(args.submission, {c.operator for c in manifest.cases}, load=args.load)
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if args.out:
        args.out.write_text(text, encoding='utf-8')
    emit(text, end='')
    return int(not report['ok'])


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
