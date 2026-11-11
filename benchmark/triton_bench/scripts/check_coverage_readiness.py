#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only static >=90% NPU launch gate. Recomputes labels and retention."""
import json
import sys
from pathlib import Path
import logging

import argparse
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from triton_eval.coverage_gate import static_readiness
from triton_eval.console import emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, default=ROOT / 'corpus/v0.16-backend-pattern-certification-candidates')
    parser.add_argument('--submission', type=Path, default=ROOT / 'examples/coverage-v16-ascend-submission')
    args = parser.parse_args()
    from triton_eval.corpusmod import load_manifest
    from triton_eval.submission_preflight import require_submission_ready
    report = static_readiness(args.corpus)
    manifest = load_manifest(args.corpus / 'manifest.json')
    report['submission_preflight'] = require_submission_ready(args.submission,
        {case.operator for case in manifest.cases})
    emit(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    main()
