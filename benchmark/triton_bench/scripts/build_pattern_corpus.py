#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Preview or freeze the opt-in four-pattern release; never skip reference gates."""
import json
import sys
from pathlib import Path
import logging

import argparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from triton_eval.pattern_release import CORPUS, SUBMISSION, build, preview
from triton_eval.console import emit


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan-only', action='store_true')
    parser.add_argument('--corpus', type=Path, default=CORPUS)
    parser.add_argument('--submission', type=Path, default=SUBMISSION)
    parser.add_argument('--out', type=Path)
    arguments = parser.parse_args()
    if not arguments.plan_only and sys.version_info < (3, 11):
        parser.error('冻结需要项目声明的 Python 3.11+；--plan-only 只生成未认证草案')
    try:
        report = preview() if arguments.plan_only else build(arguments.corpus, arguments.submission)
    except ModuleNotFoundError as exc:
        emit(f'冻结未完成：缺少依赖 {exc.name}；未跳过独立参考门禁。请使用完整开发环境重试。', file=sys.stderr)
        return 2
    text = json.dumps(report, ensure_ascii=False, indent=2) + '\n'
    if arguments.out:
        arguments.out.write_text(text)
    emit(text, end='')
    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
