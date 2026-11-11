#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""每项能力独立进程，保存真实编译失败而不影响适配实现测试。"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path
import logging
from triton_eval.console import emit


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
PROBES = ("health", "gather_float32", "gather_int32")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=Path("runs/ascend-capabilities"))
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--require-pass", action="store_true")
    parser.add_argument("--health-only", action="store_true")
    parser.add_argument("--child", choices=PROBES, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.child:
        from ascend_capability_kernels import run

        import torch
        import triton

        emit(json.dumps({"torch": torch.__version__, "triton": triton.__version__,
                                 "triton_path": triton.__file__}), flush=True)
        run(args.child)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    probes = ("health",) if args.health_only else PROBES
    report = {"python": sys.version, "platform": platform.platform(), "probes": {},
              "not_run": [name for name in PROBES if name not in probes]}
    for name in probes:
        log = args.out / f"{name}.log"
        try:
            with log.open("w", encoding="utf-8") as stream:
                process = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--child", name],
                    cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, timeout=args.timeout)
            status = "PASS" if process.returncode == 0 else "FAIL"
            result = {"status": status, "returncode": process.returncode, "log": str(log)}
        except subprocess.TimeoutExpired:
            result = {"status": "TIMEOUT", "log": str(log)}
        report["probes"][name] = result
        emit(f"{name}: {result['status']}", flush=True)
        (args.out / "capabilities.json").write_text(
            json.dumps(report, indent=2) + "\n", encoding="utf-8")
    required = probes if args.require_pass else ("health",)
    return int(any(report["probes"][name]["status"] != "PASS" for name in required))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
