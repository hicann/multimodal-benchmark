#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""f32 tl.dot 与 torch matmul 的精度分歧归因探针（真机诊断，非门禁）。

背景与完整说明见 corpus/v0.5-cov20-frozen/probes/dot_precision.py 的
模块注释（三个融合 GEMM 算子大 K f32 用例 WRONG_RESULT 的归因）。
本脚本加载该探针模块，按 K × 分块 BK 扫描，逐例隔离（单例失败不毁
整批），量化分歧量级——为「按既有机制登记算子级容差覆盖（附归因
证据）」或「调整 demo 内核分块」提供决策数据。探针不是门禁：任何
输出都是合法的观测。

产物写入 --out 指定目录（默认自建 runs/verify-dotprobe-<时间戳>）：
dot-precision-report.json + checks.log；随窗口组合跑时与其它报告共用
一个验证目录，`bash scripts/upload-verify-record.sh <目录>` 一次归档。

用法（真机）：
    source scripts/env-bootstrap.sh
    python scripts/dot_precision_probe.py [--device npu:0] [--out <目录>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from datetime import datetime, timezone
from pathlib import Path
import logging
from triton_eval.console import emit
from triton_eval.envprobe.collect import kernel_dependencies_available

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBE = PROJECT_ROOT / "corpus" / "v0.5-cov20-frozen" / "probes" / "dot_precision.py"

KS = [128, 1024, 2048, 65536]   # 覆盖「通过的小 K」到「失败的大 K」
BKS = [32, 128]                # 分块敏感性：32 与 demo 内核同款，128 对照


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--out", default=None,
                    help="报告目录（默认自建 runs/verify-dotprobe-<时间戳>）")
    args = ap.parse_args()

    if not kernel_dependencies_available(args.device):
        return 2

    spec = importlib.util.spec_from_file_location("probe_dot_precision", PROBE)
    mod = importlib.util.module_from_spec(spec)
    if not (spec.loader is not None):
        raise ValueError('spec.loader is not None')
    spec.loader.exec_module(mod)

    import torch

    dev = args.device
    stamp = datetime.now(timezone.utc).astimezone().strftime("%m%d-%H%M%S")
    if args.out:
        out_dir = Path(args.out)
        if not out_dir.is_absolute():
            out_dir = PROJECT_ROOT / out_dir
    else:
        out_dir = PROJECT_ROOT / "runs" / f"verify-dotprobe-{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    lines = ["[记] f32 tl.dot vs torch matmul 精度分歧探针"
             f"（设备 {dev}，M=64 N=64，种子固定）"]
    _measure_dot_variants(mod, dev, rows, lines)

    (out_dir / "dot-precision-report.json").write_text(
        json.dumps({"device": dev, "M": 64, "N": 64,
                    "context": "三个 f32 大 K 融合用例 WRONG_RESULT 的归因："
                               "triton 分块 dot 与 torch matmul 的求和顺序噪声"
                               "随 K 增长（近零输出被抵消放大）",
                    "rows": rows}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    (out_dir / "checks.log").write_text("\n".join(lines) + "\n",
                                        encoding="utf-8")
    emit(f"\n已写入 {out_dir}/（dot-precision-report.json + checks.log）")
    emit(f"归档：bash scripts/upload-verify-record.sh {out_dir}")
    return 0




def _measure_dot_variants(mod, dev, rows, lines):
    for k in KS:
        for bk in BKS:
            emit(f"[dot] K={k} BK={bk}: 计算中…", flush=True)
            try:
                a, w = mod.make_inputs(k, device=dev)
                ref = mod.torch_ref(a, w)
                out = mod.run_bk(a, w, bk)
                d = (out - ref).abs()
                rel = (d / ref.abs().clamp_min(1e-30)).max().item()
                ab = d.max().item()
                within = bool(rel <= 1e-4 or ab <= 1e-5)
                rows.append({"K": k, "BK": bk, "max_rel": rel, "max_abs": ab,
                             "judge_f32_rel_tol": 1e-4, "judge_f32_abs_tol": 1e-5,
                             "within_default_tolerance": within})
                line = (f"[记] K={k:>6d} BK={bk:>3d}: max_rel={rel:.3e} "
                        f"max_abs={ab:.3e} "
                        f"{'（默认档内）' if within else '（超默认档）'}")
            except Exception as exc:  # noqa: BLE001
                rows.append({"K": k, "BK": bk, "error":
                             f"{type(exc).__name__}: {str(exc)[:120]}"})
                line = (f"[记] K={k:>6d} BK={bk:>3d}: 执行失败 "
                        f"{type(exc).__name__}: {str(exc)[:100]}")
            lines.append(line)
            emit(line, flush=True)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
