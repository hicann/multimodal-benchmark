#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""P_geo 归因：区分「launch 开销主导」与「真实优化差距」。

背景：
    实测 P_geo = 0.309（Triton kernel 比 torch eager 慢约 3 倍）。
    这个数字有三种可能解释，对编译器团队的价值完全不同：

      1. launch 开销主导 —— kernel 太小，固定开销占比大，
         放大规模后 P_geo 会向 1 靠近。属于「测量方法问题」，
         不是编译器缺陷，但提示评测应选更大负载。
      2. 基线本身极快 —— torch eager 走 CANN 高度优化的 ATen 算子，
         Triton 难以超越。属于「基线选择问题」。
      3. triton-ascend 代码质量不足 —— 真优化差距，
         这才是值得提交给编译器团队的结论。

方法：
    对同一算子做规模扫描，观察 P_geo 随规模的走势。
      规模增大 → P_geo 上升   ⇒ launch 开销主导（解释 1）
      规模增大 → P_geo 不变   ⇒ 真实差距（解释 2 或 3）

用法（昇腾主机）：
    python scripts/perf_attribution.py
"""

from __future__ import annotations

import statistics
import sys
from pathlib import Path
import logging

import json


import torch
import torch_npu  # noqa: F401
import triton
import triton.language as tl

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from triton_eval.harness.npu_exec import load_module  # noqa: E402
from triton_eval.console import emit

# 规模扫描：从 64K 到 64M 元素
SIZES = [1 << 16, 1 << 18, 1 << 20, 1 << 22, 1 << 24]
WARMUP_MS = 25
REP_MS = 100


def bench(fn, repeats: int = 5) -> float:
    """中位数。"""
    import triton.testing as tt

    samples = [
        float(tt.do_bench(fn, warmup=WARMUP_MS, rep=REP_MS, return_mode="median"))
        for _ in range(repeats)
    ]
    return statistics.median(samples)


def main() -> int:
    dev = f"npu:{torch.npu.current_device()}"
    emit(f"设备: {dev}   （HBM 总容量需足够容纳 64M float32 = 256MB）")
    emit()
    emit(f"{'规模(元素)':>12} {'kernel(ms)':>12} {'eager(ms)':>12} {'加速比':>10}")
    emit("-" * 50)

    # 最小 add kernel 固定在 tests/npu_fixtures/：
    # 契约迁移后 corpus 里不再有 kernel.py（用例只描述数据，不绑实现）。
    kernel_file = PROJECT_ROOT / "tests" / "npu_fixtures" / "bench_kernel.py"
    mod = load_module(kernel_file, "attr_add")

    rows = []
    for n in SIZES:
        try:
            x = torch.randn(n, device=dev)
            y = torch.randn(n, device=dev)
            o = torch.empty_like(x)
        except Exception as exc:  # noqa: BLE001
            emit(f"{n:>12}  内存不足，跳过：{type(exc).__name__}")
            continue

        cfg = {"BLOCK": 1024}
        k_ms = bench(lambda: mod.launch([x, y], o, cfg))
        e_ms = bench(lambda: mod.reference([x, y]))
        sp = e_ms / k_ms if k_ms > 0 else float("nan")
        rows.append({"n": n, "kernel_ms": k_ms, "eager_ms": e_ms, "speedup": sp})
        emit(f"{n:>12} {k_ms:>12.4f} {e_ms:>12.4f} {sp:>10.3f}")

        del x, y, o
        torch.npu.empty_cache()

    if len(rows) < 2:
        emit("\n样本不足，无法归因")
        return 1

    ratio = _report_performance_attribution(rows)

    out = PROJECT_ROOT / "runs" / "perf_attribution.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps({"device": dev, "rows": rows, "ratio": ratio},
                   ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    emit(f"\n明细写入 {out}")
    return 0




def _report_performance_attribution(rows):
    first, last = rows[0], rows[-1]
    emit()
    emit(f"最小规模 {first['n']:>10}  加速比 {first['speedup']:.3f}")
    emit(f"最大规模 {last['n']:>10}  加速比 {last['speedup']:.3f}")

    delta = last["speedup"] - first["speedup"]
    ratio = last["speedup"] / first["speedup"] if first["speedup"] else float("nan")

    emit()
    if ratio > 1.5:
        emit(f"判定：加速比随规模上升 {ratio:.2f} 倍  →  launch 开销主导")
        emit("      （放大负载后 kernel 效率显著提升，属于测量方法问题，")
        emit("       提示评测应选用更大规模的真实负载）")
    elif ratio < 0.7:
        emit(f"判定：加速比随规模下降 {ratio:.2f} 倍  →  规模越大劣势越明显")
        emit("      （可能是内存带宽或访存模式劣势，值得深挖）")
    else:
        emit(f"判定：加速比对规模不敏感（比值 {ratio:.2f}）→  真实优化差距")
        emit("      （与 launch 开销无关，是 triton-ascend 生成代码")
        emit("       与 CANN ATen 算子之间的实际效率差，")
        emit("       这才是应提交给编译器团队的结论）")
    return ratio


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
