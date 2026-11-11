#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""优化手法原子的收益实证跑批（真机专用；无 NPU 的机器直接退出）。

对每对探针（朴素 vs 手法）：
  1. torch 对拍——两版必须算对同一道题，否则实证无效（测「少干活」
     不是测优化）；
  2. 同口径计时（warmup 后重复取中位数；scan 探针 rep=1——朴素版是
     串行本质，秒级一次）；
  3. 判定：手法版快 ≥ 1.5 倍 → 该原子「收益已实证」；否则降级为
     「声明性考察」。达标门槛见 corpus/v0.5-cov20-frozen/probes/README.md。

平台适配（2026-10-13 真机实测）：
  - triton-ascend 的 1 维 grid 上限 < 65536——脚本自动设
    TRITON_ALL_BLOCKS_PARALLEL=1（运行时自提示的解法）绕开；
  - 逐形状隔离：单个形状炸（grid/编译/设备异常）只废该形状的实证，
    不拖垮整批——其余原子照样出具结论。

输出：probes-report.json（机器可读）+ probes-report.md（人读），
写入 --out 目录（随 remote-verify 跑时入验证目录、自动归档）。
结论供覆盖率披露引用（四类分列：正确性负载 /
性能负载（已实证）/ 性能负载（声明性）/ 预留）。

用法（真机）：
    source scripts/env-bootstrap.sh
    python scripts/probe_validation.py [--device npu:0] [--out <目录>]
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import statistics
import sys
import time
from pathlib import Path
import logging
from triton_eval.console import emit
from triton_eval.envprobe.collect import kernel_dependencies_available

# triton-ascend 的 1 维 grid 上限 < 65536（真机 2026-10-13 实测：朴素
# GEMM 探针 (1024,1024)×64 = 恰 65536 个 program，压线即炸
# "grid should be less than 65536!"——运行时自提示用本变量绕开）。
# 朴素探针"每输出元素一个 program"的形态天然产生大 grid，必须解开。
os.environ.setdefault("TRITON_ALL_BLOCKS_PARALLEL", "1")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROBES_DIR = PROJECT_ROOT / "corpus" / "v0.5-cov20-frozen" / "probes"

# 探针对 → 出具结论的原子 + 载体形状 + 计时次数
# （同载体复用：一对内核出具多条原子的结论）
PROBES = [
    ("tile_gemm", ["FA-PAT-TILE-GEMM", "FA-PAT-VEC-BROADCAST"],
     [(1024, 1024), (3072, 128), (2048, 2048)], 3),
    ("split_k", ["FA-PAT-SPLIT-K"], [(64, 65536)], 3),
    ("static_unroll", ["FA-CTRL-STATIC-UNROLL"], [(64, 512)], 3),
    ("vec_load", ["FA-PAT-VEC-LOAD"], [(65536,), (1048576,)], 3),
    # MASK-TAIL 与 VEC-LOAD 同载体同内核，差异只在形状（非对齐尾块）
    ("vec_load#tail", ["FA-PAT-MASK-TAIL"], [(99971,), (1000, 999)], 3),
    ("multipass_reduce", ["FA-PAT-MULTIPASS-REDUCE"], [(1048576,)], 3),
    ("coalesce_gather", ["FA-PAT-COALESCE-GATHER"], [(1048576,)], 3),
    ("scan_blocked", ["FA-PAT-SCAN-BLOCKED"], [(1048576,)], 1),
]

SPEEDUP_GATE = 1.5     # 手法版须快到的倍数
PASS_RATIO = 0.8       # 达标形状的占比门槛


def _load(name: str):
    path = PROBES_DIR / f"{name.split('#')[0]}.py"
    spec = importlib.util.spec_from_file_location(f"probe_{name.replace('#', '_')}",
                                                  path)
    mod = importlib.util.module_from_spec(spec)
    if not (spec.loader is not None):
        raise ValueError('spec.loader is not None')
    spec.loader.exec_module(mod)
    return mod


def _sync() -> None:
    """显式设备同步：NPU/GPU 的内核启动是异步的——不同步测到的是
    提交开销（约 0.1ms 的常数、与负载无关），2026-10-13 真机第二轮
    全部 ~0.11ms 正是此因，该轮计时数据全部作废。"""
    import torch

    if hasattr(torch, "npu"):
        torch.npu.synchronize()
    elif torch.cuda.is_available():
        torch.cuda.synchronize()


def _time(fn, *args, reps: int, warmup: int = 2) -> float:
    """同评测 standard 档口径：warmup 后重复计时取中位数（毫秒）。

    每次调用后必须同步（内核启动异步；同步本身不计入样本）。"""
    for _ in range(warmup):
        fn(*args)
    _sync()
    if reps == 1:
        t0 = time.perf_counter()
        fn(*args)
        _sync()
        return (time.perf_counter() - t0) * 1000.0
    samples = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn(*args)
        _sync()
        samples.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(samples)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="npu:0")
    ap.add_argument("--out", default=str(PROBES_DIR))
    args = ap.parse_args()

    if not kernel_dependencies_available(args.device):
        return 2

    dev = args.device
    results: dict[str, dict] = {}
    for name, atoms, shapes, reps in PROBES:
        try:
            mod = _load(name)
        except Exception as exc:  # noqa: BLE001
            emit(f"[{name}] ✗ 探针加载失败：{type(exc).__name__}: {exc}",
                         file=sys.stderr)
            for a in atoms:
                results.setdefault(a, {"shapes": []})["shapes"].append(
                    {"shape": None, "valid": False,
                     "note": f"探针加载失败: {type(exc).__name__}"})
            continue
        _run_probe_shapes(name, atoms, shapes, reps, mod, dev, results)

    _write_probe_report(results, args, dev)

    return 0




def _run_probe_shapes(name, atoms, shapes, reps, mod, dev, results):
    import torch

    for shape in shapes:
        # 逐形状隔离：一个形状炸（grid 上限 / 编译器缺陷 / 设备异常）
        # 只废该形状的实证，不拖垮整批——其余原子照样出具结论
        try:
            inputs = mod.make_inputs(shape, device=dev)
            # 1. 对拍：两版都必须与 torch 参考一致（探针有效性前提）。
            #    容差按探针模块的 CROSS_TOL 查表（GEMM 族因长 K 求和
            #    顺序噪声文档化放宽到 5e-3，逐探针注明理由；默认 1e-4）
            rtol, atol = getattr(mod, "CROSS_TOL", (1e-4, 1e-4))
            ref = mod.torch_ref(*inputs).to(torch.float32)
            out_n = mod.run_naive(*inputs).to(torch.float32)
            out_p = mod.run_pattern(*inputs).to(torch.float32)
            if not _probe_outputs_match((out_n, out_p, ref), (rtol, atol)):
                emit(f"[{name}] ✗ 对拍失败（形状 {shape}）——"
                     f"探针无效，该形状不计入实证", file=sys.stderr)
                _record_invalid_probe(results, atoms, shape)
                continue
            # 2. 计时
            ms_n = _time(mod.run_naive, *inputs, reps=reps)
            ms_p = _time(mod.run_pattern, *inputs, reps=reps)
        except Exception as exc:  # noqa: BLE001
            emit(f"[{name}] ✗ 形状 {shape} 执行失败（不计入实证）："
                         f"{type(exc).__name__}: {str(exc)[:120]}", file=sys.stderr)
            _record_failed_probe(results, atoms, shape, exc)
            continue
        speedup = ms_n / ms_p if ms_p > 0 else float("inf")
        ok = speedup >= SPEEDUP_GATE
        emit(f"[{name}] {shape}: naive={ms_n:.3f}ms "
                     f"pattern={ms_p:.3f}ms speedup={speedup:.2f}x "
                     f"{'✓' if ok else '✗ 未达 1.5x'}")
        for a in atoms:
            results.setdefault(a, {"shapes": []})["shapes"].append(
                {"shape": list(shape), "valid": True,
                 "naive_ms": round(ms_n, 4),
                 "pattern_ms": round(ms_p, 4),
                 "speedup": round(speedup, 3), "pass": ok})


def _record_failed_probe(results, atoms, shape, exc):
    for a in atoms:
        results.setdefault(a, {"shapes": []})["shapes"].append(
            {"shape": list(shape), "valid": False,
             "note": f"{type(exc).__name__}: {str(exc)[:80]}"})


def _probe_outputs_match(outputs, tolerances):
    import torch

    out_n, out_p, reference = outputs
    rtol, atol = tolerances
    return (torch.allclose(out_n, reference, rtol=rtol, atol=atol)
            and torch.allclose(out_p, reference, rtol=rtol, atol=atol))


def _record_invalid_probe(results, atoms, shape):
    for atom in atoms:
        results.setdefault(atom, {"shapes": []})["shapes"].append(
            {"shape": list(shape), "valid": False})



def _write_probe_report(results, args, dev):
    # 3. 逐原子判定
    report_atoms = {}
    for atom, data in results.items():
        valid = [s for s in data["shapes"] if s.get("valid")]
        ratio = (sum(1 for s in valid if s["pass"]) / len(valid)) \
            if valid else 0.0
        report_atoms[atom] = {
            "shapes": data["shapes"],
            "pass_ratio": round(ratio, 3),
            "verdict": ("实证达标" if ratio >= PASS_RATIO and valid
                        else "声明性考察（未通过收益实证）"
                        if valid else "实证无效（探针对拍失败或形状不可执行）"),
        }

    out_dir = Path(args.out)
    (out_dir / "probes-report.json").write_text(
        json.dumps({"gate": {"speedup": SPEEDUP_GATE, "pass_ratio": PASS_RATIO},
                    "atoms": report_atoms}, ensure_ascii=False, indent=2),
        encoding="utf-8")
    lines = ["# 优化手法原子收益实证报告", "",
             f"- 门槛：手法版快 ≥ {SPEEDUP_GATE}×，达标形状占比 ≥ {PASS_RATIO:.0%}",
             f"- 设备：{dev}", "", "| 原子 | 达标占比 | 结论 |", "|---|---|---|"]
    for atom, r in sorted(report_atoms.items()):
        lines.append(f"| `{atom}` | {r['pass_ratio']:.0%} | {r['verdict']} |")
    lines.append("")
    lines.append("逐形状明细见 probes-report.json。")
    (out_dir / "probes-report.md").write_text("\n".join(lines), encoding="utf-8")
    emit(f"\n已写入 {out_dir}/probes-report.{{json,md}}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
