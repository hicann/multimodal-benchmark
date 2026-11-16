#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""采集 do_bench 的真实样本作为测试夹具（L1，需要昇腾主机）。

为什么必须是独立 .py 文件，而不是 `python - <<EOF` 内联脚本：

    @triton.jit 在装饰时会调用 inspect.getsourcelines(fn) 反查源码。
    从 stdin 加载的代码没有源文件，会报
        OSError: could not get source code
    这条已作为「工程约束一」记录在 docs/spec-conflicts.md。

用法：
    python scripts/collect_bench_fixture.py
产出：tests/fixtures/samples_ms.json
"""

from __future__ import annotations

import statistics
from pathlib import Path
import logging

import json


import torch
import torch_npu  # noqa: F401  注册 npu 后端
import triton
import triton.language as tl
import triton.testing as tt
from triton_eval.console import emit

OUT = Path(__file__).resolve().parents[1] / "tests" / "fixtures" / "samples_ms.json"


@triton.jit
def add_kernel(x_ptr, y_ptr, o_ptr, n, block: tl.constexpr):
    """模块级定义：保证 tl 在 fn.__globals__ 中可见。"""
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    tl.store(
        o_ptr + offs,
        tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
        mask=mask,
    )


def main() -> int:
    dev = f"npu:{torch.npu.current_device()}"
    n = 1 << 20
    block = 1024
    x = torch.randn(n, device=dev)
    y = torch.randn(n, device=dev)
    o = torch.empty_like(x)

    def fn() -> None:
        add_kernel[(n // block,)](x, y, o, n, block=block)

    # 重复测量得到样本数组。样本之间天然有波动，这正是不确定性来源，
    # 也是 CV / MAD 存在的意义。
    raw_samples = [
        float(tt.do_bench(fn, warmup=25, rep=100, return_mode="median"))
        for _ in range(5)
    ]
    # 分位数来自同一次测量，仅作参考，不作为样本数组的成员：
    # 不同次测量的 median 与 p50 并不相等（实测 0.0350 vs 0.0406），
    # 混在一起会让统计断言失去意义。
    q20, q50, q80 = tt.do_bench(fn, warmup=25, rep=100, quantiles=(0.2, 0.5, 0.8))

    payload = {
        "device": dev,
        "n": n,
        "block": block,
        "samples_ms": raw_samples,
        "median_ms": float(statistics.median(raw_samples)),
        "quantiles": {"p20": float(q20), "p50": float(q50), "p80": float(q80)},
        "triton_version": triton.__version__,
        "_source": (
            "昇腾主机实测：triton.testing.do_bench(fn, warmup=25, rep=100) 重复 5 次；"
            "quantiles 来自独立一次测量，与 samples_ms 不同源"
        ),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    emit(f"已写入 {OUT}: {payload}")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
