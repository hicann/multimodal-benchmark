# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""计时（L1）——只能在昇腾执行机上运行。

只负责「调用 do_bench 并取回原始样本」，统计分析在 stats.py（L0）。

关键点：do_bench 默认 return_mode="mean"，规范要的是中位数，
必须显式指定，否则口径与社区不一致且对离群点敏感。
"""

from __future__ import annotations

from typing import Callable

import triton.testing as tt


def measure(
    fn: Callable[[], None],
    repeats: int = 3,
    warmup_ms: int = 25,
    rep_ms: int = 100,
) -> list[float]:
    """重复测量，返回每次的中位耗时（毫秒）作为原始样本。

    每轮 do_bench 内部已经做了 warmup + rep 并返回该轮中位数；
    重复 repeats 轮得到样本数组，供 L0 计算整体中位数与 CV。
    """
    samples: list[float] = []
    for _ in range(max(1, repeats)):
        value = tt.do_bench(fn, warmup=warmup_ms, rep=rep_ms, return_mode="median")
        samples.append(float(value))
    return samples


def measure_baseline(
    fn: Callable[[], None],
    repeats: int = 3,
    warmup_ms: int = 25,
    rep_ms: int = 100,
) -> float | None:
    """测同会话基线（B1：torch eager）。返回样本中位数。"""
    samples = measure(fn, repeats=repeats, warmup_ms=warmup_ms, rep_ms=rep_ms)
    if not samples:
        return None
    ordered = sorted(samples)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0
