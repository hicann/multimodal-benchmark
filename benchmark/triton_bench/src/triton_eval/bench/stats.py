# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""性能统计（L0，纯计算，无 NPU 依赖）。

放在 L0 的意义：把 do_bench 的原始样本交给纯函数做统计，
这样统计逻辑可以在没有 NPU 的机器上用 fixture 完整测试。
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class PerfSummary:
    median_ms: float | None
    mad_ms: float | None
    cv: float | None
    samples: tuple[float, ...]

    @property
    def has_data(self) -> bool:
        return self.median_ms is not None


def median(samples: list[float]) -> float | None:
    if not samples:
        return None
    return float(statistics.median(samples))


def mad(samples: list[float]) -> float | None:
    """中位绝对偏差。比标准差更抗离群点，适合计时这种长尾分布。"""
    if not samples:
        return None
    med = statistics.median(samples)
    return float(statistics.median([abs(s - med) for s in samples]))


def cv(samples: list[float]) -> float | None:
    """变异系数 = 标准差 / 均值。均值为 0 时无定义，返回 None。"""
    if len(samples) < 2:
        return None
    mean = statistics.fmean(samples)
    if mean == 0:
        return None
    return float(statistics.pstdev(samples) / mean)


def speedup(baseline_ms: float | None, impl_ms: float | None) -> float | None:
    if not baseline_ms or not impl_ms:
        return None
    if baseline_ms <= 0 or impl_ms <= 0:
        return None
    return baseline_ms / impl_ms


def summarize(samples: list[float]) -> PerfSummary:
    return PerfSummary(
        median_ms=median(samples),
        mad_ms=mad(samples),
        cv=cv(samples),
        samples=tuple(samples),
    )


def is_stable(cv_value: float | None, gate: float) -> bool:
    """稳定性门禁。样本不足（cv 为 None）时不判定为不稳定，交由调用方处理。"""
    if cv_value is None:
        return True
    return cv_value <= gate


def geometric_mean(values: list[float]) -> float | None:
    """几何平均。对比例型指标无偏，是加速比聚合的正确口径。"""
    usable = [v for v in values if v and v > 0]
    if not usable:
        return None
    return math.exp(statistics.fmean([math.log(v) for v in usable]))
