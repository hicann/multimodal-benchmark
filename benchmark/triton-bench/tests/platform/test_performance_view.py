# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


import math
from triton_eval.performance_view import performance_view


def test_timing_evidence_exposes_partial_and_unstable_without_fabricated_score():
    def row(code, speedup, cv=0.01, status="PASS"):
        return dict(
            case_code=code,
            status=status,
            perf_median_ms=1,
            baseline_median_ms=2,
            speedup=speedup,
            perf_cv=cv,
        )

    rows = [
        row("a", 2),
        row("b", 0.5, 0.06),
        row("missing", None),
        row("nan", math.nan),
        row("failed", 100, status="WRONG_RESULT"),
    ]
    result = performance_view("standard", rows)
    assert result["geometric_speedup"] == 1
    assert result["measured_cases"] == 2
    assert result["missing_cases"] == 2
    assert result["unstable_cases"] == ["b"]
    assert result["cv_gate"] == 0.05
    assert result["cross_task_comparable"] is False


def test_no_timing_is_absent_not_zero():
    assert performance_view("smoke", [])["geometric_speedup"] is None
