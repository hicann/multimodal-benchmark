# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Public timing evidence; task-local scores must not imply leaderboard ranking."""

from __future__ import annotations

import math
from triton_eval.specload import load_spec


def performance_view(profile: str, records: list[dict]) -> dict:
    gate = load_spec().profile(profile).cv_gate
    passed = [row for row in records if row["status"] == "PASS"]

    def valid(row):
        values = [row.get("perf_median_ms"), row.get("baseline_median_ms"), row.get("speedup")]
        return all(
            isinstance(value, (int, float)) and math.isfinite(value) and value > 0
            for value in values
        )

    measured = [row for row in passed if valid(row)]
    unstable = [
        row["case_code"]
        for row in measured
        if row.get("perf_cv") is not None and row["perf_cv"] > gate
    ]
    return dict(
        measured_cases=len(measured),
        passed_cases=len(passed),
        missing_cases=len(passed) - len(measured),
        unstable_cases=unstable,
        cv_gate=gate,
        geometric_speedup=(
            math.exp(sum(math.log(row["speedup"]) for row in measured) / len(measured))
            if measured
            else None
        ),
        score_basis="TASK_LOCAL_FIRST_ALL_PASS",
        cross_task_comparable=False,
    )
