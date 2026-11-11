# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""比较器不能通过浮点转型或首输出容差掩盖错误。"""

import pytest

from triton_eval.comparison import compare, output_matches
from triton_eval.judge import CaseMeasurement, decide
from triton_eval.models import CaseStatus
from triton_eval.specload import load_tolerance

pytestmark = pytest.mark.torch


def test_int64_above_double_precision_is_exact():
    import torch

    expected = torch.tensor([2 ** 60], dtype=torch.int64)
    actual = expected + 1
    assert compare(actual, expected)[:2] == (float("inf"), float("inf"))
    assert not output_matches(actual, expected, load_tolerance().for_dtype("int64"))


@pytest.mark.parametrize("actual,expected,matched", [
    ([float("inf")], [float("inf")], True),
    ([float("inf")], [-float("inf")], False),
    ([float("nan")], [float("nan")], True),
    ([0.0], [float("nan")], False),
    ([], [], True),
])
def test_special_values(actual, expected, matched):
    import torch

    assert output_matches(torch.tensor(actual), torch.tensor(expected),
                          load_tolerance().for_dtype("float32")) is matched


def test_shape_error_is_distinct_from_numeric_error():
    import torch

    assert compare(torch.ones(2), torch.ones(3))[2] is False
    measurement = CaseMeasurement("a", "Cummin", outputs_match=False,
                                  max_rel_err=0, max_abs_err=0, allclose_ok=True)
    assert decide(measurement, load_tolerance().for_dtype("float32")) == CaseStatus.WRONG_RESULT
    measurement.shape_matches = False
    assert decide(measurement, load_tolerance().for_dtype("float32")) == CaseStatus.SHAPE_MISMATCH


def test_secondary_output_cannot_move_to_cpu():
    import torch
    from triton_eval.validator import check_output

    output = (torch.empty(2, device="meta"), torch.empty(2))
    violations = check_output("Custom", output, "Elementwise", torch.device("meta"))
    assert [violation.rule_id for violation in violations] == ["SUB-EXE-006"]
