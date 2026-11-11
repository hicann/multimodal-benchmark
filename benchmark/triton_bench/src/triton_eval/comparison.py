# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""可在 CPU 独立验证的逐输出比较。"""

from __future__ import annotations

from .judge import within_tolerance


def compare(out, expected) -> tuple[float, float, bool]:
    import torch

    actual = out.detach().to("cpu")
    reference = expected.detach().to("cpu")
    if actual.shape != reference.shape:
        return float("inf"), float("inf"), False
    if not actual.is_floating_point() and not reference.is_floating_point():
        equal = bool(torch.equal(actual, reference))
        return (0.0, 0.0, True) if equal else (float("inf"), float("inf"), True)
    actual = actual.to(torch.float64)
    reference = reference.to(torch.float64)
    same_special = (
        torch.equal(torch.isnan(actual), torch.isnan(reference))
        and torch.equal(torch.isposinf(actual), torch.isposinf(reference))
        and torch.equal(torch.isneginf(actual), torch.isneginf(reference))
    )
    if not same_special:
        return float("inf"), float("inf"), False
    finite = torch.isfinite(reference)
    if not bool(finite.any()):
        return 0.0, 0.0, True
    difference = (actual[finite] - reference[finite]).abs()
    relative = difference / reference[finite].abs().clamp_min(1e-30)
    return float(relative.max()), float(difference.max()), True


def allclose(out, expected, tolerance) -> bool:
    import torch

    actual = out.detach().to("cpu")
    reference = expected.detach().to("cpu")
    if actual.shape != reference.shape:
        return False
    if tolerance.is_exact or (not actual.is_floating_point() and not reference.is_floating_point()):
        return bool(torch.equal(actual, reference))
    actual = actual.to(torch.float64)
    reference = reference.to(torch.float64)
    return bool(torch.allclose(actual, reference, rtol=tolerance.rel,
                               atol=tolerance.abs, equal_nan=True))


def output_matches(out, expected, tolerance) -> bool:
    relative, absolute, special_match = compare(out, expected)
    if not special_match:
        return False
    if tolerance.is_exact:
        return allclose(out, expected, tolerance)
    return within_tolerance(relative, absolute, tolerance) or allclose(
        out, expected, tolerance)
