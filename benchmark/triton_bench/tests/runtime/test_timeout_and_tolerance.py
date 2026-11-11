# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""超时不可绕过 + 容差按输出 dtype 查表。

两条都是「判据必须对齐被判定对象」的同型问题：
    - 超时：不能在「能否中断被测代码」上赌，要事后核对实际耗时
    - 容差：不能按输入 dtype 查表，要按**被比对的输出** dtype 查
"""

from __future__ import annotations

import pytest

from triton_eval.judge import TIMEOUT_SLACK, exceeded_timeout
from triton_eval.judge import CaseMeasurement, decide, is_compile_error
from triton_eval.models import CaseStatus
from triton_eval.specload import load_tolerance

pytestmark = pytest.mark.torch

F32 = load_tolerance().for_dtype("float32")


def _m(**kw) -> CaseMeasurement:
    base = {"case_code": "x", "operator": "Exp",
            "max_rel_err": 0.0, "max_abs_err": 0.0}
    base.update(kw)
    return CaseMeasurement(**base)  # type: ignore[arg-type]


def test_timeout_slack_is_tight() -> None:
    """余量必须很小——太宽松就等于没有超时。"""
    assert 1.0 < TIMEOUT_SLACK <= 1.2


def test_elapsed_over_timeout_is_recorded() -> None:
    """impl_ms 必须被记录，否则事后无法发现「吞掉超时异常」。

    实测（ADV-F4）：被测实现 `except BaseException: 再赖 5s`，
    SIGALRM 注入的 TimeoutError 被吞掉，超时窗口失效，结果仍记 PASS。
    可靠的做法是不依赖中断：只要实际耗时超过硬超时就判失败。
    """
    m = _m(impl_ms=75_000.0)  # 硬超时 60s，实际 75s
    assert m.impl_ms is not None
    assert exceeded_timeout(m.impl_ms, 60)
    # 对照：没超时不该误判
    assert not exceeded_timeout(30_000.0, 60)
    assert not exceeded_timeout(None, 60)


def test_tolerance_follows_output_dtype_not_case_dtype() -> None:
    """容差按**参考输出**的 dtype 查表。

    实测（ADV-C1）：cross_entropy_loss 的 case.dtype 是 int32/int64
    （那是 target 的类型），但输出是 fp32 标量。按 case.dtype 取到
    「精确相等」档，于是 rel=7.18e-08 也被判 WRONG——
    更精确的实现反而拿不到分。
    """
    m = _m(output_dtype="float32")
    assert m.output_dtype == "float32"
    # 用 float32 档判定：7e-8 的误差应当通过
    assert decide(_m(output_dtype="float32", max_rel_err=7e-8,
                     max_abs_err=1.2e-7), F32) is CaseStatus.PASS


def test_exact_tolerance_rejects_tiny_float_error() -> None:
    """对照：若真按 int64（精确档）判定，同样的误差应被拒。

    这条证明「取错 dtype」确实会改变结论——不是同一档容差的重复测试。
    """
    table = load_tolerance()
    exact = table.for_dtype("int64")
    assert not (7.18e-08 <= exact.rel and 1.19e-07 <= exact.abs)


def test_compile_error_attribution_requires_triton_module() -> None:
    """归因只认异常类型，且类型必须来自 triton。"""
    assert is_compile_error(RuntimeError("triton compile failed")) is False
    assert is_compile_error(RuntimeError("CompilationError")) is False

    class CompilationError(Exception):
        pass

    assert is_compile_error(CompilationError("x")) is False
