# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""负向验证：确认系统真能检出错误，而不只是「什么都能过」。

存在意义：
    「全 PASS」只能证明链路通了，不能证明判定是对的。
    如果一套判定逻辑对算错的 kernel 也判 PASS，那么全 PASS 就是假象。

新契约下的负向验证分两层：
    1. 提交合规层：作弊实现必须被静态校验拒绝（不进入执行）
    2. 执行判定层：能跑但算错的实现必须判 WRONG_RESULT

这些用例放在 tests/npu_fixtures/，不属于冻结 corpus，不参与评分。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from triton_eval.models import CaseSource, CaseSpec, CaseStatus
from triton_eval.specload import load_tolerance
from triton_eval.validator import validate_submission

pytestmark = pytest.mark.npu

FIXTURES = Path(__file__).parents[1] / "npu_fixtures"
CORPUS = Path(__file__).resolve().parents[2] / "corpus" / "v0.3-multi-input-frozen"


def _case(code: str, operator: str) -> CaseSpec:
    return CaseSpec(
        code=code, operator=operator, source=CaseSource.PARAM_VARIANT,
        dtype="float32", shape=[1024], seed=20261008,
    )


# ---------- 提交合规层：静态拒绝 ----------


def test_cheating_submission_rejected_statically() -> None:
    """用 torch 内置 API 代算的实现必须在执行前被拒。"""
    result = validate_submission(FIXTURES / "cheating_submission")
    assert not result.ok, "作弊提交必须被拒绝"
    assert "SUB-STR-001" in [e.rule_id for e in result.errors]


def test_cheating_submission_not_counted() -> None:
    """被判不合规的算子不得计入评分分母。"""
    from triton_eval.models import is_counted

    assert is_counted(CaseStatus.SUBMISSION_INVALID) is False


# ---------- 执行判定层：算错必须检出 ----------


def test_silent_wrong_result_is_detected() -> None:
    """能编译、能运行但结果错 —— 必须判 WRONG_RESULT，绝不能 PASS。

    输入张量个数由 `make_inputs_for` 决定：CaseSpec 只有一个 shape，
    故一律是**单输入**。曾按双输入写参考实现，结果全部以
    `TypeError: ... missing 1 required positional argument` 失败——
    那是测试自身写错，不是被测方出错。
    """
    from triton_eval.harness.npu_exec import execute_case
    from triton_eval.judge import decide

    # 被测实现故意算错：应当是恒等映射，实际放大 2 倍
    def bad_scale(x, **kw):
        return x * 2

    def reference(x, **kw):
        return x

    case = _case("neg-wrong", "Exp")
    m = execute_case(case, impl=bad_scale, reference=reference)

    assert m.compiled, f"应当能编译通过：{m.compile_err}"
    assert m.executed, f"应当能运行：{m.compile_err}"

    status = decide(m, load_tolerance().for_dtype("float32"))
    assert status is CaseStatus.WRONG_RESULT, (
        f"静默错误未被检出！实际 {status.value}，rel={m.max_rel_err}。"
        "若判为 PASS，说明正确性判定失效。"
    )


def test_missing_implementation_is_submission_invalid() -> None:
    """提交中没有该算子的实现 → 提交不合规，而非编译失败。"""
    from triton_eval.harness.npu_exec import execute_case
    from triton_eval.judge import decide

    case = _case("neg-nosub", "Exp")
    m = execute_case(case, impl=None, reference=lambda x: x)

    status = decide(m, load_tolerance().for_dtype("float32"))
    assert status is CaseStatus.SUBMISSION_INVALID, (
        f"期望 SUBMISSION_INVALID，实际 {status.value}"
    )


def test_wrong_result_is_flagged_as_severe() -> None:
    from triton_eval.harness.npu_exec import execute_case
    from triton_eval.judge import build_record, decide
    from triton_eval.models import is_severe

    def bad_scale(x, **kw):
        return x * 2

    m = execute_case(_case("neg-sev", "Exp"), impl=bad_scale,
                     reference=lambda x, **kw: x)
    status = decide(m, load_tolerance().for_dtype("float32"))
    rec = build_record(run_id=0, measurement=m, status=status)
    assert is_severe(rec.status)
    assert rec.severe is True


def test_positive_control_still_passes() -> None:
    """阳性对照：正确实现仍应判 PASS。

    只有「负向被检出」且「正向仍通过」同时成立，
    才能说明判定既不过松也不过严。
    """
    from triton_eval.harness.npu_exec import execute_case
    from triton_eval.judge import decide

    def identity(x, **kw):
        return x

    case = _case("pos-ok", "Exp")
    m = execute_case(case, impl=identity, reference=identity)
    status = decide(m, load_tolerance().for_dtype("float32"))
    assert status is CaseStatus.PASS, f"阳性对照失败：{status.value} err={m.compile_err}"
