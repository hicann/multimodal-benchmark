# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""执行期兜底校验的测试（L0，无需 NPU）。

守护一条实测教训：
    SUB-EXE-001（伪张量）/ SUB-EXE-002（非连续输出）这两条规则**曾经是死的**——
    规则表里写着，测试里也测着，但 validate_submission 只跑静态+加载，
    真实评测从不调用它们。于是上传一份返回包装器或非连续视图的实现，
    能通过校验并拿到 PASS。

    静态扫描看不见运行期行为，这类不合规只有真跑一遍才暴露。
    本测试锁住「判定函数可被逐用例调用」这一前提，
    链路接入由 test_npu 的端到端用例与 cli 覆盖。
"""

from __future__ import annotations

from unittest.mock import Mock

import pytest

from triton_eval.judge import CaseMeasurement, decide
from triton_eval.models import CaseStatus
from triton_eval.operator_registry import by_name
from triton_eval.specload import load_tolerance
from triton_eval.validator import check_output, validate_exec_output

pytestmark = pytest.mark.torch

F32 = load_tolerance().for_dtype("float32")


def _m(**overrides) -> CaseMeasurement:
    base = {"case_code": "exp-f32-00", "operator": "Exp",
            "max_rel_err": 0.0, "max_abs_err": 0.0}
    base.update(overrides)
    return CaseMeasurement(**base)  # type: ignore[arg-type]


def test_check_output_is_pure_and_reusable() -> None:
    """必须是纯判定：L1 执行器手里没有 ValidationResult，只要一个答案。"""
    import torch

    assert check_output("Exp", torch.randn(4, 8), "Elementwise") == []
    assert check_output("Exp", torch.randn(4, 8), "") == []


def test_non_contiguous_output_is_violation() -> None:
    import torch

    x = torch.randn(4, 8).t()  # 非连续
    errs = check_output("Exp", x, "Elementwise")
    assert [e.rule_id for e in errs] == ["SUB-EXE-002"]


def test_non_contiguous_allowed_for_layout_transform() -> None:
    """布局变换类的核心语义就是搬运，非连续不得误杀。"""
    import torch

    x = torch.randn(4, 8).t()
    assert check_output("Transpose", x, "LayoutTransform") == []


def test_fake_tensor_is_violation() -> None:
    import torch

    class Wrapper:
        """看起来像张量的包装器。"""

        def __init__(self, t):
            self._t = t

        def __getattr__(self, name):
            return getattr(self._t, name)

    errs = check_output("Exp", Wrapper(torch.randn(4)), "Elementwise")
    assert [e.rule_id for e in errs] == ["SUB-EXE-001"]


@pytest.mark.parametrize('nested', [False, True])
def test_real_tensor_subclass_is_accepted_by_instance_policy(nested) -> None:
    import torch

    class TensorSubclass(torch.Tensor):
        pass

    original = torch.randn(4)
    subclass = original.as_subclass(TensorSubclass)
    assert isinstance(subclass, torch.Tensor)
    assert check_output('Exp', original, 'Elementwise') == []
    output = [original, subclass] if nested else subclass
    assert check_output('Exp', output, 'Elementwise') == []


def test_validate_exec_output_delegates_to_check_output() -> None:
    """追加式版本必须复用同一套判定，不能出现两份实现。"""
    import torch

    from triton_eval.validator import ValidationResult

    result = ValidationResult(submit_id="x")
    validate_exec_output("Exp", torch.randn(4, 8).t(),
                         by_name("Exp"),
                         result)
    assert [e.rule_id for e in result.errors] == ["SUB-EXE-002"]


@pytest.mark.parametrize('kind', ['tensor', 'child', 'grandchild', 'parameter'])
@pytest.mark.parametrize('nested', [False, True])
def test_tensor_instance_policy_accepts_real_tensor_variants(kind, nested):
    import torch

    class TensorChild(torch.Tensor):
        pass

    class TensorGrandchild(TensorChild):
        pass

    original = torch.randn(4)
    if kind == 'child':
        output = original.as_subclass(TensorChild)
    elif kind == 'grandchild':
        output = original.as_subclass(TensorGrandchild)
    elif kind == 'parameter':
        output = torch.nn.Parameter(original)
    else:
        output = original
    if nested:
        output = [original, (output,)]
    from triton_eval.anticheat import looks_like_fake_tensor

    assert not looks_like_fake_tensor(output)
    assert check_output('Exp', output, 'Elementwise') == []


@pytest.mark.parametrize('nested', [False, True])
def test_spoofed_tensor_proxy_still_runs_followup_output_checks(nested):
    import torch

    from triton_eval.anticheat import looks_like_fake_tensor

    proxy = Mock(spec_set=torch.Tensor)
    proxy.is_contiguous.side_effect = RuntimeError('Proxy has no tensor storage')
    assert isinstance(proxy, torch.Tensor)
    assert isinstance(proxy, Mock)
    output = [torch.randn(4), (proxy,)] if nested else proxy
    assert not looks_like_fake_tensor(output)
    with pytest.raises(RuntimeError, match='Proxy has no tensor storage'):
        check_output('Exp', output, 'Elementwise')
    proxy.is_contiguous.assert_called_once_with()


@pytest.mark.parametrize('subclassed', [False, True])
@pytest.mark.parametrize('nested', [False, True])
def test_torch_dispatch_failure_propagates_from_followup_output_checks(subclassed, nested):
    import torch
    from torch.overrides import TorchFunctionMode

    from triton_eval.anticheat import looks_like_fake_tensor

    class TensorChild(torch.Tensor):
        pass

    original = torch.randn(4)
    tensor = original.as_subclass(TensorChild) if subclassed else original
    output = [(tensor,)] if nested else tensor
    with TorchFunctionMode():
        assert not looks_like_fake_tensor(output)
        with pytest.raises(NotImplementedError):
            check_output('Exp', output, 'Elementwise')
    assert check_output('Exp', output, 'Elementwise') == []


@pytest.mark.parametrize('nested', [False, True])
def test_accepted_subclass_still_has_layout_and_device_checked(nested):
    import torch

    class TensorSubclass(torch.Tensor):
        pass

    original = torch.randn(4, 8)
    output = original.as_subclass(TensorSubclass)
    checked = [original, output] if nested else output
    spec = by_name('Exp')
    assert check_output('Exp', checked, spec.category, expected_device=torch.device('cpu')) == []
    strided = output.t()
    checked = [original, strided] if nested else strided
    errors = check_output('Exp', checked, spec.category)
    assert [error.rule_id for error in errors] == ['SUB-EXE-002']
    errors = check_output('Exp', output, spec.category, expected_device=torch.device('meta'))
    assert [error.rule_id for error in errors] == ['SUB-EXE-006']


def test_bad_output_judged_as_submission_invalid() -> None:
    """运行期不合规必须优先于精度判定：拿假张量比误差，得出的「通过」毫无意义。"""
    assert decide(_m(output_ok=False), F32) is CaseStatus.SUBMISSION_INVALID
    # 误差明明在容差内，也不得因为「看起来对了」就放过
    assert decide(_m(output_ok=False, max_rel_err=0.0, max_abs_err=0.0),
                  F32) is CaseStatus.SUBMISSION_INVALID


def test_output_guard_precedes_wrong_result() -> None:
    assert decide(_m(output_ok=False, max_rel_err=1e-1), F32) is \
        CaseStatus.SUBMISSION_INVALID
