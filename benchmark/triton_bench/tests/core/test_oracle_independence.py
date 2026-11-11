# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""参考实现的独立性（判定权归属）。

守护的是评测的根：
    参考实现是判定的锚。它一旦被替换，「期望值」就由提交方定义，
    整套评测失去意义——被测方只要跟着返回同样的假结果就能拿满分。

    实测（ADV-D7）：提交代码在 import 期执行
        torch.softmax = lambda *a, **k: torch.zeros_like(a[0])
    于是 reference_for("Softmax") 的输出和从 2.0 变成 0.0，
    而这份提交 3 个用例全部 PASS。

已知限制（必须说清，不许含糊）：
    当前是**单进程架构下的降级方案**——探测 torch 顶层关键符号
    是否被重绑定。它能拦住 ADV-D7 这一形态，但拦不住
        - 替换参考实现依赖的其它模块（torch.nn.functional.*）
        - 修改 sys.modules 里的 ref 模块对象
        - 更底层的 monkeypatch
    根治需要**进程隔离**：参考实现在独立解释器中执行、跨进程传张量。
    那是一次架构改动，此处只做当前架构下能做到的最强一层。
"""

from __future__ import annotations

import pytest

from triton_eval.oracle import _detect_module_tamper, reference_for

pytestmark = pytest.mark.torch


def test_clean_environment_not_flagged() -> None:
    """不得误报：干净环境下必须无告警，否则所有用例都会假失败。"""
    assert _detect_module_tamper() is None


def test_tampered_softmax_detected() -> None:
    import torch

    original = torch.softmax
    try:
        def poisoned_softmax(*arguments, **keywords):
            return torch.zeros_like(arguments[0])

        torch.softmax = poisoned_softmax
        found = _detect_module_tamper()
        assert found is not None, "投毒未被探测到——判定权已被接管"
        assert "softmax" in found
    finally:
        torch.softmax = original


def test_tampered_exp_detected() -> None:
    import torch

    original = torch.exp
    try:
        def poisoned_exp(*arguments, **keywords):
            return arguments[0]

        torch.exp = poisoned_exp
        assert _detect_module_tamper() is not None
    finally:
        torch.exp = original


def test_poisoned_reference_refuses_to_judge() -> None:
    """锚点被投毒时必须拒绝判定，而不是拿假参考去判他人。

    这是关键行为：宁可判 ORACLE_FAIL（评测方资产问题），
    也不能用一个已被接管的锚点给出 PASS。
    """
    import torch

    original = torch.softmax
    try:
        def poisoned_softmax(*arguments, **keywords):
            return torch.zeros_like(arguments[0])

        torch.softmax = poisoned_softmax
        with pytest.raises(RuntimeError, match="锚点不可信"):
            reference_for("Softmax")(torch.randn(2, 4))
    finally:
        torch.softmax = original


def test_reference_works_after_restore() -> None:
    """投毒撤销后必须恢复正常——探测不能把好环境永久标记成坏。"""
    import torch

    original = torch.softmax
    try:
        def poisoned_softmax(*arguments, **keywords):
            return torch.zeros_like(arguments[0])

        torch.softmax = poisoned_softmax
        assert _detect_module_tamper() is not None
    finally:
        torch.softmax = original

    assert _detect_module_tamper() is None
    out = reference_for("Softmax")(torch.randn(2, 4))
    assert float(out.sum()) == pytest.approx(2.0, abs=1e-5)


def test_reference_source_is_not_in_submission_dir() -> None:
    """参考实现绝不能来自提交目录——那是自己验证自己。"""
    from pathlib import Path

    from triton_eval.oracle import ref_path

    for op in ("Exp", "Softmax", "Gelu", "Cummin"):
        p = ref_path(op)
        assert p is not None and p.is_file()
        assert "demo-submission" not in str(p)
        assert "adversarial-submissions" not in str(p)
