# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""运行时篡改校验的测试。

守护什么：
    ApiGuard 是对抗「换写法绕过静态扫描」的最终防线（ADV-D6/D7 攻防
    结论：别名导入、链式属性、setattr 全部能过静态）。
    身份比对不看写法只看结果——这里验证的是「检出—恢复」闭环本身：
    假阳性（正常状态报警）与漏恢复（restore 之后仍是假对象）都不可接受。
"""

from __future__ import annotations

import pytest

from triton_eval.integrity import (
    CRITICAL_APIS,
    ApiGuard,
    check_identity,
    module_identity,
)

pytestmark = pytest.mark.torch


def test_critical_api_list_is_wellformed() -> None:
    """名单格式合法且非空——这是防线存在的前提。"""
    assert CRITICAL_APIS
    for module_name, attr_path in CRITICAL_APIS:
        assert module_name and attr_path
        assert not attr_path.startswith(".") and not attr_path.endswith(".")


def test_snapshot_reports_clean_initially() -> None:
    guard = ApiGuard.snapshot()
    assert guard.verify() == [], "干净环境不应报警（假阳性会毁掉整道防线）"
    assert guard.covered, "至少应守护到一批接口"


def test_key_entries_are_actually_covered() -> None:
    """钉住关键条目：名单写错模块名（如 torch.Tensor 当模块）会被静默跳过，
    没有这条测试，整类守护可能在无声中消失。
    """
    import importlib.util

    guard = ApiGuard.snapshot()
    covered = set(guard.covered)
    required = [
        "time.perf_counter", "torch.softmax", "torch.matmul",
        "torch.Tensor.exp", "torch.Tensor.to", "torch.Tensor.item",
        "torch.nn.functional.softmax",
    ]
    if importlib.util.find_spec("triton"):
        required += ["triton.jit", "triton.testing.do_bench"]
    for name in required:
        assert name in covered, f"关键接口 {name} 未被守护（名单写错或被静默跳过）"


def test_replaced_function_is_detected_and_restored(monkeypatch: pytest.MonkeyPatch) -> None:
    """ADV-D7 的正对：import 期把 torch.softmax 换成假实现。"""
    import torch

    guard = ApiGuard.snapshot()
    original = torch.softmax

    def fake_softmax(x, *a, **k):  # pragma: no cover - 不会被调用
        return x

    monkeypatch.setattr(torch, "softmax", fake_softmax, raising=True)

    tampered = guard.verify()
    assert tampered, "被替换的接口必须被检出"
    assert any("torch.softmax" in t for t in tampered)

    restored = guard.restore()
    assert torch.softmax is original, "恢复后必须是原对象（身份级，非等值）"
    assert guard.verify() == []
    assert "torch.softmax" in " ".join(restored)


def test_deleted_attribute_is_detected(monkeypatch: pytest.MonkeyPatch) -> None:
    import time as time_mod

    guard = ApiGuard.snapshot()
    original = time_mod.perf_counter
    monkeypatch.delattr(time_mod, "perf_counter", raising=True)

    tampered = guard.verify()
    assert any("time.perf_counter" in t and "删除" in t for t in tampered)

    monkeypatch.setattr(time_mod, "perf_counter", original, raising=False)
    assert guard.verify() == []


def test_tensor_class_method_guard(monkeypatch: pytest.MonkeyPatch) -> None:
    """类属性篡改（torch.Tensor.exp = ...）影响所有实例，必须同样检出。"""
    import torch

    guard = ApiGuard.snapshot()
    original = torch.Tensor.exp

    def fake_exp(self):  # pragma: no cover
        return self

    monkeypatch.setattr(torch.Tensor, "exp", fake_exp, raising=True)

    tampered = guard.verify()
    assert any("torch.Tensor.exp" in t for t in tampered), \
        "类属性替换必须被检出（比对链路依赖张量方法）"

    guard.restore()
    assert torch.Tensor.exp is original
    assert guard.verify() == []


def test_npu_api_guarded_when_available() -> None:
    """有 torch_npu 时，同步/事件接口必须在守护名单内且可解析。"""
    import importlib.util

    if importlib.util.find_spec("torch_npu") is None:
        pytest.skip("需要 torch_npu（仅昇腾环境）")
    import torch_npu  # noqa: F401
    import torch

    guard = ApiGuard.snapshot()
    # 名单声明的 torch.npu 条目都已纳入守护
    npu_entries = [name for name in guard.covered if name.startswith("torch.npu.")]
    assert npu_entries, "torch.npu.synchronize / Event 应被守护"
    assert "torch.npu.synchronize" in npu_entries


def test_public_coverage_keeps_synthetic_npu_identity_and_restoration(monkeypatch):
    import sys
    from types import ModuleType

    module = ModuleType("torch.npu")

    def synchronize():
        return None

    module.synchronize = synchronize
    module.Event = object
    monkeypatch.setitem(sys.modules, "torch.npu", module)
    guard = ApiGuard.snapshot()
    assert {"torch.npu.synchronize", "torch.npu.Event"} <= set(guard.covered)
    module.synchronize = object()
    assert "torch.npu.synchronize" in guard.verify()
    assert "torch.npu.synchronize" in guard.restore()
    assert module.synchronize is synchronize
    assert guard.verify() == []


def test_check_identity_matching_and_mismatch() -> None:
    """子进程模块身份 vs 环境指纹：一致沉默、不一致点名。"""
    ident = module_identity()
    assert set(ident) >= {"torch"}

    matching = {"torch_version": ident["torch"]["version"]}
    assert check_identity(matching) == []

    mismatches = check_identity({"torch_version": "0.0.0-definitely-not"})
    assert mismatches and "torch" in mismatches[0]


def test_check_identity_tolerates_missing_fields() -> None:
    """指纹侧缺值（如本机无 torch_npu 记录）不得制造假警报。"""
    assert check_identity({}) == []
    assert check_identity({"torch_version": ""}) == []
