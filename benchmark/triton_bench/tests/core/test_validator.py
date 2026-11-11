# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""三阶段校验测试（L0，无 NPU 依赖）。"""

from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path

import pytest

from triton_eval.submission import MANIFEST_NAME
from triton_eval.validator import (
    STAGE_EXEC,
    STAGE_LOAD,
    STAGE_STATIC,
    ValidationResult,
    list_rules,
    validate_exec_output,
    validate_submission,
)

# 合法实现：真实 triton kernel，不调内置
LEGAL_KERNEL = '''
import torch, triton
import triton.language as tl

@triton.jit
def _k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    m = offs < n
    tl.store(o_ptr + offs, tl.load(x_ptr + offs, mask=m), mask=m)

def {entry}({params}):
    return x
'''


def _make_sub(tmp_path: Path, name: str, manifest: str,
              files: dict[str, str]) -> Path:
    root = tmp_path / name
    root.mkdir(parents=True)
    (root / MANIFEST_NAME).write_text(manifest, encoding="utf-8")
    for rel, content in files.items():
        p = root / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
    return root


# ---------- 任务 5.1：规则编号与失败结果 ----------


def test_rules_have_ids_and_results() -> None:
    rules = list_rules()
    # 不写死条数：新增规则是常态，写死只会让每次扩规则都要改测试。
    # 真正要保证的是「每条都有编号与失败结果说明」。
    assert len(rules) >= 15
    for rid, desc in rules:
        assert rid.startswith("SUB-")
        assert desc, f"{rid} 缺少失败结果说明"


def test_rule_ids_unique() -> None:
    ids = [r for r, _ in list_rules()]
    assert len(ids) == len(set(ids))


# ---------- 任务 5.2：阶段顺序与短路 ----------


def test_static_failure_short_circuits_load(tmp_path: Path) -> None:
    """静态失败不得进入加载阶段。"""
    root = _make_sub(
        tmp_path, "sub-cheat",
        """
submit_id: sub-cheat
operators:
  - name: Softmax
    entry: softmax
    path: operators/softmax/kernel.py
""",
        {"operators/softmax/kernel.py": "def softmax(x, dim=-1):\n    return torch.softmax(x, dim)\n"},
    )
    result = validate_submission(root)
    assert not result.ok
    stages = {e.stage for e in result.errors}
    assert STAGE_STATIC in stages
    assert STAGE_LOAD not in stages, "静态失败后不应进入加载阶段"
    assert "SUB-STR-001" in [e.rule_id for e in result.errors]


def test_legal_submission_passes_all_stages(tmp_path: Path) -> None:
    root = _make_sub(
        tmp_path, "sub-ok",
        """
submit_id: sub-ok
operators:
  - name: Exp
    entry: exp
    path: operators/exp/kernel.py
""",
        {"operators/exp/kernel.py":
         "def exp(x, base=-1.0, scale=1.0, shift=0.0):\n    return x\n"},
    )
    result = validate_submission(root)
    assert result.ok, result.render()


# ---------- 任务 5.3：合法 / 非法样例 ----------


@pytest.mark.skipif(
    find_spec("torch") is None,
    reason="加载阶段需要 torch；纯 L0 环境只做静态校验",
)
def test_invalid_signature_count_rejected(tmp_path: Path) -> None:
    root = _make_sub(
        tmp_path, "sub-sig",
        """
submit_id: sub-sig
operators:
  - name: Exp
    entry: exp
    path: operators/exp/kernel.py
""",
        {"operators/exp/kernel.py": "def exp(x):\n    return x\n"},
    )
    result = validate_submission(root)
    assert "SUB-LOD-006" in [e.rule_id for e in result.errors]
    detail = result.errors[0].detail
    assert "期望" in detail and "实际" in detail, "必须同时给出期望与实际"


@pytest.mark.skipif(
    find_spec("torch") is None,
    reason="加载阶段需要 torch；纯 L0 环境只做静态校验",
)
def test_invalid_signature_names_rejected(tmp_path: Path) -> None:
    root = _make_sub(
        tmp_path, "sub-sig2",
        """
submit_id: sub-sig2
operators:
  - name: Exp
    entry: exp
    path: operators/exp/kernel.py
""",
        {"operators/exp/kernel.py":
         "def exp(a, b, c, d):\n    return a\n"},
    )
    result = validate_submission(root)
    assert "SUB-LOD-007" in [e.rule_id for e in result.errors]


def test_load_stage_skipped_without_torch(tmp_path: Path) -> None:
    """本地无 torch 时不应进入加载阶段。

    否则会把「环境缺依赖」误报成「提交不可调用」
    —— 那会让本地 L0 开发完全无法进行。
    """
    import importlib.util

    if importlib.util.find_spec("torch") is not None:
        pytest.skip("当前环境有 torch，不适用")

    root = _make_sub(
        tmp_path, "sub-notorch",
        """
submit_id: sub-notorch
operators:
  - name: Exp
    entry: exp
    path: operators/exp/kernel.py
""",
        {"operators/exp/kernel.py": "def exp(x):\n    return x\n"},
    )
    result = validate_submission(root)
    assert all(e.stage == STAGE_STATIC for e in result.errors), (
        "无 torch 时不得产生加载阶段的失败"
    )


def test_unknown_operator_rejected(tmp_path: Path) -> None:
    root = _make_sub(
        tmp_path, "sub-unknown",
        """
submit_id: sub-unknown
operators:
  - name: NotAnOperator
    entry: not_an_operator
    path: operators/x/kernel.py
""",
        {"operators/x/kernel.py": "def not_an_operator(x):\n    return x\n"},
    )
    result = validate_submission(root)
    assert "SUB-LOD-002" in [e.rule_id for e in result.errors]


def test_missing_manifest_reports_filename(tmp_path: Path) -> None:
    root = tmp_path / "sub-nomanifest"
    root.mkdir()
    result = validate_submission(root)
    assert "SUB-LOD-001" in [e.rule_id for e in result.errors]
    assert MANIFEST_NAME in result.errors[0].detail


# ---------- 任务 5.4：失败定位信息 ----------


def test_error_renders_rule_stage_operator_and_line(tmp_path: Path) -> None:
    root = _make_sub(
        tmp_path, "sub-loc",
        """
submit_id: sub-loc
operators:
  - name: Softmax
    entry: softmax
    path: operators/softmax/kernel.py
""",
        {"operators/softmax/kernel.py":
         "def softmax(x, dim=-1):\n    return F.softmax(x, dim=dim)\n"},
    )
    result = validate_submission(root)
    text = result.render()
    assert "SUB-STR-001" in text
    assert "Softmax" in text
    assert "静态" in text
    assert "第 2 行" in text


# ---------- 任务 5.5：异常兜底 ----------


def test_unexpected_exception_contained(tmp_path: Path) -> None:
    """校验中任何未预期异常都归入失败，不得终止整体评测。"""
    root = _make_sub(
        tmp_path, "sub-boom",
        """
submit_id: sub-boom
operators:
  - name: Exp
    entry: exp
    path: operators/exp/kernel.py
""",
        {"operators/exp/kernel.py": "def exp(x, **kw):\n    raise RuntimeError('boom')\n"},
    )
    # 导入时不会执行函数体，因此这里验证的是不会抛出到调用方
    result = validate_submission(root)
    assert isinstance(result, ValidationResult)


def test_never_raises_on_broken_source(tmp_path: Path) -> None:
    root = _make_sub(
        tmp_path, "sub-syntax",
        """
submit_id: sub-syntax
operators:
  - name: Exp
    entry: exp
    path: operators/exp/kernel.py
""",
        {"operators/exp/kernel.py": "def broken(:\n"},
    )
    result = validate_submission(root)  # 不抛异常
    assert not result.ok


# ---------- 任务 5.6：确定性 ----------


def test_validation_is_deterministic(tmp_path: Path) -> None:
    root = _make_sub(
        tmp_path, "sub-det",
        """
submit_id: sub-det
operators:
  - name: Softmax
    entry: softmax
    path: operators/softmax/kernel.py
""",
        {"operators/softmax/kernel.py": "def softmax(x, dim=-1):\n    return torch.softmax(x, dim)\n"},
    )
    a = validate_submission(root)
    b = validate_submission(root)
    assert [e.rule_id for e in a.errors] == [e.rule_id for e in b.errors]
    assert [e.detail for e in a.errors] == [e.detail for e in b.errors]
    assert a.ok == b.ok


# ---------- 执行阶段：输出校验 ----------


def test_exec_output_non_contiguous_is_structural_failure() -> None:
    """非连续输出在布局类之外属结构性失败，不是精度问题。"""
    import importlib.util

    from triton_eval.operator_registry import by_name

    if importlib.util.find_spec("torch") is None:
        pytest.skip("运行期检测需要 torch；纯 L0 环境无法验证")

    import torch

    spec = by_name("Exp")
    result = ValidationResult(submit_id="x")

    # 真实张量但非连续：应命中 SUB-EXE-002（结构性失败）
    base = torch.randn(4, 8)
    non_contig = base.t()

    validate_exec_output("Exp", non_contig, spec, result)
    assert result.errors
    assert result.errors[0].rule_id == "SUB-EXE-002"
    assert result.errors[0].stage == STAGE_EXEC


def test_exec_output_contiguous_passes() -> None:
    import importlib.util

    from triton_eval.operator_registry import by_name

    if importlib.util.find_spec("torch") is None:
        pytest.skip("运行期检测需要 torch；纯 L0 环境无法验证")

    import torch

    spec = by_name("Exp")
    result = ValidationResult(submit_id="x")
    validate_exec_output("Exp", torch.randn(4, 8), spec, result)
    assert result.errors == []


def test_exec_output_fake_tensor_detected() -> None:
    import importlib.util

    from triton_eval.operator_registry import by_name

    if importlib.util.find_spec("torch") is None:
        pytest.skip("运行期检测需要 torch；纯 L0 环境无法验证")

    spec = by_name("Exp")
    result = ValidationResult(submit_id="x")

    class Wrapper:
        pass

    validate_exec_output("Exp", Wrapper(), spec, result)
    assert result.errors
    assert result.errors[0].rule_id == "SUB-EXE-001"
