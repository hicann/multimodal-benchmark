# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""不可信代码边界的兜底测试（L0）。

守护的缺陷（ADV-E3 实测）：
    提交代码在 import 期调用 sys.exit()，整个评测进程**静默消失**——
    无告警、无报告、退出码 0。CI 会把「评测被攻击掉了」当成评测成功。

    这不是判错，是根本没有判定，属于提交闭环最致命的一类。
    根因是 SystemExit 继承 BaseException，所有 except Exception 都拦不住。
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest
from triton_eval.fixture_sources import fixture_path

from triton_eval.safeexec import SubmissionEscape, guard_call, is_escape
from triton_eval.validator import validate_submission

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ADV = fixture_path(PROJECT_ROOT / "examples" / "adversarial-submissions")


def import_exit_failure():
    source = ADV / "ADV-E3-systemexit/operators/exp/kernel.py"
    spec = importlib.util.spec_from_file_location("untrusted_import_exit", source)
    module = importlib.util.module_from_spec(spec)
    with pytest.raises(SubmissionEscape) as failure:
        guard_call(spec.loader.exec_module, module)
    return failure.value


def test_systemexit_becomes_exception() -> None:
    failure = import_exit_failure()
    assert type(failure.origin).__name__ == "SystemExit"
    assert failure.origin.code == 0
    assert failure.__cause__ is failure.origin
    assert not isinstance(failure.origin, Exception)


def test_generator_exit_becomes_exception() -> None:
    with pytest.raises(SubmissionEscape):
        guard_call(lambda: (_ for _ in ()).throw(GeneratorExit()))


def test_keyboard_interrupt_passes_through() -> None:
    """Ctrl-C 必须保留：那是操作者意图，不是提交方行为。"""
    with pytest.raises(KeyboardInterrupt):
        guard_call(lambda: (_ for _ in ()).throw(KeyboardInterrupt()))


def test_normal_call_unaffected() -> None:
    assert guard_call(lambda x: x * 2, 21) == 42


def test_is_escape_recognises_boundary_exceptions() -> None:
    failure = import_exit_failure()
    assert is_escape(failure.origin)
    assert is_escape(GeneratorExit())
    assert is_escape(failure)
    assert not is_escape(KeyboardInterrupt())
    assert not is_escape(ValueError())


def test_validate_reports_escape_instead_of_dying(capsys) -> None:
    """校验必须给出结论并拒绝，而不是让进程消失。"""
    from triton_eval import cli

    rc = cli.main(["validate", "--submission", str(ADV / "ADV-E3-systemexit")])
    assert rc == 1, "越界的提交必须返回非 0"
    out = capsys.readouterr().out
    assert "SUB-LOD-" in out, "必须报出结构化编号，而不是静默退出"
    assert "终止进程" in out, "必须说清是提交方试图终止进程"


def test_cli_never_exits_zero_on_escape() -> None:
    """端到端：真实子进程不得「无输出 + 退出码 0」。"""
    proc = subprocess.run(
        [sys.executable, "-m", "triton_eval", "validate",
         "--submission", str(ADV / "ADV-E3-systemexit")],
        cwd=PROJECT_ROOT, capture_output=True, text=True, timeout=120,
    )
    assert proc.returncode != 0, (
        "退出码 0 意味着 CI 会把被攻击掉的评测当成成功；\n"
        f"stdout={proc.stdout!r}\nstderr={proc.stderr!r}"
    )
    combined = proc.stdout + proc.stderr
    assert "SUB-LOD-" in combined, "必须给出结构化编号"
    assert "终止进程" in combined, "必须说清是提交方试图终止进程"


def test_scatter_params_tensors_before_attrs() -> None:
    """P0-2：参数序必须「张量在前、属性在后」。

    曾经写成 (data, dim, indices, updates, reduce)，执行器位置传张量后
    第 2 个实参绑到 dim，再遇关键字 dim 必然 TypeError——
    该算子在冻结集里没有可行签名（实测 5/5 RUNTIME_FAIL）。
    """
    from triton_eval.operator_registry import ALL_OPERATORS

    for spec in ALL_OPERATORS:
        tensor_names = {i.name for i in spec.inputs}
        seen_attr = False
        for p in spec.params:
            if p in tensor_names:
                assert not seen_attr, (
                    f"{spec.name} 的张量参数 {p!r} 出现在属性之后：{spec.params}"
                )
            else:
                seen_attr = True


def test_validate_param_order_rejects_bad_order() -> None:
    from triton_eval.operator_registry import SCATTER

    SCATTER.validate_param_order()  # 当前应通过

    bad = SCATTER.model_copy(update={"params": ["data", "dim", "indices",
                                                "updates", "reduce"]})
    with pytest.raises(ValueError, match="参数序"):
        bad.validate_param_order()
