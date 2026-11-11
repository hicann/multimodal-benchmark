# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""契约层守卫：路径边界 / 仅限位置参数 / 归因不可伪造。

三条都是「校验器漏检，失败却记在提交方头上」的同型缺陷：
静态阶段没拦住，执行期炸掉，最终以 RUNTIME_FAIL 或 COMPILE_FAIL 呈现，
看起来是提交方（或被测编译器）的问题，实际是评测方的校验缺口。
"""

from __future__ import annotations

from pathlib import Path

import pytest
from triton_eval.fixture_sources import fixture_path

from triton_eval.validator import validate_submission

pytestmark = pytest.mark.torch

ADV = fixture_path(Path(__file__).resolve().parents[2] / "examples" / "adversarial-submissions")


def _ids(name: str) -> list[str]:
    r = validate_submission(ADV / name)
    return [e.rule_id for e in r.errors]


def test_path_escape_rejected() -> None:
    """路径必须落在提交目录内（ADV-E1 实测读到目录外文件仍 3×PASS）。

    快照只扫 root 内部，逃逸文件完全不可见——所以必须在**读取之前**判。
    """
    assert "SUB-LOD-008" in _ids("ADV-E1-path-escape")


def test_positional_only_signature_rejected() -> None:
    """仅限位置参数无法接收关键字属性（ADV-E4 实测判成 RUNTIME_FAIL）。

    参数名与个数都对，但调用必然失败。校验阶段就该拦住，
    不该把校验器的漏检变成提交方的运行期错误。
    """
    assert "SUB-LOD-009" in _ids("ADV-E4-posonly-signature")


def test_posonly_helper_detects_marker() -> None:
    from triton_eval.validator import _positional_only_names

    def f(x, base, /, scale=1.0):
        return x

    def g(x, base=-1.0, scale=1.0):
        return x

    assert _positional_only_names(f) == ["x", "base"]
    assert _positional_only_names(g) == []


def test_attribution_uses_exception_type_not_message() -> None:
    """归因只认异常类型，绝不看 message——文本由提交方书写。

    实测（ADV-F5）：raise RuntimeError("triton compile failed")
    就把提交方的运行期缺陷判成了 COMPILE_FAIL（扣给被测编译器）。
    COMPILE_FAIL 是被测对象的负面指标，证据不足时不得给出。
    """
    # 不直接 import npu_exec：它会 import torch_npu，
    # 于是这条契约测试只能在昇腾机上跑。判定逻辑本身不需要 NPU，
    # 故按源码加载该模块，让它能在任何机器验证。
    from triton_eval.judge import is_compile_error

    # 消息里带 compile，但类型是 RuntimeError → 不是编译失败
    assert is_compile_error(RuntimeError("triton compile failed")) is False
    assert is_compile_error(RuntimeError("SyntaxError in lowering")) is False

    # 假的 CompilationError 类型（模块不属 triton）也不认
    class CompilationError(Exception):
        pass

    assert is_compile_error(CompilationError("boom")) is False


def test_input_fingerprint_detects_inplace_mutation() -> None:
    """就地修改输入会污染判定基准（ADV-F6 实测 5×PASS）。

    指纹只需发现「变了」，不需要定位哪里变了。
    """
    import torch

    from triton_eval.inputs import tensor_fp

    t = torch.randn(8)
    before = tensor_fp(t)
    t.zero_()
    assert tensor_fp(t) != before, "就地修改未被指纹发现"

    u = torch.randn(8)
    assert tensor_fp(u) == tensor_fp(u), "未修改时指纹必须稳定"


def test_mlir_compiler_errors_are_classified_by_type_and_keep_root_diagnostic():
    from triton_eval.judge import exception_detail, is_compile_error

    compiler_error = type("MLIRCompilationError", (Exception,), {"__module__": "triton.backends.ascend.compiler"})
    error = compiler_error("outer compiler context " + "x" * 6000 + " root cause: unsupported sort")
    assert is_compile_error(error)
    assert not is_compile_error(RuntimeError("MLIRCompilationError"))
    assert not is_compile_error(type("MLIRCompilationError", (Exception,), {})("user error"))
    detail = exception_detail(error)
    assert len(detail) == 2000
    assert detail.startswith("MLIRCompilationError:")
    assert detail.endswith("root cause: unsupported sort")
    assert exception_detail(ValueError("short")) == "ValueError: short"
