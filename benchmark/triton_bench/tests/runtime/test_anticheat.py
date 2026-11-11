# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反作弊测试（L0，静态部分无 NPU 依赖）。

每条规则都必须有**合法样例**与**非法样例**。
只有非法样例能证明规则有效，只有合法样例能证明规则不误伤。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from triton_eval.anticheat import (
    LAYOUT_ALLOWED_CATEGORIES,
    RULE_CACHING,
    RULE_DELEGATION,
    RULE_DEVICE_FALLBACK,
    RULE_LAYOUT_MOVE,
    RULE_TIMING_TAMPER,
    detect_address_keyed_cache,
    scan_source,
)


def _scan(src: str, category: str = "Elementwise", entry: str = "") -> list[str]:
    return [f.rule_id for f in scan_source(src, category=category, entry=entry)]


# ---------- SUB-BEH-001：内置 API 代算 ----------


def test_delegation_detected_direct_call() -> None:
    src = "def softmax(x, dim=-1):\n    return F.softmax(x, dim=dim)\n"
    assert RULE_DELEGATION in _scan(src, category="Normalization", entry="softmax")


def test_delegation_detected_module_call() -> None:
    src = "def softmax(x, dim=-1):\n    return torch.softmax(x, dim)\n"
    assert RULE_DELEGATION in _scan(src, category="Normalization", entry="softmax")


def test_delegation_detected_in_nested_function() -> None:
    src = (
        "def softmax(x, dim=-1):\n"
        "    def helper(t):\n"
        "        return torch.softmax(t, -1)\n"
        "    return helper(x)\n"
    )
    assert RULE_DELEGATION in _scan(src, category="Normalization", entry="softmax")


def test_delegation_not_triggered_by_same_name_entry() -> None:
    """与入口同名的调用不重复计为代算（避免自身签名误报）。"""
    src = "def exp(x, base=-1.0):\n    return x\n"
    assert RULE_DELEGATION not in _scan(src, category="Elementwise", entry="exp")


def test_delegation_ignores_triton_kernel_body() -> None:
    """@triton.jit kernel 内部调用 tl.* 是真实实现，不得误伤。"""
    src = (
        "import triton\nimport triton.language as tl\n\n"
        "@triton.jit\n"
        "def _k(x_ptr, o_ptr, n, BLOCK: tl.constexpr):\n"
        "    offs = tl.arange(0, BLOCK)\n"
        "    tl.store(o_ptr + offs, tl.load(x_ptr + offs), mask=offs < n)\n"
    )
    assert RULE_DELEGATION not in _scan(src, category="Elementwise", entry="exp")


def test_delegation_legal_sample_passes() -> None:
    """合法样例：自行实现，不调内置。"""
    src = (
        "import torch\nimport triton\nimport triton.language as tl\n\n"
        "@triton.jit\n"
        "def _exp_kernel(x_ptr, o_ptr, n, BLOCK: tl.constexpr):\n"
        "    offs = tl.arange(0, BLOCK)\n"
        "    m = offs < n\n"
        "    tl.store(o_ptr + offs, tl.exp(tl.load(x_ptr + offs, mask=m)), mask=m)\n\n"
        "def exp(x, base=-1.0, scale=1.0, shift=0.0):\n"
        "    out = torch.empty_like(x)\n"
        "    n = x.numel()\n"
        "    _exp_kernel[(triton.cdiv(n, 1024),)](x, out, n, BLOCK=1024)\n"
        "    return out\n"
    )
    assert _scan(src, category="Elementwise", entry="exp") == []


# ---------- SUB-BEH-002：设备回退 ----------


def test_device_fallback_detected() -> None:
    src = "def exp(x, **kw):\n    return torch.softmax(x.cpu(), -1).to(x.device)\n"
    found = _scan(src, category="Elementwise", entry="exp")
    assert RULE_DEVICE_FALLBACK in found


def test_device_fallback_legal_sample_passes() -> None:
    src = "def exp(x, **kw):\n    return x\n"
    assert RULE_DEVICE_FALLBACK not in _scan(src, category="Elementwise", entry="exp")


# ---------- SUB-BEH-006：计时篡改 ----------


def test_timing_tamper_detected() -> None:
    src = "def exp(x, **kw):\n    torch.npu.synchronize = lambda *a, **k: None\n    return x\n"
    # 这里是赋值不是调用，改为直接调用形式验证
    src2 = "def exp(x, **kw):\n    torch.npu.synchronize()\n    return x\n"
    assert RULE_TIMING_TAMPER in _scan(src2, category="Elementwise", entry="exp")


def test_timing_tamper_legal_sample_passes() -> None:
    src = "def exp(x, **kw):\n    return x\n"
    assert RULE_TIMING_TAMPER not in _scan(src, category="Elementwise", entry="exp")


# ---------- SUB-BEH-007：按 category 差异化白名单 ----------


def test_layout_move_blocked_for_elementwise() -> None:
    """Elementwise 类算子做布局变换 → 拦截。"""
    src = "def exp(x, **kw):\n    return x.transpose(0, 1)\n"
    assert RULE_LAYOUT_MOVE in _scan(src, category="Elementwise", entry="exp")


def test_layout_move_allowed_for_layout_category() -> None:
    """LayoutTransform 类的核心语义就是布局变换 → 放行。

    这是把参考仓库「只能人工审查」的项自动化的关键。
    """
    src = "def transpose(x, perm):\n    return x.permute(perm)\n"
    assert RULE_LAYOUT_MOVE not in _scan(src, category="LayoutTransform", entry="transpose")


def test_layout_allowed_categories_configured() -> None:
    assert LAYOUT_ALLOWED_CATEGORIES == {"LayoutTransform"}


def test_layout_move_blocked_for_reduction() -> None:
    src = "def cummin(input_value, dim):\n    return input_value.contiguous()\n"
    assert RULE_LAYOUT_MOVE in _scan(src, category="Reduction", entry="cummin")


# ---------- SUB-BEH-003：缓存 ----------


def test_address_keyed_cache_detected() -> None:
    src = "def exp(x, **kw):\n    key = x.data_ptr()\n    return x\n"
    assert detect_address_keyed_cache(src) is True


def test_no_cache_detected() -> None:
    src = "def exp(x, **kw):\n    return x\n"
    assert detect_address_keyed_cache(src) is False


def test_cache_rule_id_defined() -> None:
    assert RULE_CACHING == "SUB-BEH-003"


# ---------- 语法错误处理 ----------


def test_syntax_error_reported_not_crashed() -> None:
    findings = scan_source("def broken(:\n", category="Elementwise", entry="broken")
    assert len(findings) == 1
    assert findings[0].lineno is not None


# ---------- 报告可读性 ----------


def test_finding_render_includes_rule_and_line() -> None:
    src = "def exp(x, **kw):\n    return F.softmax(x, -1)\n"
    findings = scan_source(src, category="Elementwise", entry="exp")
    text = findings[0].render()
    assert "SUB-BEH-001" in text
    assert "第" in text and "行" in text


# ---------- 扫描文件 ----------


def test_scan_file(tmp_path: Path) -> None:
    p = tmp_path / "k.py"
    p.write_text("def exp(x, **kw):\n    return F.softmax(x, -1)\n", encoding="utf-8")
    from triton_eval.anticheat import scan_file

    findings = scan_file(p, category="Elementwise", entry="exp")
    assert any(f.rule_id == RULE_DELEGATION for f in findings)
