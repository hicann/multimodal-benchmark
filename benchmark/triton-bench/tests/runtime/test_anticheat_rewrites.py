# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反作弊对「换写法」的抵抗（L0，无需 NPU）。

守护一个根因缺陷：
    扫描器曾经只认 `torch.xxx()` 这一层——不解析 import 别名、
    不回溯属性链、不认方法形态、不看字符串参数、只按调用名查表。
    于是同一句作弊换个写法就能通过，而这些写法与直调**语义完全等价**。

    实测（ADV 样例）：
        import torch as _t            → _t.exp(x)            绕过
        from torch import exp         → exp(x)               绕过
        torch.nn.functional.softmax   → 属性链，只认最后一段  绕过
        x.exp()                       → 方法形态             绕过
        (x * a + b).exp()             → 接收者是表达式        绕过
        x.to("cpu")                   → 方法名是 to          绕过
        setattr(torch.npu, "syn...")  → 走函数调用           绕过

    反作弊若只惩罚老实人，比没有反作弊更糟。
"""

from __future__ import annotations

import pytest

from triton_eval.anticheat import (
    RULE_CACHING,
    RULE_DELEGATION,
    RULE_DEVICE_FALLBACK,
    RULE_TIMING_TAMPER,
    scan_source,
)

CAT = "Elementwise"


def _ids(source: str, category: str = CAT) -> list[str]:
    return [f.rule_id for f in scan_source(source, category=category, entry="exp")]


@pytest.mark.parametrize("src", [
    "import torch\ndef exp(x):\n    return torch.exp(x)",
    "import torch as _t\ndef exp(x):\n    return _t.exp(x)",
    "from torch import exp\ndef exp_(x):\n    return exp(x)",
    "import torch\ndef exp(x):\n    return torch.nn.functional.exp(x)",
    "import torch.nn.functional as F\ndef exp(x):\n    return F.exp(x)",
])
def test_import_alias_and_chains_all_caught(src: str) -> None:
    """别名、from-import、属性链——与 torch.exp() 完全等价。"""
    assert RULE_DELEGATION in _ids(src)


@pytest.mark.parametrize("src", [
    "def exp(x):\n    return x.exp()",
    "def exp(x, a=1.0, b=0.0):\n    return (x * a + b).exp()",
    "def exp(x, a=1.0):\n    return x.mul(a).exp()",
])
def test_method_form_caught(src: str) -> None:
    """方法形态：接收者是变量、表达式、链式调用都要认。"""
    assert RULE_DELEGATION in _ids(src)


def test_to_cpu_string_arg_caught() -> None:
    """x.to("cpu")：方法名是 to，只有看字符串参数才能抓到。"""
    assert RULE_DEVICE_FALLBACK in _ids(
        'def exp(x):\n    return x.to("cpu").exp()'
    )
    assert RULE_DEVICE_FALLBACK in _ids("def exp(x):\n    return x.cpu()")


def test_setattr_tamper_caught() -> None:
    """setattr 走函数调用，按调用名查表自然漏掉。"""
    assert RULE_TIMING_TAMPER in _ids(
        "import torch\ndef exp(x):\n"
        '    setattr(torch.npu, "synchronize", lambda *a: None)\n'
        "    return x"
    )


def test_id_keyed_cache_caught() -> None:
    """id() 与 data_ptr() 都是「地址键缓存」的入口。"""
    assert RULE_CACHING in _ids(
        "def exp(x, c={}):\n    k = id(x)\n    return c.setdefault(k, x)"
    )
    assert RULE_CACHING in _ids(
        "def exp(x, c={}):\n    k = x.data_ptr()\n    return c.setdefault(k, x)"
    )


def test_comment_mentioning_banned_word_not_flagged() -> None:
    """注释 / docstring 里提到被禁词不得判违规（ADV-E5 实测误拒）。

    曾经用全文本匹配，命中注释即报违规，且报错位置指向文档行。
    AST 不含注释，天然免疫。
    """
    src = (
        '"""注意：不要用 data_ptr 做缓存键。"""\n'
        "import math\n"
        "# 也不要用 id() 做缓存键\n"
        "def exp(x, base=-1.0):\n"
        "    return math.exp(x)\n"
    )
    assert _ids(src) == []


def test_stdlib_scalar_math_allowed() -> None:
    """标准库算标量常量是合法写法，不得误杀。

    误杀比漏放更伤：会让提交方放弃本来合规的实现。
    实测：合法正控制 ADV-C1 用 math.exp/math.sqrt 曾被整体判为代算。
    """
    src = (
        "import math\n"
        "def exp(x, base=-1.0, scale=1.0, shift=0.0):\n"
        "    lb = 1.0 if base in (None, -1.0) else math.log(float(base))\n"
        "    return x * float(scale) * lb\n"
    )
    assert _ids(src) == []


def test_triton_builtin_still_allowed() -> None:
    """tl.* / triton.* 是真实 kernel 语义，不得当代算。"""
    src = (
        "import triton\nimport triton.language as tl\n"
        "@triton.jit\n"
        "def _k(p, n, BLOCK: tl.constexpr):\n"
        "    o = tl.arange(0, BLOCK)\n"
        "    tl.store(p + o, tl.exp(tl.load(p + o)))\n"
        "def exp(x):\n    return x\n"
    )
    assert _ids(src) == []


def test_layout_move_allowed_for_layout_category() -> None:
    """布局类的核心语义就是搬运，不得误杀。"""
    src = "def transpose(x, perm):\n    return x.permute(tuple(perm)).contiguous()"
    assert _ids(src, category="LayoutTransform") == []
    # 非布局类仍然要拦。scan_source 返回内部行为编号，
    # 由 validator 映射成对外的 SUB-STR-004。
    from triton_eval.anticheat import RULE_LAYOUT_MOVE

    assert RULE_LAYOUT_MOVE in _ids(src, category="Elementwise")
