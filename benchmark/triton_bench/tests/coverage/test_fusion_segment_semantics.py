# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""融合段语义的数值回归钉：中间结果必须舍回输入精度（L0，只需 torch）。

为什么钉：组合参考 = 原语参考顺序调用，原语参考的出口是 .to(x.dtype)
——中间张量本来就是输入精度的。融合样例内核若全程 fp32 不舍回，半精度
用例会全错（2026-10-13 真机实测：rms_norm_linear bf16 rel 3.9 /
grouped_matmul_swiglu f16 abs 0.25，本地模拟与真机逐位吻合）。

钉法：在 CPU 上复刻内核数学（含量化点）与参考链对拍——
「按段语义舍回」必须与参考一致（快到零）、「不舍回」必须显著超容差
（双向断言：改丢舍回会被抓住，参考侧改约定也会被抓住）。
内核本身 import triton、本机不可执行——执行侧由真机窗口复核。
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from triton_eval.inputs import make_inputs_for  # noqa: E402
from triton_eval.corpusmod import load_manifest  # noqa: E402
from triton_eval.oracle import (configure_corpus, make_inputs_for as ora_mif,  # noqa: E402
                                reference_for)

pytestmark = pytest.mark.torch

V05 = Path(__file__).resolve().parents[2] / "corpus" / "v0.5-cov20-frozen"


def _case(code: str):
    manifest = load_manifest(V05 / "manifest.json")
    return next(c for c in manifest.cases if c.code == code)


def _build(code: str):
    import torch

    case = _case(code)
    configure_corpus(V05)
    inputs = make_inputs_for(case, getattr(torch, case.dtype), "cpu",
                             dict(case.attrs), ora_mif(case.operator))
    ref = reference_for(case.operator)(*inputs, **dict(case.attrs))
    return case, inputs, ref


def _metrics(sim, ref):
    import torch

    r32, s32 = ref.to(torch.float32), sim.to(torch.float32)
    d = (s32 - r32).abs()
    rel = (d / r32.abs().clamp_min(1e-30)).max().item()
    return rel, d.max().item()


# ---- RmsNormLinear：归一化段出口舍回输入精度 ----

def _rnl_sim(inputs, attrs, round_mid: bool):
    import torch

    x, gamma, w, b = inputs
    gl = [int(v) for v in attrs["group_list"]]
    eps = float(attrs.get("epsilon", 1e-6))
    x32 = x.to(torch.float32)
    norm = x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) \
        * gamma.to(torch.float32)
    if round_mid:
        norm = norm.to(x.dtype).to(torch.float32)   # 段语义：出口舍回
    outs, st = [], 0
    for g, cnt in enumerate(gl):
        outs.append(norm[st:st + cnt] @ w[g].to(torch.float32)
                    + b[g].to(torch.float32))
        st += cnt
    return torch.cat(outs).to(x.dtype)


@pytest.mark.parametrize("code", ["rms_norm_linear-bf16-S08",
                                  "rms_norm_linear-f16-S05"])
def test_rnl_segment_semantics_requires_rounding(code: str) -> None:
    case, inputs, ref = _build(code)
    rel_ok, abs_ok = _metrics(_rnl_sim(inputs, case.attrs, True), ref)
    rel_bad, abs_bad = _metrics(_rnl_sim(inputs, case.attrs, False), ref)
    assert rel_ok <= 1e-6 and abs_ok <= 1e-6, \
        "按段语义舍回后应与参考一致——参考的段出口约定变了？"
    assert rel_bad > 0.1, \
        f"不舍回的分歧应显著（真机实测 rel≈{2.9:.0f}~{3.9:.1f} 量级），" \
        f"实测 {rel_bad:.3g}——参考或模拟的口径漂了"


# ---- GroupedMatmulSwiGlu：矩阵乘段出口舍回输入精度 ----

def _gms_sim(inputs, attrs, round_mid: bool):
    import torch

    x, w, b = inputs
    gl = [int(v) for v in attrs["group_list"]]
    outs, st = [], 0
    for g, cnt in enumerate(gl):
        v = x[st:st + cnt].to(torch.float32) @ w[g].to(torch.float32) \
            + b[g].to(torch.float32)
        if round_mid:
            v = v.to(x.dtype).to(torch.float32)   # 段语义：出口舍回
        a, bb = v.chunk(2, dim=-1)
        outs.append(torch.nn.functional.silu(a) * bb)
        st += cnt
    return torch.cat(outs).to(x.dtype)


def test_gms_segment_semantics_requires_rounding() -> None:
    case, inputs, ref = _build("grouped_matmul_swiglu-f16-M09")
    rel_ok, abs_ok = _metrics(_gms_sim(inputs, case.attrs, True), ref)
    rel_bad, abs_bad = _metrics(_gms_sim(inputs, case.attrs, False), ref)
    assert rel_ok <= 1e-6 and abs_ok <= 1e-6
    assert abs_bad > 0.05, \
        f"不舍回的分歧应显著（真机实测 abs≈0.25），实测 {abs_bad:.3g}"
