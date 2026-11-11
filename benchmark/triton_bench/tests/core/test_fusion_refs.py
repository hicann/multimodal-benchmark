# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""融合算子的参考实现测试（L0 + torch，无需 NPU）。

守护什么：
    1. 参考实现自检（可执行 / 两遍一致 / 输出有限）——与原语参考同一门禁；
    2. **组合等价性**：Tier-2 参考必须与「手动串原语参考」逐元素一致
       （单一代码事实来源——组合参考走样等于偷换 oracle）；
    3. **语料解析回归**：参考必须从当前生效语料目录加载（oracle 历史
       缺陷的防回归钉——参考曾固定从旧版本语料静默加载）；
    4. Tier-1 参考与来源池 golden 的关键语义对拍（公式不走样）。
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

from triton_eval.corpusmod import load_manifest
from triton_eval.inputs import _as_tensor, _dtype_of, make_inputs_for
from triton_eval.models import CaseSpec
from triton_eval.operator_registry import ALL_OPERATORS, COMPOSITIONS, by_name
from triton_eval.oracle import (
    active_corpus,
    configure_corpus,
    make_inputs_for as op_make_inputs,
    reference_for,
)

pytestmark = pytest.mark.torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CORPUS = PROJECT_ROOT / "corpus" / "v0.5-cov20-frozen"

FUSION_OPS = [op for op in ALL_OPERATORS if op.name in COMPOSITIONS]

# 每个融合算子的代表性用例（dtype/shape/attr 变体各异，冒烟全覆盖）
_SMOKE_CASES: list[tuple[str, str, list[int], dict]] = [
    ("LinearGelu", "float32", [64, 128], {"group_list": [32, 32]}),
    ("RmsNormLinear", "float32", [64, 128], {"group_list": [64], "epsilon": 1e-3}),
    ("GroupedMatmulSwiGlu", "bfloat16", [96, 64], {"group_list": [48, 48]}),
    ("TransposeLinear", "float32", [64, 128], {"group_list": [64, 64]}),
    ("MaskedSoftmax", "float32", [8, 128], {"dim": 1}),
    ("MaskedSoftmax", "int8", [1024], {"dim": 0}),
    ("MaskedGelu", "float32", [32, 128], {"approximate": "tanh"}),
    ("GatherExp", "float32", [32, 128],
     {"dim": 1, "base": 2.0, "scale": 0.5, "shift": 1.0}),
    ("AddRmsNorm", "bfloat16", [64, 128], {"epsilon": 1e-6}),
    ("ApplyRotaryPosEmb", "float32", [64, 128], {"rotary_mode": "half"}),
    ("ApplyRotaryPosEmb", "float32", [2, 32, 128], {"rotary_mode": "interleaved"}),
    ("MoeGatingTopKSoftmax", "float32", [128, 64], {"k": 4}),
    ("MoeGatingTopKSoftmax", "bfloat16", [2, 64, 64], {"k": 8}),
    ("UnsortedSegmentSum", "float32", [1024, 64], {"num_segments": 16}),
    ("UnsortedSegmentSum", "int32", [2048], {"num_segments": 32}),
]


def _case(op: str, dtype: str, shape: list[int], attrs: dict) -> CaseSpec:
    return CaseSpec(
        code=f"smoke-{op}-{dtype}", operator=op, source="vendor_selftest",
        dtype=dtype, shape=shape, attrs=attrs, seed=20261013,
        launch={}, timeout_s=60, feature_atoms=[],
        value_range={"kind": "normal"}, contiguous=True,
    )


def _run(case: CaseSpec, device: str = "cpu"):
    spec = by_name(case.operator)
    attrs = spec.resolve_attrs(case.attrs)
    inputs = make_inputs_for(case, _dtype_of(case.dtype), device, attrs,
                             op_make_inputs(case.operator))
    kwargs = {k: v for k, v in attrs.items() if v is not None}
    out = _as_tensor(reference_for(case.operator)(*inputs, **kwargs))
    return inputs, out, kwargs


def _seg_ref(entry: str):
    """直接按文件路径加载原语参考（与组合参考的 _seg 同一手法）。"""
    from triton_eval.corpus_assets import resolve_reference_path

    path = resolve_reference_path(CORPUS, f"operators/{entry}/ref.py")
    spec_ = importlib.util.spec_from_file_location(f"t_seg_{entry}", path)
    mod = importlib.util.module_from_spec(spec_)
    assert spec_.loader is not None
    spec_.loader.exec_module(mod)
    return mod


@pytest.fixture(autouse=True)
def _use_v05_corpus():
    """全部用例在 v0.5 语料上跑（25 算子的参考都在该目录），跑完还原。

    注意必须「跑完还原」：oracle 的语料配置是进程级状态，
    泄漏会污染同进程其它测试（test_refs 钉在 v0.3）。
    """
    configure_corpus(CORPUS)
    yield
    configure_corpus(PROJECT_ROOT / "corpus" / "v0.3-multi-input-frozen")


# ---------- 1. 参考自检（门禁同款） ----------


@pytest.mark.parametrize("op,dtype,shape,attrs", _SMOKE_CASES)
def test_fusion_reference_selfcheck(op, dtype, shape, attrs) -> None:
    from triton_eval.refcheck import check_case

    assert check_case(_case(op, dtype, shape, attrs), "cpu") is None


# ---------- 2. 组合等价性：Tier-2 参考 = 原语参考的顺序调用 ----------


def test_linear_gelu_equals_staged_primitives() -> None:
    case = _case("LinearGelu", "float32", [64, 128], {"group_list": [32, 32]})
    inputs, out, kwargs = _run(case)
    gmm, gelu = _seg_ref("grouped_matmul"), _seg_ref("gelu")
    staged = gelu.reference(gmm.reference(*inputs, **kwargs))
    assert out.dtype == staged.dtype and torch_allclose(out, staged)


def test_rms_norm_linear_equals_staged_primitives() -> None:
    case = _case("RmsNormLinear", "float32", [64, 128],
                 {"group_list": [64], "epsilon": 1e-3})
    inputs, out, kwargs = _run(case)
    x, gamma, weight, bias = inputs
    rms, gmm = _seg_ref("rms_norm"), _seg_ref("grouped_matmul")
    staged = gmm.reference(rms.reference(x, gamma, kwargs["epsilon"]),
                           weight, bias, kwargs["group_list"])
    assert torch_allclose(out, staged)


def test_grouped_matmul_swiglu_equals_staged_primitives() -> None:
    case = _case("GroupedMatmulSwiGlu", "bfloat16", [96, 64],
                 {"group_list": [48, 48]})
    inputs, out, kwargs = _run(case)
    gmm, swi = _seg_ref("grouped_matmul"), _seg_ref("swi_glu")
    staged = swi.reference(gmm.reference(*inputs, **kwargs), -1)
    assert out.dtype == staged.dtype and torch_allclose(out, staged)


def test_transpose_linear_equals_staged_primitives() -> None:
    import torch

    case = _case("TransposeLinear", "float32", [64, 128],
                 {"group_list": [64, 64]})
    inputs, out, kwargs = _run(case)
    x, weight, bias = inputs
    gmm = _seg_ref("grouped_matmul")
    # 手动两段：转置（rank-2 固定 [1,0]）→ 矩阵乘
    staged = gmm.reference(x.to(torch.float32).transpose(0, 1),
                           weight, bias, kwargs["group_list"])
    # 转置后行数是 K=128：group_list 之和必须等于 K 而非 M
    assert sum(kwargs["group_list"]) == case.shape[1]
    assert out.shape == (case.shape[1], staged.shape[-1])
    assert torch_allclose(out, staged.to(out.dtype))


def test_masked_gelu_equals_staged_primitives() -> None:
    case = _case("MaskedGelu", "float32", [32, 128], {"approximate": "tanh"})
    inputs, out, kwargs = _run(case)
    import torch

    x, mask = inputs
    gelu = _seg_ref("gelu")
    staged = (gelu.reference(x, kwargs["approximate"]).to(torch.float32)
              * mask.to(torch.float32)).to(x.dtype)
    assert torch_allclose(out, staged)


def test_gather_exp_equals_staged_primitives() -> None:
    case = _case("GatherExp", "float32", [32, 128],
                 {"dim": 1, "base": 2.0, "scale": 0.5, "shift": 1.0})
    inputs, out, kwargs = _run(case)
    gather, exp = _seg_ref("gather"), _seg_ref("exp")
    staged = exp.reference(gather.reference(inputs[0], inputs[1], kwargs["dim"]),
                           kwargs["base"], kwargs["scale"], kwargs["shift"])
    assert out.dtype == staged.dtype and torch_allclose(out, staged)


# ---------- 3. Tier-1 关键语义（对拍来源池 golden 公式） ----------


def test_masked_softmax_full_masked_row_is_finite() -> None:
    """归约行全掩码时必须无 NaN——make_inputs 的保底修补是门禁前提。"""
    import torch

    case = _case("MaskedSoftmax", "float32", [8, 128], {"dim": 1})
    inputs, out, _ = _run(case)
    assert bool(torch.isfinite(out).all()), "softmax 出现 NaN/Inf"
    # 掩码行和恒为 1（归一化语义）
    assert torch.allclose(out.sum(dim=1), torch.ones(8), atol=1e-5)


def test_rope_matches_pool_formula() -> None:
    """y = x*cos_full + rotate_half(x)*sin_full，两种模式分别对拍池公式。"""
    import torch

    for mode in ("half", "interleaved"):
        case = _case("ApplyRotaryPosEmb", "float32", [4, 64],
                     {"rotary_mode": mode})
        (x, cos, sin), out, _ = _run(case)
        x32 = x.to(torch.float32)
        c, s = cos.to(torch.float32), sin.to(torch.float32)
        if mode == "interleaved":
            rot = torch.stack([-x32[..., 1::2], x32[..., ::2]], dim=-1).flatten(-2)
            cf, sf = c.repeat_interleave(2, -1), s.repeat_interleave(2, -1)
        else:
            rot = torch.cat([-x32[..., 32:], x32[..., :32]], dim=-1)
            cf, sf = torch.cat([c, c], -1), torch.cat([s, s], -1)
        pool = x32 * cf + rot * sf
        assert torch_allclose(out, pool), f"rotary_mode={mode} 与池公式不符"


def test_moe_gating_matches_pool_formula() -> None:
    """softmax→topk 的 values 与手算一致；row_idx 为展平全局位置。"""
    import torch

    case = _case("MoeGatingTopKSoftmax", "float32", [128, 64], {"k": 4})
    (x,), out, _ = _run(case)
    sm = torch.softmax(x.to(torch.float32), dim=-1)
    vals, idx = torch.topk(sm, 4, dim=-1)
    assert out.shape == (128, 4)
    assert torch_allclose(out, vals)
    row_expected = torch.arange(128 * 4).reshape(4, 128).T
    # 三输出的元组：第二个是专家下标、第三个是行位置（现行契约只判第一个）
    full = reference_for("MoeGatingTopKSoftmax")(x, 4)
    assert isinstance(full, tuple) and len(full) == 3
    assert torch.equal(full[2].to(torch.int64), row_expected)


def test_unsorted_segment_sum_matches_index_add() -> None:
    """逐段掩码求和（确定性实现）与 index_add_ 的数值结果一致。"""
    import torch

    case = _case("UnsortedSegmentSum", "float32", [1024, 64],
                 {"num_segments": 16})
    (data, ids), out, _ = _run(case)
    pool = torch.zeros((16, 64), dtype=torch.float32)
    pool.index_add_(0, ids.to(torch.int64), data.to(torch.float32))
    assert torch_allclose(out, pool.to(out.dtype), tol=1e-5)


def test_scatter_concurrent_accum_duplicate_indices() -> None:
    """reduce=add 变体（v0.5）：索引允许重复——整型下「和」精确且顺序无关。

    守护三件事：构造确实产生重复；参考与手动 scatter_add 逐元素相等
    （顺序无关的可判定性）；生成侧把浮点挡在池外（并发浮点和不可判定）。
    """
    import torch

    from triton_eval.casegen import derive_pool

    case = _case("Scatter", "int32", [1024], {"dim": 0, "reduce": "add"})
    inputs, out, _ = _run(case)
    flat = inputs[1].flatten()
    assert flat.numel() > flat.unique().numel(), \
        "并发累加变体的索引必须含重复——否则考不到顺序无关累加"
    manual = inputs[0].clone()
    manual.scatter_add_(0, inputs[1], inputs[2])
    assert torch.equal(out, manual), "整型并发和必须与写入顺序无关"

    pool = derive_pool(by_name("Scatter"))
    add_cases = [c for c in pool if c.attrs.get("reduce") == "add"]
    assert add_cases, "并发累加变体必须进池（FA-CONCUR-ACCUM 载体）"
    assert all(c.dtype in ("int32", "int64") for c in add_cases), \
        "浮点/窄整型的并发变体必须被组合合法性挡在池外"


# ---------- 4. 语料解析回归钉 ----------


def test_configure_corpus_switches_reference_source() -> None:
    """参考必须从配置的语料目录加载（历史缺陷：固定旧版语料静默加载）。

    v0.3 没有 linear_gelu 的参考，v0.5 有——切换后必须立刻可用；
    切回 v0.3 后必须恢复不可用。缓存清空是修复的一部分。
    """
    v03 = PROJECT_ROOT / "corpus" / "v0.3-multi-input-frozen"
    configure_corpus(CORPUS)
    assert active_corpus() == CORPUS
    assert reference_for("LinearGelu") is not None

    configure_corpus(v03)
    assert reference_for("LinearGelu") is None, \
        "切回 v0.3 后融合参考必须不可用——缓存未清空则旧引用仍存活"
    assert reference_for("Exp") is not None, "原语参考仍在"


def torch_allclose(a, b, tol: float = 1e-6) -> bool:
    import torch

    a, b = a.to(torch.float32), b.to(torch.float32)
    if a.shape != b.shape:
        return False
    return bool(torch.allclose(a, b, rtol=tol, atol=tol))
