# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""参考实现的自检（只需 torch，不需要 NPU）。

守护的是什么：
    参考实现是判定的锚。它一旦有问题（缺文件、参数错位、输入越界、
    结果不确定），失败会被记成 ORACLE_FAIL——那是**评测方**的资产缺口，
    却长着一张「被测方没跑通」的脸。这类问题必须在冻结之后、评测之前暴露。

为什么标 `torch` 而不是 `npu`：
    这些检查在 CPU 上就能跑完整逻辑。曾经只能等昇腾机，
    于是 12 个参考实现一次都没被执行过就进了冻结集。
"""

from __future__ import annotations

from pathlib import Path

import pytest

from triton_eval.corpusmod import load_manifest
from triton_eval.inputs import _as_tensor, _dtype_of, as_outputs, make_inputs_for
from triton_eval.operator_registry import ALL_OPERATORS, by_name
from triton_eval.oracle import make_inputs_for as op_make_inputs, ref_path, reference_for

pytestmark = pytest.mark.torch

CORPUS = Path(__file__).resolve().parents[2] / "corpus" / "v0.3-multi-input-frozen"


def _cases():
    return load_manifest(CORPUS / "manifest.json").cases


def _run(case, device="cpu"):
    spec = by_name(case.operator)
    attrs = spec.resolve_attrs(case.attrs)
    inputs = make_inputs_for(case, _dtype_of(case.dtype), device, attrs,
                             op_make_inputs(case.operator))
    kwargs = {k: v for k, v in attrs.items() if v is not None}
    out = _as_tensor(reference_for(case.operator)(*inputs, **kwargs))
    return inputs, out, kwargs


def test_every_operator_has_reference() -> None:
    """缺 ref.py 的算子，其用例必然 ORACLE_FAIL——冻结时不得留下这种洞。

    25 算子的参考实现在 v0.5 语料（14 个原语随版本迁移 + 11 个融合参考）；
    参考随语料版本冻结（oracle.configure_corpus），故对 v0.5 目录断言。
    """
    from triton_eval.corpus_assets import resolve_reference_path

    v05 = Path(__file__).resolve().parents[2] / "corpus" / "v0.5-cov20-frozen"
    v07 = Path(__file__).resolve().parents[2] / "corpus" / "v0.7-ascend-contract"
    v08 = Path(__file__).resolve().parents[2] / "corpus" / "v0.8-official-candidates"
    registered = {operator.name for operator in ALL_OPERATORS}
    covered = set()
    for corpus in (v05, v07, v08):
        manifest = load_manifest(corpus / "manifest.json")
        for operator in manifest.operators:
            covered.add(operator.name)
            assert resolve_reference_path(corpus, operator.ref_path).is_file(), (corpus.name, operator.name)
    assert covered == registered


def test_reference_outputs_stay_on_input_device() -> None:
    """参考的每个输出都必须与输入同设备——meta 设备纯度测试。

    2026-10-13 真机实测：MoE 的参考用 torch.arange 不带 device= 造
    row_idx（落 CPU），与 NPU 上的实现输出比对时抛设备不匹配、杀死
    评测子进程（后续 15 个用例全部陪葬）。CPU 上的本地测试暴露不了
    这类缺陷（全 CPU 无从错位）；meta 张量保留设备语义，可精确复现
    「参考在默认设备上建张量」的错误类。
    UnsortedSegmentSum 豁免：其参考用布尔掩码索引（nonzero），meta
    后端未实现——非设备缺陷，改为人工核对（其 zeros 已显式带 device=）。
    """
    import torch

    from triton_eval.oracle import active_corpus, configure_corpus

    v05 = Path(__file__).resolve().parents[2] / "corpus" / "v0.5-cov20-frozen"
    prev = active_corpus()
    configure_corpus(v05)
    try:
        seen: dict[str, object] = {}
        for case in load_manifest(v05 / "manifest.json").cases:
            seen.setdefault(case.operator, case)
        for op_name, case in sorted(seen.items()):
            if op_name == "UnsortedSegmentSum":
                continue
            spec = by_name(case.operator)
            attrs = spec.resolve_attrs(case.attrs)
            inputs = make_inputs_for(case, _dtype_of(case.dtype), "meta",
                                     attrs, op_make_inputs(case.operator))
            kwargs = {k: v for k, v in attrs.items() if v is not None}
            out = reference_for(case.operator)(*inputs, **kwargs)
            for t in as_outputs(out):
                assert t.device.type == "meta", (
                    f"{op_name} 的参考输出落在 {t.device}（应为输入的 meta）："
                    "参考实现创建张量时必须显式带 device=，否则 NPU 环境下"
                    "与实现输出比对即崩（2026-10-13 MoE 真机事故同源）"
                )
    finally:
        configure_corpus(prev)


def test_every_case_reference_runs() -> None:
    for case in _cases():
        inputs, out, _ = _run(case)
        assert out.numel() > 0, f"{case.code} 参考输出为空"
        assert not (out.is_floating_point() and not bool(out.isfinite().all())), (
            f"{case.code} 参考输出含 NaN/Inf，锚点本身不可信"
        )


def test_reference_is_deterministic() -> None:
    """参考实现必须可复现：不确定的锚点会让同一用例在不同轮次给出不同判定。"""
    for case in _cases():
        spec = by_name(case.operator)
        attrs = spec.resolve_attrs(case.attrs)
        kwargs = {k: v for k, v in attrs.items() if v is not None}
        inputs = make_inputs_for(case, _dtype_of(case.dtype), "cpu", attrs,
                                 op_make_inputs(case.operator))
        a = _as_tensor(reference_for(case.operator)(*inputs, **kwargs))
        b = _as_tensor(reference_for(case.operator)(*inputs, **kwargs))
        assert bool((a == b).all()), f"{case.code} 参考实现两次执行结果不一致"


def test_multi_input_operators_produce_multiple_tensors() -> None:
    """多输入算子必须真的造出多个张量——否则参考实现会缺参数而报 TypeError。"""
    expected = {
        "Maximum": 2, "MaskedScale": 2, "RmsNorm": 2, "Gather": 2,
        "Scatter": 3, "GroupedMatmul": 3, "CrossEntropyLoss": 2,
    }
    seen: dict[str, int] = {}
    for case in _cases():
        spec = by_name(case.operator)
        attrs = spec.resolve_attrs(case.attrs)
        inputs = make_inputs_for(case, _dtype_of(case.dtype), "cpu", attrs,
                                 op_make_inputs(case.operator))
        seen[case.operator] = len(inputs)
    for name, want in expected.items():
        assert seen.get(name) == want, f"{name} 输入个数应为 {want}，实际 {seen.get(name)}"


def test_gather_index_within_range() -> None:
    """越界索引在不同后端行为不一致（报错/截断/脏数据），会让判定失去意义。"""
    import torch

    for case in _cases():
        if case.operator != "Gather":
            continue
        spec = by_name(case.operator)
        attrs = spec.resolve_attrs(case.attrs)
        x, index = make_inputs_for(case, _dtype_of(case.dtype), "cpu", attrs,
                                   op_make_inputs(case.operator))
        assert int(index.min()) >= 0
        assert int(index.max()) < x.shape[attrs["dim"]], f"{case.code} 索引越界"


def test_scatter_indices_unique_along_dim() -> None:
    """同一位置被写两次时结果依赖写入顺序，用例就不再可判定。"""
    import torch

    for case in _cases():
        if case.operator != "Scatter":
            continue
        spec = by_name(case.operator)
        attrs = spec.resolve_attrs(case.attrs)
        _, index, _ = make_inputs_for(case, _dtype_of(case.dtype), "cpu", attrs,
                                      op_make_inputs(case.operator))
        dim = attrs["dim"]
        moved = index.movedim(dim, 0)
        for row in moved.reshape(moved.shape[0], -1).T:
            assert len(torch.unique(row)) == len(row), f"{case.code} 索引在 dim={dim} 上重复"


def test_argmax_inputs_have_no_ties() -> None:
    """并列最大值时返回哪个下标由实现自由决定，那种用例不具备可判定性。"""
    import torch

    for case in _cases():
        if case.operator != "ArgMax":
            continue
        spec = by_name(case.operator)
        attrs = spec.resolve_attrs(case.attrs)
        (x,) = make_inputs_for(case, _dtype_of(case.dtype), "cpu", attrs,
                               op_make_inputs(case.operator))
        if x.dtype in (torch.int32, torch.int64):
            assert len(torch.unique(x)) == x.numel(), f"{case.code} 整型输入存在并列值"


def test_grouped_matmul_groups_sum_to_rows() -> None:
    """group_list 之和必须等于 x 的行数，否则 weight 组数与行数对不上。"""
    for case in _cases():
        if case.operator != "GroupedMatmul":
            continue
        spec = by_name(case.operator)
        attrs = spec.resolve_attrs(case.attrs)
        x, weight, bias = make_inputs_for(case, _dtype_of(case.dtype), "cpu", attrs,
                                          op_make_inputs(case.operator))
        assert sum(attrs["group_list"]) == x.shape[0]
        assert weight.shape[0] == len(attrs["group_list"]) == bias.shape[0]
        assert weight.shape[1] == x.shape[1]


def test_nondeterminism_message_reports_magnitude(monkeypatch) -> None:
    """两次不一致的失败原因必须带量级（2026-10-14 真机 CE 发现后加）。

    只报"不一致"无从判断是 ULP 级噪声还是量级错误——量级决定登记口径
    （容差可吸收的平台噪声 vs 参考实现缺陷）。真机 5e 首跑发现的
    cross_entropy_loss 3 例（npu:0 两遍不一致）即靠此诊断量级。
    """
    import torch

    from triton_eval import oracle
    from triton_eval.refcheck import check_case

    case = next(c for c in _cases() if c.operator == "Exp")
    calls = {"n": 0}

    def flaky_ref(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return torch.full((4, 8), 1.5)
        return torch.full((4, 8), 2.0)

    monkeypatch.setattr(oracle, "reference_for", lambda op: flaky_ref)
    why = check_case(case, "cpu")
    assert why is not None, "两次结果不同必须报失败"
    assert "两次执行结果不一致" in why
    assert "max_abs=5.000e-01" in why, f"失败原因应带量级：{why}"


def test_independent_reference_mismatch_reports_values(monkeypatch):
    import torch
    from triton_eval import oracle
    from triton_eval.refcheck import check_case

    case = next(c for c in _cases() if c.operator == "Exp")
    monkeypatch.setattr(oracle, "reference_for", lambda op: lambda *args, **kwargs: torch.full((4, 8), 1.5))
    monkeypatch.setattr(oracle, "cross_reference_for", lambda op: lambda *args, **kwargs: torch.full((4, 8), 2.0))
    why = check_case(case, "cpu")
    assert "第二参考输出 0 不一致" in why
    assert "mismatch_count=32/32" in why
    assert "actual=1.5" in why and "expected=2.0" in why
    assert "max_abs=5.000e-01" in why
