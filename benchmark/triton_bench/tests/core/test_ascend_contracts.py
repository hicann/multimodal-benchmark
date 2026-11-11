# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""验证修订语料、独立参考与非法输入关系。"""

import importlib.util
from collections import Counter
from pathlib import Path

import pytest

from triton_eval.ascend_contracts import validate_inputs
from triton_eval.corpusmod import load_manifest, verify_corpus_files
from triton_eval.inputs import _dtype_of, make_inputs_for
from triton_eval.operator_registry import by_name
from triton_eval.oracle import active_corpus, configure_corpus, make_inputs_for as maker
from triton_eval.refcheck import check_case

pytestmark = pytest.mark.torch
ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus/v0.7-ascend-contract"


@pytest.fixture(autouse=True)
def corpus_scope():
    previous = active_corpus()
    configure_corpus(CORPUS)
    yield
    configure_corpus(previous)


def cases():
    return load_manifest(CORPUS / "manifest.json").cases


def inputs_for(operator):
    case = next(case for case in cases() if case.operator == operator)
    attrs = by_name(operator).resolve_attrs(case.attrs)
    tensors = make_inputs_for(case, _dtype_of(case.dtype), "cpu", attrs, maker(operator))
    return case, attrs, tensors


def test_guided_selection_is_frozen_and_balanced():
    module_spec = importlib.util.spec_from_file_location(
        "ascend_builder", ROOT / "scripts/build_corpus_v06_ascend_ops.py")
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    selected, report = module.select_cases()
    assert [case.code for case in selected] == [case.code for case in cases()]
    assert Counter(case.operator for case in selected) == {name: 20 for name in module.ASCEND_OP_NAMES}
    assert report["coverage"]["missing"] == []
    assert {case.dtype for case in selected if case.operator == "VectorAdd"} == {"int32", "int64", "float32"}
    assert max(case.shape[0] for case in selected if case.operator == "VectorAdd") == 4194304
    assert any(case.shape[0] == 65 for case in selected if case.operator == "InsertSlice")
    assert any(case.shape[1] == 257 for case in selected if case.operator == "CausalConv1d")
    manifest = load_manifest(CORPUS / "manifest.json")
    from triton_eval.corpusmod import compute_case_content_hash, verify_manifest

    verify_manifest(manifest)
    assert all(compute_case_content_hash(CORPUS, case) == case.content_hash for case in manifest.cases)
    verify_corpus_files(CORPUS, manifest)
    verify_corpus_files(CORPUS.relative_to(ROOT), manifest)
    assert all((CORPUS / operator.ref_path).is_file() for operator in manifest.operators)


@pytest.mark.parametrize("operator", sorted({case.operator for case in cases()}))
def test_smallest_reference_and_independent_reference_agree(operator):
    from math import prod

    case = min((case for case in cases() if case.operator == operator), key=lambda item: prod(item.shape))
    assert check_case(case) is None


def test_duplicate_request_rows_are_rejected():
    _, attrs, inputs = inputs_for("AssignReqToTokenPool")
    inputs[0][1] = inputs[0][0]
    with pytest.raises(ValueError, match="唯一"):
        validate_inputs("AssignReqToTokenPool", inputs, attrs)


def test_out_of_bounds_gather_rejected():
    _, attrs, inputs = inputs_for("DiscreteMemoryAccess")
    inputs[1][0] = inputs[0].numel()
    with pytest.raises(ValueError, match="越界"):
        validate_inputs("DiscreteMemoryAccess", inputs, attrs)


def test_inconsistent_expert_boundaries_rejected():
    _, attrs, inputs = inputs_for("PaddedGatherScatter")
    inputs[3][0] = -1
    with pytest.raises(ValueError, match="边界不一致"):
        validate_inputs("PaddedGatherScatter", inputs, attrs)


@pytest.mark.parametrize("failure, expected", [
    ("arity", "元数"),
    ("nonfinite", "输出 1 含"),
    ("nondeterministic", "输出 1 两次"),
    ("mutation", "修改了输入"),
    ("secondary_mutation", "第二参考修改"),
])
def test_refcheck_checks_all_outputs_and_input_purity(monkeypatch, failure, expected):
    import triton_eval.oracle as oracle

    case = next(case for case in cases() if case.operator == "VectorCmp")
    count = 0

    def reference(values, **kwargs):
        nonlocal count
        count += 1
        if failure == "mutation":
            values.zero_()
        if failure == "arity":
            return values.clone()
        statistic = values.new_full((values.shape[0],), float("nan") if failure == "nonfinite" else 0)
        if failure == "nondeterministic":
            statistic.fill_(count)
        return values.clone(), statistic, statistic.clone()

    def secondary(values, **kwargs):
        values.zero_()
        return values.clone(), values.new_zeros(values.shape[0]), values.new_zeros(values.shape[0])

    def reference_for_operator(operator):
        return reference

    monkeypatch.setattr(oracle, "reference_for", reference_for_operator)
    monkeypatch.setattr(
        oracle,
        "cross_reference_for",
        lambda operator: secondary if failure == "secondary_mutation" else None)
    assert expected in check_case(case)


def test_asset_change_is_detected(tmp_path, monkeypatch):
    import triton_eval.corpus_assets as assets

    monkeypatch.setattr(assets, "ROOT", tmp_path)
    monkeypatch.setattr(assets, "FRAMEWORK_FILES", ("contract.py",))
    (tmp_path / "contract.py").write_text("first")
    corpus = tmp_path / "corpus"
    corpus.mkdir()
    assets.freeze_assets(corpus, "manifest")
    (tmp_path / "contract.py").write_text("changed")
    with pytest.raises(ValueError, match="contract.py"):
        assets.verify_assets(corpus, "manifest")


def test_attention_routes_kv_heads_and_zeros_inactive_splits():
    import torch
    from triton_eval.oracle import reference_for, cross_reference_for

    query = torch.ones((1, 4, 2), dtype=torch.bfloat16)
    keys = torch.zeros((3, 2, 2), dtype=torch.bfloat16)
    values = torch.tensor([[[1, 2], [10, 20]], [[3, 4], [30, 40]],
                           [[5, 6], [50, 60]]], dtype=torch.bfloat16)
    inputs = [query, keys, values, torch.tensor([0, 1], dtype=torch.int32),
              torch.tensor([2], dtype=torch.int32), torch.tensor([1], dtype=torch.int32)]
    for reference in (reference_for("DecodeGroupedAttention"), cross_reference_for("DecodeGroupedAttention")):
        output, lse = reference(*inputs, max_kv_splits=4)
        assert torch.equal(output[0, :, 0], torch.tensor([[5, 6], [5, 6], [50, 60], [50, 60]]))
        assert torch.count_nonzero(output[:, :, 1:]) == 0
        assert torch.count_nonzero(lse) == 0


def test_conv_state_mapping_and_sequence_window():
    import torch
    from triton_eval.oracle import reference_for, cross_reference_for

    values = torch.tensor([[[3, 4]]], dtype=torch.bfloat16)
    state = torch.tensor([[[99, 98]], [[1, 2]]], dtype=torch.bfloat16)
    weight = torch.ones((1, 3), dtype=torch.bfloat16)
    bias = torch.tensor([1], dtype=torch.bfloat16)
    indices = torch.tensor([1], dtype=torch.int32)
    for reference in (reference_for("CausalConv1d"), cross_reference_for("CausalConv1d")):
        output, updated = reference(values, state, weight, bias, indices, activation="none")
        assert torch.equal(output, torch.tensor([[[7, 10]]], dtype=torch.bfloat16))
        assert torch.equal(updated[0], state[0])
        assert torch.equal(updated[1], values[0])
        assert torch.equal(state, torch.tensor([[[99, 98]], [[1, 2]]], dtype=torch.bfloat16))
