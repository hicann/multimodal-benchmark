# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The generated dimensions must change real legal inputs, not just labels."""
import sys
from pathlib import Path

import pytest

from triton_eval.official_catalog import BY_NAME, CANDIDATES
from triton_eval.official_generation import (
    GENERATOR, bounded_selection, cases_for, coverage_gaps, numeric_profiles, select_cases,
)
from triton_eval.specload import load_feature_atoms


def test_bounded_selection_is_deterministic_and_reports_budget_gaps():
    pool = cases_for(BY_NAME["Add"])
    small, report = bounded_selection(pool, [], budget=1)
    assert len(small) == 1
    assert report["lost_reachable_atoms"]
    assert report["unrepresented_strata"]
    assert bounded_selection(list(reversed(pool)), [], budget=1) == (small, report)
    with pytest.raises(ValueError, match="budget"):
        bounded_selection(pool, pool[:2], budget=1)
    with pytest.raises(ValueError, match="absent"):
        bounded_selection(pool[:3], [pool[4]], budget=1)


def test_new_generation_preserves_reachable_features_in_fixed_quotas():
    cases, reports = select_cases(CANDIDATES)
    assert len(cases) == 1360
    assert all(report["selected"] == report["budget"] == 20 for report in reports.values())
    assert all(not report["lost_reachable_atoms"] for report in reports.values())
    assert all(not report["unrepresented_strata"] for report in reports.values())
    gaps = coverage_gaps(cases, [case for c in CANDIDATES for case in cases_for(c)], load_feature_atoms())
    assert gaps["rate"] == len(gaps["covered"]) / 71
    assert gaps["required_covered"] == 64
    assert not gaps["target_met"]  # Improving generation does not imply taskbook acceptance.
    assert set(gaps["rank_counts"]) == {1, 2, 3, 4}
    assert set(gaps["size_counts"]) == {"SMALL", "MEDIUM", "LARGE", "XLARGE"}
    assert gaps["layout_counts"]["stride2"] > 0


@pytest.mark.torch
@pytest.mark.parametrize("candidate", CANDIDATES, ids=lambda candidate: candidate.name)
def test_every_advertised_profile_produces_legal_reproducible_data(candidate):
    import torch
    from triton_eval.inputs import _dtype_of, make_inputs_for
    from triton_eval.official_reference import bind, validate_inputs

    _, factory = bind(candidate.name)
    # Small shapes exercise every distribution and dtype without repeating the
    # expensive complete independent-oracle gate already run during the build.
    for case in cases_for(candidate):
        if case.shape != [4, 7] and not (candidate.family == "swiglu" and case.shape == [4, 14]):
            continue
        first = make_inputs_for(case, _dtype_of(case.dtype), "cpu", case.attrs, factory)
        repeat = make_inputs_for(case, _dtype_of(case.dtype), "cpu", case.attrs, factory)
        validate_inputs(candidate.name, first, case.attrs)
        assert all(torch.equal(x, y) for x, y in zip(first, repeat))
        main_index = 1 if candidate.family == "where" else 0
        main = first[main_index]
        assert main.is_contiguous() == case.contiguous
        profile = case.value_range["kind"]
        if profile == "zeros":
            assert bool((main == 0).all())
        elif profile == "mixed":
            assert bool((main == 0).any()) and bool((main != 0).any())
        elif profile == "extreme":
            baseline = case.model_copy(update={"value_range": {"kind": "normal", "generator": GENERATOR}})
            normal = make_inputs_for(baseline, _dtype_of(case.dtype), "cpu", case.attrs, factory)
            torch.testing.assert_close(main, normal[main_index] * 20)
        if candidate.name == "Where":
            assert first[0].dtype == torch.bool
            assert bool(first[0].any()) and bool((~first[0]).any())
        if candidate.name in ("Div", "FloorDiv", "Mod"):
            assert bool((first[1] != 0).all())


@pytest.mark.torch
@pytest.mark.parametrize("name", ["Log", "Log2", "Log10", "Rsqrt", "Pow"])
def test_strict_positive_domains_reject_false_zero_and_mixed_labels(name):
    import torch
    from triton_eval.inputs import make_inputs_for
    from triton_eval.official_reference import bind

    _, factory = bind(name)
    assert "zeros" not in numeric_profiles(BY_NAME[name], "float32")
    case = cases_for(BY_NAME[name])[0].model_copy(update={"value_range": {"kind": "zeros", "generator": GENERATOR}})
    with pytest.raises(ValueError, match="unsupported numeric profile"):
        make_inputs_for(case, torch.float32, "cpu", case.attrs, factory)


@pytest.mark.torch
@pytest.mark.parametrize("shape", [(257,), (3, 7), (2, 3, 7), (2, 3, 5, 7)])
def test_strided_construction_preserves_all_logical_values(shape):
    import torch
    from triton_eval.inputs import _make_strided

    values = torch.arange(torch.tensor(shape).prod().item(), dtype=torch.float32).reshape(shape)
    values.reshape(-1)[0] = -0.0
    result = _make_strided(values)
    assert result.shape == values.shape and not result.is_contiguous()
    assert result.stride()[-1] == 2
    assert torch.equal(result, values)
    assert torch.equal(torch.signbit(result), torch.signbit(values))


@pytest.mark.torch
def test_contract_rejects_arbitrary_stride_and_non_elementwise_layout():
    import torch
    from triton_eval.official_reference import validate_inputs

    with pytest.raises(ValueError, match="stride=2"):
        validate_inputs("Add", [torch.ones(7, 4).T, torch.ones(4, 7)], {})
    with pytest.raises(ValueError, match="only the elementwise"):
        validate_inputs("ReduceSum", [torch.ones(7, 4).T], {"dim": -1})


def test_whole_operator_selection_keeps_rare_labels_without_admitting_failures():
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
    from qualify_official_results import choose_operators
    cases, _ = select_cases(CANDIDATES)
    eligible = set(BY_NAME)
    selected = choose_operators(eligible, cases=cases)
    assert len(selected) == 50 and "Gelu" in selected
    eligible_cases = [case for case in cases if case.operator in eligible]
    selected_cases = [case for case in cases if case.operator in selected]
    all_atoms = {atom for case in eligible_cases for atom in case.feature_atoms}
    chosen_atoms = {atom for case in selected_cases for atom in case.feature_atoms}
    assert chosen_atoms == all_atoms
    eligible.remove("Gelu")
    assert "Gelu" not in choose_operators(eligible, cases=cases)
