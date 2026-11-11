# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Versioned official candidate generation and bounded, auditable coverage selection.

Only input dimensions supported by the declared contracts are offered. Profiles
are applied by the domain-aware input factory, never by relabeling an old case.
"""

from __future__ import annotations

import json
import math
from collections import Counter

from .coverage import _case_signature, coverage_stats, label_case, size_class
from .models import CaseSource, CaseSpec
from .official_catalog import ELEMENTWISE_FAMILIES, specification

GENERATOR = "official-domain-v2"
SHAPES = ((1, 1), (2, 2), (3, 3), (4, 7), (5, 8), (7, 15), (8, 16), (9, 31),
          (15, 32), (16, 33), (17, 63), (16, 64), (32, 65), (33, 127), (32, 128),
          (64, 129), (65, 255), (7, 256), (17, 257), (33, 511), (65, 512), (65, 513))
POSITIVE_INPUT = frozenset(("Log", "Log2", "Log10", "Rsqrt", "Pow"))
# ±20σ profiles only where the full computation remains in the legal finite
# domain; exclusions are explicit rather than silently clamping dangerous data.
EXTREME_INPUT = frozenset(("Abs", "Neg", "Add", "Sub", "Mul", "Div", "Maximum", "Minimum",
                           "Equal", "NotEqual", "Greater", "GreaterEqual", "Less", "LessEqual",
                           "Where", "Clamp", "Relu", "Sigmoid", "Tanh", "Softmax", "RmsNorm", "LayerNorm"))
ANCHORS = (0, 2, 3, 21)


def numeric_profiles(candidate, dtype):
    profiles = ["normal", "boundary"]
    if candidate.name not in POSITIVE_INPUT:
        profiles.extend(("zeros", "mixed"))
    if dtype == "float32" and candidate.name in EXTREME_INPUT:
        profiles.append("extreme")
    return tuple(profiles)


def _attrs(candidate, shape, index, spec):
    family = candidate.family
    attrs = {}
    if family in ("reduce", "arg", "scan", "softmax", "gather", "swiglu", "concat", "flip", "sort", "topk"):
        attrs["dim"] = -1 if index % 2 else 1
    if family in ("reduce", "arg") and spec.attr("keepdim"):
        attrs["keepdim"] = bool(index % 2)
    if family == "transpose":
        attrs["perm"] = [1, 0]
    if family in ("rms", "layer"):
        attrs["epsilon"] = 1e-6 if index % 2 else 1e-4
    if family == "sort":
        attrs["descending"] = bool(index % 2)
    if family == "topk":
        attrs["k"] = max(1, shape[-1] // 3) if index % 3 else shape[-1]
    attrs.update(_operator_attrs(candidate.name, index))
    return attrs


def _operator_attrs(name, index):
    attrs = {}
    if name == "Gelu":
        attrs["approximate"] = "none" if index % 2 else "tanh"
    if name == "Exp":
        attrs.update(base=-1.0 if index % 2 else 2.0, scale=1.0 if index % 3 else 0.5,
                     shift=0.0 if index % 3 else -0.25)
    if name == "Clamp":
        attrs.update(lower=-0.5 if index % 2 else -1.0, upper=0.5 if index % 2 else 1.0)
    return attrs


def cases_for(candidate):
    from itertools import product

    spec = specification(candidate)
    shapes = list(SHAPES)
    if candidate.family in ELEMENTWISE_FAMILIES:
        shapes.extend(((257,), (3, 7, 17), (2, 3, 5, 17), (2, 4096), (2, 4097),
                       (128, 4096), (513, 4096)))
    elif candidate.family == "matmul":
        # Weight shape is (K, M+3): bound M instead of accidentally creating
        # a quadratic output by reusing the tall row-operation workloads.
        shapes.extend(((128, 128), (128, 2048), (129, 2048)))
    else:
        shapes.extend(((256, 257), (257, 1024), (2049, 1024)))
    result = []
    for index, base_shape in enumerate(shapes):
        shape = list(base_shape)
        if candidate.family == "swiglu":
            shape[-1] *= 2
        attrs = _attrs(candidate, shape, index, spec)
        for dtype_index, dtype in enumerate(candidate.dtypes):
            spec.validate_case_params(dtype, shape, attrs)
            profiles = numeric_profiles(candidate, dtype) if index in (3, 11) else ("normal",)
            layouts = (
                True,
                False) if candidate.family in ELEMENTWISE_FAMILIES and index in (
                3,
                23,
                24) else (
                True,
            )
            for profile, contiguous in product(profiles, layouts):
                case = CaseSpec(
                    code=f"official-{candidate.entry}-{dtype}-{index:02d}-{profile}-{'c' if contiguous else 's2'}",
                    operator=candidate.name, source=CaseSource.VENDOR_SELFTEST, dtype=dtype,
                    shape=shape, attrs=attrs, value_range={"kind": profile, "generator": GENERATOR},
                    contiguous=contiguous, seed=2026102400 + index * 7 + dtype_index, timeout_s=120)
                case.feature_atoms = label_case(case, spec)
                result.append(case)
    return result


def _strata(case):
    return {("dtype", case.dtype), ("size", size_class(math.prod(case.shape))),
            ("rank", len(case.shape)), ("layout", case.contiguous),
            ("profile", case.value_range["kind"])}


def _attributes(case):
    return {(name, json.dumps(value, sort_keys=True)) for name, value in case.attrs.items()}


def bounded_selection(pool, mandatory, budget=20):
    """Greedy marginal coverage with a hard budget and already-covered anchors.

    Labels, dtype/profile/shape/layout strata, attribute diversity and then cost
    determine priority, in that order. Seed-only variants cannot consume slots.
    No promise of an exact set-cover optimum; all remaining gaps are reported.
    """
    unique = {}
    for case in sorted(pool, key=lambda item: item.code):
        unique.setdefault(_case_signature(case), case)
    candidates = list(unique.values())
    if len({case.code for case in mandatory}) != len(mandatory):
        raise ValueError("duplicate mandatory cases")
    if len(mandatory) > budget or len(candidates) < budget:
        raise ValueError("mandatory cases or candidate count incompatible with hard budget")
    if any(case not in candidates for case in mandatory):
        raise ValueError("mandatory case absent from deduplicated pool")
    selected, atoms, strata, attributes, trace = [], set(), set(), set(), []

    def add(case, reason):
        trace.append({"code": case.code, "reason": reason,
                      "new_atoms": sorted(set(case.feature_atoms) - atoms),
                      "new_strata": sorted([list(item) for item in _strata(case) - strata], key=str)})
        selected.append(case)
        atoms.update(case.feature_atoms)
        strata.update(_strata(case))
        attributes.update(_attributes(case))
        candidates.remove(case)

    for case in sorted(mandatory, key=lambda item: item.code):
        add(case, "mandatory_boundary")
    while len(selected) < budget:
        def priority(case):
            return (-len(set(case.feature_atoms) - atoms), -len(_strata(case) - strata),
                    -len(_attributes(case) - attributes), math.prod(case.shape), case.code)
        add(min(candidates, key=priority), "marginal_coverage_and_strata")
    reachable = sorted({atom for case in pool for atom in case.feature_atoms})
    all_strata = set().union(*(_strata(case) for case in pool))
    report = {"policy": "bounded-marginal-v2", "budget": budget, "pool_size": len(pool),
              "deduplicated_pool_size": len(unique), "selected": len(selected),
              "mandatory_cases": [case.code for case in mandatory], "selection_trace": trace,
              "reachable_atoms": reachable, "lost_reachable_atoms": sorted(set(reachable) - atoms),
              "unrepresented_strata": sorted([list(item) for item in all_strata - strata], key=str),
              "coverage": coverage_stats(selected, reachable)}
    return sorted(selected, key=lambda item: item.code), report


def select_cases(candidates):
    selected, reports = [], {}
    for candidate in candidates:
        pool = cases_for(candidate)
        mandatory = [case for case in pool if _is_anchor_case(case, candidate)]
        cases, report = bounded_selection(pool, mandatory)
        report["profile_exclusions"] = _profile_exclusions(candidate)
        selected.extend(cases)
        reports[candidate.name] = report
    return selected, reports


def coverage_gaps(cases, pool, atom_specs):
    """Global union coverage over the unchanged active denominator, with gaps."""
    active = [item["id"] for item in atom_specs["atoms"] if item.get("status", "active") == "active"]
    reachable = {atom for case in pool for atom in case.feature_atoms} & set(active)
    covered = {atom for case in cases for atom in case.feature_atoms} & set(active)
    required = math.ceil(0.9 * len(active))
    return dict(coverage_stats(cases, active),
                coverage_basis="static case-label union; not executed API/IR/instruction coverage or pass rate",
                target_fraction=0.9, required_covered=required, target_met=len(covered) >= required,
                remaining_to_target=max(0, required - len(covered)), reachable_atoms=sorted(reachable),
                unreachable_atoms=sorted(set(active) - reachable),
                lost_reachable_atoms=sorted(reachable - covered),
                per_operator=_operator_coverage(cases, active),
                profile_counts=dict(Counter(case.value_range["kind"] for case in cases)),
                rank_counts=dict(Counter(len(case.shape) for case in cases)),
                layout_counts=dict(Counter("contiguous" if case.contiguous else "stride2" for case in cases)),
                size_counts=dict(Counter(size_class(math.prod(case.shape)) for case in cases)))


def _is_anchor_case(case, candidate):
    if case.dtype != candidate.dtypes[0]:
        return False
    if case.value_range["kind"] != "normal":
        return False
    if not case.contiguous:
        return False
    return int(case.code.split("-")[-3]) in ANCHORS


def _profile_exclusions(candidate):
    exclusions = {}
    for dtype in candidate.dtypes:
        profiles = {}
        for kind in ("normal", "boundary", "zeros", "mixed", "extreme"):
            if kind not in numeric_profiles(candidate, dtype):
                profiles[kind] = ("main input must be strictly positive"
                                  if candidate.name in POSITIVE_INPUT and kind in ("zeros", "mixed")
                                  else "no finite domain safety contract for this dtype/operator")
        exclusions[dtype] = profiles
    return exclusions


def _operator_coverage(cases, active):
    result = {}
    for name in sorted({case.operator for case in cases}):
        selected = [case for case in cases if case.operator == name]
        result[name] = coverage_stats(selected, active)
    return result
