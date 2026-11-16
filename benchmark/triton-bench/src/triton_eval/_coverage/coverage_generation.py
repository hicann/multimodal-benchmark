# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""v0.11: preserve the complete v0.9 baseline and add bounded feature witnesses.

Static labels retain the v1.1-fusion interpretation, including its declared
structure proxies. They never serve as evidence of compiled IR or execution.
"""
from __future__ import annotations

import math
from collections import Counter

from .coverage import _case_signature, coverage_stats, label_case
from .models import CaseSource, CaseSpec
from .operator_registry import by_name

VERSION = "v0.16-backend-pattern-certification-candidates"
EXTENSIONS = ("Scatter", "Cummin", "GroupedMatmul", "CrossEntropyLoss",
              "MaskedSoftmax", "ApplyRotaryPosEmb", "Transpose")
MAX_ADDITIONS = 64
MAX_PER_OPERATOR = 16


def extension_pool():
    plans = []
    # Each dtype is the actual data/update dtype, not merely an index dtype.
    for dtype in ("float32", "float16", "bfloat16", "int32", "int64"):
        for dim in (0, 1, 2):
            plans.append(("Scatter", dtype, [4, 6, 8], {"dim": dim}))
        if dtype in ("int32", "int64"):
            for dim in (0, 1, 2):
                plans.append(("Scatter", dtype, [4, 6, 8], {"dim": dim, "reduce": "add"}))
    for dtype in ("float32", "float16", "bfloat16"):
        for dim in (0, 1, 2):
            plans.append(("Cummin", dtype, [3, 5, 17], {"dim": dim}))
    for dtype in ("float32", "float16", "bfloat16"):
        for groups in ([3, 5, 8], [8, 8]):
            plans.append(("GroupedMatmul", dtype, [16, 32], {"group_list": groups}))
        for reduction in ("mean", "sum"):
            plans.append(("CrossEntropyLoss", dtype, [7, 33], {"reduction": reduction, "ignore_index": -100}))
        for dim in (0, 1, 2):
            plans.append(("MaskedSoftmax", dtype, [3, 5, 17], {"dim": dim}))
        for mode in ("half", "interleaved"):
            plans.append(("ApplyRotaryPosEmb", dtype, [3, 5, 32], {"rotary_mode": mode}))
    for dtype in ("int8", "uint8"):
        for dim in (0, 1, 2):
            plans.append(("MaskedSoftmax", dtype, [3, 5, 17], {"dim": dim}))
    for shape in ([5, 17], [7, 19]):
        plans.append(("Transpose", "int16", shape, {"perm": [1, 0]}))
    cases = []
    for index, (name, dtype, shape, attrs) in enumerate(plans):
        spec = by_name(name)
        spec.validate_case_params(dtype, shape, attrs)
        case = CaseSpec(code=f"coverage-{spec.entry}-{index:03d}", operator=name,
                        source=CaseSource.VENDOR_SELFTEST, dtype=dtype, shape=shape,
                        attrs=attrs, value_range={"kind": "normal"},
                        seed=2026102500 + index, timeout_s=120)
        case.feature_atoms = label_case(case, spec)
        cases.append(case)
    return cases


def select_extensions(baseline, pool, active, *, budget=MAX_ADDITIONS,
                      per_operator=MAX_PER_OPERATOR, target=0.9, witnesses=2):
    """Hard baseline retention; deterministic, bounded cover with non-seed witnesses.

    Every extension must gain an atom or strengthen an atom's (operator, dtype,
    shape, attrs, layout) witness. Stop as soon as all reachable missing atoms
    have two distinct structural witnesses. Never fill a quota with random cases.
    Failure to meet the target is a hard error, with no denominator adjustment.
    """
    if not 0 < target <= 1 or witnesses < 1:
        raise ValueError("invalid target/witness requirement")
    active = set(active)
    baseline_atoms = {a for c in baseline for a in c.feature_atoms} & active
    reachable = {a for c in pool for a in c.feature_atoms} & active
    wanted = reachable - baseline_atoms
    selected, trace = [], []
    counts = Counter()
    signatures = {_case_signature(c) for c in baseline}
    remaining = sorted(pool, key=lambda c: c.code)
    atom_witnesses = {atom: set() for atom in wanted}
    while len(selected) < budget:
        def gains(case):
            signature = _case_signature(case)
            candidates = [atom for atom in case.feature_atoms if atom in wanted]
            return [atom for atom in candidates if _needs_witness(atom_witnesses[atom], witnesses, signature)]
        available = _available_coverage_cases(remaining, counts, per_operator, signatures, gains)
        if not available:
            break

        def priority(case):
            gain = gains(case)
            fresh = sum(not atom_witnesses[a] for a in gain)
            return (-fresh, -len(gain), counts[case.operator], math.prod(case.shape), case.code)
        case = min(available, key=priority)
        gain = gains(case)
        trace.append({"case": case.code, "operator": case.operator,
                      "new_atoms": sorted(a for a in gain if not atom_witnesses[a]),
                      "strengthened_atoms": sorted(a for a in gain if atom_witnesses[a]),
                      "reason": "missing_feature_or_distinct_structural_witness"})
        selected.append(case)
        counts[case.operator] += 1
        signature = _case_signature(case)
        signatures.add(signature)
        for atom in gain:
            atom_witnesses[atom].add(signature)
        remaining.remove(case)
    stats = coverage_stats([*baseline, *selected], sorted(active))
    short = {a: len(v) for a, v in atom_witnesses.items() if len(v) < witnesses}
    report = dict(stats, policy="baseline-preserving-feature-witness-v1",
                  baseline_cases=len(baseline), retained_baseline_cases=len(baseline),
                  added_cases=len(selected), max_additions=budget, per_operator_budget=per_operator,
                  required_covered=math.ceil(len(active) * target), target=target,
                  target_met=len(stats["covered"]) >= math.ceil(len(active) * target),
                  witness_requirement=witnesses, insufficient_witnesses=short,
                  additions_by_operator=dict(sorted(counts.items())), selection_trace=trace,
                  coverage_basis=("static case-label union; includes declared structure proxies; not compiled API/IR "
                                  "coverage"),
                  compilation_state="NOT_RUN", execution_state="NOT_RUN",
                  agreement_state="AWAITING_JOINT_CONFIRMATION")
    if not report["target_met"] or short:
        raise ValueError(f"static coverage/witness budget gate failed: {len(stats['covered'])}/{len(active)}; {short}")
    return selected, report


def _needs_witness(existing, witnesses, signature):
    return len(existing) < witnesses and signature not in existing


def _available_coverage_cases(remaining, counts, per_operator, signatures, gains):
    available = []
    for case in remaining:
        if not counts[case.operator] < per_operator:
            continue
        if _case_signature(case) in signatures:
            continue
        if gains(case):
            available.append(case)
    return available
