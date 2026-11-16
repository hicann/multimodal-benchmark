# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Baseline-preserving, coverage-guided completion of sparse extension workloads."""

from __future__ import annotations
import math
from collections import Counter
from dataclasses import dataclass
from typing import NamedTuple
from functools import partial
from triton_eval.coverage import _case_signature, label_case, size_class
from triton_eval.models import CaseSource, CaseSpec
from triton_eval.operator_registry import by_name

VERSION = "v0.19-workload-completion-candidates"
PARENT = "v0.18-scan-certification-candidates"
PARENT_HASH = "0890185080c5ad36eb0bb8722f975b4b2bd820e99feb702d328f1d40b19258a0"
TARGETS = (
    "Scatter",
    "Cummin",
    "GroupedMatmul",
    "CrossEntropyLoss",
    "MaskedSoftmax",
    "ApplyRotaryPosEmb",
)
TIERS = ("SMALL", "MEDIUM", "LARGE", "XLARGE")


SHAPES = {
    "Scatter": [[4, 6, 8], [7, 9, 17], [8, 16, 64], [16, 32, 64], [64, 64, 128], [65, 65, 512]],
    "Cummin": [[3, 5, 17], [7, 9, 31], [16, 32, 32], [64, 64, 32], [128, 129, 32], [257, 257, 32]],
    "GroupedMatmul": [[16, 32], [31, 32], [256, 32], [1024, 32], [16385, 32], [65537, 32]],
    "CrossEntropyLoss": [[7, 33], [17, 65], [33, 257], [128, 513], [513, 1024], [2049, 1024]],
    "MaskedSoftmax": [
        [3, 5, 17],
        [7, 9, 31],
        [8, 16, 64],
        [16, 32, 64],
        [64, 65, 128],
        [129, 129, 128],
    ],
    "ApplyRotaryPosEmb": [
        [3, 5, 32],
        [7, 9, 64],
        [8, 16, 64],
        [16, 32, 64],
        [64, 65, 128],
        [257, 129, 64],
    ],
}


def _case_attributes(name, dtype, shape):
    if name == "Scatter":
        attributes = [dict(dim=dim) for dim in (0, 1, 2)]
        if dtype in ("int32", "int64"):
            attributes += [dict(dim=dim, reduce="add") for dim in (0, 1, 2)]
        return attributes
    if name == "Cummin":
        return [dict(dim=dim) for dim in range(3) if shape[dim] <= 32]
    if name == "GroupedMatmul":
        rows = shape[0]
        return [dict(group_list=[rows]), dict(group_list=[rows // 2, rows - rows // 2])]
    if name == "CrossEntropyLoss":
        return [dict(reduction=reduction, ignore_index=-100) for reduction in ("mean", "sum")]
    if name == "MaskedSoftmax":
        return [dict(dim=dim) for dim in (0, 1, 2) if shape[dim] <= 128]
    return [dict(rotary_mode=mode) for mode in ("half", "interleaved")]


def _make_candidate(name, dtype, shape, parameters, ordinal):
    spec = by_name(name)
    spec.validate_case_params(dtype, shape, parameters)
    case = CaseSpec(
        code=f"workload-{spec.entry}-{ordinal:04d}",
        operator=name,
        source=CaseSource.VENDOR_SELFTEST,
        dtype=dtype,
        shape=shape,
        attrs=parameters,
        value_range={"kind": "normal"},
        seed=2026111300 + ordinal,
        timeout_s=120,
    )
    case.feature_atoms = label_case(case, spec)
    return case


def candidate_pool():
    output = []
    for name in TARGETS:
        dtypes = (
            ("float32", "float16", "bfloat16", "int32", "int64")
            if name == "Scatter"
            else ("float32", "float16", "bfloat16")
        )
        for dtype in dtypes:
            _append_candidates(output, name, dtype)
    return output


def _append_candidates(output, name, dtype):
    for shape in SHAPES[name]:
        for parameters in _case_attributes(name, dtype, shape):
            output.append(_make_candidate(name, dtype, shape, parameters, len(output)))
    return output


@dataclass
class SelectionContext:
    counts: Counter
    pairs: set
    attributes: set


def _selection_context(rows):
    pairs = set()
    attributes = set()
    counts = Counter()
    for case in rows:
        tier = size_class(math.prod(case.shape))
        counts[tier] += 1
        attributes.add(str(sorted(case.attrs.items())))
        for atom in case.feature_atoms:
            pairs.add((atom, case.dtype, tier))
    return SelectionContext(counts, pairs, attributes)


class CandidatePriority(NamedTuple):
    tier_deficit: int
    tier_count: int
    feature_gain: int
    attribute_gain: int
    elements: int
    code: str


def _priority(context, case):
    tier = size_class(math.prod(case.shape))
    gain = sum((atom, case.dtype, tier) not in context.pairs for atom in case.feature_atoms)
    return CandidatePriority(
        -(context.counts[tier] < 3),
        context.counts[tier],
        -gain,
        -(str(sorted(case.attrs.items())) not in context.attributes),
        math.prod(case.shape),
        case.code,
    )


def _selection_record(context, case, name):
    tier = size_class(math.prod(case.shape))
    return dict(
        case=case.code,
        operator=name,
        tier=tier,
        reason="size_dtype_attribute_feature_gain",
        previous_tier_count=context.counts[tier],
        new_feature_dtype_tier_pairs=sum(
            (atom, case.dtype, tier) not in context.pairs for atom in case.feature_atoms
        ),
    )


def _complete_operator(name, rows, pool, signatures, target):
    selected, trace = [], []
    while len(rows) < target:
        context = _selection_context(rows)
        remaining = [
            case
            for case in pool
            if case.operator == name and _case_signature(case) not in signatures
        ]
        if not remaining:
            raise ValueError(f"insufficient distinct candidates: {name}")
        case = min(remaining, key=partial(_priority, context))
        trace.append(_selection_record(context, case, name))
        rows.append(case)
        selected.append(case)
        signatures.add(_case_signature(case))
    if set(TIERS) - {size_class(math.prod(case.shape)) for case in rows}:
        raise ValueError("missing workload tier")
    return selected, trace


def select_completion(baseline, pool=None, target=20):
    pool = candidate_pool() if pool is None else pool
    signatures = {_case_signature(case) for case in baseline}
    selected, trace = [], []
    for name in TARGETS:
        rows = [case for case in baseline if case.operator == name]
        additions, events = _complete_operator(name, rows, pool, signatures, target)
        selected.extend(additions)
        trace.extend(events)
    return selected, dict(
        policy="baseline-preserving-workload-completion-v1",
        target_per_operator=target,
        additions_by_operator=dict(Counter(case.operator for case in selected)),
        selection_trace=trace,
    )
