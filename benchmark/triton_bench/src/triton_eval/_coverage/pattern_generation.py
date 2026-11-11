# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bounded, target-obligation selection for the four remaining patterns."""
from __future__ import annotations

import json
import math
from collections import Counter
from typing import NamedTuple

VERSION = 'v0.17-pattern-completion-candidates'
SCAN_REPAIR_VERSION = 'v0.18-scan-certification-candidates'
PATTERN_VERSIONS = (VERSION, SCAN_REPAIR_VERSION)
PARENT = 'v0.16-backend-pattern-certification-candidates'
PARENT_MANIFEST_HASH = 'e758eaef22e06c06a0d533bc5dabb1ee2bd35b74ff2af490dd2d718c57eff546'
TARGETS = {
    'FA-PAT-MULTIPASS-REDUCE': 'ReduceMax',
    'FA-PAT-SCAN-BLOCKED': 'Cummin',
    'FA-PAT-SPLIT-K': 'Matmul',
    'FA-PAT-VEC-BROADCAST': 'GroupedMatmul',
}
MAX_ADDITIONS = 16
MAX_PER_OPERATOR = 4
MAX_DEVICE_BYTES = 64 * 1024 * 1024


def _invalid_selection_limits(targets, active, witnesses, budget, per_operator, max_device_bytes):
    if (not targets):
        return True
    if (not targets <= set(TARGETS)):
        return True
    if (not targets <= active):
        return True
    if (witnesses < 1):
        return True
    if (budget < 1):
        return True
    if (per_operator < 1):
        return True
    return bool(max_device_bytes < 1)


def _is_small_output_large_inner(case, rows):
    if case.operator != 'Matmul':
        return False
    if len(case.shape) != 2:
        return False
    if not (rows <= 64):
        return False
    return bool(case.shape[1] >= 2**16)


def _is_large_row_reduction(case, axis):
    if case.operator != 'ReduceMax':
        return False
    if len(case.shape) != 2:
        return False
    if axis != 1:
        return False
    return bool(case.shape[1] >= 2**20)


class _StructuralSignature(NamedTuple):
    operator: str
    dtype: str
    shape: tuple
    attrs: str
    contiguous: bool
    value_range: str


def structural_signature(case):
    return _StructuralSignature(
        operator=case.operator, dtype=case.dtype, shape=tuple(case.shape),
        attrs=json.dumps(case.attrs, sort_keys=True, separators=(',', ':')),
        contiguous=bool(case.contiguous), value_range=json.dumps(case.value_range, sort_keys=True))


def target_features(case):
    if not case.contiguous or not case.shape or any(size < 1 for size in case.shape):
        return set()
    rows = case.shape[0]
    axis = case.attrs.get('dim', -1) % len(case.shape)
    if _is_large_row_reduction(case, axis):
        return {'FA-PAT-MULTIPASS-REDUCE'}
    if case.operator == 'Cummin' and case.shape[axis] >= 2**20:
        return {'FA-PAT-SCAN-BLOCKED'}
    if _is_small_output_large_inner(case, rows):
        return {'FA-PAT-SPLIT-K'}
    if case.operator == 'GroupedMatmul' and len(case.shape) == 2 and rows >= 2048:
        groups = case.attrs.get('group_list', [])
        if groups and all(size > 0 for size in groups) and sum(groups) == rows:
            return {'FA-PAT-VEC-BROADCAST'}
    return set()


def device_bytes(case):
    width = {'float16': 2, 'bfloat16': 2, 'float32': 4}.get(case.dtype)
    if width is None:
        raise ValueError('pattern candidates require a reviewed floating dtype')
    elements = math.prod(case.shape)
    if case.operator == 'Cummin':
        chunks = math.ceil(case.shape[case.attrs['dim'] % len(case.shape)] / 512)
        lines = elements // case.shape[case.attrs['dim'] % len(case.shape)]
        return elements * (2 * width + 20) + lines * chunks * 24
    if case.operator == 'ReduceMax':
        return elements * width + case.shape[0] * (math.ceil(case.shape[1] / 1024) + 1) * 4
    if case.operator == 'Matmul':
        rows, inner = case.shape
        columns = rows + 3
        return (elements + inner * columns + rows * columns) * width + 8 * rows * columns * 4
    if case.operator == 'GroupedMatmul':
        groups = len(case.attrs['group_list'])
        return (elements + groups * case.shape[1] * 16 + groups * 16 + case.shape[0] * 16) * width
    raise ValueError('unknown pattern carrier')


def candidate_plans():
    return [
        ('ReduceMax', 'float32', [1, 2**20], {'dim': -1, 'keepdim': False}),
        ('ReduceMax', 'float32', [2, 2**20 + 257], {'dim': 1, 'keepdim': True}),
        ('Cummin', 'float32', [1, 2**20], {'dim': 1}),
        ('Cummin', 'float32', [1, 2**20 + 257], {'dim': -1}),
        ('Matmul', 'float16', [16, 2**16], {}),
        ('Matmul', 'float16', [32, 2**16 + 64], {}),
        ('GroupedMatmul', 'float16', [2048, 32], {'group_list': [2048]}),
        ('GroupedMatmul', 'float16', [2064, 64], {'group_list': [1024, 1040]}),
    ]


def label_pattern_case(case, spec):
    from .coverage import label_case

    atoms = set(label_case(case, spec))
    if 'FA-PAT-VEC-BROADCAST' in atoms and 'FA-PAT-VEC-BROADCAST' not in target_features(case):
        atoms.remove('FA-PAT-VEC-BROADCAST')
    return sorted(atoms)


def extension_pool():
    from .models import CaseSource, CaseSpec
    from .operator_registry import by_name

    pool = []
    for index, (operator, dtype, shape, attrs) in enumerate(candidate_plans()):
        spec = by_name(operator)
        spec.validate_case_params(dtype, shape, attrs)
        case = CaseSpec(code=f'pattern-{spec.entry}-{index:03d}', operator=operator,
                        source=CaseSource.VENDOR_SELFTEST, dtype=dtype, shape=shape,
                        attrs=attrs, seed=2026102700 + index, timeout_s=180)
        case.feature_atoms = label_pattern_case(case, spec)
        if not target_features(case) <= set(case.feature_atoms):
            raise ValueError('target workload is not supported by derived static labels')
        pool.append(case)
    return pool


def select_patterns(baseline, pool, active, *, targets=None, witnesses=2,
                    budget=MAX_ADDITIONS, per_operator=MAX_PER_OPERATOR,
                    max_device_bytes=MAX_DEVICE_BYTES):
    targets = set(TARGETS if targets is None else targets)
    active = set(active)
    if (_invalid_selection_limits(targets, active, witnesses, budget, per_operator, max_device_bytes)):
        raise ValueError('invalid explicit pattern obligations or budget')
    if len({case.code for case in baseline}) != len(baseline):
        raise ValueError('duplicate baseline case codes')
    if len({case.code for case in pool}) != len(pool):
        raise ValueError('duplicate candidate case codes')
    signatures = {structural_signature(case) for case in baseline}
    codes = {case.code for case in baseline}
    seen = {atom: set() for atom in targets}
    selected, trace = [], []
    counts = Counter()
    remaining = sorted(pool, key=lambda case: case.code)
    reachable = set().union(*(target_features(case) & set(case.feature_atoms) for case in remaining))
    if targets - reachable:
        raise ValueError(f'unreachable required pattern targets: {sorted(targets - reachable)}')
    while len(selected) < budget:
        def gains(case):
            signature = structural_signature(case)
            candidates = target_features(case) & set(case.feature_atoms) & targets
            return {atom for atom in candidates if _needs_witness(seen[atom], witnesses, signature)}

        available = _available_pattern_cases(
            remaining, codes, signatures, counts, per_operator, max_device_bytes, gains)
        if not available:
            break
        case = min(available, key=lambda candidate: (
            -sum(not seen[atom] for atom in gains(candidate)), -len(gains(candidate)),
            device_bytes(candidate), counts[candidate.operator], candidate.code))
        gain = gains(case)
        signature = structural_signature(case)
        trace.append(dict(case=case.code, operator=case.operator,
                          target_obligations=sorted(gain), estimated_device_bytes=device_bytes(case),
                          reason='required_pattern_distinct_structure',
                          compilation_state='NOT_RUN', execution_state='NOT_RUN'))
        for atom in gain:
            seen[atom].add(signature)
        selected.append(case)
        signatures.add(signature)
        codes.add(case.code)
        counts[case.operator] += 1
        remaining.remove(case)
    return _pattern_selection_report(
        baseline, selected, seen, witnesses, active, targets, budget, per_operator, max_device_bytes, trace)


def _needs_witness(existing, witnesses, signature):
    return len(existing) < witnesses and signature not in existing


def _available_pattern_cases(remaining, codes, signatures, counts, per_operator, max_device_bytes, gains):
    available = []
    for case in remaining:
        if case.code in codes or structural_signature(case) in signatures:
            continue
        if not counts[case.operator] < per_operator or not device_bytes(case) <= max_device_bytes:
            continue
        if gains(case):
            available.append(case)
    return available


def _pattern_selection_report(
        baseline, selected, seen, witnesses, active, targets, budget, per_operator, max_device_bytes, trace):
    short = {atom: len(signatures) for atom, signatures in seen.items() if len(signatures) < witnesses}
    if short:
        raise ValueError(f'required pattern witnesses not satisfied within budget: {short}')
    covered = {atom for case in [*baseline, *selected] for atom in case.feature_atoms} & active
    if len(covered) < math.ceil(0.9 * len(active)):
        raise ValueError('global static coverage target not met')
    return selected, dict(policy='explicit-pattern-obligation-v1', parent_version=PARENT,
                          retained_parent_cases=len(baseline), added_cases=len(selected),
                          required_targets=sorted(targets), witness_requirement=witnesses,
                          target_witnesses={atom: len(items) for atom, items in sorted(seen.items())},
                          max_additions=budget, per_operator_budget=per_operator,
                          max_device_bytes=max_device_bytes, selection_trace=trace,
                          covered=sorted(covered), missing=sorted(active - covered),
                          rate=len(covered) / len(active), compilation_state='NOT_RUN',
                          execution_state='NOT_RUN', agreement_state='AWAITING_JOINT_CONFIRMATION')
