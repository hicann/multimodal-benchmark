# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Recompute static readiness; reports alone cannot open the NPU launch gate."""
from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path

from .buildlog import read_log
from .corpus_assets import verify_assets
from .corpusmod import (
    compute_case_content_hash,
    load_manifest,
    verify_corpus_files,
    verify_manifest,
)
from .coverage import coverage_stats, label_case
from .operator_registry import by_name
from .specload import load_feature_atoms

BASELINE_HASH = '885788c6a9be45b09b8e5e08def0b7b904648dd5de0a726111013b20c7dd7d8a'


def static_readiness(corpus):
    corpus = Path(corpus)
    manifest = load_manifest(corpus / 'manifest.json')
    if manifest.version in ('v0.17-pattern-completion-candidates', 'v0.18-scan-certification-candidates'):
        from .pattern_release import static_readiness as pattern_readiness
        return pattern_readiness(corpus)
    verify_corpus_files(corpus, manifest)
    verify_assets(corpus, manifest.manifest_hash)
    plan = json.loads((corpus / 'coverage-plan.json').read_text())
    atoms = load_feature_atoms()
    active = [a['id'] for a in atoms['atoms'] if a.get('status', 'active') == 'active']
    if plan['active_atoms'] != sorted(active) or manifest.feature_atoms_version != str(atoms['version']):
        raise ValueError('feature denominator/version drift')
    if plan['baseline_manifest_hash'] != BASELINE_HASH:
        raise ValueError('unrecognized baseline identity')
    cases = {c.code: c for c in manifest.cases}
    if len(cases) != len(manifest.cases):
        raise ValueError('duplicate case codes')
    parent, parents = _baseline_cases(cases, plan)
    _validate_static_cases(manifest.cases, parents)
    additions = [c for c in manifest.cases if c.code not in parents]
    _validate_additions(corpus, plan, parent, active, additions)
    _validate_generation(corpus, manifest.manifest_hash, len(cases))
    stats = coverage_stats(manifest.cases, active)
    required = math.ceil(0.9 * len(active))
    if len(manifest.cases) < 1000 or len(stats['covered']) < required:
        raise ValueError(f'static launch gate failed: {len(stats["covered"])}/{len(active)}, requires {required}')
    return dict(stats, manifest_hash=manifest.manifest_hash, required_covered=required,
                cases=len(cases), preserved_cases=len(parents), added_cases=len(additions),
                static_target_met=True, compilation_state='NOT_RUN', execution_state='NOT_RUN',
                agreement_state='AWAITING_JOINT_CONFIRMATION',
                coverage_basis=plan['coverage_basis'])


def _baseline_cases(cases, plan):
    from .corpus_assets import ROOT
    parent = load_manifest(ROOT / 'corpus/v0.9-official-candidates/manifest.json')
    verify_manifest(parent)
    if parent.manifest_hash != BASELINE_HASH:
        raise ValueError('canonical baseline identity mismatch')
    parents = {c.code: c.content_hash for c in parent.cases}
    if parents != plan['baseline_case_hashes']:
        raise ValueError('baseline hash map differs from canonical frozen corpus')
    if len(parents) != 1360 or any(code not in cases or cases[code].content_hash != h for code, h in parents.items()):
        raise ValueError('baseline case removed or changed')
    baseline_counts = Counter(cases[code].operator for code in parents)
    if len(baseline_counts) != 68 or set(baseline_counts.values()) != {20}:
        raise ValueError('must retain all original 20 cases for every baseline operator')
    return parent, parents


def _validate_static_cases(cases, parents):
    from .official_catalog import BY_NAME, specification
    for case in cases:
        spec = specification(BY_NAME[case.operator]) if case.code in parents else by_name(case.operator)
        if case.feature_atoms != label_case(case, spec):
            raise ValueError(f'non-derived static labels: {case.code}')
        # Every case must be runnable within the registered specification.
        spec.validate_case_params(case.dtype, case.shape, case.attrs)


def _validate_additions(corpus, plan, parent, active, additions):
    from .coverage_generation import (
        MAX_ADDITIONS,
        MAX_PER_OPERATOR,
        extension_pool,
        select_extensions,
    )
    if plan['max_additions'] != MAX_ADDITIONS or plan['per_operator_budget'] != MAX_PER_OPERATOR:
        raise ValueError('selection budget policy drift')
    if len(additions) > plan['max_additions'] or any(n > plan['per_operator_budget']
                                                     for n in Counter(c.operator for c in additions).values()):
        raise ValueError('extension budget exceeded')
    expected, selection = select_extensions(parent.cases, extension_pool(), active)
    if [c.code for c in additions] != [c.code for c in expected]:
        raise ValueError('supplements differ from reproducible coverage-guided selection')
    if any(compute_case_content_hash(corpus, c) != compute_case_content_hash(corpus, e)
           for c, e in zip(additions, expected)):
        raise ValueError('supplement parameters differ from generated witnesses')
    if any(plan.get(key) != value for key, value in selection.items()):
        raise ValueError('selection trace/witness policy mismatch')


def _validate_generation(corpus, manifest_hash, case_count):
    generation = read_log(corpus / "generation-log.yaml")
    starts = [i for i, event in enumerate(generation) if event.get('step') == 'build_start']
    attempt = generation[starts[-1]:] if starts else []
    gates = [event for event in attempt if event.get('step') == 'independent_cpu_reference_gate']
    if not attempt or attempt[-1].get('step') != 'build_end':
        raise ValueError('missing successful generation reference gate')
    if attempt[-1].get('manifest_hash') != manifest_hash or len(gates) != 1:
        raise ValueError('missing successful generation reference gate')
    if gates[0].get('failures') != {} or gates[0].get('cases') != case_count:
        raise ValueError('missing successful generation reference gate')
