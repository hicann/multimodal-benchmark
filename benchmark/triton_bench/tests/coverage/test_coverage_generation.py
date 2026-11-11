# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Coverage generation must preserve history and add real, independent witnesses."""
import json
import math
import shutil
from collections import Counter
from pathlib import Path

import pytest

from triton_eval.corpusmod import load_manifest, manifest_digest, compute_case_content_hash
from triton_eval.coverage import _case_signature
from triton_eval.coverage_generation import extension_pool, select_extensions
from triton_eval.coverage_gate import static_readiness
from triton_eval.specload import load_feature_atoms
from triton_eval.buildlog import read_log, write_log

from triton_eval.fixture_sources import fixture_path

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / 'corpus/v0.18-scan-certification-candidates'
LEGACY = ROOT / 'corpus/v0.16-backend-pattern-certification-candidates'
BASE = ROOT / 'corpus/v0.9-official-candidates'


def active_atoms():
    return sorted(a['id'] for a in load_feature_atoms()['atoms'] if a.get('status', 'active') == 'active')


def test_selection_is_bounded_deterministic_and_preserves_entire_baseline():
    baseline = load_manifest(BASE / 'manifest.json').cases
    pool = extension_pool()
    selected, report = select_extensions(baseline, pool, active_atoms())
    again, audit = select_extensions(baseline, list(reversed(pool)), active_atoms())
    assert selected == again and report == audit
    assert len(pool) == 65 and len(selected) == 17
    assert math.isclose(report['rate'], 67 / 71, rel_tol=0.0, abs_tol=0.0)
    assert report['retained_baseline_cases'] == 1360
    assert report['insufficient_witnesses'] == {}
    assert all(t['new_atoms'] or t['strengthened_atoms'] for t in report['selection_trace'])
    assert max(Counter(c.operator for c in selected).values()) <= 16
    frozen = load_manifest(LEGACY / 'manifest.json')
    assert len(frozen.cases) == 1377 and len(frozen.operators) == 74
    assert frozen.cases[:1360] == baseline
    assert len({_case_signature(c) for c in selected}) == len(selected)
    assert {c.code for c in selected} == {c.code for c in frozen.cases[1360:]}


@pytest.mark.parametrize('kwargs', [{'budget': 1}, {'per_operator': 1}, {'target': 1.0}])
def test_insufficient_budget_or_unreachable_target_fails(kwargs):
    with pytest.raises(ValueError, match='budget gate failed'):
        select_extensions(load_manifest(BASE / 'manifest.json').cases, extension_pool(), active_atoms(), **kwargs)


def test_seed_changes_cannot_supply_a_second_structural_witness():
    case = extension_pool()[0]
    duplicate = case.model_copy(deep=True)
    duplicate.code += '-another-seed'
    duplicate.seed += 1
    with pytest.raises(ValueError, match='budget gate failed'):
        select_extensions([], [case, duplicate], case.feature_atoms, target=1)


def test_local_static_gate_recomputes_all_static_labels_without_npu():
    report = static_readiness(CORPUS)
    assert report['rate'] == 1 and report['required_covered'] == 64
    assert report['cases'] == 1385
    assert report['preserved_cases'] == 1377 and report['added_cases'] == 8
    assert report['compilation_state'] == report['execution_state'] == 'NOT_RUN'
    assert report['agreement_state'] == 'AWAITING_JOINT_CONFIRMATION'


@pytest.mark.parametrize('change,match', [
    ('delete', 'baseline case removed'), ('alter', 'baseline case removed'),
    ('denominator', 'denominator/version drift'), ('parent-map', 'canonical frozen corpus'),
    ('invent-tag', 'non-derived static labels'), ('budget', 'budget policy drift'),
    ('trace', 'trace/witness policy mismatch'), ('reference', 'generation reference gate'),
    ('supplement', 'supplement parameters'),
])
def test_static_gate_rejects_rehashed_policy_or_corpus_tampering(tmp_path, change, match):
    """Refreezing a modified test fixture cannot bypass the semantic gate."""
    from triton_eval.corpus_assets import freeze_assets
    destination = tmp_path / 'corpus'
    shutil.copytree(LEGACY, destination)
    (destination / 'evaluation-assets.json').unlink()
    manifest = load_manifest(destination / 'manifest.json')
    plan = json.loads((destination / 'coverage-plan.json').read_text())
    if change == 'delete':
        manifest.cases.pop(0)
    elif change == 'alter':
        manifest.cases[0].seed += 1
    elif change == 'denominator':
        plan['active_atoms'].pop()
    elif change == 'parent-map':
        plan['baseline_case_hashes'][manifest.cases[0].code] = '0' * 64
    elif change == 'invent-tag':
        manifest.cases[-1].feature_atoms.append('FA-PAT-SPLIT-K')
    elif change == 'budget':
        plan['max_additions'] += 100
    elif change == 'trace':
        plan['selection_trace'][0]['new_atoms'] = []
    elif change == 'reference':
        log = read_log(destination / 'generation-log.yaml')
        next(event for event in log if event.get('step') == 'independent_cpu_reference_gate')[
            'failures'] = {'fixture': 'bad reference'}
        write_log(destination / 'generation-log.yaml', log)
    elif change == 'supplement':
        manifest.cases[-1].seed += 1
    for case in manifest.cases:
        case.content_hash = compute_case_content_hash(destination, case)
    manifest.manifest_hash = manifest_digest(manifest)
    log = read_log(destination / 'generation-log.yaml')
    log[-1]['manifest_hash'] = manifest.manifest_hash
    write_log(destination / 'generation-log.yaml', log)
    (destination / 'manifest.json').write_text(manifest.model_dump_json())
    (destination / 'coverage-plan.json').write_text(json.dumps(plan))
    migration_path = destination / 'evaluation-assets-migration.json'
    if manifest.manifest_hash != json.loads(migration_path.read_text())['manifest_hash']:
        with pytest.raises(ValueError, match='previous asset identity or case manifest differs'):
            freeze_assets(destination, manifest.manifest_hash)
        migration_path.unlink()
    freeze_assets(destination, manifest.manifest_hash)
    with pytest.raises(ValueError, match=match):
        static_readiness(destination)


@pytest.mark.torch
def test_all_extension_candidates_have_reproducible_legal_inputs_and_independent_oracles():
    import torch
    from triton_eval import coverage_reference as refs
    from triton_eval.oracle import active_corpus, configure_corpus
    from triton_eval.refcheck import check_case
    previous = active_corpus()
    try:
        configure_corpus(CORPUS)
        for case in extension_pool():
            kwargs = dict(seed=case.seed, dtype=getattr(torch, case.dtype), shape=case.shape,
                          device='cpu', attrs=case.attrs, value_range=case.value_range)
            inputs = refs.make_inputs(case.operator, **kwargs)
            again = refs.make_inputs(case.operator, **kwargs)
            assert all(torch.equal(a, b) for a, b in zip(inputs, again)), case.code
            assert any(str(t.dtype) == 'torch.' + case.dtype for t in inputs), case.code
            refs.validate_inputs(case.operator, inputs, case.attrs)
            assert check_case(case) is None, case.code
            if case.operator == 'Scatter' and case.attrs.get('reduce') == 'add':
                assert bool((inputs[1] == 0).all())
                result = refs.reference(case.operator, *inputs, **case.attrs)
                assert torch.equal(result.select(case.attrs['dim'], 1), inputs[0].select(case.attrs['dim'], 1))
            if case.operator == 'Cummin':
                assert torch.equal(inputs[0].select(case.attrs['dim'], 0), inputs[0].select(case.attrs['dim'], 1))
            if case.operator == 'MaskedSoftmax':
                assert bool((inputs[1] == 0).any()) and bool((inputs[1] != 0).any())
    finally:
        configure_corpus(previous)


@pytest.mark.torch
def test_biased_extension_reference_is_rejected(monkeypatch):
    from triton_eval import coverage_reference as refs
    from triton_eval.oracle import active_corpus, configure_corpus
    from triton_eval.refcheck import check_case
    original, previous = refs.reference, active_corpus()
    monkeypatch.setattr(refs, 'reference', lambda name, *args, **kwargs: original(name, *args, **kwargs) + 1)
    try:
        configure_corpus(CORPUS)
        assert '第二参考输出' in check_case(extension_pool()[0])
    finally:
        configure_corpus(previous)


def test_coverage_submission_passes_static_rules():
    from triton_eval.validator import ValidationResult, validate_static
    result = ValidationResult(submit_id='coverage-test')
    validate_static(fixture_path(ROOT / 'examples/coverage-v18-ascend-submission'), result)
    assert result.ok, result.errors
