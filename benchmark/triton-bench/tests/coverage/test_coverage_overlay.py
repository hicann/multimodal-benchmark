# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Historical overlays are analytical, scoped replacements, never suite acceptance."""
import json
from pathlib import Path

import pytest
import yaml

from triton_eval.coverage_evidence import _rate, sha
from triton_eval.coverage_overlay import ENVIRONMENT_KEYS, overlay_results
from triton_eval.corpusmod import build_manifest
from triton_eval.models import CaseSource, CaseSpec


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload))


@pytest.fixture
def overlay_pair(tmp_path):
    atoms = ['FA-MATH-ADD', 'FA-PROG-GRID1D', 'FA-DT-FP32']
    cases = [CaseSpec(code='add', operator='Add', source=CaseSource.VENDOR_SELFTEST,
                      dtype='float32', shape=[2, 3], seed=1, feature_atoms=[atoms[0], atoms[2]]),
             CaseSpec(code='neg', operator='Neg', source=CaseSource.VENDOR_SELFTEST,
                      dtype='float32', shape=[2, 3], seed=2, feature_atoms=[atoms[1], atoms[2]])]
    roots, submissions = [], []
    for repaired in (False, True):
        root = tmp_path / ('repair' if repaired else 'baseline')
        submission = tmp_path / ('repair-submission' if repaired else 'baseline-submission')
        root.mkdir()
        submission.mkdir()
        (submission / 'kernel.py').write_text('def kernel():\n    pass\n')
        payload = dict(operators=[dict(name=name, path='kernel.py', entry=name.lower()) for name in ('Add', 'Neg')])
        if repaired:
            (submission / 'new.py').write_text('def neg(input):\n    return input\n')
            payload['operators'][1]['path'] = 'new.py'
        (submission / 'manifest.yaml').write_text(yaml.safe_dump(payload))
        snapshot = {path.name: sha(path.read_bytes()) for path in submission.iterdir()}
        manifest = build_manifest(root, 'synthetic-repair' if repaired else 'synthetic-baseline', 'test', [], cases)
        (root / 'manifest.json').write_text(manifest.model_dump_json())
        selected = ['Neg'] if repaired else ['Add', 'Neg']
        summary = dict(manifest_hash=manifest.manifest_hash, selected_operators=selected,
                       operators={name: dict(reference_gate_returncodes={'cpu': 0, 'npu:0': 0}) for name in selected})
        write_json(root / 'suite-summary.json', summary)
        write_json(root / 'coverage-plan.json', dict(active_atoms=atoms))
        write_json(root / 'submission-sha256.json', snapshot)
        asset_names = ['spec/spec.yaml', 'spec/tolerance.yaml', 'spec/feature_atoms.yaml',
                       'src/triton_eval/inputs.py', 'src/triton_eval/comparison.py',
                       'src/triton_eval/coverage_reference.py', 'src/triton_eval/official_reference.py']
        write_json(root / 'evaluation-assets.json', dict(files={name: 'synthetic' for name in asset_names}))
        for operator in selected:
            status = 'PASS' if operator == 'Add' or repaired else 'COMPILE_FAIL'
            write_json(root / operator / 'run-1/results.jsonl', dict(case_code=operator.lower(), operator=operator,
                                                                     run_id=1, status=status))
            write_json(root / operator / 'run-1/env.json', {key: 'synthetic' for key in ENVIRONMENT_KEYS})
        witnessed_atoms = atoms[1:] if repaired else [atoms[0], atoms[2]]
        witness_case = 'neg' if repaired else 'add'
        witnesses = {atom: [witness_case] for atom in witnessed_atoms}
        hashes = {str(path.relative_to(root)): sha(path.read_bytes()) for path in root.rglob('*') if path.is_file()}
        report = dict(manifest_hash=manifest.manifest_hash, static=_rate(atoms, atoms),
                      compilation=_rate(witnesses, atoms), execution_validation=_rate(witnesses, atoms),
                      compilation_witnesses=witnesses, execution_witnesses=witnesses,
                      selected_operator_validation_passed=repaired, evidence_sha256=hashes, submission_sha256=snapshot)
        write_json(root / 'coverage-validation.json', report)
        roots.append(root)
        submissions.append(submission)
    return dict(baseline=roots[0], repair=roots[1],
                baseline_submission=submissions[0], repair_submission=submissions[1])


def revise_index(root, relative, payload):
    write_json(root / relative, payload)
    report = json.loads((root / 'coverage-validation.json').read_text())
    report['evidence_sha256'][relative] = sha((root / relative).read_bytes())
    write_json(root / 'coverage-validation.json', report)


def test_overlay_replaces_failed_partition_deduplicates_labels_and_keeps_origins(overlay_pair):
    report = overlay_results(**overlay_pair)
    assert report['compilation']['covered_count'] == report['execution_validation']['covered_count'] == 3
    assert report['passed_cases'] == report['expected_cases'] == 2
    assert report['newly_confirmed_features'] == ['FA-PROG-GRID1D']
    assert not report['missing_cases'] and not report['nonpass_cases']
    assert report['analysis_only'] and not report['verified_npu_coverage_target_met']
    assert not report['taskbook_accepted']
    assert report['feature_origins']['compilation']['FA-PROG-GRID1D'][0]['source'] == 'repair'


def test_lock_hash_difference_is_disclosed_without_hiding_confirmed_features(overlay_pair):
    root = overlay_pair['repair']
    environment = {key: 'synthetic' for key in ENVIRONMENT_KEYS}
    environment['venv_lock_hash'] = 'changed'
    revise_index(root, 'Neg/run-1/env.json', environment)
    report = overlay_results(**overlay_pair)
    assert not report['single_locked_environment']
    assert report['environment_differences'] == {'venv_lock_hash': ['changed', 'synthetic']}
    assert report['compilation']['covered_count'] == 3


@pytest.mark.parametrize('change', ['hash', 'toolchain', 'reference', 'witness', 'untested_source', 'repair_failed'])
def test_overlay_rejects_incompatible_or_unverified_inputs(overlay_pair, change):
    root = overlay_pair['repair']
    if change == 'hash':
        (root / 'Neg/run-1/results.jsonl').write_text('tampered')
    elif change == 'toolchain':
        environment = {key: 'synthetic' for key in ENVIRONMENT_KEYS}
        environment['compiler_version'] = 'different'
        revise_index(root, 'Neg/run-1/env.json', environment)
    elif change == 'reference':
        assets = json.loads((root / 'evaluation-assets.json').read_text())
        assets['files']['spec/tolerance.yaml'] = 'relaxed'
        revise_index(root, 'evaluation-assets.json', assets)
    elif change == 'untested_source':
        (overlay_pair['repair_submission'] / 'kernel.py').write_text('changed untested Add')
    else:
        report = json.loads((root / 'coverage-validation.json').read_text())
        if change == 'witness':
            report['execution_witnesses']['FA-PROG-GRID1D'] = ['unknown']
        else:
            report['selected_operator_validation_passed'] = False
        write_json(root / 'coverage-validation.json', report)
    with pytest.raises((ValueError, KeyError)):
        overlay_results(**overlay_pair)
