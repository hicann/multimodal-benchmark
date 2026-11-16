# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Real CPU reference and version-scoped domain gates for the pattern release."""
import importlib.util
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from test_coverage_evidence import configure_target_suite

from triton_eval.ascend_contracts import validate_inputs
from triton_eval.coverage_reference import make_inputs, validate_inputs as validate_previous
from triton_eval.oracle import active_corpus, configure_corpus
from triton_eval.pattern_contracts import validate_inputs as validate_pattern
from triton_eval.pattern_generation import SCAN_REPAIR_VERSION as VERSION, candidate_plans
from triton_eval.buildlog import read_log, write_log


torch = pytest.importorskip('torch')

ROOT = Path(__file__).resolve().parents[2]
PATTERNS = [plan for plan in candidate_plans() if plan[0] in ('Cummin', 'GroupedMatmul')]


@pytest.fixture(params=[VERSION, 'v0.17-pattern-completion-candidates'])
def use_pattern_corpus(tmp_path, request):
    previous = active_corpus()
    (tmp_path / 'coverage-plan.json').write_text(json.dumps(dict(version=request.param)))
    configure_corpus(tmp_path)
    yield tmp_path
    configure_corpus(previous)


@pytest.mark.parametrize('operator,dtype,shape,attrs', PATTERNS)
def test_explicit_new_domains_pass_without_changing_the_input_factory(
        use_pattern_corpus, operator, dtype, shape, attrs):
    inputs = make_inputs(operator, seed=123, dtype=getattr(torch, dtype), shape=shape, device='cpu', attrs=attrs)
    validate_inputs(operator, inputs, attrs)
    repeated = make_inputs(operator, seed=123, dtype=getattr(torch, dtype), shape=shape, device='cpu', attrs=attrs)
    assert all(torch.equal(first, second) for first, second in zip(inputs, repeated))


@pytest.mark.parametrize('operator,dtype,shape,attrs', [PATTERNS[0], PATTERNS[1], PATTERNS[3]])
def test_previous_release_keeps_rejecting_new_domains(operator, dtype, shape, attrs):
    inputs = make_inputs(operator, seed=123, dtype=getattr(torch, dtype), shape=shape, device='cpu', attrs=attrs)
    with pytest.raises(ValueError):
        validate_previous(operator, inputs, attrs)


@pytest.mark.parametrize('mutation', ['bias', 'weight', 'dtype', 'partition', 'layout', 'arity'])
def test_new_grouped_domain_rejects_broken_tensor_relations(use_pattern_corpus, mutation):
    operator, dtype, shape, attrs = PATTERNS[3]
    inputs = make_inputs(operator, seed=123, dtype=getattr(torch, dtype), shape=shape, device='cpu', attrs=attrs)
    attrs = dict(attrs)
    if mutation == 'bias':
        inputs[2] = inputs[2][:1]
    elif mutation == 'weight':
        inputs[1] = inputs[1][:, :-1]
    elif mutation == 'dtype':
        inputs[2] = inputs[2].float()
    elif mutation == 'partition':
        attrs['group_list'] = [1024, 1039]
    elif mutation == 'layout':
        inputs[0] = inputs[0].t().contiguous().t()
    else:
        inputs.pop()
    with pytest.raises(ValueError):
        validate_inputs(operator, inputs, attrs)


def test_unqualified_large_scan_does_not_bypass_old_bounds(use_pattern_corpus):
    with pytest.raises(ValueError, match='bounded cummin'):
        validate_inputs('Cummin', [torch.zeros((1, 2**20 + 1))], dict(dim=1))


@pytest.mark.parametrize('operator,shape,attrs', [
    ('Cummin', [4, 8, 16], dict(dim=1)),
    ('GroupedMatmul', [12, 32], dict(group_list=[5, 7])),
])
def test_existing_domains_still_use_the_unchanged_validator(operator, shape, attrs):
    inputs = make_inputs(operator, seed=123, dtype=torch.float32, shape=shape, device='cpu', attrs=attrs)
    validate_pattern(operator, inputs, attrs)
    validate_previous(operator, inputs, attrs)


def test_dispatch_requires_the_new_version_not_merely_a_plan_file(use_pattern_corpus):
    (use_pattern_corpus / 'coverage-plan.json').write_text(json.dumps(dict(version='v0.16')))
    with pytest.raises(ValueError, match='bounded cummin'):
        validate_inputs('Cummin', [torch.zeros((1, 2**20))], dict(dim=1))


@pytest.mark.parametrize('certified', [False, True])
def test_four_pattern_gate_controls_exit_even_when_all_52_cases_pass(tmp_path, monkeypatch, certified):
    from triton_eval.pattern_generation import TARGETS, target_features
    from triton_eval.corpusmod import load_manifest

    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module_spec = importlib.util.spec_from_file_location('pattern_suite', ROOT / 'scripts/run_ascend_suite.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    corpus = ROOT / 'corpus' / VERSION
    manifest = load_manifest(corpus / 'manifest.json')
    output = tmp_path / 'suite'
    selected = sorted(TARGETS.values())
    witnesses = {atom: [case.code for case in manifest.cases if atom in target_features(case)]
                 for atom in TARGETS}

    def execute(command, path, timeout=None):
        command = list(map(str, command))
        if 'run' in command:
            operator = command[command.index('--operators') + 1]
            run = Path(command[command.index('--runs') + 1]) / 'run-1'
            run.mkdir()
            rows = [dict(case_code=case.code, operator=operator, run_id=1, status='PASS')
                    for case in manifest.cases if case.operator == operator]
            (run / 'results.jsonl').write_text('\n'.join(map(json.dumps, rows)))
        if any('audit_coverage_results.py' in argument for argument in command):
            proof = dict(selected_operator_validation_passed=True, selected_operators=selected,
                         manifest_hash=manifest.manifest_hash,
                         compilation=dict(covered=sorted(TARGETS)),
                         execution_validation=dict(covered=sorted(TARGETS)),
                         compilation_witnesses=witnesses,
                         execution_witnesses=witnesses if certified else {})
            (output / 'coverage-validation.json').write_text(json.dumps(proof))
        return 0

    configure_target_suite(monkeypatch, module, corpus, output, selected)
    monkeypatch.setattr(module, 'run_logged', execute)
    assert module.main() == int(not certified)
    summary = json.loads((output / 'suite-summary.json').read_text())
    assert summary['all_pass'] and summary['expected_cases'] == 52 and summary['corpus_cases'] == 1385
    assert summary['target_pattern_gate_required']
    proof = json.loads((output / 'target-pattern-validation.json').read_text())
    assert proof['passed'] is certified


def test_pattern_shell_fixes_scope_capture_and_independent_gate(tmp_path):
    capture = tmp_path / 'arguments'
    executable = tmp_path / 'bash'
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$TRITON_EVAL_REQUIRE_TARGET_PATTERNS" '
                          '"$TRITON_EVAL_REQUIRE_BACKEND_PATTERNS" "$TRITON_EVAL_CAPTURE_PASSES" '
                          '"$TRITON_EVAL_CAPTURE_BACKEND" "$@" > "$CAPTURE"\n')
    executable.chmod(0o755)
    preflight = tmp_path / 'preflight'
    preflight.write_text('#!/bin/sh\nexit 0\n')
    preflight.chmod(0o755)
    environment = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}', CAPTURE=str(capture),
                       TRITON_EVAL_PREFLIGHT_PYTHON=str(preflight))
    command = ['/bin/bash', str(ROOT / 'scripts/coverage-patterns-verify.sh')]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == ['1',
                                                '0',
                                                '1',
                                                '1',
                                                str(ROOT / 'scripts/ascend-suite-verify.sh'),
                                                '--profile',
                                                'smoke',
                                                '--corpus',
                                                str(ROOT / 'corpus/v0.17-pattern-completion-candidates'),
                                                '--submission',
                                                str(ROOT / 'examples/coverage-v17-ascend-submission'),
                                                '--operators',
                                                'ReduceMax,Cummin,Matmul,GroupedMatmul',
                                                '--isolate-reference-gates']
    capture.unlink()
    assert subprocess.run(command + ['--operators', 'Abs'], env=environment, capture_output=True).returncode == 2
    assert not capture.exists()


@pytest.mark.parametrize('mutation,reason', [
    ('remove-parent', 'parent cases removed'),
    ('candidate', 'pattern candidate content differs'),
    ('labels', 'pattern candidate content differs'),
    ('policy', 'selection policy'),
    ('specification', 'parent specification'),
    ('cpu-gate', 'independent CPU reference gate'),
    ('denominator', 'parent identity or feature denominator'),
])
def test_pattern_semantic_gate_rejects_refingerprinted_tampering(tmp_path, mutation, reason):
    from triton_eval.corpus_assets import freeze_assets
    from triton_eval.corpusmod import load_manifest, manifest_digest, compute_case_content_hash
    from triton_eval.pattern_release import static_readiness

    corpus = tmp_path / 'corpus'
    shutil.copytree(ROOT / 'corpus' / VERSION, corpus)
    (corpus / 'evaluation-assets.json').unlink()
    manifest = load_manifest(corpus / 'manifest.json')
    plan = json.loads((corpus / 'coverage-plan.json').read_text())
    events = read_log(corpus / 'generation-log.yaml')
    if mutation == 'remove-parent':
        manifest.cases.pop(0)
    elif mutation == 'candidate':
        manifest.cases[-1].seed += 1
    elif mutation == 'labels':
        manifest.cases[-1].feature_atoms.pop()
    elif mutation == 'policy':
        plan['selection_trace'] = []
    elif mutation == 'specification':
        manifest.feature_atoms_version = 'unconfirmed'
    elif mutation == 'cpu-gate':
        gate = next(event for event in reversed(events) if event.get('step') == 'independent_cpu_reference_gate')
        gate['failures'] = {'fixture': 'not accepted'}
    else:
        plan['active_atoms'].pop()
    for case in manifest.cases:
        case.content_hash = compute_case_content_hash(corpus, case)
    manifest.manifest_hash = manifest_digest(manifest)
    events[-1]['manifest_hash'] = manifest.manifest_hash
    write_log(corpus / 'generation-log.yaml', events)
    (corpus / 'coverage-plan.json').write_text(json.dumps(plan))
    (corpus / 'manifest.json').write_text(manifest.model_dump_json())
    migration_path = corpus / 'evaluation-assets-migration.json'
    if manifest.manifest_hash != json.loads(migration_path.read_text())['manifest_hash']:
        with pytest.raises(ValueError, match='previous asset identity or case manifest differs'):
            freeze_assets(corpus, manifest.manifest_hash)
        migration_path.unlink()
    freeze_assets(corpus, manifest.manifest_hash)
    with pytest.raises(ValueError, match=reason):
        static_readiness(corpus)


def test_failed_cpu_reference_gate_cannot_freeze_a_release(tmp_path, monkeypatch):
    from triton_eval.pattern_release import build

    checked = []

    def reject(cases, device, corpus_dir):
        checked.append((len(cases), device))
        return {cases[-1].code: 'synthetic independent reference failure'}

    monkeypatch.setattr('triton_eval.refcheck.check_all', reject)
    corpus = tmp_path / 'draft'
    with pytest.raises(ValueError, match='independent CPU reference gate failed'):
        build(corpus, tmp_path / 'submission', version=VERSION)
    assert checked == [(1385, 'cpu')]
    assert not (corpus / 'manifest.json').exists()
    assert not (corpus / 'evaluation-assets.json').exists()
    events = read_log(corpus / 'generation-log.yaml')
    assert all(event.get('step') != 'build_end' for event in events)
