# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Two-operator regression scope must not imply full-corpus qualification."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from test_coverage_evidence import write_pass_case_rows

from triton_eval.fixture_sources import fixture_path


ROOT = Path(__file__).resolve().parents[2]


@pytest.mark.parametrize('failed', [False, True])
def test_regression_runs_only_selected_partitions_and_keeps_coverage_separate(tmp_path, monkeypatch, failed):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('regression_suite', ROOT / 'scripts/run_ascend_suite.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus = ROOT / 'corpus/v0.18-scan-certification-candidates'
    payload = json.loads((corpus / 'manifest.json').read_text())
    output = tmp_path / 'regression'
    calls = []
    audited = []

    def execute(command, path, timeout=None):
        command = list(map(str, command))
        calls.append(command)
        if 'run' in command:
            operator = command[command.index('--operators') + 1]
            run = Path(command[command.index('--runs') + 1]) / 'run-1'
            run.mkdir()
            rows = [dict(case_code=case['code'], operator=operator, run_id=1,
                         status='COMPILE_FAIL' if failed else 'PASS')
                    for case in payload['cases'] if case['operator'] == operator]
            (run / 'results.jsonl').write_text('\n'.join(map(json.dumps, rows)))
        if any('audit_coverage_results.py' in argument for argument in command):
            assert '--operator-regression' in command
            audited.append((output / 'suite-summary.json').read_bytes())
            return int(failed)
        return 0

    monkeypatch.setattr(module, 'run_logged', execute)
    monkeypatch.setattr(sys, 'argv', ['suite', '--corpus', str(corpus),
                                      '--submission',
                                      str(fixture_path(ROOT / 'examples/coverage-v18-ascend-submission')),
                                      '--operators', 'FloorDiv,GroupedMatmul', '--out', str(output)])
    assert module.main() == int(failed)
    summary = json.loads((output / 'suite-summary.json').read_text())
    assert summary['selected_operators'] == ['FloorDiv', 'GroupedMatmul']
    assert set(summary['operators']) == {'FloorDiv', 'GroupedMatmul'}
    assert summary['expected_cases'] == 24 and summary['corpus_cases'] == 1385
    assert summary['all_pass'] is not failed
    assert summary['validation_scope'] == 'operator_regression'
    assert summary['coverage_target_gate_applied'] is False
    assert audited == [(output / 'suite-summary.json').read_bytes()]
    references = [command for command in calls if any('verify_refs.py' in arg for arg in command)]
    assert len(references) == 4
    assert {command[command.index('--operators') + 1] for command in references} == {'FloorDiv', 'GroupedMatmul'}
    capabilities = [command for command in calls if any('probe_ascend_capabilities.py' in arg for arg in command)]
    assert len(capabilities) == 1 and '--health-only' in capabilities[0]


def test_regression_shell_entry_fixes_exact_operator_selection(tmp_path):
    capture = tmp_path / 'arguments'
    executable = tmp_path / 'bash'
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$@" > "$CAPTURE"\n')
    executable.chmod(0o755)
    environment = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}', CAPTURE=str(capture))
    command = ['/bin/bash', str(ROOT / 'scripts/coverage-regression-verify.sh')]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == [str(ROOT / 'scripts/coverage-suite-verify.sh'),
                                                '--operators', 'FloorDiv,GroupedMatmul']
    capture.unlink()
    rejected = subprocess.run(command + ['--operators', 'CrossEntropyLoss'], env=environment,
                              capture_output=True, text=True)
    assert rejected.returncode == 2 and not capture.exists()


def test_gap_regression_selects_only_four_carriers_and_requests_lowered_ir(tmp_path):
    capture = tmp_path / 'arguments'
    executable = tmp_path / 'bash'
    executable.write_text(
        '#!/bin/sh\nprintf "%s\\n" "$MLIR_ENABLE_DUMP" "$TRITON_EVAL_CAPTURE_PASSES" "$@" > "$CAPTURE"\n')
    executable.chmod(0o755)
    environment = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}', CAPTURE=str(capture))
    command = ['/bin/bash', str(ROOT / 'scripts/coverage-gap-verify.sh')]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == ['1', '1', str(ROOT / 'scripts/coverage-suite-verify.sh'),
                                                '--operators', 'Neg,Matmul,MaskedSoftmax,CrossEntropyLoss']
    capture.unlink()
    rejected = subprocess.run(command + ['--operators', 'Abs'], env=environment, capture_output=True, text=True)
    assert rejected.returncode == 2 and not capture.exists()


def test_backend_regression_selects_only_two_operators_and_forty_unchanged_cases(tmp_path):
    capture = tmp_path / 'arguments'
    executable = tmp_path / 'bash'
    executable.write_text(
        ('#!/bin/sh\nprintf "%s\\n" "$TRITON_EVAL_CAPTURE_BACKEND" "$TRITON_EVAL_REQUIRE_BACKEND_PATTERNS" "$@" > '
         '"$CAPTURE"\n'))
    executable.chmod(0o755)
    environment = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}', CAPTURE=str(capture))
    command = ['/bin/bash', str(ROOT / 'scripts/coverage-backend-verify.sh')]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == ['1', '1', str(ROOT / 'scripts/coverage-suite-verify.sh'),
                                                '--operators', 'Neg,Matmul']
    capture.unlink()
    assert subprocess.run(command + ['--operators', 'Abs'], env=environment, capture_output=True).returncode == 2
    assert not capture.exists()
    manifest = json.loads((ROOT / 'corpus/v0.16-backend-pattern-certification-candidates/manifest.json').read_text())
    assert sum(case['operator'] in ('Neg', 'Matmul') for case in manifest['cases']) == 40


@pytest.mark.parametrize('missing', [None, 'FA-PAT-VEC-LOAD', 'FA-PAT-TILE-GEMM', 'report', 'functionality'])
def test_backend_gate_cannot_confuse_functionality_with_pattern_certification(tmp_path, missing):
    spec = importlib.util.spec_from_file_location('backend_gate_suite', ROOT / 'scripts/run_ascend_suite.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    atoms = ['FA-PAT-VEC-LOAD', 'FA-PAT-TILE-GEMM']
    proof = dict(selected_operator_validation_passed=missing != 'functionality',
                 compilation=dict(covered=atoms),
                 execution_validation=dict(covered=[atom for atom in atoms if atom != missing]))
    if missing != 'report':
        (tmp_path / 'coverage-validation.json').write_text(json.dumps(proof))
    summary = b'{"all_pass": true}\n'
    (tmp_path / 'suite-summary.json').write_bytes(summary)
    assert module.backend_pattern_gate(tmp_path) == int(missing is not None)
    result = json.loads((tmp_path / 'backend-pattern-validation.json').read_text())
    assert result['passed'] is (missing is None)
    assert (tmp_path / 'suite-summary.json').read_bytes() == summary


@pytest.mark.parametrize('certified', [False, True])
def test_backend_entry_fails_on_missing_patterns_even_when_all_selected_cases_pass(tmp_path, monkeypatch, certified):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('backend_gate_integration', ROOT / 'scripts/run_ascend_suite.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus = ROOT / 'corpus/v0.18-scan-certification-candidates'
    payload = json.loads((corpus / 'manifest.json').read_text())
    output = tmp_path / 'backend-suite'
    audited = []

    def execute(command, path, timeout=None):
        command = list(map(str, command))
        write_pass_case_rows(command, payload['cases'])
        if any('audit_coverage_results.py' in argument for argument in command):
            atoms = ['FA-PAT-VEC-LOAD', 'FA-PAT-TILE-GEMM'] if certified else []
            proof = dict(selected_operator_validation_passed=True,
                         compilation=dict(covered=atoms), execution_validation=dict(covered=atoms))
            (output / 'coverage-validation.json').write_text(json.dumps(proof))
            audited.append((output / 'suite-summary.json').read_bytes())
        return 0
    monkeypatch.setenv('TRITON_EVAL_REQUIRE_BACKEND_PATTERNS', '1')
    monkeypatch.setattr(module, 'run_logged', execute)
    monkeypatch.setattr(sys, 'argv', ['suite', '--corpus', str(corpus),
                                      '--submission',
                                      str(fixture_path(ROOT / 'examples/coverage-v18-ascend-submission')),
                                      '--operators', 'Neg,Matmul', '--out', str(output)])
    assert module.main() == int(not certified)
    summary = json.loads((output / 'suite-summary.json').read_text())
    assert summary['all_pass'] and summary['expected_cases'] == 42
    assert summary['backend_pattern_gate_required']
    assert audited == [(output / 'suite-summary.json').read_bytes()]
    assert json.loads((output / 'backend-pattern-validation.json').read_text())['passed'] is certified


def test_v16_preserves_every_case_reference_tolerance_and_previously_verified_kernels():
    from triton_eval.corpus_assets import reference_layout, resolve_reference_path
    from triton_eval.corpusmod import load_manifest
    from triton_eval.coverage_gate import static_readiness

    before = ROOT / 'corpus/v0.15-backend-capture-fix-candidates'
    after = ROOT / 'corpus/v0.16-backend-pattern-certification-candidates'
    previous = load_manifest(before / 'manifest.json')
    current = load_manifest(after / 'manifest.json')
    assert {case.code: case.content_hash for case in previous.cases} == {
        case.code: case.content_hash for case in current.cases}
    references = reference_layout(before)
    assert references and references.keys() == reference_layout(after).keys()
    for relative in references:
        assert (resolve_reference_path(before, relative).read_bytes()
                == resolve_reference_path(after, relative).read_bytes())
    old_assets = json.loads((before / 'evaluation-assets.json').read_text())['files']
    new_assets = json.loads((after / 'evaluation-assets.json').read_text())['files']
    for path in ('spec/spec.yaml', 'spec/tolerance.yaml', 'spec/feature_atoms.yaml',
                 'src/triton_eval/coverage_reference.py'):
        assert old_assets[path] == new_assets[path]
    for kernel in (fixture_path(ROOT / 'examples/coverage-v15-ascend-submission')).rglob('*.py'):
        relative = kernel.relative_to(fixture_path(ROOT / 'examples/coverage-v15-ascend-submission'))
        previous = fixture_path(ROOT / 'examples/coverage-v16-ascend-submission' / relative)
        assert kernel.read_bytes() == previous.read_bytes()
    assert static_readiness(after)['static_target_met'] is True


def test_new_submission_reroutes_only_two_operators_and_preserves_frozen_inputs():
    import yaml
    from triton_eval.corpusmod import load_manifest

    previous = fixture_path(ROOT / 'examples/coverage-v111-ascend-submission')
    current = fixture_path(ROOT / 'examples/coverage-v12-ascend-submission')
    before = yaml.safe_load((previous / 'manifest.yaml').read_text())
    after = yaml.safe_load((current / 'manifest.yaml').read_text())
    old_entries = {entry['name']: entry for entry in before['operators']}
    new_entries = {entry['name']: entry for entry in after['operators']}
    assert set(old_entries) == set(new_entries)
    assert {name for name in old_entries if old_entries[name] != new_entries[name]} == {'FloorDiv', 'GroupedMatmul'}
    for path in previous.rglob('*.py'):
        assert path.read_bytes() == (current / path.relative_to(previous)).read_bytes()
    old_manifest = load_manifest(ROOT / 'corpus/v0.11.1-contract-candidates/manifest.json')
    new_manifest = load_manifest(ROOT / 'corpus/v0.12-compat-candidates/manifest.json')
    assert {case.code: case.content_hash for case in old_manifest.cases} == {
        case.code: case.content_hash for case in new_manifest.cases}
    assert new_manifest.manifest_hash != old_manifest.manifest_hash


def test_full_corpus_run_still_fails_when_coverage_target_fails(tmp_path, monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    spec = importlib.util.spec_from_file_location('full_coverage_suite', ROOT / 'scripts/run_ascend_suite.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    corpus = ROOT / 'corpus/v0.18-scan-certification-candidates'
    payload = json.loads((corpus / 'manifest.json').read_text())
    output = tmp_path / 'full-suite'

    def execute(command, path, timeout=None):
        command = list(map(str, command))
        write_pass_case_rows(command, payload['cases'])
        if any('audit_coverage_results.py' in argument for argument in command):
            assert '--operator-regression' not in command
            return 1
        return 0

    monkeypatch.setattr(module, 'run_logged', execute)
    monkeypatch.setattr(sys, 'argv', ['suite', '--corpus', str(corpus), '--submission',
                        str(fixture_path(ROOT / 'examples/coverage-v18-ascend-submission')), '--out', str(output)])
    assert module.main() == 1
    summary = json.loads((output / 'suite-summary.json').read_text())
    assert summary['all_pass'] and summary['complete']
    assert summary['validation_scope'] == 'full_corpus'
    assert summary['coverage_target_gate_applied'] is True


@pytest.mark.parametrize('regression,passed,expected', [(True, True, 0), (True, False, 1), (False, True, 1)])
def test_audit_exit_code_distinguishes_regression_from_global_target(
        tmp_path, monkeypatch, regression, passed, expected):
    spec = importlib.util.spec_from_file_location('regression_audit_cli', ROOT / 'scripts/audit_coverage_results.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    report = dict(selected_operator_validation_passed=passed, verified_npu_coverage_target_met=False)
    monkeypatch.setattr(module, 'audit_suite', lambda *args, **kwargs: report)
    arguments = ['audit', '--results', str(tmp_path)]
    if regression:
        arguments.append('--operator-regression')
    monkeypatch.setattr(sys, 'argv', arguments)
    assert module.main() == expected
    assert json.loads((tmp_path / 'coverage-validation.json').read_text()) == report


def test_health_only_probe_does_not_launch_gather_checks(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location(
        'regression_health_probe',
        ROOT / 'scripts/probe_ascend_capabilities.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    calls = []

    def execute(command, **kwargs):
        calls.append(command)
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(module.subprocess, 'run', execute)
    monkeypatch.setattr(module.platform, 'platform', lambda: 'synthetic-unit-only')
    monkeypatch.setattr(sys, 'argv', ['probe', '--health-only', '--require-pass', '--out', str(tmp_path)])
    assert module.main() == 0
    assert len(calls) == 1 and calls[0][-2:] == ['--child', 'health']
    report = json.loads((tmp_path / 'capabilities.json').read_text())
    assert set(report['probes']) == {'health'}
    assert report['not_run'] == ['gather_float32', 'gather_int32']
