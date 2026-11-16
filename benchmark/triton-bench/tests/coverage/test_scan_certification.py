# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Replay archived real IR; these CPU tests are not new NPU certification."""
import copy
import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from test_coverage_evidence import configure_target_suite

from triton_eval.pattern_evidence import pattern_witnesses
from triton_eval.pattern_generation import SCAN_REPAIR_VERSION, VERSION, TARGETS
from triton_eval.pattern_release import pattern_gate, required_pattern_targets


ROOT = Path(__file__).resolve().parents[2]
FIXTURES = ROOT / 'tests/fixtures/ascend_scan'
ATOM = 'FA-PAT-SCAN-BLOCKED'


def captured_scan(index):
    fixture = json.loads((FIXTURES / 'provenance.json').read_text())
    observed = copy.deepcopy(fixture['cases'][index])
    observed['case'] = SimpleNamespace(**observed['case'])
    for launch in observed['launches']:
        raw = (FIXTURES / launch.pop('ttir_file')).read_bytes()
        assert hashlib.sha256(raw).hexdigest() == launch.pop('ttir_sha256')
        launch['stage_text'] = {'ttir': raw.decode()}
    return observed


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('spelling', ['tt.scan', 'tt.associative_scan'])
def test_actual_scan_ttir_and_tensor_chain_are_recognized(index, spelling):
    observed = captured_scan(index)
    for launch in observed['launches']:
        launch['stage_text']['ttir'] = launch['stage_text']['ttir'].replace('"tt.scan"', '"' + spelling + '"')
    proof = pattern_witnesses(observed)[ATOM]
    assert proof['verified'], proof['reason']
    assert proof['chunks'] == [2048, 2049][index]
    assert proof['compiler_stage'] == 'ttir' and proof['stages'] == 3
    assert proof['tie_rule'] == 'last occurrence'
    assert proof['performance_gain'] == 'NOT_ESTABLISHED'


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('before,after', [
    ('"tt.scan"', '"tt.scan_extra"'),
    ('%13:2 = "tt.scan"', '%13 = "tt.scan"'),
    ('reverse = false', 'reverse = true'),
    ('axis = 0 : i32', 'axis = 1 : i32'),
    ('arith.cmpi sgt,', 'arith.cmpi slt,'),
    ('tt.scan.return %40, %41', 'tt.scan.return %41, %40'),
    ('tt.store %16, %13#0', 'tt.store %16, %cst'),
])
def test_unknown_dead_or_wrong_scan_is_rejected(index, before, after):
    observed = captured_scan(index)
    launch = observed['launches'][0]
    assert before in launch['stage_text']['ttir']
    launch['stage_text']['ttir'] = launch['stage_text']['ttir'].replace(before, after)
    assert not pattern_witnesses(observed)[ATOM]['verified']


@pytest.mark.parametrize('index', [0, 1])
@pytest.mark.parametrize('before,after', [
    ('%23 = tt.splat %19 : f32 -> tensor<512xf32>', '%23 = tt.splat %11 : f32 -> tensor<512xf32>'),
    ('%23 = tt.splat %19 : f32 -> tensor<512xf32>', '%23 = tt.broadcast %19 : f32 -> tensor<512xf32>'),
    ('%23 = tt.splat %19 : f32 -> tensor<512xf32>', '%23 = tt.splat %19 : f32 -> tensor<256xf32>'),
    ('%23 = tt.splat %19 : f32 -> tensor<512xf32>',
     '%changed = arith.addf %19, %19 : f32\n    %23 = tt.splat %changed : f32 -> tensor<512xf32>'),
    ('%26 = tt.splat %22 : i64 -> tensor<512xi64>', '%26 = tt.splat %15 : i64 -> tensor<512xi64>'),
    ('arith.select %29, %26, %15', 'arith.select %29, %15, %26'),
    ('arith.cmpi sgt, %26, %15', 'arith.cmpi slt, %26, %15'),
    ('arith.cmpf olt, %23, %11', 'arith.cmpf ogt, %23, %11'),
])
def test_fixup_requires_exact_carry_broadcast_and_pair_semantics(index, before, after):
    observed = captured_scan(index)
    launch = observed['launches'][2]
    assert before in launch['stage_text']['ttir']
    launch['stage_text']['ttir'] = launch['stage_text']['ttir'].replace(before, after)
    assert not pattern_witnesses(observed)[ATOM]['verified']


def test_scan_gate_requires_both_structural_witnesses_and_both_metrics(tmp_path):
    from triton_eval.corpusmod import load_manifest

    manifest = load_manifest(ROOT / 'corpus' / SCAN_REPAIR_VERSION / 'manifest.json')
    (tmp_path / 'manifest.json').write_text(manifest.model_dump_json())
    codes = ['pattern-cummin-002', 'pattern-cummin-003']
    proof = dict(manifest_hash=manifest.manifest_hash, selected_operators=['Cummin'],
                 selected_operator_validation_passed=True,
                 compilation=dict(covered=[ATOM]), execution_validation=dict(covered=[ATOM]),
                 compilation_witnesses={ATOM: codes}, execution_witnesses={ATOM: codes})
    path = tmp_path / 'coverage-validation.json'
    path.write_text(json.dumps(proof))
    assert pattern_gate(tmp_path) == 0
    saved = json.loads((tmp_path / 'target-pattern-validation.json').read_text())
    assert saved['scope'] == 'blocked_scan_repair' and set(saved['checks']) == {ATOM}
    assert saved['coverage_report_sha256'] == hashlib.sha256(path.read_bytes()).hexdigest()
    for key in ['compilation_witnesses', 'execution_witnesses']:
        for missing in [[], codes[:1], [codes[0], codes[0]], [*codes, 'unknown']]:
            changed = copy.deepcopy(proof)
            changed[key][ATOM] = missing
            path.write_text(json.dumps(changed))
            assert pattern_gate(tmp_path) == 1


def test_scan_scope_cannot_weaken_v17_or_a_full_run():
    assert required_pattern_targets(SCAN_REPAIR_VERSION, ['Cummin']) == (ATOM,)
    assert set(required_pattern_targets(VERSION, ['Cummin'])) == set(TARGETS)
    assert set(required_pattern_targets(SCAN_REPAIR_VERSION, TARGETS.values())) == set(TARGETS)


def test_scan_shell_pins_four_cases_and_rejects_arguments(tmp_path):
    capture = tmp_path / 'arguments'
    executable = tmp_path / 'bash'
    executable.write_text('#!/bin/sh\nprintf "%s\\n" "$TRITON_EVAL_REQUIRE_TARGET_PATTERNS" '
                          '"$TRITON_EVAL_REQUIRE_BACKEND_PATTERNS" "$TRITON_EVAL_CAPTURE_PASSES" '
                          '"$TRITON_EVAL_CAPTURE_BACKEND" "$@" > "$CAPTURE"\n')
    executable.chmod(0o755)
    environment = dict(os.environ, PATH=f'{tmp_path}:{os.environ["PATH"]}', CAPTURE=str(capture),
                       TRITON_EVAL_PREFLIGHT_PYTHON=sys.executable)
    command = ['/bin/bash', str(ROOT / 'scripts/coverage-scan-verify.sh')]
    result = subprocess.run(command, env=environment, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert capture.read_text().splitlines() == ['1', '0', '1', '1',
                                                str(ROOT / 'scripts/ascend-suite-verify.sh'), '--profile', 'smoke',
                                                '--corpus', str(ROOT / 'corpus' / SCAN_REPAIR_VERSION), '--submission',
                                                str(ROOT / 'examples/coverage-v18-ascend-submission'), '--operators',
                                                'Cummin', '--isolate-reference-gates']
    capture.unlink()
    assert subprocess.run(command + ['--operators', 'Neg'], env=environment, capture_output=True).returncode == 2
    assert not capture.exists()


@pytest.mark.parametrize('certified', [False, True])
def test_four_cummin_passes_do_not_replace_scan_certification(tmp_path, monkeypatch, certified):
    from triton_eval.corpusmod import load_manifest

    monkeypatch.syspath_prepend(str(ROOT / 'scripts'))
    module_spec = importlib.util.spec_from_file_location('scan_suite', ROOT / 'scripts/run_ascend_suite.py')
    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    corpus = ROOT / 'corpus' / SCAN_REPAIR_VERSION
    manifest = load_manifest(corpus / 'manifest.json')
    output = tmp_path / 'suite'

    def execute(command, path, timeout=None):
        command = list(map(str, command))
        if 'run' in command:
            assert command[command.index('--operators') + 1] == 'Cummin'
            run = Path(command[command.index('--runs') + 1]) / 'run-1'
            run.mkdir()
            rows = [dict(case_code=case.code, operator='Cummin', run_id=1, status='PASS')
                    for case in manifest.cases if case.operator == 'Cummin']
            (run / 'results.jsonl').write_text('\n'.join(map(json.dumps, rows)))
        if any('audit_coverage_results.py' in argument for argument in command):
            codes = ['pattern-cummin-002', 'pattern-cummin-003']
            proof = dict(selected_operator_validation_passed=True, selected_operators=['Cummin'],
                         manifest_hash=manifest.manifest_hash, compilation=dict(covered=[ATOM]),
                         execution_validation=dict(covered=[ATOM]), compilation_witnesses={ATOM: codes},
                         execution_witnesses={ATOM: codes} if certified else {})
            (output / 'coverage-validation.json').write_text(json.dumps(proof))
        return 0

    configure_target_suite(monkeypatch, module, corpus, output, ['Cummin'])
    monkeypatch.setattr(module, 'run_logged', execute)
    assert module.main() == int(not certified)
    summary = json.loads((output / 'suite-summary.json').read_text())
    assert summary['all_pass'] and summary['expected_cases'] == 4 and summary['corpus_cases'] == 1385
    assert json.loads((output / 'target-pattern-validation.json').read_text())['passed'] is certified
