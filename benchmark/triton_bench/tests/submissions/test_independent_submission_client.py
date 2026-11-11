# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Serialized HTTP and real scoring with explicit device-result fixtures, not NPU evidence."""

from __future__ import annotations

import copy
import hashlib
import importlib.util
import json
import sys
import urllib.error
from io import BytesIO
from pathlib import Path
from urllib.parse import urlsplit

import pytest

from test_submission_http import raw_http
from test_submission_service_worker import engine_fixture, intercept_engine_processes
from triton_eval.submission_service import EvaluationService
from triton_eval.submission_simulation import DEFAULT_CORPUS, build_suite, check_outcome


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/submission_simulate.py'
SPEC = importlib.util.spec_from_file_location('independent_submission_client', SCRIPT)
CLIENT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CLIENT)


def test_new_zip_http_isolation_replay_scope_and_artifact_chain_with_fixture_device_results(tmp_path, monkeypatch):
    suite = tmp_path / 'suite'
    expectations = build_suite(suite)
    selected = {'NEW-REPEAT-A', 'NEW-REPEAT-B', 'NEW-UNICODE-ID', 'NEW-SCOPED'}
    expectations['scenarios'] = [item for item in expectations['scenarios'] if item['code'] in selected]
    (suite / 'expectations.json').write_text(json.dumps(expectations))
    service = EvaluationService(tmp_path / 'service', DEFAULT_CORPUS, execute=True)

    def invoke(task, phase, command):
        if phase == 'run':
            engine_fixture(service, task, Path(command[4]).parent)
            return [sys.executable, '-c', 'raise SystemExit(0)']
        return None

    def transport(request, timeout):
        path = urlsplit(request.full_url).path
        if request.get_method() == 'GET' and path.startswith('/api/v1/evaluations/') and path.count('/') == 4:
            service.work_once()
        code, body = raw_http(service, request.get_method(), path, request.data or b'',
                              request.get_header('Content-type', 'application/json'), dict(request.header_items()))
        if code >= 400:
            raise urllib.error.HTTPError(request.full_url, code, 'HTTP fixture transport', {}, BytesIO(body))
        return BytesIO(body)

    intercept_engine_processes(monkeypatch, service, invoke)
    monkeypatch.setattr(CLIENT.urllib.request, 'urlopen', transport)
    output = tmp_path / 'observations.json'
    try:
        assert CLIENT.main(['run', '--suite', str(suite), '--mode', 'independent', '--output', str(output)]) == 0
        report = json.loads(output.read_text())
        assert report['matched'] == 4 and report['mismatched'] == 0
        assert report['acceptance']['all_expectations_matched']
        assert report['acceptance']['universal_legal_submission_support'] == 'NOT_PROVEN'
        records = {item['scenario']: item for item in report['observations']}
        first, second = records['NEW-REPEAT-A'], records['NEW-REPEAT-B']
        assert first['submission']['package_sha256'] == second['submission']['package_sha256']
        assert first['submission']['id'] != second['submission']['id']
        assert first['task']['id'] != second['task']['id']
        scoped = records['NEW-SCOPED']['results']
        assert set(scoped['declared_operators']) == {'Exp', 'Add'}
        assert scoped['scope_operators'] == ['Exp'] and scoped['completed_cases'] == 20
        for item in report['observations']:
            assert item['submission']['id'] == item['upload_replay_id']
            assert item['task']['id'] == item['task_replay_id']
            assert {artifact['name'] for artifact in item['artifacts']} >= {'report.md', 'report.html', 'results.jsonl'}
            for artifact in item['artifacts']:
                path = output.parent / (output.stem + '-artifacts') / item['scenario'] / artifact['name']
                assert hashlib.sha256(path.read_bytes()).hexdigest() == artifact['sha256']
    finally:
        service.close()


def test_empty_independent_suite_cannot_pass_without_running_anything(tmp_path):
    (tmp_path / 'expectations.json').write_text(json.dumps(dict(scenarios=[])))
    output = tmp_path / 'observations.json'
    assert CLIENT.main(['run', '--suite', str(tmp_path), '--mode', 'independent', '--output', str(output)]) == 1
    report = json.loads(output.read_text())
    assert not report['all_selected_observed'] and report['selected_scenario_count'] == 0


@pytest.mark.parametrize('fault', ['scope', 'package', 'engine', 'score', 'cases'])
def test_positive_acceptance_rejects_wrong_scope_identity_errors_and_missing_cases(tmp_path, fault):
    report = build_suite(tmp_path)
    scenario = next(item for item in report['scenarios'] if item['code'] == 'NEW-EXP')
    cases = [dict(operator='Exp', case_code=code, status='PASS') for code in scenario['case_codes']]
    task = dict(
        status='COMPLETED',
        planned_cases=len(cases),
        profile='smoke',
        manifest_hash=report['manifest_hash'],
        case_codes=scenario['case_codes'],
        package_sha256=scenario['sha256'])
    results = dict(cases=cases, package_sha256=scenario['sha256'], declared_operators=['Exp'],
                   scope_operators=['Exp'], score=dict(coverage_ops_core=100, coverage_strict=100))
    assert check_outcome(scenario, dict(status='READY'), task, results) == []
    results = copy.deepcopy(results)
    if fault == 'scope':
        results['scope_operators'] = ['Add']
    elif fault == 'package':
        results['package_sha256'] = 'wrong-package'
    elif fault == 'engine':
        results['engine_error'] = 'ENGINE-DATABASE'
    elif fault == 'score':
        results['score']['coverage_ops_core'] = 0
    else:
        results['cases'].pop()
    assert check_outcome(scenario, dict(status='READY'), task, results)
