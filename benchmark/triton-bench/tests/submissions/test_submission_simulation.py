# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import hashlib
import json
import signal
import time
import sqlite3
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from triton_eval.package import ingest_zip
from triton_eval.submission_service import EvaluationService, ServiceError
from triton_eval.submission_simulation import (
    DEFAULT_CORPUS, build_suite, check_outcome, scenarios,
)


@pytest.fixture(scope='module')
def suite(tmp_path_factory):
    directory = tmp_path_factory.mktemp('submission-simulation')
    report = build_suite(directory)
    return directory, {record['code']: record for record in report['scenarios']}


@pytest.fixture
def service(tmp_path):
    instance = EvaluationService(tmp_path / 'service', DEFAULT_CORPUS)
    yield instance
    instance.close()


def package(suite, code):
    directory, records = suite
    return (directory / records[code]['package']).read_bytes()


def test_catalog_has_independent_unique_content_scenarios(suite):
    directory, records = suite
    assert len(records) == len(scenarios()) == 73
    report = json.loads((directory / 'expectations.json').read_text())
    assert report['evidence_kind'] == 'EXPECTATIONS_NOT_HARDWARE_RESULTS'
    assert report['source_hashes']
    assert {record['tier'] for record in records.values()} == {'normal', 'independent', 'full', 'slow', 'disruptive'}
    assert all(not (record['rule'] or '').startswith('SUB-PKG-') for record in records.values())


def test_generated_archives_use_existing_ingest_without_package_negatives(suite, tmp_path):
    directory, records = suite
    for record in records.values():
        archive = directory / record['package']
        assert hashlib.sha256(archive.read_bytes()).hexdigest() == record['sha256']
        ingested = ingest_zip(archive, tmp_path / record['code'])
        assert (ingested.dest / 'manifest.yaml').is_file()


def test_build_is_reproducible(suite, tmp_path):
    directory, records = suite
    rebuilt = build_suite(tmp_path / 'rebuilt')
    assert {record['code']: record['sha256'] for record in rebuilt['scenarios']} == {
        code: record['sha256'] for code, record in records.items()}
    assert (directory / 'expectations.json').read_bytes() == (tmp_path / 'rebuilt/expectations.json').read_bytes()


@pytest.mark.parametrize('scenario', scenarios(), ids=lambda scenario: scenario.code)
def test_admission_reaches_intended_layer(service, suite, scenario):
    submission = service.upload(package(suite, scenario.code))
    if scenario.stage == 'content':
        assert submission['stage'] == 'content'
        assert check_outcome(asdict(scenario), submission, None, None) == []
        assert {error['rule_id'] for error in submission['errors']} == {scenario.rule}
    else:
        assert submission['status'] == 'READY', submission['errors']
        assert not submission['errors']


def test_upload_never_imports_user_code(service, suite):
    assert service.upload(package(suite, 'LOAD-EXIT'))['status'] == 'READY'
    assert service.upload(package(suite, 'LOAD-RAISE'))['status'] == 'READY'
    assert service.upload(package(suite, 'COMPILE-UNDEFINED'))['status'] == 'READY'


def test_package_identity_and_idempotency_are_not_submit_id(service, suite):
    data = package(suite, 'OK-EXP')
    first = service.upload(data, key='upload-one')
    assert service.upload(data, key='upload-one')['id'] == first['id']
    second = service.upload(data)
    assert first['id'] != second['id']
    assert first['declared_submit_id'] == second['declared_submit_id']
    with pytest.raises(ServiceError, match='不同内容'):
        service.upload(package(suite, 'OK-ADD'), key='upload-one')


def test_scope_uses_all_frozen_cases_for_selected_operator(service, suite):
    submission = service.upload(package(suite, 'OK-MULTI'))
    task = service.create_task(dict(submission_id=submission['id'], operators=['Exp']), key='task')
    expected = [case.code for case in service.manifest.cases if case.operator == 'Exp']
    assert task['case_codes'] == expected
    assert task['planned_cases'] == len(expected)
    assert task['manifest_hash'] == service.manifest.manifest_hash
    assert task['package_sha256'] == submission['package_sha256']
    assert service.create_task(dict(submission_id=submission['id'], operators=['Exp']), key='task')['id'] == task['id']
    with pytest.raises(ServiceError) as conflict:
        service.create_task(dict(submission_id=submission['id'], operators=['Add']), key='task')
    assert conflict.value.code == 'IDEMPOTENCY-CONFLICT'
    full = service.create_task(dict(submission_id=submission['id'], scope='full'))
    assert full['planned_cases'] == 1385
    assert len(full['operators']) == 74


@pytest.mark.parametrize('update', [dict(operators=[]), dict(operators=['Exp', 'Exp']),
                                    dict(operators=['Matmul']), 
                                    dict(scope='full', operators=['Exp']), dict(profile='official'),
                                    dict(corpus='/tmp/mine'), dict(limit=1), dict(scope=[]), dict(profile={})])
def test_request_cannot_override_contract_or_silently_shrink_scope(service, suite, update):
    submission = service.upload(package(suite, 'OK-EXP'))
    with pytest.raises(ServiceError):
        service.create_task(dict(submission_id=submission['id'], **update))


def test_rejected_submission_cannot_create_job(service, suite):
    submission = service.upload(package(suite, 'META-EMPTY'))
    with pytest.raises(ServiceError) as rejected:
        service.create_task(dict(submission_id=submission['id']))
    assert rejected.value.code == 'SUBMISSION-REJECTED'


def test_queued_cancellation_and_event_cursor(service, suite):
    submission = service.upload(package(suite, 'OK-EXP'))
    task = service.create_task(dict(submission_id=submission['id']))
    initial = service.events(task['id'])
    assert initial['items'][0]['status'] == 'QUEUED'
    cancelled = service.cancel(task['id'])
    assert cancelled['status'] == 'CANCELLED'
    assert service.cancel(task['id'])['status'] == 'CANCELLED'
    more = service.events(task['id'], initial['next_cursor'])
    assert [event['status'] for event in more['items']] == ['CANCELLED']
    assert service.results(task['id'])['score'] is None
    assert not service.work_once()


def test_restart_retains_queued_jobs_and_marks_interrupted_jobs(tmp_path, suite):
    root = tmp_path / 'persistent'
    original = EvaluationService(root, DEFAULT_CORPUS, execute=True)
    try:
        submission = original.upload(package(suite, 'OK-EXP'))
        interrupted = original.create_task(dict(submission_id=submission['id']))
        queued = original.create_task(dict(submission_id=submission['id']))
    finally:
        original.close()

    marker = tmp_path / 'worker-ready.json'
    probe = Path(__file__).with_name('worker_restart_probe.py')
    with subprocess.Popen([sys.executable, str(probe), str(root), str(DEFAULT_CORPUS),
                           interrupted['id'], str(marker)],
                          stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as worker:
        try:
            deadline = time.monotonic() + 30
            while not marker.exists() and worker.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert marker.exists(), 'worker did not reach the pre-device execution boundary'
            current = json.loads(marker.read_text())
            assert current['id'] == interrupted['id'] and current['status'] == 'RUNNING'
            worker.terminate()
            worker.communicate(timeout=10)
            assert worker.returncode == -signal.SIGTERM
        finally:
            if worker.poll() is None:
                worker.kill()
                worker.communicate(timeout=10)

    restored = EvaluationService(root, DEFAULT_CORPUS)
    try:
        assert restored.task(queued['id'])['status'] == 'QUEUED'
        assert restored.task(interrupted['id'])['status'] == 'INTERRUPTED'
        assert restored.task(queued['id'])['worker_paused']
        assert restored.submission(submission['id'])['package_sha256'] == submission['package_sha256']
    finally:
        restored.close()


def test_service_directory_has_single_owner(service):
    with pytest.raises(ServiceError) as conflict:
        EvaluationService(service.root, DEFAULT_CORPUS)
    assert conflict.value.code == 'SERVICE-IN-USE'


def finish_engine_processes(monkeypatch, service, finished):
    import triton_eval.submission_service as worker
    from test_submission_service_worker import engine_fixture

    original = worker.subprocess.Popen

    def spawn(command, **options):
        assert command[:4] == [sys.executable, '-m', 'triton_eval', '--db']
        task = service.task(Path(command[4]).parent.name)
        phase = command[5]
        assert phase in {'run', 'score', 'report'}
        replacement = command
        if phase == 'run':
            engine_fixture(service, task, Path(command[4]).parent)
            replacement = [sys.executable, '-c', 'pass']
        process = original(replacement, **options)
        try:
            process.wait(timeout=30)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait()
            raise
        finished(task, phase)
        return process

    monkeypatch.setattr(worker.subprocess, 'Popen', spawn)


class FrozenTaskRead:
    def __init__(self, row):
        self.row = row

    def fetchone(self):
        row, self.row = self.row, None
        return row


def cancel_after_final_snapshot(monkeypatch, service, identifier):
    armed = []
    original_artifacts = service.artifacts
    original_connect = sqlite3.connect
    observed = []

    def artifacts(task_id):
        result = original_artifacts(task_id)
        armed.append(task_id)
        return result

    class SnapshotConnection(sqlite3.Connection):
        def execute(self, query, parameters=()):
            cursor = super().execute(query, parameters)
            if (armed and query == 'SELECT payload FROM objects WHERE kind=? AND id=?'
                    and parameters == ('task', identifier)):
                armed.clear()
                row = cursor.fetchone()
                cursor.close()
                assert row is not None
                observed.append(json.loads(row[0]))
                service.cancel(identifier)
                return FrozenTaskRead(row)
            return cursor

    def connect(*arguments, **options):
        return original_connect(*arguments, **options, factory=SnapshotConnection)

    monkeypatch.setattr(service, 'artifacts', artifacts)
    monkeypatch.setattr(sqlite3, 'connect', connect)
    return observed


@pytest.mark.parametrize('cancel_point', ['after-run', 'after-final-snapshot'])
def test_task_transitions_do_not_lose_concurrent_cancel(service, suite, monkeypatch, cancel_point):
    submission = service.upload(package(suite, 'OK-EXP'))
    task = service.create_task(dict(submission_id=submission['id']))
    service.execute = True
    phases = []

    def finished(current, phase):
        phases.append(phase)
        if cancel_point == 'after-run' and phase == 'run':
            service.cancel(task['id'])
        if cancel_point == 'after-run':
            assert service.task(current['id'])['cancel_requested']

    with monkeypatch.context() as patch:
        observed = (cancel_after_final_snapshot(patch, service, task['id'])
                    if cancel_point == 'after-final-snapshot' else [])
        finish_engine_processes(patch, service, finished)
        assert service.work_once()
    if cancel_point == 'after-final-snapshot':
        assert len(observed) == 1 and not observed[0]['cancel_requested']
        assert observed[0]['status'] == 'REPORTING'
    assert phases == ['run', 'score', 'report']
    assert service.task(task['id'])['cancel_requested']
    assert service.task(task['id'])['status'] == 'CANCELLED'
    assert service.results(task['id'])['score'] is None
    statuses = [event['status'] for event in service.events(task['id'])['items']]
    assert 'SCORING' in statuses and 'REPORTING' in statuses and statuses[-1] == 'CANCELLED'


def test_artifact_allowlist_never_accepts_paths_or_symlinks(service, suite, tmp_path):
    submission = service.upload(package(suite, 'OK-EXP'))
    task = service.create_task(dict(submission_id=submission['id']))
    directory = service.root / 'evaluations' / task['id']
    directory.mkdir(parents=True)
    (directory / 'run.log').write_text('local evidence')
    assert service.artifact(task['id'], 'run.log') == b'local evidence'
    with pytest.raises(ServiceError):
        service.artifact(task['id'], '../../service.sqlite')
    (directory / 'run.log').unlink()
    (directory / 'run.log').symlink_to(tmp_path / 'outside')
    assert not service.artifacts(task['id'])


def test_infrastructure_errors_never_match_expected_compile_failure():
    scenario = asdict(next(item for item in scenarios() if item.code == 'COMPILE-UNDEFINED'))
    failures = check_outcome(scenario, dict(status='READY'), dict(status='FAILED'), dict(cases=[]))
    assert failures


def test_corrupt_engine_database_does_not_hide_persistent_task_state(service, suite):
    submission = service.upload(package(suite, 'OK-EXP'))
    task = service.create_task(dict(submission_id=submission['id']))
    directory = service.root / 'evaluations' / task['id']
    directory.mkdir(parents=True)
    (directory / 'engine.sqlite').write_bytes(b'corrupt engine database')
    assert service.task(task['id'])['status'] == 'QUEUED'
    results = service.results(task['id'])
    assert results['engine_error'] == 'ENGINE-DATABASE'
    assert results['score'] is None


def test_independent_expectations_check_scores_not_only_case_status():
    scenario = asdict(next(item for item in scenarios() if item.code == 'OK-EXP'))
    task = dict(status='COMPLETED', planned_cases=1, profile='smoke')
    results = dict(cases=[dict(operator='Exp', status='PASS', case_code='fixture')],
                   score=dict(coverage_ops_core=100, coverage_strict=100))
    assert check_outcome(scenario, dict(status='READY'), task, results) == []
    results['score']['coverage_ops_core'] = 0
    assert check_outcome(scenario, dict(status='READY'), task, results)


@pytest.mark.parametrize('code,operator', [('OK-CUMMIN', 'Cummin'), ('OK-ARGMAX', 'ArgMax'),
                                         ('OK-SWIGLU', 'SwiGlu'), ('OK-CROSSENTROPY', 'CrossEntropyLoss')])
def test_parameter_migration_operators_reach_normal_zip_evaluation(suite, code, operator):
    _, records = suite
    record = records[code]
    assert record['tier'] == 'normal' and record['stage'] == 'execution'
    assert tuple(record['operators']) == (operator,)
    assert record['case_codes'] and tuple(record['statuses']) == ('PASS',)
