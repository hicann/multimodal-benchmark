# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import importlib.util
import json
import os
import shlex
import shutil
import subprocess
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_eval import submission_http, submission_service


SCRIPT = Path(__file__).resolve().parents[2] / 'scripts/submission_simulate.py'
SPEC = importlib.util.spec_from_file_location('submission_verify_entry', SCRIPT)
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


def fixture_git(directory, *arguments):
    executable = shutil.which('git')
    if executable is None:
        raise RuntimeError('Git is required for publication regression tests')
    return subprocess.check_output([executable, '-C', str(directory), *arguments], text=True).strip()


@pytest.fixture
def shell_project(tmp_path):
    project = tmp_path / 'platform with spaces'
    scripts = project / 'scripts'
    scripts.mkdir(parents=True)
    for name in ('submission-simulation-verify.sh', 'publish_ascend_results.py'):
        shutil.copy2(SCRIPT.parent / name, scripts / name)
    (scripts / 'submission-simulation-verify.sh').chmod(0o755)
    package = project / 'src/triton_eval'
    package.mkdir(parents=True)
    for name in ('__init__.py', 'console.py'):
        shutil.copy2(SCRIPT.parents[1] / 'src/triton_eval' / name, package / name)
    (scripts / 'env-bootstrap.sh').write_text('export PYTHONPATH="$PWD/src"\n')
    (scripts / 'submission_simulate.py').write_text(
        'import os, sys\nraise SystemExit(int(os.environ["FIXTURE_EVALUATION_RC"]))\n')
    binaries = tmp_path / 'binaries'
    binaries.mkdir()
    for name in ('python', 'python3'):
        interpreter = binaries / name
        interpreter.write_text('#!/bin/sh\nexec ' + shlex.quote(sys.executable) + ' -S "$@"\n')
        interpreter.chmod(0o755)
    remote = tmp_path / 'remote.git'
    fixture_git(tmp_path, 'init', '--bare', '-q', str(remote))
    fixture_git(project, 'init', '-q')
    fixture_git(project, '-c', 'user.name=Regression', '-c', 'user.email=regression@example.invalid',
                'commit', '--allow-empty', '-qm', 'Shell regression fixture')
    fixture_git(project, 'remote', 'add', 'origin', str(remote))
    environment = dict(os.environ)
    excluded = {'PYTHONPATH', 'TRITON_EVAL_PUSH_URL', 'GITCODE_TOKEN'}
    for key in tuple(environment):
        if key in excluded or key.startswith('GIT_'):
            environment.pop(key)
    environment['PATH'] = str(binaries) + os.pathsep + environment['PATH']
    return SimpleNamespace(root=project, remote=remote, env=environment, outside=tmp_path)


def run_shell_fixture(project, returncode):
    project.env['FIXTURE_EVALUATION_RC'] = str(returncode)
    return subprocess.run([str(project.root / 'scripts/submission-simulation-verify.sh'),
                           'independent'], cwd=project.outside, env=project.env,
                          capture_output=True, text=True, timeout=30)


@pytest.mark.parametrize('evaluation_returncode', [0, 1])
def test_exit_trap_publishes_without_inheriting_bootstrap_pythonpath(shell_project, evaluation_returncode):
    result = run_shell_fixture(shell_project, evaluation_returncode)
    assert result.returncode == evaluation_returncode, result.stdout + result.stderr
    index = json.loads(fixture_git(shell_project.remote, 'show', 'npu-results:rolling-results.json'))
    execution = json.loads(fixture_git(shell_project.remote, 'show',
                                      f'npu-results:records/{index["latest"]}/execution.json'))
    assert execution['state'] == 'finished'
    assert execution['returncode'] == evaluation_returncode
    assert execution['mode'] == 'independent'
    assert execution['code_commit'] == fixture_git(shell_project.root, 'rev-parse', 'HEAD')
    assert not (shell_project.root / 'results-upload-failed.txt').exists()


def test_upload_failure_prints_retry_executable_outside_project(shell_project):
    fixture_git(shell_project.root, 'remote', 'set-url', 'origin', str(shell_project.outside / 'missing.git'))
    result = run_shell_fixture(shell_project, 1)
    assert result.returncode == 2, result.stdout + result.stderr
    retry = next(line.split('补传：', 1)[1] for line in result.stdout.splitlines()
                 if line.startswith('[simulation] 上传失败；补传：'))
    fixture_git(shell_project.root, 'remote', 'set-url', 'origin', str(shell_project.remote))
    uploaded = subprocess.run(shlex.split(retry), cwd=shell_project.outside, env=shell_project.env,
                              capture_output=True, text=True, timeout=30)
    assert uploaded.returncode == 0, uploaded.stdout + uploaded.stderr
    index = json.loads(fixture_git(shell_project.remote, 'show', 'npu-results:rolling-results.json'))
    execution = json.loads(fixture_git(shell_project.remote, 'show',
                                      f'npu-results:records/{index["latest"]}/execution.json'))
    assert execution['returncode'] == 1


@pytest.fixture
def runtime(monkeypatch):
    events = []
    stopped = threading.Event()

    class Service:
        def __init__(self, root, corpus, *, execute):
            assert execute
            self.root = root
            root.mkdir(parents=True)

        @staticmethod
        def start_worker():
            events.append('worker')

        def close(self):
            events.append('close-service')
            (self.root / 'worker-paused.txt').write_text('health check required\n')

    class Server:
        server_port = 12345

        @staticmethod
        def serve_forever():
            stopped.wait()

        @staticmethod
        def shutdown():
            events.append('shutdown')
            stopped.set()

        @staticmethod
        def server_close():
            events.append('close-server')

    def build(dest):
        dest.mkdir(parents=True)
        (dest / 'expectations.json').write_text('{}\n')

    monkeypatch.setattr(ENTRY, 'build_suite', build)
    monkeypatch.setattr(submission_service, 'EvaluationService', Service)
    monkeypatch.setattr(submission_http, 'make_server', lambda service, port: Server())
    return SimpleNamespace(events=events)


@pytest.mark.parametrize('returncode', [0, 1])
def test_verify_runs_normal_and_preserves_client_status(tmp_path, monkeypatch, runtime, returncode):
    def client(arguments):
        assert arguments[0] == 'run'
        assert arguments[arguments.index('--mode') + 1] == 'normal'
        assert arguments[arguments.index('--url') + 1] == 'http://127.0.0.1:12345'
        return returncode

    monkeypatch.setattr(ENTRY, 'main', client)
    record = tmp_path / 'records/batch'
    assert ENTRY.verify(record, tmp_path / 'work/batch') == returncode
    assert (record / 'expectations.json').read_text() == '{}\n'
    assert (record / 'worker-paused.txt').is_file()
    assert runtime.events == ['worker', 'shutdown', 'close-server', 'close-service']


def test_verify_interrupt_closes_http_and_worker(tmp_path, monkeypatch, runtime):
    def interrupted(arguments):
        raise KeyboardInterrupt

    monkeypatch.setattr(ENTRY, 'main', interrupted)
    with pytest.raises(KeyboardInterrupt):
        ENTRY.verify(tmp_path / 'records/batch', tmp_path / 'work/batch')
    assert runtime.events == ['worker', 'shutdown', 'close-server', 'close-service']


def test_verify_bind_failure_closes_service(tmp_path, monkeypatch, runtime):
    def denied(service, port):
        raise PermissionError('loopback forbidden')

    monkeypatch.setattr(submission_http, 'make_server', denied)
    with pytest.raises(PermissionError):
        ENTRY.verify(tmp_path / 'records/batch', tmp_path / 'work/batch')
    assert runtime.events == ['close-service']


@pytest.mark.parametrize('pause_path', ['work/old/service/worker-paused.txt', 'service/worker-paused.txt'])
def test_verify_does_not_bypass_previous_pause(tmp_path, runtime, pause_path):
    pause = tmp_path / pause_path
    pause.parent.mkdir(parents=True)
    pause.write_text('manual health check required')
    with pytest.raises(RuntimeError, match='暂停标记'):
        ENTRY.verify(tmp_path / 'records/new', tmp_path / 'work/new')
    assert runtime.events == []
    assert pause.exists()


def test_verify_requires_isolation_acknowledgement(tmp_path):
    result = subprocess.run(
        [sys.executable, str(SCRIPT), 'verify',
         '--record', str(tmp_path / 'record'), '--work', str(tmp_path / 'work')],
        capture_output=True, text=True, timeout=30,
    )
    assert result.returncode == 2
    assert '--acknowledge-isolation' in result.stderr
    assert not (tmp_path / 'record').exists()
    assert not (tmp_path / 'work').exists()


def test_verify_independent_mode_is_separate_from_historical_normal(tmp_path, monkeypatch, runtime):
    def client(arguments):
        assert arguments[arguments.index('--mode') + 1] == 'independent'
        assert arguments[arguments.index('--output') + 1].endswith('independent-observations.json')
        return 0

    monkeypatch.setattr(ENTRY, 'main', client)
    assert ENTRY.verify(tmp_path / 'records/batch', tmp_path / 'work/batch', 'independent') == 0


def test_task_wait_reports_status_changes_and_waiting_progress(monkeypatch, capsys):
    states = iter(['RUNNING', 'SCORING', 'REPORTING', 'COMPLETED'])
    clock = [0]
    monkeypatch.setattr(ENTRY.time, 'monotonic', lambda: clock[0])

    def advance(seconds):
        clock[0] += 16

    def poll(base, path):
        return dict(id='test-task', status=next(states), worker_enabled=True, worker_paused=False)

    monkeypatch.setattr(ENTRY.time, 'sleep', advance)
    monkeypatch.setattr(ENTRY, 'request', poll)
    args = SimpleNamespace(timeout=120, url='http://127.0.0.1')
    result = ENTRY.wait_simulation_task(args, dict(id='test-task', status='RUNNING',
                                                 worker_enabled=True, worker_paused=False))
    assert result['status'] == 'COMPLETED'
    printed = capsys.readouterr().out
    assert printed.count('status=RUNNING') == 2
    assert all('status=' + status in printed for status in ('SCORING', 'REPORTING', 'COMPLETED'))
    assert 'elapsed=16s' in printed


def test_verify_forwards_exact_scenario_filters(tmp_path, monkeypatch, runtime):
    """The server workflow preserves filters when invoking its HTTP client."""
    requested = ('OK-EXP', 'EXEC-PARTIAL')

    def client(arguments):
        """Inspect the real client command boundary without executing NPU work."""
        selected = [arguments[index + 1] for index, value in enumerate(arguments) if value == '--scenario']
        assert selected == list(requested)
        return 0

    monkeypatch.setattr(ENTRY, 'main', client)
    assert ENTRY.verify(tmp_path / 'record', tmp_path / 'work/batch', scenarios=requested) == 0
    assert runtime.events == ['worker', 'shutdown', 'close-server', 'close-service']


@pytest.mark.parametrize('command', ['run', 'verify'])
def test_filter_arguments_accept_repeated_exact_codes(tmp_path, command, monkeypatch):
    """Exercise repeated filters through public CLI dispatch rather than private parsing."""
    codes = ['OK-EXP', 'EXEC-PARTIAL']
    suite = dict(scenarios=[dict(code=code, tier='normal', stage='execution', description=code) for code in codes])
    (tmp_path / 'expectations.json').write_text(json.dumps(suite), encoding='utf-8')
    observed = []

    def verify_selected(record, work, mode, scenarios):
        """Observe verify dispatch without starting a device service."""
        assert mode == 'normal'
        observed.extend(scenarios)
        return 0

    def observe(args, scenario, run_id, report, persist, observation):
        """Observe the filters used by the HTTP run loop."""
        observed.append(scenario['code'])

    monkeypatch.setattr(ENTRY, 'verify', verify_selected)
    monkeypatch.setattr(ENTRY, '_check_simulation_catalog', lambda *arguments: False)
    monkeypatch.setattr(ENTRY, '_observe_scenario', observe)
    required = ['--suite', str(tmp_path), '--output', str(tmp_path / 'report.json'), '--mode', 'normal']
    if command == 'verify':
        required = ['--record', str(tmp_path / 'record'), '--work', str(tmp_path / 'work'),
                    '--acknowledge-isolation']
    assert ENTRY.main([command, *required, '--scenario', codes[0], '--scenario', codes[1]]) == 0
    assert observed == codes


@pytest.mark.parametrize('requested', [[], ['OK-ADD'], ['OK-ADD', 'OK-EXP']])
def test_selection_preserves_entire_scenarios_and_original_catalog_order(requested):
    """Filtering keeps each scenario object, including its cases and score assertions."""
    records = [dict(code='OK-EXP', tier='normal', stage='execution', case_codes=['a', 'b']),
               dict(code='INDEPENDENT', tier='independent', stage='execution'),
               dict(code='OK-ADD', tier='normal', stage='execution', expected_score={'functional_score': 100})]
    selected = ENTRY.select_scenarios(dict(scenarios=records), 'normal', requested)
    expected = [record for record in records if record['tier'] == 'normal']
    if requested:
        expected = [record for record in expected if record['code'] in requested]
    assert selected == expected
    assert all(actual is original for actual, original in zip(selected, expected, strict=True))


@pytest.mark.parametrize('requested', [['TYPO'], ['OK-*'], ['INDEPENDENT'], ['OK-EXP', 'OK-EXP']])
def test_selection_rejects_unknown_out_of_mode_and_duplicate_codes(requested):
    """A bad filter cannot silently produce an empty or misleading success."""
    suite = dict(scenarios=[dict(code='OK-EXP', tier='normal', stage='execution'),
                            dict(code='INDEPENDENT', tier='independent', stage='execution')])
    with pytest.raises(ValueError, match='scenario codes'):
        ENTRY.select_scenarios(suite, 'normal', requested)


@pytest.mark.parametrize('suite', ['default', 'failed-last', 'full'])
@pytest.mark.parametrize('returncode', [0, 1])
def test_quality_shell_forwards_selected_suite_and_publishes_scope(shell_project, suite, returncode):
    """The reusable shell defaults to focused NPU runs and still publishes results."""
    scripts = shell_project.root / 'scripts'
    executable = scripts / 'codecheck-regression-verify.sh'
    shutil.copy2(SCRIPT.parent / executable.name, executable)
    executable.chmod(0o755)
    (scripts / 'verify_codecheck_regression.py').write_text(
        'import json, sys\nfrom pathlib import Path\n'
        'record = Path(sys.argv[sys.argv.index("--record") + 1])\n'
        'record.mkdir(parents=True)\n'
        '(record / "arguments.json").write_text(json.dumps(sys.argv[1:]))\n'
        f'if __name__ == "__main__":\n    raise SystemExit({returncode})\n')
    arguments = [] if suite == 'default' else [suite]
    result = subprocess.run([str(executable), *arguments],
                            cwd=shell_project.outside, env=shell_project.env,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode == returncode, result.stdout + result.stderr
    index = json.loads(fixture_git(shell_project.remote, 'show', 'npu-results:rolling-results.json'))
    prefix = 'npu-results:records/' + index['latest'] + '/'
    forwarded = json.loads(fixture_git(shell_project.remote, 'show', prefix + 'arguments.json'))
    actual_suite = 'failed-last' if suite == 'default' else suite
    assert forwarded[forwarded.index('--suite') + 1] == actual_suite
    execution = json.loads(fixture_git(shell_project.remote, 'show', prefix + 'execution.json'))
    expected_scope = 'SUBMISSION_E2E' if suite == 'full' else 'SUBMISSION_E2E_FAILED_LAST'
    assert execution['scope'] == expected_scope
    assert execution['returncode'] == returncode


def test_filtered_http_client_observes_only_requested_scenarios(tmp_path, monkeypatch):
    """The actual run loop filters before uploading and persists the chosen scope."""
    suite = dict(scenarios=[dict(code='OK-EXP', tier='normal', stage='execution', description='selected'),
                            dict(code='OK-ADD', tier='normal', stage='execution', description='skipped')])
    (tmp_path / 'expectations.json').write_text(json.dumps(suite), encoding='utf-8')
    monkeypatch.setattr(ENTRY, '_check_simulation_catalog', lambda *arguments: False)
    observed = []

    def observe(args, scenario, run_id, report, persist, observation):
        """Record upload-boundary selection without needing a hardware service."""
        observed.append(scenario['code'])

    monkeypatch.setattr(ENTRY, '_observe_scenario', observe)
    output = tmp_path / 'observations.json'
    assert ENTRY.main(['run', '--suite', str(tmp_path), '--output', str(output),
                       '--mode', 'normal', '--scenario', 'OK-EXP']) == 0
    assert observed == ['OK-EXP']
    report = json.loads(output.read_text())
    assert report['requested_scenarios'] == ['OK-EXP']
    assert report['selected_scenario_count'] == report['matched'] == 1
    assert report['mismatched'] == 0 and report['all_selected_observed'] is True
