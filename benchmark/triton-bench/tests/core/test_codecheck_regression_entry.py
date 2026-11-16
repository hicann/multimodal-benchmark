# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fail-closed evidence handling for the real-device regression entry."""

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location('quality_regression_entry',
                                            ROOT / 'scripts/verify_codecheck_regression.py')
ENTRY = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ENTRY)


@pytest.mark.parametrize('count,skipped,expected', [(54, False, 'PASS'), (54, True, 'FAIL'),
                                                  (53, False, 'FAIL'), (0, False, 'FAIL')])
def test_device_results_require_all_cases_without_skips(tmp_path, count, skipped, expected):
    content = '<testsuites><testsuite>' + ''.join(
        '<testcase name="case' + str(index) + '">' +
        ('<skipped/>' if skipped and index == 0 else '') + '</testcase>' for index in range(count)
    ) + '</testsuite></testsuites>'
    (tmp_path / 'npu.xml').write_text(content)
    result = ENTRY.execute_step(dict(name='device', command=[sys.executable, '-c', 'pass'],
                                    timeout=10, junit='npu.xml', required_cases=54), tmp_path, dict(os.environ))
    assert result['state'] == expected
    assert result['returncode'] == (0 if expected == 'PASS' else 1)


def test_subprocess_timeout_is_a_failure_and_is_reaped(tmp_path):
    result = ENTRY.execute_step(dict(name='timeout', command=[sys.executable, '-c',
                               'import time; time.sleep(30)'], timeout=0.1), tmp_path, dict(os.environ))
    assert result['state'] == 'TIMEOUT' and result['returncode'] == 124


def test_failure_stops_later_stages_and_preserves_status(tmp_path, monkeypatch):
    monkeypatch.setattr(ENTRY.subprocess, 'check_output', lambda *args, **kwargs: 'fixture-commit')
    monkeypatch.setattr(ENTRY, 'static_checks', lambda record: None)
    monkeypatch.setattr(ENTRY.platform, 'platform', lambda: 'fixture-platform')
    invoked = []

    def fail(step, record, environment):
        running = json.loads((record / 'execution.json').read_text())
        assert next(item for item in running['steps'] if item['name'] == step['name'])['state'] == 'RUNNING'
        invoked.append(step['name'])
        return dict(name=step['name'], state='FAIL', returncode=7)

    monkeypatch.setattr(ENTRY, 'execute_step', fail)
    record = tmp_path / 'record'
    assert ENTRY.run(record, tmp_path / 'work') == 7
    status = json.loads((record / 'execution.json').read_text())
    assert invoked == ['simulation-independent']
    assert status['steps'][0]['state'] == 'FAIL'
    assert all(step['state'] == 'NOT_RUN' for step in status['steps'][1:])
    assert status['codecheck_remote_rescan'] == 'NOT_RUN'
    assert status['state'] == 'finished' and status['returncode'] == 7


def test_device_entry_runs_only_existing_http_workflows(tmp_path):
    plan = ENTRY.steps(tmp_path / 'record', tmp_path / 'work', cpu_only=False)
    assert [step['name'] for step in plan] == ['simulation-independent', 'simulation-normal']
    for step in plan:
        assert step['command'][1] == str(ENTRY.ROOT / 'scripts/submission_simulate.py')
        assert step['command'][2] == 'verify'
        assert '--acknowledge-isolation' in step['command']
    assert [step['name'] for step in ENTRY.steps(tmp_path, tmp_path, cpu_only=True)] == ['cpu']


def test_stage_output_reaches_console_and_log(tmp_path, capsys):
    command = [sys.executable, '-u', '-c', 'print("ZIP accepted; score complete", flush=True)']
    result = ENTRY.execute_step(dict(name='http', command=command, timeout=10), tmp_path, dict(os.environ))
    assert result['state'] == 'PASS'
    assert 'ZIP accepted; score complete' in capsys.readouterr().out
    assert 'ZIP accepted; score complete' in (tmp_path / 'http.log').read_text()


@pytest.mark.parametrize('defect', ['missing', 'empty', 'incomplete', 'mismatch', 'valid'])
def test_http_step_requires_complete_observation_evidence(tmp_path, defect):
    report = dict(evidence_kind='HTTP_OBSERVATIONS', mode='normal', selected_scenario_count=1,
                  all_selected_observed=True, matched=1, mismatched=0,
                  observations=[dict(scenario='OK-EXP', failures=[])])
    if defect == 'empty':
        report.update(selected_scenario_count=0, matched=0, observations=[])
    elif defect == 'incomplete':
        report.update(all_selected_observed=False)
    elif defect == 'mismatch':
        report['observations'][0]['failures'] = ['wrong score']
    if defect != 'missing':
        (tmp_path / 'observations.json').write_text(json.dumps(report))
    step = dict(name='http', command=[sys.executable, '-c', 'pass'], timeout=10,
                observations='observations.json', mode='normal')
    result = ENTRY.execute_step(step, tmp_path, dict(os.environ))
    assert result['state'] == ('PASS' if defect == 'valid' else 'FAIL')


def test_progress_streams_before_process_exits_and_preserves_utf8(tmp_path, monkeypatch, capsys):
    import codecs

    log_path = tmp_path / 'progress.log'
    message = '任务正在执行'.encode('utf-8')
    log_path.write_bytes(message[:2])

    class RunningProcess:
        checks = 0

        def wait(self, timeout):
            if self.checks == 0:
                assert capsys.readouterr().out == ''
                with log_path.open('ab') as writer:
                    writer.write(message[2:])
                self.checks += 1
                raise ENTRY.subprocess.TimeoutExpired(['fixture'], timeout)
            assert capsys.readouterr().out == '任务正在执行'
            return 0

    with log_path.open('rb') as output:
        decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
        assert ENTRY.wait_with_progress(RunningProcess(), output, decoder,
                                        dict(name='http', command=['fixture'], timeout=10)) == 0


@pytest.mark.parametrize('available', [True, False])
def test_git_uses_a_resolved_absolute_executable(tmp_path, monkeypatch, available):
    executable = tmp_path / 'git'
    executable.touch()
    monkeypatch.setattr(ENTRY.shutil, 'which', lambda name: str(executable) if available else None)
    commands = []

    def capture(command, **kwargs):
        commands.append(command)
        return 'fixture-commit\n'

    monkeypatch.setattr(ENTRY.subprocess, 'check_output', capture)
    if not available:
        with pytest.raises(FileNotFoundError, match='Git executable'):
            ENTRY.git_output('rev-parse', 'HEAD')
        assert commands == []
        return
    assert ENTRY.git_output('rev-parse', 'HEAD') == 'fixture-commit'
    assert commands == [[str(executable.resolve()), '-C', str(ENTRY.ROOT), 'rev-parse', 'HEAD']]


def test_stage_messages_use_logging(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.INFO, logger=ENTRY.LOGGER.name):
        result = ENTRY.execute_step(dict(name='logged', command=[sys.executable, '-c', 'pass'], timeout=10),
                                    tmp_path, dict(os.environ))
    assert result['state'] == 'PASS'
    assert '[quality] starting logged; log:' in caplog.text
    assert '[quality] logged: PASS' in caplog.text


def test_heartbeat_uses_logging_without_changing_child_output(tmp_path, monkeypatch, caplog, capsys):
    import codecs
    import logging

    times = iter([0, 16, 16, 16, 17])
    monkeypatch.setattr(ENTRY.time, 'monotonic', lambda: next(times))

    class HeartbeatProcess:
        checks = 0

        def wait(self, timeout):
            self.checks += 1
            if self.checks == 1:
                raise ENTRY.subprocess.TimeoutExpired(['fixture'], timeout)
            return 0

    path = tmp_path / 'progress.log'
    path.write_text('子进程输出')
    with path.open('rb') as output, caplog.at_level(logging.INFO, logger=ENTRY.LOGGER.name):
        code = ENTRY.wait_with_progress(HeartbeatProcess(), output,
                                       codecs.getincrementaldecoder('utf-8')(),
                                       dict(name='http', command=['fixture'], timeout=60))
    assert code == 0
    assert '[quality] http: running, 16s elapsed' in caplog.text
    assert capsys.readouterr().out == '子进程输出'


@pytest.mark.parametrize('outcome', ['PASS', 'TIMEOUT', 'INTERRUPTED', 'keyboard', 'error', 'paused'])
def test_run_preserves_final_state_and_pause_markers(tmp_path, monkeypatch, outcome):
    monkeypatch.setattr(ENTRY, 'git_output', lambda *args: 'fixture-commit')
    monkeypatch.setattr(ENTRY, 'static_checks', lambda record: None)
    invoked = []
    work = tmp_path / 'work'
    record = tmp_path / 'record'
    if outcome == 'paused':
        marker = tmp_path / 'previous/service/worker-paused.txt'
        marker.parent.mkdir(parents=True)
        marker.write_text('inspect before retry')

    def execute(step, directory, environment):
        invoked.append(step['name'])
        if outcome == 'keyboard':
            raise KeyboardInterrupt()
        if outcome == 'error':
            raise OSError('fixture failure')
        code = dict(PASS=0, TIMEOUT=124, INTERRUPTED=130)[outcome]
        return dict(name=step['name'], state=outcome, returncode=code)

    monkeypatch.setattr(ENTRY, 'execute_step', execute)
    expected = dict(PASS=0, TIMEOUT=124, INTERRUPTED=130, keyboard=130, error=1, paused=1)[outcome]
    assert ENTRY.run(record, work) == expected
    status = json.loads((record / 'execution.json').read_text())
    assert status['state'] == 'finished' and status['returncode'] == expected
    assert status['scope'] == 'SUBMISSION_E2E'
    assert status['code_commit'] == 'fixture-commit'
    assert (work / 'service/worker-paused.txt').is_file() == (outcome in ('TIMEOUT', 'INTERRUPTED', 'keyboard'))
    assert len(invoked) == (2 if outcome == 'PASS' else 0 if outcome == 'paused' else 1)


def test_process_stop_escalates_and_reaps(monkeypatch):
    signals = []
    waits = []
    monkeypatch.setattr(ENTRY.os, 'killpg', lambda pid, signum: signals.append((pid, signum)))

    class StubbornProcess:
        pid = 12345

        @staticmethod
        def wait(timeout):
            waits.append(timeout)
            if len(waits) == 1:
                raise ENTRY.subprocess.TimeoutExpired(['fixture'], timeout)
            return -ENTRY.signal.SIGKILL

    ENTRY.stop_process(StubbornProcess())
    assert signals == [(12345, ENTRY.signal.SIGTERM), (12345, ENTRY.signal.SIGKILL)]
    assert waits == [10, 10]


@pytest.mark.parametrize('content', [None, '<testsuites><testcase><failure/></testcase></testsuites>'])
def test_junit_missing_or_failed_is_not_success(tmp_path, content):
    if content is not None:
        (tmp_path / 'result.xml').write_text(content)
    result = ENTRY.execute_step(dict(name='junit', command=[sys.executable, '-c', 'pass'], timeout=10,
                                    junit='result.xml'), tmp_path, dict(os.environ))
    assert result['state'] == 'FAIL' and result['returncode'] == 1


def test_process_interrupt_keeps_code_and_flushes_tail(tmp_path, monkeypatch, capsys):
    import codecs

    stopped = []
    monkeypatch.setattr(ENTRY, 'wait_with_progress', lambda *args: (_ for _ in ()).throw(KeyboardInterrupt()))
    monkeypatch.setattr(ENTRY, 'stop_process', lambda process: stopped.append(process))
    path = tmp_path / 'tail.log'
    path.write_bytes('最后一条日志'.encode('utf-8'))
    process = object()
    with path.open('rb') as output:
        result = ENTRY.process_outcome(process, output, codecs.getincrementaldecoder('utf-8')(), {})
    assert result == (130, 'INTERRUPTED')
    assert stopped == [process]
    assert capsys.readouterr().out == '最后一条日志'


@pytest.mark.parametrize('field,value', [
    ('evidence_kind', 'OTHER'), ('mode', 'independent'), ('selected_scenario_count', 0),
    ('observations', []), ('all_selected_observed', False), ('all_selected_observed', 1),
    ('mismatched', 1), ('matched', 0), ('observations', [dict(failures=['wrong score'])]),
    ('observations', [dict()]),
])
def test_http_evidence_rejects_each_invalid_condition(tmp_path, field, value):
    report = dict(evidence_kind='HTTP_OBSERVATIONS', mode='normal', selected_scenario_count=1,
                  all_selected_observed=True, matched=1, mismatched=0, observations=[dict(failures=[])])
    report[field] = value
    path = tmp_path / 'observations.json'
    path.write_text(json.dumps(report), encoding='utf-8')
    with pytest.raises(ValueError, match='HTTP scenarios are incomplete or have mismatches'):
        ENTRY.check_http_observations(path, 'normal')


@pytest.mark.parametrize('mode', ['independent', 'normal'])
def test_http_evidence_accepts_complete_report(tmp_path, mode):
    report = dict(evidence_kind='HTTP_OBSERVATIONS', mode=mode, selected_scenario_count=2,
                  all_selected_observed=True, matched=2, mismatched=0,
                  observations=[dict(failures=[]), dict(failures=[])])
    path = tmp_path / 'observations.json'
    path.write_text(json.dumps(report), encoding='utf-8')
    assert ENTRY.check_http_observations(path, mode) == dict(selected=2, matched=2, mismatched=0)


def test_logging_transport_preserves_chunks_and_flushes_decoder_tail(tmp_path, capsys):
    import codecs

    content = '任务 50%\r完成\n无换行'.encode('utf-8') + b'\xe4'
    path = tmp_path / 'child.log'
    path.write_bytes(content[:2])
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    with path.open('rb') as output:
        ENTRY.forward_output(output, decoder)
        assert capsys.readouterr().out == ''
        with path.open('ab') as writer:
            writer.write(content[2:])
        ENTRY.forward_output(output, decoder)
        assert capsys.readouterr().out == '任务 50%\r完成\n无换行'
        ENTRY.forward_output(output, decoder, final=True)
        assert capsys.readouterr().out == '\ufffd'
    assert path.read_bytes() == content


def test_failed_last_plan_preserves_whole_scenarios_and_excludes_passing_modes(tmp_path):
    """Focused runs forward exactly the six previously failing normal scenarios."""
    plan = ENTRY.steps(tmp_path / 'record', tmp_path / 'work', False, 'failed-last')
    assert len(plan) == 1
    step = plan[0]
    assert step['name'] == 'simulation-normal'
    assert step['scenarios'] == ENTRY.FAILED_LAST_SCENARIOS
    command = step['command']
    selected = [command[index + 1] for index, argument in enumerate(command) if argument == '--scenario']
    assert selected == list(ENTRY.FAILED_LAST_SCENARIOS)
    assert '--acknowledge-isolation' in command
    assert 'pytest' not in command


@pytest.mark.parametrize('defect', ['missing', 'duplicate', 'unexpected', 'valid'])
def test_focused_evidence_requires_exact_scenario_identities(tmp_path, defect):
    """Matching counts alone cannot hide duplicated or replaced focused scenarios."""
    codes = list(ENTRY.FAILED_LAST_SCENARIOS)
    if defect == 'missing':
        codes.pop()
    elif defect == 'duplicate':
        codes[-1] = codes[0]
    elif defect == 'unexpected':
        codes[-1] = 'OK-SOFTMAX'
    report = dict(evidence_kind='HTTP_OBSERVATIONS', mode='normal', selected_scenario_count=len(codes),
                  all_selected_observed=True, matched=len(codes), mismatched=0,
                  observations=[dict(scenario=code, failures=[]) for code in codes])
    path = tmp_path / 'observations.json'
    path.write_text(json.dumps(report), encoding='utf-8')
    if defect == 'valid':
        assert ENTRY.check_http_observations(path, 'normal', ENTRY.FAILED_LAST_SCENARIOS)['matched'] == 6
    else:
        with pytest.raises(ValueError, match='scenario identities'):
            ENTRY.check_http_observations(path, 'normal', ENTRY.FAILED_LAST_SCENARIOS)


def test_failed_last_run_records_partial_scope_and_baseline(tmp_path, monkeypatch):
    """Focused success is explicitly not evidence of a full regression."""
    monkeypatch.setattr(ENTRY, 'git_output', lambda *arguments: 'fixture-commit')
    monkeypatch.setattr(ENTRY, 'record_environment', lambda record: None)
    monkeypatch.setattr(ENTRY, 'static_checks', lambda record: None)
    invoked = []

    def execute(step, record, environment):
        """Record the selected step without claiming device execution."""
        invoked.append(step['name'])
        return dict(name=step['name'], state='PASS', returncode=0)

    monkeypatch.setattr(ENTRY, 'execute_step', execute)
    record = tmp_path / 'record'
    assert ENTRY.run(record, tmp_path / 'work', suite='failed-last') == 0
    status = json.loads((record / 'execution.json').read_text())
    assert invoked == ['simulation-normal']
    assert status['scope'] == 'SUBMISSION_E2E_FAILED_LAST'
    assert status['selection'] == dict(baseline_record=ENTRY.FAILED_LAST_RECORD,
                                       scenarios=list(ENTRY.FAILED_LAST_SCENARIOS),
                                       skipped_previously_matched=64, full_regression=False)


def test_failed_last_rejects_cpu_combination(tmp_path):
    """An explicitly focused device run must not silently become a CPU run."""
    result = ENTRY.subprocess.run(
        [sys.executable, str(ENTRY.ROOT / 'scripts/verify_codecheck_regression.py'),
         '--record', str(tmp_path / 'record'), '--work', str(tmp_path / 'work'),
         '--suite', 'failed-last', '--cpu-only'],
        cwd=ENTRY.ROOT, env=dict(os.environ, PYTHONPATH=str(ENTRY.ROOT / 'src')),
        capture_output=True, text=True, timeout=30)
    assert result.returncode == 2
    assert '--suite failed-last cannot be combined with --cpu-only' in result.stderr
    assert not (tmp_path / 'record').exists()
    assert not (tmp_path / 'work').exists()


@pytest.mark.parametrize('cause', ['paused', 'static', 'interrupted'])
def test_preflight_failure_logs_reason_and_keeps_pause_protection(tmp_path, monkeypatch, caplog, cause):
    """Preflight rejection must be visible without opening the published JSON file."""
    import logging

    monkeypatch.setattr(ENTRY, 'git_output', lambda *arguments: 'fixture-commit')
    monkeypatch.setattr(ENTRY, 'record_environment', lambda record: None)
    marker = tmp_path / 'previous/service/worker-paused.txt'
    invoked = []

    def inspect(record):
        """Supply a deterministic preflight failure without touching the device."""
        invoked.append(record)
        if cause == 'static':
            raise ValueError('fixture static verification failed')
        raise KeyboardInterrupt()

    monkeypatch.setattr(ENTRY, 'static_checks', inspect)
    if cause == 'paused':
        marker.parent.mkdir(parents=True)
        marker.write_text('inspect before retry\n')
    record = tmp_path / 'record'
    with caplog.at_level(logging.ERROR, logger=ENTRY.LOGGER.name):
        returncode = ENTRY.run(record, tmp_path / 'work', suite='failed-last')
    status = json.loads((record / 'execution.json').read_text())
    assert returncode == (130 if cause == 'interrupted' else 1)
    assert status['error'] in caplog.text
    assert str(record / 'execution.json') in caplog.text
    assert '[quality] stopped:' in caplog.text
    assert status['steps'] == [dict(name='simulation-normal', state='NOT_RUN')]
    if cause == 'paused':
        assert marker.read_text() == 'inspect before retry\n'
        assert not invoked
