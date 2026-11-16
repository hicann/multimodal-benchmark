#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Verify ZIP submission, parsing, device evaluation, scoring and report downloads."""

from __future__ import annotations

import argparse
import codecs
import hashlib
import importlib.metadata
import json
import logging
import os
import platform
import shutil
import signal
import subprocess
import sys
import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from pathlib import Path

from triton_eval.console import emit

LOGGER = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
FAILED_LAST_SCENARIOS = ('OK-EXP', 'OK-ADD', 'OK-GROUPED', 'OK-MULTI', 'MIXED-RUNTIME', 'EXEC-PARTIAL')
FAILED_LAST_RECORD = 'codecheck-20261110-073225-597670'
OAT_PATHS = (
    'spec/feature_atoms.yaml', 'spec/spec.yaml', 'spec/tolerance.yaml',
    'examples/adversarial-submissions/ADV-E1-path-escape/manifest.yaml',
    'examples/adversarial-submissions/ADV-E2-empty-operators/manifest.yaml',
    'examples/ascend-ops-submission/manifest.yaml',
    'examples/invalid-submissions/SUB-LOD-002/manifest.yaml',
    'examples/adversarial-submissions/ADV-C1-fullhouse/manifest.yaml',
    'src/triton_eval/templates/report.html.j2', 'src/triton_eval/templates/report.md.j2',
)


def write_json(path, value):
    temporary = path.with_suffix('.tmp')
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def git_output(*arguments):
    executable = shutil.which('git')
    if executable is None:
        raise FileNotFoundError('Git executable was not found in PATH')
    executable = str(Path(executable).resolve(strict=True))
    return subprocess.check_output([executable, '-C', str(ROOT), *arguments], text=True).strip()


def static_checks(record):
    import yaml
    from jinja2 import Environment
    from triton_eval.corpus_assets import verify_assets
    from triton_eval.fixture_sources import LAYOUT, fixture_files, registry

    repository = Path(git_output('rev-parse', '--show-toplevel'))
    tree = ET.parse(repository / 'OAT.xml')
    policy = tree.find('.//filefilter[@name="binaryFileTypePolicyFilter"]')
    expected = {str((ROOT / relative).relative_to(repository)) for relative in OAT_PATHS}
    actual = [] if policy is None else list(policy)
    if {item.get('name') for item in actual} != expected or len(actual) != len(expected):
        raise ValueError('OAT binary policy must contain exactly the ten reviewed file paths')
    for item in actual:
        if item.get('type') != 'filepath' or not item.get('desc'):
            raise ValueError('OAT policy requires a precise filepath and an explanation')
    hashes = {}
    for relative in OAT_PATHS:
        content = (ROOT / relative).read_bytes()
        text = content.decode('utf-8')
        if '\x00' in text:
            raise ValueError('NUL byte in text configuration: ' + relative)
        if relative.endswith('.yaml'):
            yaml.safe_load(text)
        else:
            Environment().parse(text)
        hashes[relative] = hashlib.sha256(content).hexdigest()
    _, layout = registry(ROOT)
    fixture_files(tuple(layout['files']), ROOT)
    corpora = []
    for index in sorted((ROOT / 'corpus').glob('*/evaluation-assets.json')):
        active = json.loads(index.read_text())
        verify_assets(index.parent, active['manifest_hash'])
        corpora.append(dict(corpus=index.parent.name, assets_hash=active['assets_hash'],
                            manifest_hash=active['manifest_hash']))
    for relative in ('src/triton_eval/safeexec.py', 'src/triton_eval/operator_registry.py', LAYOUT):
        hashes[relative] = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
    hashes['@repository/OAT.xml'] = hashlib.sha256((repository / 'OAT.xml').read_bytes()).hexdigest()
    write_json(record / 'static-checks.json', dict(text_files=hashes, corpora=corpora,
                                                materialized_files=len(layout['files']),
                                                codecheck_remote_rescan='NOT_RUN'))


def junit_counts(path):
    cases = list(ET.parse(path).getroot().iter('testcase'))
    return dict(tests=len(cases), skipped=sum(case.find('skipped') is not None for case in cases),
                failures=sum(case.find('failure') is not None or case.find('error') is not None
                             for case in cases))


def forward_output(output, decoder, final=False):
    # The logging-backed console preserves partial text without adding prefixes or newlines.
    text = decoder.decode(output.read(), final=final)
    if text:
        emit(text, end='', flush=True)


def report_heartbeat(step, elapsed, heartbeat):
    if time.monotonic() - heartbeat < 15:
        return heartbeat
    LOGGER.info('[quality] %s: running, %ss elapsed', step['name'], int(elapsed))
    return time.monotonic()


def wait_with_progress(process, output, decoder, step):
    started = time.monotonic()
    heartbeat = started
    while True:
        forward_output(output, decoder)
        elapsed = time.monotonic() - started
        if elapsed >= step['timeout']:
            raise subprocess.TimeoutExpired(step['command'], step['timeout'])
        try:
            return process.wait(timeout=min(0.25, step['timeout'] - elapsed))
        except subprocess.TimeoutExpired:
            heartbeat = report_heartbeat(step, elapsed, heartbeat)


def check_http_observations(path, mode, expected_scenarios=()):
    report = json.loads(path.read_text(encoding='utf-8'))
    observations = report.get('observations', [])
    selected = report.get('selected_scenario_count', 0)
    if report.get('evidence_kind') != 'HTTP_OBSERVATIONS' or report.get('mode') != mode:
        raise ValueError('HTTP scenarios are incomplete or have mismatches: ' + str(path))
    if selected <= 0 or len(observations) != selected or report.get('all_selected_observed') is not True:
        raise ValueError('HTTP scenarios are incomplete or have mismatches: ' + str(path))
    if report.get('mismatched') != 0 or report.get('matched') != selected:
        raise ValueError('HTTP scenarios are incomplete or have mismatches: ' + str(path))
    if any(item.get('failures') != [] for item in observations):
        raise ValueError('HTTP scenarios are incomplete or have mismatches: ' + str(path))
    if expected_scenarios:
        actual = [item.get('scenario') for item in observations]
        if len(actual) != len(expected_scenarios) or set(actual) != set(expected_scenarios):
            raise ValueError('HTTP scenario identities do not match the requested subset: ' + str(path))
    return dict(selected=selected, matched=report['matched'], mismatched=report['mismatched'])


def stop_process(process):
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def process_outcome(process, output, decoder, step):
    try:
        returncode = wait_with_progress(process, output, decoder, step)
        return returncode, 'PASS' if returncode == 0 else 'FAIL'
    except (subprocess.TimeoutExpired, KeyboardInterrupt) as error:
        stop_process(process)
        if isinstance(error, subprocess.TimeoutExpired):
            return 124, 'TIMEOUT'
        return 130, 'INTERRUPTED'
    finally:
        forward_output(output, decoder, final=True)


def check_junit_result(step, record, result):
    if not step.get('junit'):
        return
    path = record / step['junit']
    if not path.is_file():
        result.update(state='FAIL', returncode=1, reason='missing JUnit results')
        return
    counts = junit_counts(path)
    result['junit'] = counts
    if counts['tests'] == 0 or counts['failures']:
        result.update(state='FAIL', returncode=1)
    if step.get('required_cases') is not None and (
            counts['tests'] != step['required_cases'] or counts['skipped']):
        result.update(state='FAIL', returncode=1, reason='all required cases must execute without skips')


def check_step_evidence(step, record, result):
    check_junit_result(step, record, result)
    if not step.get('observations') or result['state'] != 'PASS':
        return
    try:
        result['http'] = check_http_observations(
            record / step['observations'], step['mode'], step.get('scenarios', ()))
    except (OSError, ValueError, TypeError, KeyError) as error:
        result.update(state='FAIL', returncode=1, reason=str(error))


def execute_step(step, record, environment):
    log_path = record / (step['name'] + '.log')
    LOGGER.info('[quality] starting %s; log: %s', step['name'], log_path)
    started = time.monotonic()
    decoder = codecs.getincrementaldecoder('utf-8')(errors='replace')
    with log_path.open('wb') as log, log_path.open('rb') as output:
        with subprocess.Popen(step['command'], cwd=ROOT, env=environment, stdout=log,
                              stderr=subprocess.STDOUT, start_new_session=True) as process:
            returncode, state = process_outcome(process, output, decoder, step)
    result = dict(name=step['name'], state=state, returncode=returncode,
                  seconds=round(time.monotonic() - started, 3), command=step['command'])
    check_step_evidence(step, record, result)
    LOGGER.info('[quality] %s: %s', step['name'], result['state'])
    return result


def steps(record, work, cpu_only, suite='full'):
    python = sys.executable
    cpu = dict(name='cpu', command=[python, '-m', 'pytest', '-q', '-p', 'no:cacheprovider',
               '-m', 'not npu and not triton', '--junitxml=' + str(record / 'cpu.xml'),
               '--basetemp=' + str(work / 'pytest-cpu')], timeout=1800, junit='cpu.xml')
    if cpu_only:
        return [cpu]
    plan = []
    modes = ('normal',) if suite == 'failed-last' else ('independent', 'normal')
    for mode in modes:
        scenarios = FAILED_LAST_SCENARIOS if suite == 'failed-last' else ()
        command = [python, str(ROOT / 'scripts/submission_simulate.py'), 'verify',
                   '--record', str(record / mode), '--work', str(work / mode),
                   '--mode', mode, '--acknowledge-isolation']
        for code in scenarios:
            command.extend(['--scenario', code])
        plan.append(dict(name='simulation-' + mode, command=command, timeout=7200,
                         mode=mode, observations=mode + '/' + mode + '-observations.json',
                         scenarios=scenarios))
    return plan


def record_environment(record):
    packages = {}
    for name in ('torch', 'torch-npu', 'triton', 'triton-ascend', 'pytest', 'pydantic'):
        try:
            packages[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            packages[name] = None
    write_json(record / 'environment.json', dict(python=sys.version, executable=sys.executable,
               platform=platform.platform(), packages=packages,
               visible_devices=os.environ.get('ASCEND_VISIBLE_DEVICES')))


def check_previous_run(work):
    markers = list(work.parent.glob('*/service/worker-paused.txt'))
    markers += list(work.parent.glob('*/*/service/worker-paused.txt'))
    markers += list((ROOT / 'runs/submission-simulation/work').glob('*/service/worker-paused.txt'))
    if markers:
        raise RuntimeError('Previous run requires process and device inspection: ' + str(markers[0]))


def mark_interrupted(work):
    marker = work / 'service/worker-paused.txt'
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text('Regression interrupted; inspect worker processes and device before retrying.\n')


def execute_plan(plan, record, work, status):
    environment = dict(os.environ, PYTHONDONTWRITEBYTECODE='1', PYTHONUNBUFFERED='1')
    environment['PYTHONPATH'] = str(ROOT / 'src') + os.pathsep + environment.get('PYTHONPATH', '')
    for index, step in enumerate(plan):
        status['steps'][index].update(state='RUNNING', command=step['command'])
        write_json(record / 'execution.json', status)
        result = execute_step(step, record, environment)
        status['steps'][index] = result
        status['returncode'] = result['returncode']
        write_json(record / 'execution.json', status)
        if result['state'] == 'PASS':
            continue
        if result['state'] in ('TIMEOUT', 'INTERRUPTED'):
            mark_interrupted(work)
        break


def run(record, work, cpu_only=False, suite='full'):
    record, work = record.resolve(), work.resolve()
    if record.exists() or work.exists():
        raise ValueError('record and work must be new directories; existing evidence is never overwritten')
    record.mkdir(parents=True)
    work.mkdir(parents=True)
    commit = git_output('rev-parse', 'HEAD')
    record_environment(record)
    plan = steps(record, work, cpu_only, suite)
    status = dict(kind='codecheck-regression', code_commit=commit, scope='CPU_ONLY' if cpu_only else 'SUBMISSION_E2E',
                  started_at=datetime.now(timezone.utc).isoformat(), state='running', returncode=1,
                  codecheck_remote_rescan='NOT_RUN',
                  steps=[dict(name=step['name'], state='NOT_RUN') for step in plan])
    if not cpu_only and suite == 'failed-last':
        status.update(scope='SUBMISSION_E2E_FAILED_LAST', selection=dict(
            baseline_record=FAILED_LAST_RECORD, scenarios=list(FAILED_LAST_SCENARIOS),
            skipped_previously_matched=64, full_regression=False))
    write_json(record / 'execution.json', status)
    try:
        if not cpu_only:
            check_previous_run(work)
        LOGGER.info('[quality] checking text files and frozen assets')
        static_checks(record)
        LOGGER.info('[quality] static checks: PASS')
        execute_plan(plan, record, work, status)
    except KeyboardInterrupt:
        status.update(returncode=130, error='Regression interrupted')
        LOGGER.error('[quality] stopped: %s; details: %s', status['error'], record / 'execution.json')
        if not cpu_only:
            mark_interrupted(work)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as error:
        status.update(returncode=1, error=str(error))
        LOGGER.error('[quality] stopped: %s; details: %s', status['error'], record / 'execution.json')
    finally:
        status.update(state='finished', finished_at=datetime.now(timezone.utc).isoformat())
        write_json(record / 'execution.json', status)
    return status['returncode']


def main(argv=None):
    logging.basicConfig(level=logging.INFO, format='%(message)s', stream=sys.stdout)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--record', type=Path, required=True)
    parser.add_argument('--work', type=Path, required=True)
    parser.add_argument('--cpu-only', action='store_true')
    parser.add_argument('--suite', choices=['full', 'failed-last'], default='full')
    parser.add_argument('--acknowledge-isolation', action='store_true')
    args = parser.parse_args(argv)
    if args.cpu_only and args.suite == 'failed-last':
        parser.error('--suite failed-last cannot be combined with --cpu-only')
    if not args.cpu_only and not args.acknowledge_isolation:
        parser.error('Real device submission tests require --acknowledge-isolation')
    return run(args.record, args.work, args.cpu_only, args.suite)


def interrupt_run(signum, frame):
    raise KeyboardInterrupt('Regression received signal ' + str(signum))


if __name__ == '__main__':
    signal.signal(signal.SIGTERM, interrupt_run)
    signal.signal(signal.SIGHUP, interrupt_run)
    raise SystemExit(main())
