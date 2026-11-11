#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""构建内容级提交 ZIP，并通过真实 HTTP 接口执行独立预期对照。"""

from __future__ import annotations

import hashlib
import json
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from urllib.parse import urlsplit
import logging

import argparse

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / 'src'))


from triton_eval.submission_service import TERMINAL
from triton_eval.submission_simulation import DEFAULT_CORPUS, DEFAULT_SOURCE, build_suite, check_outcome
from triton_eval.console import emit


def request(base, path, payload=None, *, zip_data=None, key=None):
    headers = {}
    if zip_data is not None:
        body = zip_data
        headers['Content-Type'] = 'application/zip'
    elif payload is not None:
        body = json.dumps(payload).encode()
        headers['Content-Type'] = 'application/json'
    else:
        body = None
    if key:
        headers['Idempotency-Key'] = key
    call = urllib.request.Request(base.rstrip('/') + path, data=body, headers=headers)
    with urllib.request.urlopen(call, timeout=120) as response:
        return json.load(response)


def verify(record: Path, work: Path, mode: str = 'normal', scenarios=()) -> int:
    from triton_eval.submission_http import make_server
    from triton_eval.submission_service import EvaluationService

    paused = sorted(work.parent.glob('*/service/worker-paused.txt'))
    legacy_pause = work.parent.parent / 'service/worker-paused.txt'
    if legacy_pause.exists():
        paused.append(legacy_pause)
    if paused:
        raise RuntimeError(f'存在暂停标记，必须先人工确认进程和设备健康：{paused[0]}')
    record.mkdir(parents=True, exist_ok=True)
    suite = work / 'suite'
    emit(f'[simulation] 构建 ZIP 提交场景：{suite}', flush=True)
    build_suite(suite)
    emit('[simulation] ZIP 场景构建完成，正在启动评测服务', flush=True)
    (record / 'expectations.json').write_bytes((suite / 'expectations.json').read_bytes())
    service = EvaluationService(work / 'service', DEFAULT_CORPUS, execute=True)
    try:
        server = make_server(service, 0)
        try:
            thread = threading.Thread(target=server.serve_forever)
            thread.start()
            try:
                service.start_worker()
                emit(f'联调接口：http://127.0.0.1:{server.server_port}；仅执行 {mode} 场景', flush=True)
                return main(['run', '--suite', str(suite),
                             '--url', f'http://127.0.0.1:{server.server_port}',
                             '--mode', mode, '--output', str(record / f'{mode}-observations.json'),
                             *scenario_arguments(scenarios)])
            finally:
                server.shutdown()
                thread.join()
        finally:
            server.server_close()
    finally:
        try:
            service.close()
        finally:
            pause = work / 'service/worker-paused.txt'
            if pause.exists():
                (record / pause.name).write_bytes(pause.read_bytes())


def main(argv=None) -> int:
    parser, args = _simulation_arguments(argv)

    if args.command == 'verify':
        if not args.acknowledge_isolation:
            parser.error('执行不可信提交前必须提供 --acknowledge-isolation')
        return verify(args.record, args.work, args.mode, args.scenario)
    if args.command == 'build':
        suite = build_suite(args.dest, args.source, args.corpus)
        emit(f"已生成 {len(suite['scenarios'])} 个提交包；仅含独立预期，不是真机认证结果")
        return 0
    if urlsplit(args.url).hostname not in {'127.0.0.1', 'localhost'}:
        parser.error('联调客户端只连接 loopback；跨主机使用 SSH 端口转发')
    if args.timeout <= 0:
        parser.error('--timeout 必须为正数')
    suite = json.loads((args.suite / 'expectations.json').read_text(encoding='utf-8'))
    selected_scenarios = select_scenarios(suite, args.mode, args.scenario)
    run_id = str(time.time_ns())
    report = dict(schema_version=1, mode=args.mode, evidence_kind='HTTP_OBSERVATIONS',
                  expectations_sha256=hashlib.sha256((args.suite / 'expectations.json').read_bytes()).hexdigest(),
                  profile=args.profile, selected_scenario_count=len(selected_scenarios),
                  requested_scenarios=args.scenario,
                  automatic_hardware_certification=False, observations=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def persist():
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')

    if _check_simulation_catalog(args, suite, selected_scenarios, report, persist):
        return 1

    for position, scenario in enumerate(selected_scenarios, 1):
        emit(f"[simulation] {position}/{len(selected_scenarios)} {scenario['code']}: "
             f"开始提交 ZIP（{scenario['description']}）", flush=True)
        observation = dict(scenario=scenario['code'], failures=[], submission=None, task=None, results=None)
        report['observations'].append(observation)
        try:
            _observe_scenario(args, scenario, run_id, report, persist, observation)
        except (OSError, ValueError, RuntimeError) as exc:
            observation['failures'].append(f'{type(exc).__name__}: {exc}')
        persist()
        emit(f"{scenario['code']}: {'MISMATCH' if observation['failures'] else 'MATCH'}", flush=True)
        if observation.get('task') and observation['task'].get('worker_paused'):
            report['stopped_reason'] = 'WORKER_PAUSED'
            persist()
            break
    _summarize_simulation(report, selected_scenarios, args)

    persist()
    return 0 if report['all_selected_observed'] and report['mismatched'] == 0 else 1




def scenario_arguments(scenarios):
    """Forward exact scenario names without interpreting them as patterns."""
    arguments = []
    for code in scenarios:
        arguments.extend(['--scenario', code])
    return arguments


def select_scenarios(suite, mode, requested):
    """Select whole scenarios, rejecting duplicate, unknown and out-of-mode names."""
    eligible = []
    for scenario in suite['scenarios']:
        selected = scenario['stage'] == 'content' if mode == 'content' else scenario['tier'] == mode
        if selected:
            eligible.append(scenario)
    if not requested:
        return eligible
    if len(set(requested)) != len(requested):
        raise ValueError('Duplicate requested scenario codes')
    available = {scenario['code'] for scenario in eligible}
    unknown = set(requested) - available
    if unknown:
        raise ValueError('Unknown or out-of-mode scenario codes: ' + ', '.join(sorted(unknown)))
    return [scenario for scenario in eligible if scenario['code'] in requested]


def _simulation_arguments(argv):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    build = commands.add_parser('build')
    build.add_argument('--dest', type=Path, required=True)
    build.add_argument('--source', type=Path, default=DEFAULT_SOURCE)
    build.add_argument('--corpus', type=Path, default=DEFAULT_CORPUS)
    run = commands.add_parser('run')
    run.add_argument('--suite', type=Path, required=True)
    run.add_argument('--url', default='http://127.0.0.1:8765')
    run.add_argument('--output', type=Path, required=True)
    run.add_argument(
        '--mode',
        choices=[
            'content',
            'normal',
            'independent',
            'full',
            'slow',
            'disruptive'],
        default='content')
    run.add_argument('--timeout', type=int, default=14400)
    run.add_argument('--profile', choices=['smoke', 'standard'], default='smoke')
    verify_command = commands.add_parser('verify')
    verify_command.add_argument('--record', type=Path, required=True)
    verify_command.add_argument('--work', type=Path, required=True)
    verify_command.add_argument('--acknowledge-isolation', action='store_true')
    verify_command.add_argument('--mode', choices=['normal', 'independent'], default='normal')
    for command in (run, verify_command):
        command.add_argument('--scenario', action='append', default=[],
                             help='Exact scenario code; repeat to select a subset within the mode')
    args = parser.parse_args(argv)
    return parser, args


def _observe_scenario(args, scenario, run_id, report, persist, observation):
    package = (args.suite / scenario['package']).read_bytes()
    if hashlib.sha256(package).hexdigest() != scenario['sha256']:
        raise ValueError('提交包与预期清单的指纹不一致')
    submission = request(args.url, '/api/v1/submissions', zip_data=package,
                         key=f'{run_id}-{scenario["code"]}-upload')
    observation['submission'] = submission
    emit(f"[simulation] {scenario['code']}: ZIP 解析结果 {submission['status']}; "
         f"submission={submission['id']}", flush=True)
    _check_upload_replay(args, scenario, run_id, report, observation, package, submission)

    task = None
    result = None
    if scenario['stage'] != 'content' and submission['status'] == 'READY':
        task, result = _observe_task(args, scenario, run_id, report, persist, observation, submission)
    observation['task'] = task
    observation['results'] = result
    if result is not None:
        emit(f"[simulation] {scenario['code']}: 评测结果 {task['status']}; "
             f"score={result.get('score')}", flush=True)
    observation['failures'] += check_outcome(scenario, submission, task, result)


def _observe_task(args, scenario, run_id, report, persist, observation, submission):
    task_request = dict(submission_id=submission['id'],
                        scope='full' if scenario['tier'] == 'full' else 'operators', profile=args.profile)
    if scenario.get('request_operators'):
        task_request['operators'] = scenario['request_operators']
    task = request(args.url, '/api/v1/evaluations', task_request,
                   key=f'{run_id}-{scenario["code"]}-task')
    observation['task'] = task
    emit(f"[simulation] {scenario['code']}: 评测任务已创建 task={task['id']}", flush=True)
    if scenario.get('origin') == 'independent':
        replay = request(args.url, '/api/v1/evaluations', task_request,
                         key=f'{run_id}-{scenario["code"]}-task')
        observation['task_replay_id'] = replay['id']
        if replay['id'] != task['id']:
            observation['failures'].append('相同幂等键的任务创建没有复用任务记录')
        if any((previous.get('task') or {}).get('id') == task['id'] for previous in report['observations'][:-1]):
            observation['failures'].append('不同评测请求被错误合并为同一任务')
    persist()
    task = wait_simulation_task(args, task)

    result = request(args.url, f'/api/v1/evaluations/{task["id"]}/results')
    observation['results'] = result
    if scenario.get('origin') == 'independent':
        events = request(args.url, f'/api/v1/evaluations/{task["id"]}/events')
        observation['events'] = events['items']
        states = [event['status'] for event in events['items']]
        if states != ['QUEUED', 'RUNNING', 'SCORING', 'REPORTING', 'COMPLETED']:
            observation['failures'].append(f'合法提交状态事件链不完整：{states}')
    _download_simulation_artifacts(args, scenario, task, observation)

    return task, result


def wait_simulation_task(args, task):
    started = time.monotonic()
    deadline = started + args.timeout
    last_status = None
    heartbeat = started
    while task['status'] not in TERMINAL:
        now = time.monotonic()
        if task['status'] != last_status or now - heartbeat >= 15:
            emit(f"[simulation] task={task['id']} status={task['status']} "
                 f"elapsed={int(now - started)}s", flush=True)
            last_status = task['status']
            heartbeat = now
        if not task['worker_enabled'] or (task['worker_paused'] and task['status'] == 'QUEUED'):
            raise RuntimeError('Worker 未启用或已暂停；该任务不算验证成功')
        if time.monotonic() >= deadline:
            request(args.url, f'/api/v1/evaluations/{task["id"]}/cancel', {})
            raise TimeoutError('客户端等待超时，已请求取消；须确认终止与设备健康')
        time.sleep(0.5)
        task = request(args.url, f'/api/v1/evaluations/{task["id"]}')
    emit(f"[simulation] task={task['id']} status={task['status']} "
         f"elapsed={int(time.monotonic() - started)}s", flush=True)
    return task


def _download_simulation_artifacts(args, scenario, task, observation):
    artifacts = request(args.url, f'/api/v1/evaluations/{task["id"]}/artifacts')
    observation['artifacts'] = artifacts['items']
    available = {artifact['name'] for artifact in artifacts['items']}
    if task['status'] == 'COMPLETED' and not {'report.md', 'report.html', 'results.jsonl'} <= available:
        observation['failures'].append('完整链路缺少报告或原始结果产物')
    for artifact in artifacts['items']:
        call = urllib.request.Request(args.url.rstrip('/') + f'/api/v1/evaluations/{task["id"]}'
                                      + f'/artifacts/{artifact["name"]}')
        with urllib.request.urlopen(call, timeout=120) as response:
            data = response.read()
        if hashlib.sha256(data).hexdigest() != artifact['sha256']:
            observation['failures'].append(f'下载产物指纹不符：{artifact["name"]}')
        destination = args.output.parent / (args.output.stem + '-artifacts') / scenario['code']
        destination.mkdir(parents=True, exist_ok=True)
        (destination / artifact['name']).write_bytes(data)
    emit(f"[simulation] {scenario['code']}: 已下载 {len(artifacts['items'])} 项报告和结果产物", flush=True)


def _check_upload_replay(args, scenario, run_id, report, observation, package, submission):
    if scenario.get('origin') == 'independent':
        replay = request(args.url, '/api/v1/submissions', zip_data=package,
                         key=f'{run_id}-{scenario["code"]}-upload')
        observation['upload_replay_id'] = replay['id']
        if replay['id'] != submission['id']:
            observation['failures'].append('相同幂等键的上传没有复用提交记录')
        if _seen_simulation_id(report, 'submission', submission['id']):
            observation['failures'].append('不同上传请求被错误合并为同一提交')
        if submission.get('declared_submit_id') != (
                scenario.get('declared_submit_id') or f'simulation-{scenario["code"].lower()}'):
            observation['failures'].append('合法提交标识没有原样保留')


def _summarize_simulation(report, selected_scenarios, args):
    report['all_selected_observed'] = len(report['observations']) == len(selected_scenarios)
    report['matched'] = sum(not item['failures'] for item in report['observations'])
    report['mismatched'] = sum(bool(item['failures']) for item in report['observations'])
    if args.mode == 'independent':
        report['acceptance'] = dict(
            scope='INDEPENDENT_IMPLEMENTATIONS_ON_FROZEN_CASES',
            all_legal_submissions_accepted=(report['all_selected_observed']
                                            and _all_simulation_statuses(report, 'submission', 'READY')),
            all_tasks_completed=(report['all_selected_observed']
                                 and _all_simulation_statuses(report, 'task', 'COMPLETED')),
            all_expectations_matched=report['all_selected_observed'] and report['mismatched'] == 0,
            universal_legal_submission_support='NOT_PROVEN',
            out_of_domain_inputs_tested=False)




def _seen_simulation_id(report, kind, identifier):
    for previous in report['observations'][:-1]:
        if (previous.get(kind) or {}).get('id') == identifier:
            return True
    return False


def _all_simulation_statuses(report, kind, status):
    for item in report['observations']:
        if (item.get(kind) or {}).get('status') != status:
            return False
    return True




def _check_simulation_catalog(args, suite, selected_scenarios, report, persist):
    try:
        if not selected_scenarios:
            raise ValueError('选定档位没有任何场景，不能判为验证成功')
        catalog = request(args.url, '/api/v1/catalog')
        report['server_catalog'] = catalog
        if catalog['manifest_hash'] != suite['manifest_hash']:
            raise RuntimeError('服务绑定版本与独立预期清单不一致，未提交任何任务')
        if args.mode != 'content' and (not catalog['worker_enabled'] or catalog['worker_paused']):
            raise RuntimeError('Worker 未启用或已暂停，未提交任何任务')
    except (OSError, ValueError, RuntimeError) as exc:
        report.update(stopped_reason=str(exc), all_selected_observed=False, matched=0, mismatched=0)
        persist()
        emit(report['stopped_reason'], file=sys.stderr)
        return 1

    return 0


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
