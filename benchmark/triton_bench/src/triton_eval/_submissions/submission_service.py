# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Loopback integration service: persistent jobs around the existing evaluator.

This is orchestration, not a security sandbox or a second scoring implementation.
User code is only imported by the existing evaluation subprocess on an isolated host.
"""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
import signal
import sqlite3
import subprocess
import sys
import threading
import time
import traceback
import uuid
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path

from .corpusmod import load_manifest, verify_corpus_files
from .models import CaseStatus, RunScoreRecord, is_counted
from .package import PackageError, ingest_zip
from .store import Store
from .submission import discover
from .validator import ValidationResult, validate_static

ACTIVE = {'RUNNING', 'SCORING', 'REPORTING'}
TERMINAL = {'COMPLETED', 'REJECTED', 'FAILED', 'CANCELLED', 'INTERRUPTED'}


CHILD_CONTROL_ENV = {
    'TRITON_EVAL_CHILD_CMD', 'TRITON_EVAL_CASE_TIMEOUT_S', 'TRITON_EVAL_IDLE_MARGIN_S', 'TRITON_EVAL_GRACE_S',
}


def _case_timeout_budget(cases, case_codes):
    selected = set(case_codes)
    return sum(case.timeout_s + 180 for case in cases if case.code in selected)


def _invalid_artifact_path(path, directory):
    if (path is None):
        return True
    if (not path.is_file()):
        return True
    if (path.is_symlink()):
        return True
    return not path.resolve().is_relative_to(directory)


def _invalid_requested_operators_part_2(operators):
    if (not isinstance(operators, list)):
        return True
    if (not operators):
        return True
    if (any(not isinstance(name, str) for name in operators)):
        return True
    return bool(len(operators) != len(set(operators)))


def _invalid_requested_operators(operators):
    if not (operators is not None):
        return False
    return bool(_invalid_requested_operators_part_2(operators))


class ServiceError(Exception):
    def __init__(self, code: str, detail: str, http_status: int = 400):
        self.code = code
        self.detail = detail
        self.http_status = http_status
        super().__init__(detail)


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class EvaluationService:
    def __init__(self, root: Path, corpus: Path, *, execute: bool = False):
        self.root = Path(root).resolve()
        self.corpus = Path(corpus).resolve()
        self.manifest = load_manifest(self.corpus / 'manifest.json')
        verify_corpus_files(self.corpus, self.manifest)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.process_lock = (self.root / 'service.lock').open('a')
        try:
            fcntl.flock(self.process_lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.process_lock.close()
            raise ServiceError('SERVICE-IN-USE', '该工作目录已有服务进程', 409) from None
        self.database = self.root / 'service.sqlite'
        with self._db() as connection:
            connection.executescript('''
                CREATE TABLE IF NOT EXISTS objects (
                    kind TEXT NOT NULL, id TEXT PRIMARY KEY, payload TEXT NOT NULL,
                    idem TEXT, digest TEXT NOT NULL, UNIQUE(kind, idem)
                );
                CREATE TABLE IF NOT EXISTS events (
                    seq INTEGER PRIMARY KEY AUTOINCREMENT, task_id TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
            ''')
        self.execute = execute
        self.stopping = threading.Event()
        self.thread = None
        self._recover()

    @staticmethod
    def _terminate(process: subprocess.Popen) -> None:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            pass
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait()

    @staticmethod
    def _event(task: dict, connection: sqlite3.Connection) -> None:
        event = dict(time=_now(), status=task['status'], error=task.get('error'),
                     cancel_requested=task['cancel_requested'])
        connection.execute('INSERT INTO events(task_id,payload) VALUES (?,?)',
                           (task['id'], json.dumps(event)))

    def close(self) -> None:
        self.stopping.set()
        if self.thread is not None:
            self.thread.join()
        fcntl.flock(self.process_lock, fcntl.LOCK_UN)
        self.process_lock.close()

    def upload(self, data: bytes, *, key: str | None = None) -> dict:
        digest = hashlib.sha256(data).hexdigest()
        with self.lock:
            existing = self._existing('submission', key, digest)
            if existing:
                return existing
            identifier = uuid.uuid4().hex
            directory = self.root / 'submissions' / identifier
            directory.mkdir(parents=True)
            package = directory / 'submission.zip'
            package.write_bytes(data)
            errors = []
            names = []
            declared_id = None
            stage = 'content'
            try:
                ingested = ingest_zip(package, directory / 'ingest')
                validation = ValidationResult(identifier)
                validate_static(ingested.dest, validation)
                errors = [asdict(error) for error in validation.errors]
                if not errors:
                    submission = discover(ingested.dest)
                    declared_id = submission.submit_id
                    names = submission.operator_names()
                    self._check_declared_operators(names, errors)
            except PackageError as exc:
                stage = 'package'
                errors = [dict(rule_id=exc.rule_id, stage=stage, operator='-', detail=exc.detail)]
            value = dict(id=identifier, package_sha256=digest, created_at=_now(),
                         declared_submit_id=declared_id, operators=names,
                         status='REJECTED' if errors else 'READY',
                         stage=stage, errors=errors)
            return self._insert('submission', value, key, digest)

    def create_task(self, request: dict, *, key: str | None = None) -> dict:
        if not isinstance(request, dict) or set(request) - {'submission_id', 'scope', 'operators', 'profile'}:
            raise ServiceError('REQUEST-SCHEMA', '只接受 submission_id/scope/operators/profile')
        identifier = request.get('submission_id')
        scope = request.get('scope', 'operators')
        profile = request.get('profile', 'smoke')
        operators = request.get('operators')
        if not isinstance(identifier, str) or scope not in (
                'operators', 'full') or profile not in ('smoke', 'standard'):
            raise ServiceError('REQUEST-SCHEMA', '提交 ID、范围或运行档位非法')
        if _invalid_requested_operators(operators):
            raise ServiceError('REQUEST-SCHEMA', 'operators 必须是非空、不重复的字符串列表')
        if scope == 'full' and operators is not None:
            raise ServiceError('REQUEST-SCOPE', '全量评测不能缩小算子范围')
        with self.lock:
            submission = self._get('submission', identifier)
            if submission['status'] != 'READY':
                raise ServiceError('SUBMISSION-REJECTED', '提交未通过内容检查', 409)
            names = ({operator.name for operator in self.manifest.operators}
                     if scope == 'full' else set(operators or submission['operators']))
            if scope == 'operators' and not names <= set(submission['operators']):
                raise ServiceError('REQUEST-SCOPE', '专项范围必须来自已声明算子')
            cases = [case.code for case in self.manifest.cases if case.operator in names]
            if not cases:
                raise ServiceError('REQUEST-SCOPE', '评测范围内没有冻结用例')
            normalized = dict(submission_id=identifier, scope=scope, operators=sorted(names),
                              profile=profile, manifest_hash=self.manifest.manifest_hash)
            digest = hashlib.sha256(json.dumps(normalized, sort_keys=True).encode()).hexdigest()
            existing = self._existing('task', key, digest)
            if existing:
                return self.task(existing['id'])
            task = dict(id=uuid.uuid4().hex, **normalized, status='QUEUED',
                        corpus_version=self.manifest.version,
                        package_sha256=submission['package_sha256'],
                        created_at=_now(), updated_at=_now(), planned_cases=len(cases),
                        case_codes=cases, cancel_requested=False, error=None)
            self._insert('task', task, key, digest)
            return self.task(task['id'])

    def submission(self, identifier: str) -> dict:
        return self._get('submission', identifier)

    def catalog(self) -> dict:
        from .operator_registry import by_name

        counts = Counter(case.operator for case in self.manifest.cases)
        operators = []
        for operator in self.manifest.operators:
            spec = by_name(operator.name)
            operators.append(dict(name=operator.name, weight=operator.weight,
                                  case_count=counts[operator.name],
                                  entry=spec.entry if spec else None,
                                  params=spec.params if spec else [],
                                  signature=spec.signature if spec else None))
        return dict(corpus_version=self.manifest.version, manifest_hash=self.manifest.manifest_hash,
                    case_count=len(self.manifest.cases), operator_count=len(operators),
                    operators=operators, profiles=['smoke', 'standard'], scopes=['operators', 'full'],
                    worker_enabled=self.execute, worker_paused=(self.root / 'worker-paused.txt').exists(),
                    device_health='NOT_PROBED', full_suite_accepted=False)

    def task(self, identifier: str) -> dict:
        task = self._get('task', identifier)
        records = self._engine(identifier)['cases']
        codes = set(task['case_codes'])
        completed = len({record['case_code'] for record in records if record['case_code'] in codes})
        return {**task, 'completed_cases': completed,
                'progress': completed / task['planned_cases'],
                'worker_enabled': self.execute,
                'worker_paused': (self.root / 'worker-paused.txt').exists()}

    def results(self, identifier: str) -> dict:
        task = self.task(identifier)
        engine = self._engine(identifier)
        records = engine['cases']
        counted = sum(is_counted(CaseStatus(record['status'])) for record in records)
        score = engine['scores'][-1] if engine['scores'] and counted and task['status'] == 'COMPLETED' else None
        if score is not None:
            score = dict(score)
            score.pop('id', None)
            for group in ('by_category', 'by_level'):
                score[group] = json.loads(score.pop(group + '_json', None) or 'null')
            score = RunScoreRecord.model_validate(score).model_dump(mode='json')
        for record in records:
            if record['log_path']:
                record['log_path'] = f'case-{record["case_code"]}.log'
        return dict(task_id=identifier, status=task['status'], scope=task['scope'],
                    error=task['error'], engine_error=engine.get('error'),
                    corpus_version=task['corpus_version'], manifest_hash=task['manifest_hash'],
                    package_sha256=task['package_sha256'],
                    declared_operators=self.submission(task['submission_id'])['operators'],
                    counted_operators=sorted(_counted_operator_names(records)),
                    scope_operators=task['operators'],
                    planned_cases=task['planned_cases'], completed_cases=task['completed_cases'],
                    counted_cases=counted, status_counts=dict(Counter(record['status'] for record in records)),
                    score=score, score_reason=(None if score else 'TASK_NOT_COMPLETED'
                                               if task['status'] != 'COMPLETED' else 'NO_COUNTED_CASES'
                                               if not counted else 'SCORE_NOT_AVAILABLE'),
                    performance_available=bool(score and score['perf_score'] is not None),
                    certification=dict(static='NOT_COMPUTED', compilation='NOT_COMPUTED',
                                       execution='NOT_COMPUTED', full_suite_accepted=False),
                    validation=engine['validation'], cases=records)

    def events(self, identifier: str, after: int = 0) -> dict:
        self._get('task', identifier)
        with self._db() as connection:
            rows = connection.execute('SELECT seq,payload FROM events WHERE task_id=? AND seq>? '
                                      'ORDER BY seq LIMIT 200', (identifier, after)).fetchall()
        return dict(items=[dict(sequence=row[0], **json.loads(row[1])) for row in rows],
                    next_cursor=rows[-1][0] if rows else after)

    def cancel(self, identifier: str) -> dict:
        with self.lock:
            task = self._get('task', identifier)
            if task['status'] in TERMINAL:
                return self.task(identifier)
            if task['status'] == 'QUEUED':
                self._transition(task, 'CANCELLED')
            else:
                task['cancel_requested'] = True
                self._save('task', task, emit_event=True)
            return self.task(identifier)

    def artifacts(self, identifier: str) -> list[dict]:
        directory = self._directory(identifier)
        output = []
        for name, path in self._artifact_paths(identifier).items():
            if path.is_file() and not path.is_symlink() and path.resolve().is_relative_to(directory):
                data = path.read_bytes()
                output.append(dict(name=name, bytes=len(data), sha256=hashlib.sha256(data).hexdigest()))
        return output

    def artifact(self, identifier: str, name: str) -> bytes:
        directory = self._directory(identifier)
        path = self._artifact_paths(identifier).get(name)
        if _invalid_artifact_path(path, directory):
            raise ServiceError('NOT-FOUND', '产物不存在', 404)
        return path.read_bytes()

    def work_once(self) -> bool:
        task = self._claim_task()
        if task is None:
            return False
        try:
            self._execute_task(task)
        except ServiceError as exc:
            self._transition(task, 'CANCELLED' if exc.code == 'TASK-CANCELLED' else 'FAILED', exc.code)
        except Exception:
            directory = self._directory(task['id'])
            directory.mkdir(parents=True, exist_ok=True)
            with (directory / 'worker-error.log').open('a', encoding='utf-8') as log:
                traceback.print_exc(file=log)
            self._pause('Worker 未预期异常；检查日志及设备后再恢复')
            self._transition(task, 'FAILED', 'WORKER-FAILURE')
        return True

    def start_worker(self) -> None:
        if self.thread is not None:
            return

        def loop():
            while not self.stopping.is_set():
                if not self.work_once():
                    self.stopping.wait(0.2)

        self.thread = threading.Thread(target=loop, name='evaluation-worker', daemon=True)
        self.thread.start()

    def _check_declared_operators(self, names, errors):
        if not names:
            errors.append(dict(rule_id='SERVICE-EMPTY', stage='content',
                               operator='-', detail='专项评测至少声明一个算子'))
        if len(set(names)) != len(names):
            errors.append(dict(rule_id='SERVICE-DUPLICATE', stage='content',
                               operator='-', detail='服务不接受重复算子声明'))
        unsupported = set(names) - {operator.name for operator in self.manifest.operators}
        if unsupported:
            errors.append(dict(rule_id='SERVICE-CORPUS-OPERATOR', stage='content',
                               operator='-', detail=f'不在当前测试集：{sorted(unsupported)}'))

    def _db(self):
        return sqlite3.connect(self.database, timeout=30)

    def _get(self, kind: str, identifier: str) -> dict:
        with self._db() as connection:
            row = connection.execute('SELECT payload FROM objects WHERE kind=? AND id=?',
                                     (kind, identifier)).fetchone()
        if row is None:
            raise ServiceError('NOT-FOUND', '记录不存在', 404)
        return json.loads(row[0])

    def _save(self, kind: str, value: dict, *, emit_event: bool = False) -> None:
        with self._db() as connection:
            connection.execute('UPDATE objects SET payload=? WHERE kind=? AND id=?',
                               (json.dumps(value), kind, value['id']))
            if emit_event:
                self._event(value, connection)

    def _existing(self, kind: str, key: str | None, digest: str) -> dict | None:
        if not key:
            return None
        if len(key) > 200:
            raise ServiceError('IDEMPOTENCY-KEY', '幂等键最长 200 字符')
        with self._db() as connection:
            row = connection.execute('SELECT payload, digest FROM objects WHERE kind=? AND idem=?',
                                     (kind, key)).fetchone()
        if row and row[1] != digest:
            raise ServiceError('IDEMPOTENCY-CONFLICT', '相同幂等键对应不同内容', 409)
        return json.loads(row[0]) if row else None

    def _insert(self, kind: str, value: dict, key: str | None, digest: str) -> dict:
        with self._db() as connection:
            connection.execute('INSERT INTO objects VALUES (?,?,?,?,?)',
                               (kind, value['id'], json.dumps(value), key or None, digest))
            if kind == 'task':
                self._event(value, connection)
        return value

    def _transition(self, task: dict, status: str, error: str | None = None) -> None:
        with self.lock:
            task['cancel_requested'] = self._get('task', task['id'])['cancel_requested']
            if status == 'COMPLETED' and task['cancel_requested']:
                status = 'CANCELLED'
            task.update(status=status, updated_at=_now(), error=error)
            self._save('task', task, emit_event=True)

    def _recover(self) -> None:
        with self.lock, self._db() as connection:
            rows = connection.execute("SELECT payload FROM objects WHERE kind='task'").fetchall()
            for row in rows:
                task = json.loads(row[0])
                if task['status'] in ACTIVE:
                    self._pause('服务中断；必须人工确认遗留进程和设备健康，不自动重跑')
                    self._transition(task, 'INTERRUPTED', 'SERVICE-RESTART')

    def _pause(self, reason: str) -> None:
        (self.root / 'worker-paused.txt').write_text(reason + '\n', encoding='utf-8')

    def _directory(self, identifier: str) -> Path:
        self._get('task', identifier)
        return self.root / 'evaluations' / identifier

    def _engine(self, identifier: str) -> dict:
        database = self._directory(identifier) / 'engine.sqlite'
        output = dict(run_status=None, validation=[], cases=[], scores=[], package_sha256=None)
        if not database.is_file():
            return output
        try:
            with Store(database) as store:
                runs = store.conn.execute('SELECT id FROM eval_run ORDER BY id').fetchall()
                if not runs:
                    return output
                if len(runs) != 1:
                    output['error'] = 'ENGINE-IDENTITY'
                    return output
                run_id = runs[0]['id']
                run = store.get_run(run_id)
                output['run_status'] = run['status']
                output['run_id'] = run_id
                output['cases'] = [record.model_dump(mode='json') for record in store.case_results(run_id)]
                output['scores'] = store.run_scores(run_id)
                record_id = run.get('submission_record_id')
                if record_id:
                    record = store.submission_record(record_id)
                    output['validation'] = json.loads(record['validation_json'])
                    output['package_sha256'] = record['package_sha256']
        except sqlite3.Error:
            output['error'] = 'ENGINE-DATABASE'
        return output

    def _artifact_paths(self, identifier: str) -> dict[str, Path]:
        directory = self._directory(identifier)
        allowed = {'report.md': directory / 'reports/report-run-1.md',
                   'report.html': directory / 'reports/report-run-1.html',
                   'results.jsonl': directory / 'runs/run-1/results.jsonl',
                   'run.log': directory / 'run.log',
                   'load.log': directory / 'runs/run-1/logs/load.log',
                   'score.log': directory / 'score.log',
                   'report.log': directory / 'report.log'}
        for code in self._get('task', identifier)['case_codes']:
            allowed[f'case-{code}.log'] = directory / 'runs/run-1/logs' / f'{code}.log'
        return allowed

    def _command(self, task: dict, phase: str) -> list[str]:
        directory = self._directory(task['id'])
        base = [sys.executable, '-m', 'triton_eval', '--db', str(directory / 'engine.sqlite')]
        if phase == 'run':
            package = self.root / 'submissions' / task['submission_id'] / 'submission.zip'
            command = base + ['run', '--submission', str(package), '--corpus', str(self.corpus),
                              '--runs', str(directory / 'runs'), '--ingest-dir', str(directory / 'ingest'),
                              '--profile', task['profile']]
            if task['scope'] != 'full':
                command += ['--operators', ','.join(task['operators'])]
            return command
        if phase == 'score':
            return base + ['score', '--run', '1']
        return base + ['report', '--run', '1', '--out', str(directory / 'reports')]

    def _invoke(self, task: dict, phase: str) -> int:
        directory = self._directory(task['id'])
        directory.mkdir(parents=True, exist_ok=True)
        environment = {name: value for name, value in os.environ.items() if name not in CHILD_CONTROL_ENV}
        with (directory / f'{phase}.log').open('wb') as log:
            process = subprocess.Popen(self._command(task, phase), stdout=log, stderr=log,
                                       start_new_session=True, env=environment)
            started = time.monotonic()
            budget = _case_timeout_budget(self.manifest.cases, task['case_codes']) + 600
            while process.poll() is None:
                current = self._get('task', task['id'])
                if self.stopping.is_set() or current['cancel_requested'] or time.monotonic() - started > budget:
                    self._pause('Worker 进程组被终止；人工确认进程及设备健康后才可解除暂停')
                    self._terminate(process)
                    raise ServiceError('TASK-CANCELLED' if current['cancel_requested'] else 'WORKER-INTERRUPTED',
                                       '运行被终止，保留已有证据并暂停后续派发', 409)
                time.sleep(0.1)
            return process.returncode

    def _claim_task(self):
        if not self.execute or (self.root / 'worker-paused.txt').exists():
            return None
        with self.lock, self._db() as connection:
            if self.stopping.is_set():
                return None
            rows = connection.execute("SELECT payload FROM objects WHERE kind='task' ORDER BY rowid").fetchall()
            if any(json.loads(row[0])['status'] in ACTIVE for row in rows):
                return None
            task = next((json.loads(row[0]) for row in rows if json.loads(row[0])['status'] == 'QUEUED'), None)
            if task is None:
                return None
            self._transition(task, 'RUNNING')
        return task

    def _validate_worker_result(self, task, engine):
        actual = [record['case_code'] for record in engine['cases']]
        if engine['package_sha256'] != task['package_sha256']:
            raise ServiceError('RESULT-IDENTITY', '执行器记录的包指纹与提交不一致')
        if len(actual) != len(set(actual)) or set(actual) != set(task['case_codes']):
            raise ServiceError('RESULT-SCOPE', '实际结果与冻结任务范围不一致')
        if any(record['status'] in {'ENV_ABORT', 'ORACLE_FAIL', 'SPEC_ERROR'} for record in engine['cases']):
            self._pause('评测方或环境异常，不能作为正常得分发布')
            raise ServiceError('EVALUATOR-FAILURE', '评测方异常，不认定为提交方失败')

    def _publish_task(self, task, engine):
        counted = any(is_counted(CaseStatus(record['status'])) for record in engine['cases'])
        if counted:
            self._transition(task, 'SCORING')
            if self._invoke(task, 'score') != 0:
                raise ServiceError('ENGINE-SCORE', '评分失败，详见本机日志')
            if not self._engine(task['id'])['scores']:
                raise ServiceError('ENGINE-SCORE-MISSING', '评分进程未保存分数')
        self._transition(task, 'REPORTING')
        if self._invoke(task, 'report') != 0:
            raise ServiceError('ENGINE-REPORT', '报告生成失败，详见本机日志')
        available = {artifact['name'] for artifact in self.artifacts(task['id'])}
        if not {'report.md', 'report.html', 'results.jsonl'} <= available:
            raise ServiceError('ENGINE-ARTIFACTS', '评测产物不完整')
        if self._get('task', task['id'])['cancel_requested']:
            self._transition(task, 'CANCELLED')
        else:
            self._transition(task, 'COMPLETED')

    def _execute_task(self, task):
        verify_corpus_files(self.corpus, self.manifest)
        if load_manifest(self.corpus / 'manifest.json').manifest_hash != task['manifest_hash']:
            raise ServiceError('CORPUS-CHANGED', '测试集版本发生变化')
        submission = self.submission(task['submission_id'])
        package = self.root / 'submissions' / submission['id'] / 'submission.zip'
        if hashlib.sha256(package.read_bytes()).hexdigest() != submission['package_sha256']:
            raise ServiceError('PACKAGE-CHANGED', '上传后包指纹发生变化')
        return_code = self._invoke(task, 'run')
        engine = self._engine(task['id'])
        if return_code != 0 or engine['run_status'] != 'OK':
            if engine['run_status'] == 'INVALID' and engine['validation']:
                self._transition(task, 'REJECTED', 'ENGINE-VALIDATION')
            else:
                self._pause('评测未正常完成；需人工确认环境健康')
                self._transition(task, 'FAILED', engine.get('error') or 'ENGINE-RUN')
            return
        self._validate_worker_result(task, engine)
        self._publish_task(task, engine)


def _counted_operator_names(records):
    operators = set()
    for record in records:
        if is_counted(CaseStatus(record['status'])):
            operators.add(record['operator'])
    return operators
