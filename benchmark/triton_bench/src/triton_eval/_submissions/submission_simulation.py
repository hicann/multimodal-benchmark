# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Deterministic, package-valid user submissions and independent expectations."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path

from collections import Counter, defaultdict

import yaml

from . import PROJECT_ROOT
from .submission import load_manifest
from .corpusmod import load_manifest as load_corpus, verify_corpus_files
from .independent_submissions import KERNEL_SOURCE, case_boundary, definitions, entries as independent_entries

from .fixture_sources import fixture_path, require_source_dependencies

DEFAULT_SOURCE = fixture_path(PROJECT_ROOT / 'examples/coverage-v18-ascend-submission')
DEFAULT_CORPUS = PROJECT_ROOT / 'corpus/v0.18-scan-certification-candidates'
EXP_SIGNATURE = 'def exp(x, base=-1.0, scale=1.0, shift=0.0):\n'


@dataclass(frozen=True)
class Scenario:
    code: str
    description: str
    operators: tuple[str, ...] = ('Exp',)
    mutation: str = ''
    stage: str = 'execution'
    rule: str | None = None
    statuses: tuple[str, ...] = ('PASS',)
    tier: str = 'normal'
    status_mode: str = 'exact'
    origin: str = 'historical'
    layout: str = 'shared'
    request_operators: tuple[str, ...] = ()
    declared_submit_id: str | None = None


def scenarios() -> list[Scenario]:
    output = _initial_scenarios()

    output += _content_scenarios()
    output += _load_scenarios()
    output += _runtime_scenarios()
    output += [
        Scenario('COMPILE-UNDEFINED', '真实 JIT 中未定义符号', mutation='compile_undefined',
                 stage='compile', statuses=('COMPILE_FAIL',)),
        Scenario('LAUNCH-ARGUMENT', '内核启动缺少必需参数', mutation='launch_argument',
                 statuses=('RUNTIME_FAIL', 'COMPILE_FAIL'), status_mode='allowed'),
        Scenario('MIXED-RUNTIME', '一个算子正确、另一个执行失败', ('Exp', 'Add'),
                 mutation='mixed_runtime', statuses=('PASS', 'RUNTIME_FAIL')),
        Scenario('MIXED-LOAD', '一个入口错误时观察现有整包拒收策略', ('Exp', 'Add'),
                 mutation='mixed_load', stage='load', rule='SUB-LOD-005', statuses=()),
        Scenario('EXEC-TIMEOUT', '等待触发现有超时；单独显式执行',
                 mutation='import time\n' + EXP_SIGNATURE + '    time.sleep(3600)\n    return x\n',
                 statuses=('RUNTIME_FAIL',), tier='slow'),
        Scenario('EXEC-EXIT', '执行期 SystemExit；预期非自然结束并暂停派发',
                 mutation=EXP_SIGNATURE + '    raise SystemExit(0)\n',
                 stage='termination', statuses=('RUNTIME_FAIL',), tier='disruptive'),
        Scenario('EXEC-DEVICE', '返回 CPU 输出，触发设备契约',
                 mutation='import torch\n' + EXP_SIGNATURE
                 + '    return torch.empty(x.shape, device="cpu", dtype=x.dtype)\n',
                 statuses=('SUBMISSION_INVALID',), rule='SUB-EXE-006'),
        Scenario('EXEC-INPUT', '修改输入，触发输入指纹契约',
                 mutation=EXP_SIGNATURE + '    x.add_(1)\n    return x\n',
                 statuses=('SUBMISSION_INVALID',), rule='SUB-EXE-004'),
        Scenario('EXEC-CUMMIN-MISSING', '多输出算子只交第一个输出', ('Cummin',),
                 mutation='cummin_missing', statuses=('SHAPE_MISMATCH',)),
        Scenario('EXEC-CUMMIN-INDICES', '主输出正确、索引输出错误', ('Cummin',),
                 mutation='cummin_indices', statuses=('WRONG_RESULT',)),
        Scenario('EXEC-PARTIAL', '同一算子仅部分 dtype 执行失败', mutation='partial',
                 statuses=('PASS', 'RUNTIME_FAIL')),
    ]
    return output + [Scenario(**record) for record in definitions()]


def _entries(source: Path, scenario: Scenario) -> dict[str, bytes]:
    if scenario.origin == 'independent':
        return independent_entries(scenario)
    source = fixture_path(source)
    manifest = load_manifest(source)
    selected = [entry.model_dump() for entry in manifest.operators
                if not scenario.operators or entry.name in scenario.operators]
    expected = set(scenario.operators)
    if expected - {entry['name'] for entry in selected}:
        raise ValueError(f'基线缺少算子：{sorted(expected)}')
    entries = {str(path.relative_to(source)): path.read_bytes()
               for path in sorted(source.rglob('*.py')) if '__pycache__' not in path.parts}
    raw = dict(submit_id=f'simulation-{scenario.code.lower()}', operators=copy.deepcopy(selected))
    mutation = scenario.mutation
    if mutation == 'yaml':
        entries['manifest.yaml'] = b'operators: [\n'
        return entries
    _mutate_entries(entries, raw, scenario, mutation)
    entries['manifest.yaml'] = yaml.safe_dump(raw, sort_keys=False).encode()
    return entries


def build_suite(destination: Path, source: Path = DEFAULT_SOURCE, corpus: Path = DEFAULT_CORPUS) -> dict:
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = load_corpus(corpus / 'manifest.json')
    verify_corpus_files(corpus, manifest)
    source = fixture_path(source)
    for module in sorted(source.rglob('*.py')):
        require_source_dependencies(module.read_text(encoding='utf-8'), str(module.relative_to(source)))
    records = []
    baseline = {str(path.relative_to(source)): hashlib.sha256(path.read_bytes()).hexdigest()
                for path in sorted(source.rglob('*'))
                if path.is_file() and path.suffix in {'.py', '.yaml'} and '__pycache__' not in path.parts}
    for scenario in scenarios():
        path = destination / f'{scenario.code}.zip'
        with zipfile.ZipFile(path, 'w', compression=zipfile.ZIP_STORED) as archive:
            for name, data in sorted(_entries(source, scenario).items()):
                info = zipfile.ZipInfo(name, date_time=(2026, 1, 1, 0, 0, 0))
                info.external_attr = 0o100644 << 16
                archive.writestr(info, data)
        record = asdict(scenario)
        record['package'] = path.name
        record['sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        record['expected_score'] = None
        record['manifest_hash'] = manifest.manifest_hash
        selected_operators = scenario.request_operators or scenario.operators
        selected_cases = [case for case in manifest.cases
                          if not selected_operators or case.operator in selected_operators]
        record['case_codes'] = [case.code for case in selected_cases]
        record['case_boundaries'] = [case_boundary(case) for case in selected_cases]
        if scenario.origin == 'independent':
            record['expected_score'] = dict(coverage_ops_core=100.0, coverage_strict=100.0)
        record['operator_weights'] = {operator.name: operator.weight for operator in manifest.operators
                                      if not selected_operators or operator.name in selected_operators}
        records.append(record)
    report = dict(schema_version=1, evidence_kind='EXPECTATIONS_NOT_HARDWARE_RESULTS',
                  corpus_version=manifest.version, manifest_hash=manifest.manifest_hash,
                  source=str(source.resolve()), source_hashes=baseline, scenarios=records)
    report['independent_source_sha256'] = hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest()
    report['unverified_boundaries'] = ['unregistered_operators', 'out_of_domain_inputs',
                                       'all_possible_legal_implementations', 'performance_acceptance']
    (destination / 'expectations.json').write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    return report


def check_outcome(scenario: dict, submission: dict, task: dict | None, results: dict | None) -> list[str]:
    """Compare independent expectations; infrastructure failures never count as success."""
    failures = []
    expected_stage = scenario['stage']
    if expected_stage == 'content':
        if submission['status'] != 'REJECTED':
            failures.append('内容负例未在执行前拒收')
        rules = {error['rule_id'] for error in submission['errors']}
        if scenario['rule'] not in rules:
            failures.append(f"未命中预期规则 {scenario['rule']}")
        return failures
    if submission['status'] != 'READY':
        return ['非内容负例被提前拒收']
    if not task or not results:
        return ['缺少真实任务或结果']
    if scenario.get('manifest_hash') and task['manifest_hash'] != scenario['manifest_hash']:
        failures.append('任务未绑定预期冻结版本')
    if scenario.get('case_codes') and task['case_codes'] != scenario['case_codes']:
        failures.append('任务计划用例与独立清单不符')
    if expected_stage in {'load', 'termination'}:
        return _check_rejected_task(expected_stage, scenario, task, results, failures)
    if task['status'] != 'COMPLETED':
        return [f"任务未自然完成：{task['status']}"]
    rows = results['cases']
    _check_completed_rows(scenario, task, rows, failures)

    _check_scenario_scope(scenario, task, results, rows, failures)
    _check_simulation_score(scenario, results, rows, failures)
    if task.get('profile') == 'smoke' and results.get('performance_available'):
        failures.append('smoke 档不应给出性能验收分数')
    return failures


def _mutate_entries(entries, raw, scenario, mutation):
    if mutation == 'missing_id':
        raw.pop('submit_id')
        return
    if mutation == 'operators_type':
        raw['operators'] = 'Exp'
        return
    if mutation.startswith('missing_') and mutation != 'missing_file':
        raw['operators'][0].pop(mutation.removeprefix('missing_'))
        return
    if mutation == 'extra':
        raw['unexpected'] = True
        return
    if mutation == 'unknown':
        raw['operators'][0]['name'] = 'SimulationUnknownOperator'
        return
    if mutation == 'missing_file':
        raw['operators'][0]['path'] = 'operators/missing.py'
        return
    if mutation == 'duplicate':
        raw['operators'].append(copy.deepcopy(raw['operators'][0]))
        return
    if mutation == 'empty':
        raw['operators'] = []
        return
    _mutate_execution(entries, raw, scenario, mutation)


def _mutate_execution(entries, raw, scenario, mutation):
    if mutation in {'mixed_runtime', 'mixed_load'}:
        declaration = next(entry for entry in raw['operators'] if entry['name'] == 'Add')
        declaration['path'] = 'operators/broken_add.py'
        body = ('def add(x, y):\n    raise RuntimeError("simulation mixed failure")\n'
                if mutation == 'mixed_runtime' else 'add = 7\n')
        entries[declaration['path']] = body.encode()
        return
    if mutation in {'cummin_missing', 'cummin_indices', 'partial'}:
        declaration = raw['operators'][0]
        path = declaration['path']
        if mutation == 'partial':
            wrapper = ('\n_simulation_original_exp = exp\n\n' + EXP_SIGNATURE
                       + '    if x.dtype == torch.float16:\n'
                       + '        raise RuntimeError("simulation partial dtype failure")\n'
                       + '    return _simulation_original_exp(x, base, scale, shift)\n')
        else:
            result = 'values' if mutation == 'cummin_missing' else '(values, torch.full_like(indices, -1))'
            wrapper = ('\n_simulation_original_cummin = cummin\n\n'
                       + 'def cummin(input_value, dim):\n'
                       + '    values, indices = _simulation_original_cummin(input_value, dim)\n'
                       + f'    return {result}\n')
        entries[path] += wrapper.encode()
        return
    if mutation in {'compile_undefined', 'launch_argument'}:
        body = ('    values = simulation_undefined_symbol(offsets)\n'
                if mutation == 'compile_undefined' else '    values = tl.load(x + offsets, offsets < count)\n')
        launch = '_broken[(triton.cdiv(x.numel(), 128),)](x, out, x.numel(), BLOCK=128)'
        if mutation == 'launch_argument':
            launch = '_broken[(1,)](x, out, BLOCK=128)'
        kernel = ('import torch\nimport triton\nimport triton.language as tl\n\n'
                  '@triton.jit\ndef _broken(x, out, count, BLOCK: tl.constexpr):\n'
                  '    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)\n'
                  + body + '    tl.store(out + offsets, values, offsets < count)\n\n'
                  + EXP_SIGNATURE + '    out = torch.empty_like(x)\n    ' + launch + '\n    return out\n')
        raw['operators'][0]['path'] = 'operators/simulation.py'
        entries['operators/simulation.py'] = kernel.encode()
        return
    if mutation and scenario.stage in {'content', 'load', 'execution', 'termination'}:
        raw['operators'][0]['path'] = 'operators/simulation.py'
        entries['operators/simulation.py'] = mutation.encode()
        return


def _check_rejected_task(expected_stage, scenario, task, results, failures):
    if expected_stage == 'load':
        if task['status'] != 'REJECTED':
            failures.append('加载负例未被现有加载契约拒收')
        if scenario['rule'] not in {error['rule_id'] for error in results['validation']}:
            failures.append(f"未命中加载规则 {scenario['rule']}")
        return failures
    if expected_stage == 'termination':
        if task['status'] != 'FAILED' or task.get('error') != 'ENGINE-RUN' or not task.get('worker_paused'):
            failures.append('异常退出后没有保留失败状态并暂停 Worker')
        if not results['cases'] or any(row['status'] != 'RUNTIME_FAIL' for row in results['cases']):
            failures.append('异常退出后缺失用例没有按既有监督机制补记')
        return failures

    return failures


def _check_scenario_scope(scenario, task, results, rows, failures):
    if scenario.get('origin') == 'independent':
        if task.get('package_sha256') != scenario['sha256'] or results.get('package_sha256') != scenario['sha256']:
            failures.append('独立实现的包指纹没有贯穿任务和结果')
        if results.get('engine_error') or results.get('error') or results.get('validation'):
            failures.append('合法提交出现引擎错误或校验错误')
        if set(results.get('declared_operators', [])) != set(scenario['operators']):
            failures.append('结果没有保留合法提交声明的算子范围')
        expected_scope = set(scenario.get('request_operators') or scenario['operators'])
        if set(results.get('scope_operators', [])) != expected_scope:
            failures.append('评测范围与合法专项请求不一致')
    if scenario['code'] == 'MIXED-RUNTIME':
        if any(row['status'] != ('PASS' if row['operator'] == 'Exp' else 'RUNTIME_FAIL') for row in rows):
            failures.append('混合提交的算子隔离结果不符')


def _score_matches(actual, expected):
    """Compare independently recomputed scores at the scorer's four-decimal precision."""
    return math.isfinite(actual) and math.isclose(actual, round(expected, 4), rel_tol=0.0, abs_tol=1e-8)


def _check_simulation_score(scenario, results, rows, failures):
    counted_statuses = {'PASS', 'COMPILE_FAIL', 'RUNTIME_FAIL', 'WRONG_RESULT', 'SHAPE_MISMATCH'}
    groups = defaultdict(list)
    for row in rows:
        if row['status'] in counted_statuses:
            groups[row['operator']].append(row)
    score = results.get('score')
    if not groups:
        if score is not None:
            failures.append('没有有效计分项却返回了数值得分')
    elif score is None:
        failures.append('存在有效计分项却缺少得分')
    else:
        weights = scenario.get('operator_weights', {})
        total_weight = sum(weights.get(operator, 1.0) for operator in groups)
        expected = 100 * sum(weights.get(operator, 1.0)
                             * sum(row['status'] == 'PASS' for row in items) / len(items)
                             for operator, items in groups.items()) / total_weight
        strict = 100 * sum(weights.get(operator, 1.0) for operator, items in groups.items()
                           if all(row['status'] == 'PASS' for row in items)) / total_weight
        if (not _score_matches(score['coverage_ops_core'], expected)
                or not _score_matches(score['coverage_strict'], strict)):
            failures.append('独立重算的功能得分或严格通过率与评分器不符')
        expected_scores = scenario.get('expected_score') or {}
        matches_expected = all(_score_matches(score.get(key, -1), value) for key, value in expected_scores.items())
        if not matches_expected:
            failures.append('独立正确实现没有取得预期功能分和严格通过率')


def _content_scenarios():
    static = [
        ('STATIC-SYNTAX', '源码语法错误', 'def exp(:\n', 'SUB-STR-006'),
        ('STATIC-DELEGATE', '框架内置接口代算',
         'import torch\n' + EXP_SIGNATURE + '    return torch.exp(x)\n', 'SUB-STR-001'),
        ('STATIC-CPU', '主机侧往返', EXP_SIGNATURE + '    return x.cpu()\n', 'SUB-STR-002'),
    ]
    output = [Scenario(code, desc, mutation=source, stage='content', rule=rule, statuses=())
               for code, desc, source, rule in static]
    return output


def _load_scenarios():
    loading = [
        ('LOAD-MISSING', '入口缺失', 'other = 1\n', 'SUB-LOD-004'),
        ('LOAD-NONCALLABLE', '入口不可调用', 'exp = 7\n', 'SUB-LOD-005'),
        ('LOAD-COUNT', '入口参数个数错误', 'def exp(x):\n    return x\n', 'SUB-LOD-006'),
        ('LOAD-NAMES', '入口属性名错误',
         'def exp(x, factor=-1.0, scale=1.0, shift=0.0):\n    return x\n', 'SUB-LOD-007'),
        ('LOAD-POSONLY', '仅限位置参数不能接收属性关键字',
         'def exp(x, base=-1.0, scale=1.0, shift=0.0, /):\n    return x\n', 'SUB-LOD-009'),
        ('LOAD-IMPORT', '缺少依赖；不得误报成编译失败',
         'import simulation_missing_dependency_93d2\n' + EXP_SIGNATURE + '    return x\n',
         'SUB-LOD-005'),
        ('LOAD-RAISE', '导入期异常', 'raise RuntimeError("simulation import failure")\n',
         'SUB-LOD-005'),
        ('LOAD-EXIT', '导入期 SystemExit', 'raise SystemExit(0)\n', 'SUB-LOD-005'),
    ]
    output = [Scenario(code, desc, mutation=source, stage='load', rule=rule, statuses=())
               for code, desc, source, rule in loading]
    return output


def _runtime_scenarios():
    runtime = [
        ('EXEC-RAISE', '执行期普通异常', '    raise RuntimeError("simulation runtime failure")\n',
         ('RUNTIME_FAIL',), None),
        ('EXEC-FAKE-COMPILE', '异常文本不得伪装成编译异常',
         '    raise RuntimeError("CompilationError: fake text")\n', ('RUNTIME_FAIL',), None),
        ('EXEC-WRONG', '连续输出；只引入数学语义错误', '    return torch.zeros(x.shape, device=x.device, dtype=x.dtype)\n',
         ('WRONG_RESULT',), None),
        ('EXEC-SHAPE', '输出形状错误', '    return torch.empty((0,), device=x.device, dtype=x.dtype)\n',
         ('SHAPE_MISMATCH',), None),
        ('EXEC-DTYPE', '输出 dtype 错误',
         '    return torch.empty(x.shape, device=x.device, dtype=torch.int64)\n',
         ('SUBMISSION_INVALID',), 'SUB-EXE-003'),
        ('EXEC-NONTENSOR', '返回非张量；不受编译证据开关影响', '    return 7\n',
         ('SUBMISSION_INVALID',), 'SUB-EXE-001'),
        ('EXEC-ARITY', '单输出算子返回两个输出', '    return (x, x)\n',
         ('SHAPE_MISMATCH',), None),
    ]
    output = [Scenario(code, desc, mutation='import torch\n' + EXP_SIGNATURE + body,
                        statuses=statuses, rule=rule)
               for code, desc, body, statuses, rule in runtime]
    return output


def _initial_scenarios():
    valid = [
        Scenario('OK-EXP', '带属性的逐元素算子'),
        Scenario('OK-ADD', '双输入算子', ('Add',)),
        Scenario('OK-SOFTMAX', '归约归一化算子', ('Softmax',)),
        Scenario('OK-TRANSPOSE', '布局变换算子', ('Transpose',)),
        Scenario('OK-MATMUL', '矩阵乘法', ('Matmul',)),
        Scenario('OK-CUMMIN', '多输出及分块扫描', ('Cummin',)),
        Scenario('OK-ARGMAX', '参数迁移后的索引归约算子', ('ArgMax',)),
        Scenario('OK-SWIGLU', '参数迁移后的门控激活算子', ('SwiGlu',)),
        Scenario('OK-CROSSENTROPY', '参数迁移后的交叉熵损失算子', ('CrossEntropyLoss',)),
        Scenario('OK-FLOORDIV', '历史修复算子', ('FloorDiv',)),
        Scenario('OK-GROUPED', '分组矩阵乘法', ('GroupedMatmul',)),
        Scenario('OK-MULTI', '共享源码、多算子', ('Exp', 'Add', 'Softmax')),
        Scenario('OK-FULL', '全量提交；仅在 full 档执行', (), tier='full'),
    ]
    metadata = [
        ('META-YAML', 'YAML 语法错误', 'yaml', 'SUB-LOD-001'),
        ('META-ID', '缺少提交标识', 'missing_id', 'SUB-LOD-001'),
        ('META-OPS-TYPE', '算子列表类型错误', 'operators_type', 'SUB-LOD-001'),
        ('META-NAME', '缺少算子名', 'missing_name', 'SUB-LOD-001'),
        ('META-ENTRY', '缺少入口字段', 'missing_entry', 'SUB-LOD-001'),
        ('META-PATH', '缺少实现路径字段', 'missing_path', 'SUB-LOD-001'),
        ('META-EXTRA', '未知元数据字段', 'extra', 'SUB-LOD-001'),
        ('META-UNKNOWN', '未登记算子', 'unknown', 'SUB-LOD-002'),
        ('META-FILE', '声明的内核文件缺失', 'missing_file', 'SUB-LOD-003'),
        ('META-DUP', '重复声明；服务编排策略拒绝', 'duplicate', 'SERVICE-DUPLICATE'),
        ('META-EMPTY', '空提交；服务编排策略拒绝', 'empty', 'SERVICE-EMPTY'),
    ]
    output = valid + [Scenario(code, desc, mutation=mutation, stage='content',
                               rule=rule, statuses=())
                      for code, desc, mutation, rule in metadata]
    return output


def _check_completed_rows(scenario, task, rows, failures):
    actual = {row['status'] for row in rows}
    matches = (bool(actual) and actual <= set(scenario['statuses'])
               if scenario.get('status_mode') == 'allowed' else actual == set(scenario['statuses']))
    if not matches:
        failures.append(f"用例终态不符：{sorted(actual)} != {scenario['statuses']}")
    if len(rows) != task['planned_cases']:
        failures.append('用例数量与任务计划不一致')
    if scenario.get('case_codes') and Counter(row['case_code'] for row in rows) != Counter(scenario['case_codes']):
        failures.append('实际用例与独立冻结清单不一致')
    if scenario['rule'] and not any(scenario['rule'] in (row.get('compile_err') or '') for row in rows):
        failures.append('缺少执行规则诊断证据')
