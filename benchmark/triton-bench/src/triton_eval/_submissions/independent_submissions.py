# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""New implementations and positive submission boundaries, never hardware evidence."""

from __future__ import annotations

import ast

import yaml

from . import PROJECT_ROOT
from .operator_registry import by_name
from .official_catalog import submission_entry_name


KERNEL_SOURCE = PROJECT_ROOT / 'scripts/templates/independent_submission_kernels.py'


def definitions() -> list[dict]:
    singles = [
        ('ABS', 'Abs', '正负值及非连续主输入'),
        ('NEG', 'Neg', '零值、尾块及高阶输入'),
        ('EXP', 'Exp', '非默认属性及非连续主输入'),
        ('ADD', 'Add', '双输入及独立输入步长'),
        ('SOFTMAX', 'Softmax', '单列、非二次幂归约及数值稳定性'),
        ('REDUCE', 'ReduceSum', '正负轴及 keepdim 输出形状'),
        ('TRANSPOSE', 'Transpose', '连续布局输出及 int16 数据'),
        ('MATMUL', 'Matmul', '不整齐矩阵及长归约轴'),
        ('CUMMIN', 'Cummin', '双输出、不同轴、百万长度及跨块索引'),
    ]
    records = _single_scenarios(singles)
    records += [
        dict(code='NEW-TRANSPOSE-VIEW', description='合法非连续视图输出',
             operators=('Transpose',), layout='view'),
        dict(code='NEW-MULTI-SHARED', description='新实现多算子共享源码',
             operators=('Exp', 'Add', 'Softmax', 'Transpose')),
        dict(code='NEW-MULTI-SPLIT', description='同一语义的新实现分文件提交',
             operators=('Exp', 'Add', 'Softmax', 'Transpose'), layout='split'),
        dict(code='NEW-ENTRY', description='入口名称与算子名称不同但签名合法',
             operators=('Exp',), layout='renamed'),
        dict(code='NEW-KEYWORD-ATTR', description='属性只接受关键字传参的合法入口',
             operators=('Exp',), layout='keyword'),
        dict(code='NEW-KWARGS', description='显式契约参数加可选 kwargs 的合法入口',
             operators=('Exp',), layout='kwargs'),
        dict(code='NEW-UNICODE-ID', description='合法 Unicode 提交标识',
             operators=('Exp',), declared_submit_id='独立实现-边界验证-α'),
        dict(code='NEW-REORDERED', description='声明顺序不按注册表排序',
             operators=('Neg', 'Abs', 'Add')),
        dict(code='NEW-SCOPED', description='多算子提交只评选定算子完整用例集',
             operators=('Exp', 'Add'), request_operators=('Exp',)),
        dict(code='NEW-REPEAT-A', description='同内容第一次独立提交',
             operators=('Exp',), declared_submit_id='independent-repeat'),
        dict(code='NEW-REPEAT-B', description='相同 ZIP 和提交标识再次独立提交',
             operators=('Exp',), declared_submit_id='independent-repeat'),
    ]
    return [dict(record, origin='independent', tier='independent') for record in records]


def _source(entry_names: list[str]) -> bytes:
    source = KERNEL_SOURCE.read_text(encoding='utf-8')
    tree = ast.parse(source)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    selected = set(entry_names)
    pending = list(entry_names)
    while pending:
        node = functions[pending.pop()]
        dependencies = {child.id for child in ast.walk(node) if isinstance(child, ast.Name)} & functions.keys()
        for dependency in sorted(dependencies - selected):
            selected.add(dependency)
            pending.append(dependency)
    tree.body = [node for node in tree.body if _selected_source_node(node, selected)]
    return (ast.unparse(tree) + '\n').encode()


def entries(scenario) -> dict[str, bytes]:
    declarations = []
    functions = []
    for operator in scenario.operators:
        entry = submission_entry_name(operator, by_name(operator).entry)
        if scenario.layout == 'view':
            entry = 'transpose_view'
        functions.append(entry)
        path = f'operators/{operator.lower()}.py' if scenario.layout == 'split' else 'operators/user_impl.py'
        declarations.append(dict(name=operator, entry=entry, path=path))
    output = {}
    for declaration, entry in zip(declarations, functions):
        path = declaration['path']
        if path not in output:
            output[path] = _source([entry] if scenario.layout == 'split' else functions)
    if scenario.layout in {'renamed', 'keyword', 'kwargs'}:
        declaration = declarations[0]
        declaration['entry'] = 'evaluate_exponential'
        parameters = ('x, *, base=-1.0, scale=1.0, shift=0.0' if scenario.layout == 'keyword'
                      else 'x, base=-1.0, scale=1.0, shift=0.0, **extras' if scenario.layout == 'kwargs'
                      else 'x, base=-1.0, scale=1.0, shift=0.0')
        output[declaration['path']] += (
            f'\ndef evaluate_exponential({parameters}):\n'
            '    return exp(x, base, scale, shift)\n').encode()
    manifest = dict(submit_id=scenario.declared_submit_id or f'simulation-{scenario.code.lower()}',
                    operators=declarations)
    output['manifest.yaml'] = yaml.safe_dump(manifest, sort_keys=False, allow_unicode=True).encode()
    return output


def case_boundary(case) -> dict:
    tags = []
    if not case.contiguous:
        tags.append('noncontiguous_input')
    if any(size == 1 for size in case.shape):
        tags.append('singleton_dimension')
    if len(case.shape) > 2:
        tags.append('higher_rank')
    if any(size > 1 and size & (size - 1) for size in case.shape):
        tags.append('non_power_of_two_dimension')
    if case.value_range.get('kind') in {'zeros', 'mixed', 'boundary', 'extreme'}:
        tags.append('values_' + case.value_range['kind'])
    if case.attrs:
        tags.append('attributes')
    if case.operator == 'Cummin':
        tags.append('multiple_outputs')
        if case.shape[case.attrs['dim']] > 1024:
            tags.append('cross_block_scan')
    return dict(case_code=case.code, operator=case.operator, dtype=case.dtype,
                shape=list(case.shape), attrs=case.attrs, contiguous=case.contiguous,
                seed=case.seed, tags=tags)


def _single_scenarios(singles):
    scenarios = []
    for code, operator, description in singles:
        scenarios.append(dict(code=f'NEW-{code}', description=description, operators=(operator,)))
    return scenarios


def _selected_source_node(node, selected):
    return (isinstance(node, (ast.Import, ast.ImportFrom))
            or isinstance(node, ast.FunctionDef) and node.name in selected)
