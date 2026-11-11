# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Import-free entry checks, followed by optional real loading in a subprocess.

The source check supports direct, undecorated Python entries in the bundled
submissions. It never evaluates defaults, annotations, imports or function bodies.
It does not certify that imports, compilation or device execution will succeed.
"""
from __future__ import annotations

import ast
import inspect
from dataclasses import asdict
from pathlib import Path

from .fixture_sources import fixture_path
from .operator_registry import by_name
from .submission import discover
from .validator import ValidationResult, signature_errors, validate_load, validate_static


def source_callable(node: ast.FunctionDef):
    """Expose an AST signature to the exact rules used by validate_load."""
    arguments = node.args
    positional = [*arguments.posonlyargs, *arguments.args]
    required_count = len(positional) - len(arguments.defaults)
    optional = {parameter.arg for parameter in positional[required_count:]}
    optional.update(parameter.arg for parameter, value in zip(arguments.kwonlyargs, arguments.kw_defaults)
                    if value is not None)
    groups = (
        (arguments.posonlyargs, inspect.Parameter.POSITIONAL_ONLY),
        (arguments.args, inspect.Parameter.POSITIONAL_OR_KEYWORD),
        ([arguments.vararg] if arguments.vararg else [], inspect.Parameter.VAR_POSITIONAL),
        (arguments.kwonlyargs, inspect.Parameter.KEYWORD_ONLY),
        ([arguments.kwarg] if arguments.kwarg else [], inspect.Parameter.VAR_KEYWORD),
    )
    parameters = [inspect.Parameter(parameter.arg, kind,
                                    default=None if parameter.arg in optional else inspect.Parameter.empty)
                  for group, kind in groups for parameter in group]

    def entry(*args, **kwargs):
        raise RuntimeError('source preflight never executes an entry')
    entry.__signature__ = inspect.Signature(parameters)
    return entry


def submission_readiness(root, required_operators=(), *, load=False):
    root = fixture_path(Path(root).resolve())
    validation = ValidationResult(root.name)
    validate_static(root, validation)
    report = dict(ok=False, source_checked=0, declared_operators=0,
                  load_state='NOT_RUN', errors=[])
    if validation.ok:
        sub = discover(root)
        names = sub.operator_names()
        report['declared_operators'] = len(names)
        missing = sorted(set(required_operators) - set(names))
        if missing:
            report['errors'].append(dict(rule_id='PREFLIGHT-MISSING', operator='-',
                                         detail=f'missing corpus operators: {missing}'))
        if len(set(names)) != len(names):
            report['errors'].append(dict(rule_id='PREFLIGHT-DUPLICATE', operator='-',
                                         detail='duplicate operator declarations'))
        _check_entry_sources(root, sub, validation, report)
        if load and validation.ok and not report['errors']:
            validate_load(sub, validation)
            report['load_state'] = 'PASS' if validation.ok else 'FAIL'
    report['errors'].extend(asdict(error) for error in validation.errors)
    report['ok'] = not report['errors']
    return report


def require_submission_ready(root, required_operators=()):
    report = submission_readiness(root, required_operators)
    if not report['ok']:
        details = '\n'.join(f"[{e['rule_id']}] {e['operator']}: {e['detail']}" for e in report['errors'])
        raise ValueError(f'submission entry preflight failed:\n{details}')
    return report


def _entry_functions(tree, entry):
    functions = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == entry:
            functions.append(node)
    return functions


def _check_entry_sources(root, sub, validation, report):
    trees = {}
    for declaration in sub.manifest.operators:
        path = root / declaration.path
        if path not in trees:
            trees[path] = ast.parse(path.read_text(encoding='utf-8'))
        nodes = _entry_functions(trees[path], declaration.entry)
        if len(nodes) != 1 or not isinstance(nodes[0], ast.FunctionDef) or nodes[0].decorator_list:
            report['errors'].append(
                dict(
                    rule_id='PREFLIGHT-SOURCE',
                    operator=declaration.name,
                    detail=(f'{declaration.path}:{declaration.entry}: '
                            'requires one direct, undecorated synchronous entry')))
            continue
        func = source_callable(nodes[0])
        spec = by_name(declaration.name)
        errors = signature_errors(func, spec)
        validation.errors.extend(errors)
        # Check the actual calling convention as well: input tensors are
        # positional, all attributes are keywords. No values are executed.
        if not errors:
            try:
                inspect.signature(func).bind(*[None for _ in spec.inputs],
                                             **{attr.name: None for attr in spec.attrs})
            except TypeError as exc:
                report['errors'].append(dict(rule_id='PREFLIGHT-BIND', operator=declaration.name,
                                             detail=str(exc)))
        report['source_checked'] += 1
