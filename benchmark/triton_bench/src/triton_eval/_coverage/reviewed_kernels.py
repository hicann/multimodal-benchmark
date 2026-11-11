# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Bind flat and grouped reviewed parameters without changing raw evidence."""

from __future__ import annotations

import ast
import hashlib
import json

from triton_eval.fixture_sources import fixture_path

from . import PROJECT_ROOT
from .coverage_evidence import source_fingerprints

ROOT = PROJECT_ROOT
REGISTRY_SHA256 = "5b8a79eb79972dfeac808a42b2d198625fc76daae02adc7aae224d1c22fe9067"


def _registry():
    raw = (ROOT / "assets/reviewed-kernels.json").read_bytes()
    if hashlib.sha256(raw).hexdigest() != REGISTRY_SHA256:
        raise ValueError("historical reviewed-source registry identity differs")
    return json.loads(raw)["kernels"]


def reviewed_source(source, name):
    """Only exact historical or current reviewed ASTs are accepted."""
    record = _registry().get(name)
    if record is None:
        return False
    actual = source_fingerprints(source)
    if actual == source_fingerprints(record["source"]):
        return True
    nodes = ast.parse((fixture_path(ROOT / "scripts/templates" / record["template"])).read_text()).body
    expected = next((node for node in nodes if isinstance(node, ast.FunctionDef) and node.name == name), None)
    return expected is not None and actual == source_fingerprints(ast.unparse(expected))


def _member_group(expression):
    """Recognize a direct historical group or explicit constexpr payload access."""
    if isinstance(expression, ast.Name):
        return expression.id
    if isinstance(expression, ast.Attribute) and expression.attr == 'value':
        if isinstance(expression.value, ast.Name):
            return expression.value.id
    return None


def parameter_groups(function):
    """Recognize reviewed historical groups and explicitly typed payload members."""
    groups = {}
    for statement in function.body:
        if isinstance(statement, ast.Assign) and len(statement.targets) == 1:
            target = statement.targets[0]
            if not isinstance(target, ast.Tuple) or not isinstance(statement.value, ast.Name):
                break
            if not all(isinstance(item, ast.Name) for item in target.elts):
                raise ValueError("unsupported reviewed parameter group")
            groups[statement.value.id] = [item.id for item in target.elts]
        elif isinstance(statement, ast.AnnAssign) and isinstance(statement.value, ast.Subscript):
            member = statement.value
            group_name = _member_group(member.value)
            if group_name is None or ast.unparse(statement.annotation) != 'tl.constexpr':
                break
            if not isinstance(member.slice, ast.Constant):
                raise ValueError("unsupported reviewed parameter index")
            if not isinstance(member.slice.value, int) or isinstance(member.slice.value, bool):
                raise ValueError("unsupported reviewed parameter index")
            names = groups.setdefault(group_name, [])
            if member.slice.value != len(names) or not isinstance(statement.target, ast.Name):
                raise ValueError("nonsequential reviewed parameter index")
            names.append(statement.target.id)
        else:
            break
    return groups


def _parameter_layout(function, roles):
    groups = parameter_groups(function)
    layout, position = {}, 0
    for argument in function.args.args:
        size = len(groups.get(argument.arg, [argument.arg]))
        if size < 1 or position + size > len(roles):
            raise ValueError("reviewed kernel parameter count differs")
        if argument.arg in groups and (argument.annotation is None or
                                      ast.unparse(argument.annotation) != 'tl.constexpr'):
            raise ValueError("runtime parameter groups are not reviewed")
        layout[argument.arg] = roles[position:position + size]
        position += size
    if position != len(roles):
        raise ValueError("reviewed kernel parameter count differs")
    return layout, groups


def _expanded_arguments(arguments, layout, groups):
    result = {}
    for name, roles in layout.items():
        if name not in groups:
            if name in arguments:
                result[roles[0]] = arguments[name]
            continue
        record = arguments.get(name, {})
        if not isinstance(record, dict):
            raise ValueError("invalid reviewed constexpr tuple record")
        items = record.get('items')
        if record.get('kind') != 'tuple' or record.get('constexpr') is not True:
            raise ValueError("missing reviewed constexpr tuple binding")
        if not isinstance(items, list) or len(items) != len(roles):
            raise ValueError("reviewed constexpr tuple size differs")
        for role, item in zip(roles, items):
            if not isinstance(item, dict) or item.get('kind') != 'scalar' or item.get('constexpr') is not True:
                raise ValueError("unobserved reviewed constexpr tuple member")
            if type(item.get('value')) not in (bool, int, float, str):
                raise ValueError("invalid reviewed constexpr tuple value")
            result[role] = dict(item)
    return result


def canonical_launch(launch):
    """Return an analysis view with immutable semantic roles for parameters.

    Positional correspondence is used only after the complete function AST
    matches a reviewed source. Captured source, IR and raw artifacts stay intact.
    """
    name = launch.get("name")
    source = launch.get("source_text", "")
    if not reviewed_source(source, name):
        raise ValueError("unreviewed pattern kernel source")
    nodes = [node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef) and node.name == name]
    if len(nodes) != 1:
        raise ValueError("ambiguous reviewed kernel definition")
    roles = _registry()[name]["parameters"]
    layout, groups = _parameter_layout(nodes[0], roles)
    arguments = launch.get("arguments", {})
    runtime = launch.get("runtime_arg_names", [])
    if any(key not in layout for key in arguments) or any(key not in layout for key in runtime):
        raise ValueError("kernel argument is absent from reviewed signature")
    if len(set(runtime)) != len(runtime):
        raise ValueError("duplicate runtime argument binding")
    if any(name in groups for name in runtime):
        raise ValueError("constexpr tuple incorrectly marked as runtime")
    result = dict(launch)
    result["arguments"] = _expanded_arguments(arguments, layout, groups)
    result["runtime_arg_names"] = [layout[key][0] for key in runtime]
    return result
