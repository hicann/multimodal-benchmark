# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Materialize trusted repository fixtures into independent, complete source trees.

Imports between maintained fixture modules are build-time source inclusions.
Uploaded ZIP packages never use this registry or import from the repository.
"""

from __future__ import annotations

import ast
import atexit
import builtins
import hashlib
import io
import json
import shutil
import symtable
import tempfile
import tokenize
from functools import lru_cache
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = 'scripts/example_sources/layout.json'
PREFIXES = ('scripts.templates.', 'scripts.example_sources.', 'src.triton_eval.inputs')


def safe_source(root, name):
    relative = PurePosixPath(name)
    if not name or relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Unsafe fixture source path: ' + name)
    if '\\' in name or relative.as_posix() != name:
        raise ValueError('Unsafe fixture source path: ' + name)
    path = root / name
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Fixture source escapes its root: ' + name)
    if any(part.is_symlink() for part in (path, *path.parents) if part.is_relative_to(root)):
        raise ValueError('Fixture sources cannot be symbolic links: ' + name)
    return path


def ast_hash(source):
    return hashlib.sha256(ast.dump(ast.parse(source), include_attributes=False).encode()).hexdigest()


def registry(root):
    content = safe_source(root, LAYOUT).read_bytes()
    layout = json.loads(content)
    if layout.get('format') != 1 or not layout.get('files') or not layout.get('sources'):
        raise ValueError('Invalid fixture source layout')
    for name, expected in layout['sources'].items():
        source = safe_source(root, name)
        if not source.is_file() or hashlib.sha256(source.read_bytes()).hexdigest() != expected:
            raise ValueError('Fixture source identity differs: ' + name)
    for name, record in layout['files'].items():
        safe_source(root, name)
        if record['source'] not in layout['sources']:
            raise ValueError('Unregistered fixture source')
    return content, layout


def defines_binding(node, name):
    if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
        return node.name == name
    if isinstance(node, ast.Import):
        return any((alias.asname or alias.name.split('.')[0]) == name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        return any((alias.asname or alias.name) == name for alias in node.names)
    return False


def imported_definition(root, layout, module, name):
    path = module.replace('.', '/') + '.py'
    if path not in layout['sources']:
        raise ValueError('Unregistered build-time import: ' + module)
    source = safe_source(root, path).read_text(encoding='utf-8')
    nodes = [node for node in ast.parse(source).body if defines_binding(node, name)]
    if len(nodes) != 1:
        raise ValueError('Ambiguous or missing fixture definition: ' + module + '.' + name)
    node = nodes[0]
    if isinstance(node, (ast.Import, ast.ImportFrom)) and len(node.names) != 1:
        raise ValueError('Shared fixture dependencies require one binding per declaration')
    start = min([node.lineno, *[part.lineno for part in getattr(node, 'decorator_list', [])]]) - 1
    return ''.join(source.splitlines(True)[start:node.end_lineno]).rstrip() + '\n'


def referenced_globals(scope):
    """Collect global reads recursively without importing or executing a fixture."""
    names = {symbol.get_name() for symbol in scope.get_symbols() if symbol.is_global() and symbol.is_referenced()}
    for child in scope.get_children():
        names.update(referenced_globals(child))
    return names


def require_source_dependencies(source, filename):
    """Reject incomplete trusted generated modules; this is not a user-code sandbox."""
    scope = symtable.symtable(source, filename, 'exec')
    bindings = {symbol.get_name() for symbol in scope.get_symbols() if symbol.is_assigned() or symbol.is_imported()}
    implicit = {'__name__', '__file__', '__package__', '__doc__', '__builtins__', '__spec__', '__annotations__'}
    missing = referenced_globals(scope) - bindings - set(dir(builtins)) - implicit
    if missing:
        raise ValueError(f'Unbound fixture dependencies in {filename}: {sorted(missing)}')


def assemble(root, layout, name):
    record = layout['files'][name]
    source = safe_source(root, record['source']).read_text(encoding='utf-8')
    lines = source.splitlines(True)
    for node in reversed(ast.parse(source).body):
        if not isinstance(node, ast.ImportFrom) or not (node.module or '').startswith(PREFIXES):
            continue
        if node.level or any(alias.asname not in (None, alias.name) or alias.name == '*' for alias in node.names):
            raise ValueError('Unsupported fixture build-time import')
        definitions = [imported_definition(root, layout, node.module, alias.name) for alias in node.names]
        lines[node.lineno - 1:node.end_lineno] = ['\n'.join(definitions)]
    result = ''.join(lines)
    if 'statement_order' in record:
        result = ordered_source(result, record['statement_order'])
    if ast_hash(result) != record['assembled_ast_sha256']:
        raise ValueError('Materialization changed fixture semantics: ' + name)
    if name.startswith('examples/'):
        require_source_dependencies(result, name)
    return result.encode('utf-8')


def ordered_source(source, order):
    tree = ast.parse(source)
    lines = source.splitlines(True)
    statements = {}
    covered = set()
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(isinstance(target, ast.Name) and target.id == '__all__'
                                                for target in node.targets):
            continue
        start = min([node.lineno, *[part.lineno for part in getattr(node, 'decorator_list', [])]])
        covered.update(range(start, node.end_lineno + 1))
        key = hashlib.sha256(ast.dump(node, include_attributes=False).encode()).hexdigest()
        statements.setdefault(key, []).append(''.join(lines[start - 1:node.end_lineno]).rstrip())
    body = []
    for key in order:
        if not statements.get(key):
            raise ValueError('Fixture statement was removed or changed')
        body.append(statements[key].pop(0))
    if any(statements.values()):
        raise ValueError('Fixture has unexpected statements')
    comments = [token.string for token in tokenize.generate_tokens(io.StringIO(source).readline)
                if token.type == tokenize.COMMENT and token.start[0] not in covered]
    return '\n'.join(comments) + '\n\n' + '\n\n'.join(body) + '\n'


def copy_fixture_inputs(root, destination):
    for name in ('examples', 'scripts/templates', 'reference'):
        source = root / name
        if not source.is_dir():
            continue
        for path in sorted(source.rglob('*')):
            if '__pycache__' in path.parts or path.suffix == '.pyc' or not path.is_file():
                continue
            relative = path.relative_to(root).as_posix()
            safe_source(root, relative)
            target = safe_source(destination, relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)


@lru_cache(maxsize=128)
def expected_source(root, content, name):
    return assemble(root, json.loads(content), name)


@lru_cache(maxsize=1)
def materialized(root, content):
    layout = json.loads(content)
    destination = Path(tempfile.mkdtemp(prefix='triton-fixture-sources-'))
    try:
        copy_fixture_inputs(root, destination)
        for name in layout['files']:
            target = safe_source(destination, name)
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(expected_source(root, content, name))
    except BaseException:
        shutil.rmtree(destination)
        raise
    atexit.register(shutil.rmtree, destination, ignore_errors=True)
    return destination


def fixture_files(names, root=ROOT):
    """Validate one trusted source snapshot and resolve a batch of logical files."""
    if not (root / LAYOUT).is_file():
        return {}
    content, layout = registry(root)
    destination = materialized(root, content)
    outputs = {}
    for name in names:
        if name not in layout['files']:
            continue
        target = safe_source(destination, name)
        if target.read_bytes() != expected_source(root, content, name):
            raise ValueError('Materialized fixture changed: ' + name)
        outputs[name] = target
    return outputs


def fixture_path(path, root=None):
    """Resolve only trusted local example/template paths; leave user inputs alone."""
    root = ROOT if root is None else Path(root).resolve()
    path = Path(path).absolute()
    if not path.is_relative_to(root):
        return path
    relative = path.relative_to(root).as_posix()
    if not (relative == 'examples' or relative.startswith(('examples/', 'scripts/templates/'))
            or relative == 'scripts/templates'):
        return path
    if not (root / LAYOUT).is_file():
        return path
    content, layout = registry(root)
    destination = materialized(root, content)
    for name in layout['files']:
        if name == relative or name.startswith(relative + '/'):
            target = safe_source(destination, name)
            if target.read_bytes() != expected_source(root, content, name):
                raise ValueError('Materialized fixture changed: ' + name)
    return safe_source(destination, relative)


def main():
    """Export a standalone example without modifying the maintained source tree."""
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('destination', type=Path)
    args = parser.parse_args()
    source = args.source.absolute()
    if not source.is_relative_to(ROOT / 'examples'):
        parser.error('source must be a trusted repository example directory')
    source = fixture_path(source)
    if not source.is_dir():
        parser.error('source must be a directory')
    if args.destination.exists():
        parser.error('destination already exists; nothing overwritten')
    shutil.copytree(source, args.destination)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
