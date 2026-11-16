# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Regression checks for shared sources and complete independent submissions."""

import ast
import hashlib
import importlib.util
import json
import shutil
from pathlib import Path

import pytest

from triton_eval import fixture_sources

ROOT = Path(__file__).resolve().parents[2]
LAYOUT = json.loads((ROOT / fixture_sources.LAYOUT).read_text())


@pytest.mark.parametrize('logical', sorted(LAYOUT['files']))
def test_materialized_sources_bind_ast_and_preserve_jit_definitions(logical):
    _, layout = fixture_sources.registry(ROOT)
    record = layout['files'][logical]
    source = fixture_sources.assemble(ROOT, layout, logical).decode()
    assert fixture_sources.ast_hash(source) == record['assembled_ast_sha256']
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            assert not (node.module or '').startswith(fixture_sources.PREFIXES)
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name, expected in record['jit_asts'].items():
        actual = hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest()
        assert actual == expected
    assert fixture_sources.fixture_path(ROOT / logical).read_text() == source


def local_registry(tmp_path):
    root = tmp_path / 'trusted'
    source = root / 'scripts/example_sources/component.py'
    source.parent.mkdir(parents=True)
    source.write_text('def kernel(value):\n    return value + 1\n')
    logical = 'examples/example/kernel.py'
    layout = {
        'format': 1,
        'sources': {source.relative_to(root).as_posix(): hashlib.sha256(source.read_bytes()).hexdigest()},
        'files': {logical: {
            'source': source.relative_to(root).as_posix(),
            'assembled_ast_sha256': fixture_sources.ast_hash(source.read_text()),
            'jit_asts': {},
        }},
    }
    (root / fixture_sources.LAYOUT).write_text(json.dumps(layout))
    return root, source, logical, layout


@pytest.mark.parametrize('damage', ['modified', 'missing', 'source-symlink', 'layout-symlink', 'traversal'])
def test_registry_fails_closed_for_changed_or_escaping_sources(tmp_path, damage):
    root, source, _, layout = local_registry(tmp_path)
    if damage == 'modified':
        source.write_text('def kernel(value):\n    return value - 1\n')
    elif damage == 'missing':
        source.unlink()
    elif damage == 'source-symlink':
        outside = tmp_path / 'outside.py'
        shutil.copyfile(source, outside)
        source.unlink()
        source.symlink_to(outside)
    elif damage == 'layout-symlink':
        outside = tmp_path / 'outside.json'
        shutil.move(root / fixture_sources.LAYOUT, outside)
        (root / fixture_sources.LAYOUT).symlink_to(outside)
    else:
        layout['sources']['../outside.py'] = '0' * 64
        (root / fixture_sources.LAYOUT).write_text(json.dumps(layout))
    with pytest.raises((ValueError, FileNotFoundError)):
        fixture_sources.registry(root)


def test_assembly_rejects_unregistered_imports_and_definition_drift(tmp_path):
    root, source, logical, layout = local_registry(tmp_path)
    source.write_text('from scripts.example_sources.missing import kernel\n')
    with pytest.raises(ValueError, match='Unregistered'):
        fixture_sources.assemble(root, layout, logical)
    source.write_text('def kernel(value):\n    return value - 1\n')
    with pytest.raises(ValueError, match='changed fixture semantics'):
        fixture_sources.assemble(root, layout, logical)


@pytest.mark.parametrize('declaration,binding', [
    ('import math', 'math'),
    ('import triton.language as tl', 'tl'),
    ('from itertools import product', 'product'),
    ('class TensorSubclass:\n    pass', 'TensorSubclass'),
])
def test_shared_dependencies_are_inlined_as_original_statements(tmp_path, declaration, binding):
    root, source, logical, layout = local_registry(tmp_path)
    dependency = source.with_name('dependencies.py')
    dependency.write_text(declaration + '\n')
    layout['sources'][dependency.relative_to(root).as_posix()] = hashlib.sha256(dependency.read_bytes()).hexdigest()
    source.write_text('from scripts.example_sources.dependencies import ' + binding + ' as ' + binding + '\n')
    layout['files'][logical].pop('statement_order', None)
    layout['files'][logical]['assembled_ast_sha256'] = fixture_sources.ast_hash(declaration)
    assembled = fixture_sources.assemble(root, layout, logical).decode()
    assert ast.dump(ast.parse(assembled)) == ast.dump(ast.parse(declaration))


@pytest.mark.parametrize('declaration,imported,match', [
    ('import math', 'math as renamed', 'Unsupported'),
    ('import math', '*', 'Unsupported'),
    ('import math', 'missing', 'Ambiguous or missing'),
    ('import math\nimport math', 'math', 'Ambiguous or missing'),
    ('import math, os', 'math', 'one binding'),
])
def test_shared_dependencies_reject_renaming_wildcards_and_ambiguous_bindings(tmp_path, declaration, imported, match):
    root, source, logical, layout = local_registry(tmp_path)
    dependency = source.with_name('dependencies.py')
    dependency.write_text(declaration + '\n')
    layout['sources'][dependency.relative_to(root).as_posix()] = hashlib.sha256(dependency.read_bytes()).hexdigest()
    source.write_text('from scripts.example_sources.dependencies import ' + imported + '\n')
    with pytest.raises(ValueError, match=match):
        fixture_sources.assemble(root, layout, logical)


def test_cached_materialization_tampering_is_rejected(tmp_path):
    root, _, logical, _ = local_registry(tmp_path)
    projected = fixture_sources.fixture_path(root / logical, root)
    projected.write_text('def kernel(value):\n    return value - 1\n')
    with pytest.raises(ValueError, match='Materialized fixture changed'):
        fixture_sources.fixture_path(root / logical, root)


def test_external_user_package_cannot_enable_trusted_build_imports(tmp_path):
    root, _, logical, _ = local_registry(tmp_path)
    user_file = root / logical
    user_file.parent.mkdir(parents=True)
    user_file.write_text('from scripts.example_sources.component import kernel\n')
    assert fixture_sources.fixture_path(user_file) == user_file
    assert 'import kernel' in user_file.read_text()


def load_variant(name, suffix):
    path = fixture_sources.fixture_path(ROOT / 'examples/adversarial-submissions' / name / suffix)
    spec = importlib.util.spec_from_file_location('variant_' + name.replace('-', '_'), path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.torch
@pytest.mark.parametrize('axis', [0, 1, -1, -2])
@pytest.mark.parametrize('dtype_name', ['float32', 'float64'])
def test_scalar_softmax_scaffolding_preserves_attack_behaviors(axis, dtype_name):
    import torch

    tensor = torch.tensor([[2, 1, 1], [4, 2, 2]], dtype=getattr(torch, dtype_name))
    expected = torch.softmax(tensor, axis)
    for name in ['ADV-C1-fullhouse', 'ADV-D5-id-cache', 'ADV-D9-tensor-subclass',
                 'ADV-F3-wrong-dtype', 'ADV-F7-dim-blind']:
        suffix = 'operators/scalar_kernels.py' if name == 'ADV-C1-fullhouse' else 'operators/softmax/kernel.py'
        module = load_variant(name, suffix)
        actual = module.softmax(tensor, dim=axis)
        if name == 'ADV-F7-dim-blind':
            target = torch.softmax(tensor, -1)
        else:
            target = expected
        torch.testing.assert_close(actual.to(tensor.dtype), target, check_device=True)
        assert actual.dtype == (torch.float64 if name == 'ADV-F3-wrong-dtype' else tensor.dtype)
        assert (actual.__class__ is not torch.Tensor) == (name == 'ADV-D9-tensor-subclass')
        if name == 'ADV-D5-id-cache':
            before = actual.clone()
            tensor[0, 0] += 2
            assert module.softmax(tensor, dim=axis) is actual
            torch.testing.assert_close(actual, before)
            tensor[0, 0] -= 2


@pytest.mark.torch
@pytest.mark.parametrize('axis', [0, 1, -1, -2])
def test_scalar_cummin_retains_ties_values_and_deliberately_bad_arity(axis):
    import torch

    tensor = torch.tensor([[2., 1., 1.], [4., 2., 2.]])
    values, indices = torch.cummin(tensor, axis)
    normal = load_variant('ADV-C1-fullhouse', 'operators/scalar_kernels.py').cummin(tensor, dim=axis)
    torch.testing.assert_close(normal, (values, indices))
    wrong_indices = load_variant('ADV-F1-garbage-indices', 'operators/cummin/kernel.py').cummin(tensor, dim=axis)
    torch.testing.assert_close(wrong_indices[0], values)
    assert torch.count_nonzero(wrong_indices[1]) == 0
    single = load_variant('ADV-F2-single-output', 'operators/cummin/kernel.py').cummin(tensor, dim=axis)
    assert isinstance(single, torch.Tensor)
    torch.testing.assert_close(single, values)


def test_historical_tutorial_is_the_only_maintained_loader_source():
    record = LAYOUT['files']['examples/ascend-ops-submission/_tut.py']
    assert record['source'] == 'corpus/v0.6-ascend-ops/_tut.py'


def test_exception_compatibility_keeps_canonical_class_identity(tmp_path):
    from triton_eval import validator

    path = ROOT / 'src/triton_eval/harness/npu_exec.py'
    tree = ast.parse(path.read_text())
    definitions = [node for node in tree.body if isinstance(node, (ast.ClassDef, ast.FunctionDef))]
    tree.body = [node for node in definitions if node.name in ('EvaluationTimeoutError', '__getattr__')]
    selected = tmp_path / 'exception_compatibility_only.py'
    selected.write_text(ast.unparse(tree))
    spec = importlib.util.spec_from_file_location('exception_compatibility_only', selected)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    assert module.TimeoutError_ is module.EvaluationTimeoutError
    assert validator.ValidationError_ is validator.SubmissionValidationError
    with pytest.raises(AttributeError):
        assert validator.missing_exception is None


def test_missing_torch_verifier_returns_nonzero_without_import_time_exit(monkeypatch, capsys):
    import sys

    path = ROOT / 'scripts/verify_refs.py'
    spec = importlib.util.spec_from_file_location('verify_refs_no_torch', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, 'torch', None)
    assert module.main() == 2
    assert 'torch' in capsys.readouterr().err


def test_materialization_preserves_comment_only_security_evidence():
    from triton_eval.validator import ValidationResult, validate_static

    package = fixture_sources.fixture_path(ROOT / 'examples/adversarial-submissions/ADV-E5-comment-falsepositive')
    source = (package / 'operators/exp/kernel.py').read_text()
    assert '# 提示：不要用 data_ptr 做缓存键' in source
    result = ValidationResult(submit_id='comment-only-control')
    validate_static(package, result)
    assert 'SUB-STR-003' not in [error.rule_id for error in result.errors]


def test_cached_output_never_bypasses_source_digest_validation(tmp_path):
    root, source, logical, _ = local_registry(tmp_path)
    fixture_sources.fixture_path(root / logical, root)
    source.write_text('def kernel(value):\n    return value - 1\n')
    with pytest.raises(ValueError, match='identity differs'):
        fixture_sources.fixture_path(root / logical, root)


def is_export_assignment(node):
    if not isinstance(node, ast.Assign):
        return False
    names = [target.id for target in node.targets if isinstance(target, ast.Name)]
    return '__all__' in names


@pytest.mark.parametrize('source,expected', [
    ('__all__ = []', True),
    ('other = __all__ = []', True),
    ('other = []', False),
    ('instance.__all__ = []', False),
    ('__all__: list[str] = []', False),
    ('import torch', False),
])
def test_export_assignment_selector_preserves_top_level_assignment_semantics(source, expected):
    assert is_export_assignment(ast.parse(source).body[0]) is expected


def test_explicit_export_order_keeps_future_imports_first_and_imports_executable():
    checked = 0
    for name in LAYOUT['sources']:
        if not name.startswith(('scripts/example_sources/', 'scripts/templates/')):
            continue
        source = (ROOT / name).read_text()
        tree = ast.parse(source)
        exports = [node for node in tree.body if is_export_assignment(node)]
        if not exports:
            continue
        assert len(exports) == 1
        assert all(isinstance(name, str) for name in ast.literal_eval(exports[0].value))
        for node in tree.body:
            if isinstance(node, ast.ImportFrom) and node.module == '__future__':
                assert node.lineno < exports[0].lineno
            elif isinstance(node, (ast.Import, ast.ImportFrom)):
                assert node.lineno > exports[0].end_lineno
        compile(source, name, 'exec')
        checked += 1
    assert checked == 19


def reference_functions(source):
    torch = pytest.importorskip('torch')
    tree = ast.parse(source)
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = {'torch': torch}
    exec(compile(tree, '<reference-differential>', 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('splits,scale', [(1, 0.0), (4, 0.0), (4, 0.25)])
@pytest.mark.parametrize('strided', [False, True])
def test_attention_reference_refactor_is_bit_exact_for_split_boundaries(splits, scale, strided):
    torch = pytest.importorskip('torch')
    receipt = json.loads((ROOT / 'corpus/reference-revisions/20261108-quality-repairs.json').read_text())
    record = next(item for item in receipt['references'] if 'decode_grouped_attention' in item['path'])
    before = reference_functions(record['before_source'])['reference']
    after = reference_functions((ROOT / record['path']).read_text())['reference']
    generator = torch.Generator().manual_seed(23)
    stride = 2 if strided else 1
    query = torch.randn(4, 2, 8 * stride, generator=generator).to(torch.bfloat16)[..., ::stride]
    keys = torch.randn(115, 1, 8 * stride, generator=generator).to(torch.bfloat16)[..., ::stride]
    values = torch.randn(115, 1, 5 * stride, generator=generator).to(torch.bfloat16)[..., ::stride]
    offsets = torch.tensor([0, 0, 17, 50, 115], dtype=torch.int32)
    arguments = (query, keys, values, offsets, torch.arange(115), torch.full((4,), splits))
    expected = before(*arguments, max_kv_splits=splits, sm_scale=scale)
    actual = after(*arguments, max_kv_splits=splits, sm_scale=scale)
    for original, repaired in zip(expected, actual):
        torch.testing.assert_close(repaired, original, rtol=0, atol=0)


def reference_ast_without_function_docstrings(source):
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and ast.get_docstring(node) is not None:
            node.body = node.body[1:]
    return ast.dump(tree, include_attributes=False)


def test_reference_docstring_normalization_keeps_executable_changes_visible():
    before = 'def reference(value):\n    """old description"""\n    return value + 1\n'
    documentation = before.replace('old description', 'new description')
    executable = before.replace('value + 1', 'value - 1')
    original = reference_ast_without_function_docstrings(before)
    assert reference_ast_without_function_docstrings(documentation) == original
    assert reference_ast_without_function_docstrings(executable) != original


def test_document_only_reference_changes_preserve_all_executable_statements():
    receipt = json.loads((ROOT / 'corpus/reference-revisions/20261108-quality-repairs.json').read_text())
    for record in receipt['references']:
        if '/legacy/' not in record['path']:
            continue
        before = reference_ast_without_function_docstrings(record['before_source'])
        after = reference_ast_without_function_docstrings((ROOT / record['path']).read_text())
        assert before == after


def test_trusted_raw_example_path_passes_the_real_source_preflight():
    from triton_eval.submission_preflight import submission_readiness

    report = submission_readiness(ROOT / 'examples/coverage-v18-ascend-submission')
    assert report['ok'], report['errors']
    assert report['declared_operators'] == report['source_checked'] == 74


def test_trusted_raw_positive_submission_validates_without_manual_projection():
    from triton_eval.validator import validate_submission

    result = validate_submission(ROOT / 'examples/adversarial-submissions/ADV-C1-fullhouse')
    assert result.errors == []
