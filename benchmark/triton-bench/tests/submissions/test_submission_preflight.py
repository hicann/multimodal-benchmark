# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU reproduction of the v0.11 signature rejection and release gates."""
import ast
import importlib.util
import inspect
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import yaml
from triton_eval.fixture_sources import fixture_path

from triton_eval.submission import discover
from triton_eval.submission_preflight import source_callable, submission_readiness
from triton_eval.validator import ValidationResult, validate_load

ROOT = Path(__file__).resolve().parents[2]
SUBMISSION = fixture_path(ROOT / 'examples/coverage-v18-ascend-submission')


def make_submission(tmp_path, source):
    (tmp_path / 'kernel.py').write_text(source)
    payload = dict(submit_id='preflight-test', operators=[dict(name='Neg', entry='neg', path='kernel.py')])
    (tmp_path / 'manifest.yaml').write_text(yaml.safe_dump(payload))
    return tmp_path


def test_previous_release_reproduces_exact_name_error():
    report = submission_readiness(fixture_path(ROOT / 'examples/coverage-v11-ascend-submission'))
    assert report['source_checked'] == report['declared_operators'] == 74
    assert [(e['operator'], e['rule_id']) for e in report['errors']] == [('Neg', 'SUB-LOD-007')]
    assert "['x']" in report['errors'][0]['detail']
    assert "['input_value']" in report['errors'][0]['detail']
    assert report['load_state'] == 'NOT_RUN'


def test_all_current_release_entries_match_without_imports():
    report = submission_readiness(SUBMISSION)
    assert report['ok'], report
    assert report['source_checked'] == report['declared_operators'] == 74
    assert report['load_state'] == 'NOT_RUN'


@pytest.mark.parametrize('definition,rule', [
    ('def neg(input): pass', 'SUB-LOD-007'),
    ('def neg(x, extra): pass', 'SUB-LOD-006'),
    ('def neg(x, /): pass', 'SUB-LOD-009'),
    ('def neg(*, x): pass', 'PREFLIGHT-BIND'),
    ('async def neg(x): pass', 'PREFLIGHT-SOURCE'),
    ('@unknown\ndef neg(x): pass', 'PREFLIGHT-SOURCE'),
    ('def other(x): pass', 'PREFLIGHT-SOURCE'),
])
def test_source_rejects_contract_errors_without_executing_any_code(tmp_path, definition, rule):
    root = make_submission(tmp_path, 'raise RuntimeError("must never import")\n' + definition + '\n')
    report = submission_readiness(root)
    assert not report['ok']
    assert report['errors'][0]['rule_id'] == rule
    assert report['load_state'] == 'NOT_RUN'


@pytest.mark.parametrize('definition', ['def neg(input): pass', 'def neg(x, extra): pass',
                                        'def neg(x, /): pass', 'def neg(x, **attrs): pass'])
def test_source_signature_uses_identical_runtime_rules(tmp_path, monkeypatch, definition):
    root = make_submission(tmp_path, definition + '\n')
    func = source_callable(ast.parse(definition).body[0])
    monkeypatch.setattr('triton_eval.submission.load_operator', lambda *_: SimpleNamespace(func=func))
    result = ValidationResult('runtime-rule-comparison')
    validate_load(discover(root), result)
    source = submission_readiness(root)
    assert [e['rule_id'] for e in source['errors']] == [e.rule_id for e in result.errors]


def test_real_loading_catches_import_error_after_source_passes(tmp_path):
    root = make_submission(tmp_path, 'raise RuntimeError("import failed")\ndef neg(x): pass\n')
    assert submission_readiness(root)['ok']
    report = submission_readiness(root, load=True)
    assert not report['ok'] and report['load_state'] == 'FAIL'
    assert report['errors'][0]['rule_id'] == 'SUB-LOD-005'
    assert 'import failed' in report['errors'][0]['detail']


def test_real_loading_does_not_call_entry(tmp_path):
    root = make_submission(tmp_path, 'def neg(x):\n    raise RuntimeError("entry must never run")\n')
    report = submission_readiness(root, load=True)
    assert report['ok'] and report['load_state'] == 'PASS'


def test_missing_corpus_operator_is_rejected(tmp_path):
    root = make_submission(tmp_path, 'def neg(x): pass\n')
    report = submission_readiness(root, {'Neg', 'Add'})
    assert not report['ok']
    assert report['errors'][0]['rule_id'] == 'PREFLIGHT-MISSING'


def test_builder_rejects_bad_entry_before_reference_gate_or_freeze(tmp_path, monkeypatch):
    spec = importlib.util.spec_from_file_location('contract_gate_builder', ROOT / 'scripts/build_coverage_corpus.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    original = module.shutil.copyfile

    def copy_bad_entry(src, dst, *args, **kwargs):
        result = original(src, dst, *args, **kwargs)
        dst = Path(dst)
        if dst.name == 'implementation_witness_kernels.py':
            dst.write_text(dst.read_text().replace('def neg(x):', 'def neg(input):'))
        return result
    monkeypatch.setattr(module.shutil, 'copyfile', copy_bad_entry)
    called = []
    monkeypatch.setattr('triton_eval.refcheck.check_all', lambda *a, **k: called.append(a))
    corpus = tmp_path / 'corpus'
    with pytest.raises(ValueError, match='SUB-LOD-007.*Neg'):
        module.build(corpus, tmp_path / 'submission')
    assert not called
    assert not (corpus / 'manifest.json').exists()
    assert not (corpus / 'evaluation-assets.json').exists()


@pytest.mark.parametrize('declaration,reference', [
    ('def entry(): pass', lambda: None),
    ('def entry(first, second): pass', lambda first, second: None),
    ('def entry(first=1, second=None): pass', lambda first=1, second=None: None),
    ('def entry(first, /, second=2): pass', lambda first, /, second=2: None),
    ('def entry(first=1, /, second=2): pass', lambda first=1, /, second=2: None),
    ('def entry(first, /, second, *values, required, optional=None, **options): pass',
     lambda first, /, second, *values, required, optional=None, **options: None),
    ('def entry(*, required, optional=3): pass', lambda *, required, optional=3: None),
    ('def entry(*values, **options): pass', lambda *values, **options: None),
    ('def entry(first=1, /, *values, required, optional=2, **options): pass',
     lambda first=1, /, *values, required, optional=2, **options: None),
])
def test_source_parameter_groups_preserve_runtime_binding(declaration, reference):
    runtime = inspect.signature(reference)
    expected = runtime.replace(parameters=[
        parameter.replace(default=None) if parameter.default is not inspect.Parameter.empty else parameter
        for parameter in runtime.parameters.values()
    ])
    actual = inspect.signature(source_callable(ast.parse(declaration).body[0]))
    assert actual == expected
    for positional_count in range(4):
        for keywords in ({}, dict(required=None), dict(first=None), dict(required=None, extra=None)):
            values = [None] * positional_count
            assert binding_outcome(actual, values, keywords) == binding_outcome(expected, values, keywords)


def binding_outcome(signature, values, keywords):
    try:
        return signature.bind(*values, **keywords).arguments
    except TypeError:
        return None


def test_source_signature_never_evaluates_defaults_annotations_or_body():
    declaration = (
        'def entry(first: unknown_type = fail(), /, second=1 / 0, *, required, optional=fail()) -> fail():\n'
        '    raise RuntimeError("must not execute")\n'
    )
    entry = source_callable(ast.parse(declaration).body[0])
    parameters = inspect.signature(entry).parameters
    assert list(parameters) == ['first', 'second', 'required', 'optional']
    assert [parameter.default for parameter in parameters.values()] == [None, None, inspect.Parameter.empty, None]
    assert all(parameter.annotation is inspect.Parameter.empty for parameter in parameters.values())
    with pytest.raises(RuntimeError, match='source preflight never executes an entry'):
        entry()
