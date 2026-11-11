# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Cross-operator regressions for packaging, compile-time metadata and output precedence."""

import ast
import json
from types import MethodType, SimpleNamespace

import pytest

from triton_eval.fixture_sources import ROOT, fixture_path, require_source_dependencies
from triton_eval.judge import CaseMeasurement, decide
from triton_eval.models import CaseStatus
from triton_eval.specload import load_tolerance
from triton_eval.validator import check_output


@pytest.mark.parametrize('source', [
    'def entry(value): return omitted(value)',
    'def entry(value):\n    def nested(): return missing\n    return nested()',
    'values = [missing(value) for value in range(3)]',
    'def entry(): return [missing(value) for value in range(3)]',
    'def entry(): return undeclared',
    'class Holder:\n    def entry(self): return missing',
])
def test_dependency_gate_rejects_global_reads_even_in_nested_scopes(source):
    """Dependency gate rejects global reads even in nested scopes."""
    with pytest.raises(ValueError, match='Unbound fixture dependencies'):
        require_source_dependencies(source, 'generated.py')


@pytest.mark.parametrize('source', [
    'from math import sin\ndef entry(value): return sin(value)',
    'def entry(value): return helper(value)\ndef helper(value): return value',
    'def entry(value):\n    offset = 1\n    def nested(): return value + offset\n    return nested()',
    'values = [value for value in range(3)]',
    'def entry(value): return len(value)',
    'def entry(): raise RuntimeError("not executed")',
    'raise RuntimeError("not imported")\ndef entry(): return __file__',
])
def test_dependency_gate_respects_python_scopes_without_executing_source(source):
    """Dependency gate respects python scopes without executing source."""
    require_source_dependencies(source, 'generated.py')


def test_reference_helper_is_present_in_every_materialized_dispatch_module():
    """Reference helper is present in every materialized dispatch module."""
    layout = json.loads((ROOT / 'scripts/example_sources/layout.json').read_text())
    checked = 0
    for name in layout['files']:
        if not name.startswith('examples/'):
            continue
        source = fixture_path(ROOT / name).read_text()
        require_source_dependencies(source, name)
        functions = {node.name: node for node in ast.parse(source).body if isinstance(node, ast.FunctionDef)}
        if '_validate_dispatch_inputs' in functions:
            assert 'validate_elementwise_layout' in functions
            checked += 1
    assert checked == 11


def kernel_functions(source):
    """Kernel functions."""
    functions = []
    for node in ast.parse(source).body:
        if not isinstance(node, ast.FunctionDef):
            continue
        if any(ast.unparse(decorator) == 'triton.jit' for decorator in node.decorator_list):
            functions.append(node)
    return functions


def assert_constexpr_extraction(function):
    """Assert constexpr extraction."""
    constants = set()
    for argument in function.args.args:
        if argument.annotation and ast.unparse(argument.annotation) == 'tl.constexpr':
            constants.add(argument.arg)
    checked = 0
    for statement in function.body:
        if isinstance(statement, ast.Assign) and isinstance(statement.value, ast.Name):
            assert statement.value.id not in constants or not isinstance(statement.targets[0], ast.Tuple)
        if not isinstance(statement, ast.AnnAssign) or not isinstance(statement.value, ast.Subscript):
            continue
        member = statement.value.value
        if isinstance(member, ast.Name):
            assert member.id not in constants, 'constexpr wrappers must not be directly subscripted'
            continue
        if not isinstance(member, ast.Attribute) or not isinstance(member.value, ast.Name):
            continue
        if member.value.id not in constants:
            continue
        assert member.attr == 'value'
        assert ast.unparse(statement.annotation) == 'tl.constexpr'
        checked += 1
    return checked


def test_all_shared_kernels_preserve_constexpr_when_reading_grouped_metadata():
    """All shared kernels preserve constexpr when reading grouped metadata."""
    checked = 0
    for folder in ('scripts/templates', 'scripts/example_sources'):
        for path in (ROOT / folder).glob('*.py'):
            checked += sum(assert_constexpr_extraction(function) for function in kernel_functions(path.read_text()))
    assert checked == 52


def runtime_function(name):
    """Runtime function."""
    source = ROOT / 'src/triton_eval/harness/npu_exec.py'
    functions = [node for node in ast.parse(source.read_text()).body if isinstance(node, ast.FunctionDef)]
    function = next(node for node in functions if node.name == name)
    return ast.Module(body=[function], type_ignores=[])


@pytest.mark.torch
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('expected_count,actual_count', [(1, 2), (2, 1), (2, 3)])
def test_runtime_arity_status_is_independent_of_layout(strided, expected_count, actual_count):
    """Runtime arity status is independent of layout."""
    import time
    import torch
    from triton_eval import comparison, inputs

    tensor = torch.arange(16, dtype=torch.float32).reshape(2, 8)[:, ::2] if strided else torch.ones(2, 4)
    actual = tuple(tensor for _ in range(actual_count))
    expected = tuple(torch.ones(2, 4) for _ in range(expected_count))
    namespace = dict(__package__='triton_eval.harness', time=time, tensor_fp=inputs.tensor_fp,
                     as_outputs=inputs.as_outputs, CaseMeasurement=CaseMeasurement, compare=comparison.compare,
                     load_tolerance=load_tolerance, _allclose=comparison.allclose,
                     output_matches=comparison.output_matches)
    for name in ('_check_case_output_contract', '_compare_case_outputs', '_check_output_dtypes'):
        exec(compile(runtime_function(name), '<runtime-contract>', 'exec'), namespace)
    case = SimpleNamespace(code='ARITY', operator='Exp', dtype='float32')
    measurement = CaseMeasurement(case.code, case.operator)
    options = SimpleNamespace(category='Elementwise')
    assert namespace['_check_case_output_contract'](
        case, options, [tensor], [inputs.tensor_fp(tensor)], (actual, expected), measurement, time.time())
    namespace['_compare_case_outputs'](case, actual, expected, measurement)
    assert '输出元数不符' in measurement.compile_err
    assert decide(measurement, load_tolerance().for_dtype('float32')) == CaseStatus.SHAPE_MISMATCH
    assert measurement.output_ok


@pytest.mark.torch
@pytest.mark.parametrize('output_kind,rule', [('layout', 'SUB-EXE-002'), ('device', 'SUB-EXE-006'),
                                              ('type', 'SUB-EXE-001')])
def test_arity_policy_does_not_remove_output_security_checks(output_kind, rule):
    """Arity policy does not remove output security checks."""
    import torch

    tensor = torch.ones(2, 8)[:, ::2]
    output = (tensor, tensor)
    if output_kind == 'device':
        output = (tensor, torch.ones(2, 4, device='meta'))
    elif output_kind == 'type':
        output = (tensor, object())
    expected_count = 2 if output_kind == 'layout' else 1
    errors = check_output('Exp', output, 'Elementwise', expected_device=torch.device('cpu'),
                          expected_count=expected_count)
    assert [error.rule_id for error in errors] == [rule]


@pytest.mark.torch
@pytest.mark.parametrize('mode', ['half', 'interleaved'])
@pytest.mark.parametrize('columns', [8, 48, 128])
def test_rope_host_launch_defines_power_of_two_block(mode, columns):
    """Rope host launch defines power of two block."""
    import torch

    path = fixture_path(ROOT / 'examples/demo-submission/operators/apply_rotary_pos_emb/kernel.py')
    functions = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)]
    functions = [node for node in functions if not node.decorator_list]
    calls = []

    class KernelRecorder:
        @staticmethod
        def __getitem__(grid):
            """Return a launch recorder without invoking a compiler."""
            def launch(*arguments):
                """Launch."""
                calls.append((grid, arguments))
            return launch

    namespace = dict(torch=torch, triton=SimpleNamespace(next_power_of_2=lambda value: 1 << (value - 1).bit_length()),
                     _rope_kernel=KernelRecorder())
    exec(compile(ast.Module(body=functions, type_ignores=[]), '<rope-host>', 'exec'), namespace)
    value = torch.ones(2, 3, columns)
    cos = torch.ones(2, 3, columns // 2)
    result = namespace['apply_rotary_pos_emb'](value, cos, cos, rotary_mode=mode)
    assert result.shape == value.shape and result.is_contiguous()
    assert len(calls) == 1
    assert calls[0][1][-1] == (mode == 'interleaved', max(16, 1 << (columns - 1).bit_length()))


@pytest.mark.torch
@pytest.mark.parametrize('output', [[], (), [[1]], 1, {'value': 1}])
def test_invalid_container_structure_is_rejected_without_normalization_errors(output):
    """Invalid container structure is rejected without normalization errors."""
    errors = check_output('Exp', output, 'Elementwise', expected_count=1)
    assert [error.rule_id for error in errors] == ['SUB-EXE-001']


@pytest.mark.torch
def test_tensor_nesting_and_metadata_cannot_hide_output_arity():
    """Tensor nesting and metadata cannot hide output arity."""
    import torch

    value = torch.ones(4)
    for output in ([[value]], (value, {'metadata': 1})):
        errors = check_output('Exp', output, 'Elementwise', expected_count=1)
        assert [error.rule_id for error in errors] == ['SUB-EXE-001']


@pytest.mark.parametrize('payload_access', [False, True])
@pytest.mark.parametrize('assignments', [
    'first: tl.constexpr = configuration[1]',
    'first: tl.constexpr = configuration[-1]',
    'first: tl.constexpr = configuration[True]',
    'first: tl.constexpr = configuration[unknown]',
    'first: tl.constexpr = configuration[0]\n    second: tl.constexpr = configuration[0]',
])
def test_evidence_parameter_groups_reject_unordered_or_noninteger_members(assignments, payload_access):
    """Evidence parameter groups reject unordered or noninteger members."""
    from triton_eval.reviewed_kernels import parameter_groups

    if payload_access:
        assignments = assignments.replace('configuration[', 'configuration.value[')
    function = ast.parse('def kernel(configuration):\n    ' + assignments).body[0]
    with pytest.raises(ValueError):
        parameter_groups(function)


def test_dependency_read_does_not_treat_an_unbound_module_reference_as_a_definition():
    """Dependency read does not treat an unbound module reference as a definition."""
    with pytest.raises(ValueError, match='missing'):
        require_source_dependencies('missing\ndef entry(): return missing', 'generated.py')


@pytest.mark.torch
def test_security_device_check_precedes_unrelated_layout_error():
    """Security device check precedes unrelated layout error."""
    import torch

    output = (torch.ones(2, 8)[:, ::2], torch.ones(2, 4, device='meta'))
    errors = check_output('Exp', output, 'Elementwise', expected_device=torch.device('cpu'), expected_count=2)
    assert [error.rule_id for error in errors] == ['SUB-EXE-006']


@pytest.mark.torch
@pytest.mark.parametrize('capture_enabled', [False, True])
@pytest.mark.parametrize('output_kind', ['scalar', 'empty', 'nested', 'mixed', 'valid'])
def test_capture_does_not_change_output_contract_classification(capture_enabled, output_kind):
    """Instrumentation must never classify malformed return values as execution failures."""
    from triton_eval.inputs import as_outputs
    import torch

    value = torch.ones(4)
    outputs = dict(scalar=7, empty=[], nested=[[value]], mixed=(value, object()), valid=value)
    actual = outputs[output_kind]
    bound = []
    observation = SimpleNamespace(payload=dict(diagnostics=[]), bind_outputs=bound.append) if capture_enabled else None
    namespace = dict(as_outputs=as_outputs)
    exec(compile(runtime_function('_bind_completed_observation'), '<capture-contract>', 'exec'), namespace)
    namespace['_bind_completed_observation'](observation, actual)
    errors = check_output('Exp', actual, 'Elementwise', expected_device=value.device, expected_count=1)
    assert [error.rule_id for error in errors] == ([] if output_kind == 'valid' else ['SUB-EXE-001'])
    if capture_enabled:
        assert observation.payload['synchronized'] is True
        assert bool(observation.payload['diagnostics']) is (output_kind in {'scalar', 'empty', 'nested'})


def test_nontensor_simulation_requires_type_contract_not_instrumentation_failure():
    """The negative scenario demands the specific type violation rather than any failure."""
    from triton_eval.submission_simulation import scenarios

    scenario = next(item for item in scenarios() if item.code == 'EXEC-NONTENSOR')
    assert scenario.statuses == ('SUBMISSION_INVALID',) and scenario.rule == 'SUB-EXE-001'


def grouped_metadata_in_file(path):
    """Collect the reviewed constexpr-reading kernels from one maintained source."""
    selected = []
    for function in kernel_functions(path.read_text()):
        if assert_constexpr_extraction(function):
            selected.append((path.name, function))
    return selected


def grouped_metadata_functions():
    """Enumerate maintained kernels that read compile-time tuple payloads."""
    selected = []
    for folder in ('scripts/templates', 'scripts/example_sources'):
        for path in sorted((ROOT / folder).glob('*.py')):
            selected.extend(grouped_metadata_in_file(path))
    return selected


def test_grouped_metadata_collection_preserves_kernel_and_member_coverage():
    """Keep all thirteen kernels and fifty-two member reads in compiler test coverage."""
    functions = grouped_metadata_functions()
    assert len(functions) == 13
    assert len({(name, function.name) for name, function in functions}) == 13
    assert sum(assert_constexpr_extraction(function) for _, function in functions) == 52


@pytest.mark.triton
@pytest.mark.parametrize('source_name,function', grouped_metadata_functions(),
                         ids=[item[0] + ':' + item[1].name for item in grouped_metadata_functions()])
@pytest.mark.parametrize('wrapped_members', [False, True])
def test_installed_frontend_extracts_constexpr_payload(monkeypatch, source_name, function, wrapped_members):
    """Use installed compiler visitors, not Python execution, to check member extraction."""
    from triton.compiler.code_generator import CodeGenerator
    from triton import language as tl

    visitor = object.__new__(CodeGenerator)
    visitor.lscope = {'tl': tl}
    visitor.dereference_name = visitor.lscope.__getitem__
    monkeypatch.setattr(visitor, 'visit', MethodType(ast.NodeVisitor.visit, visitor))
    members = [node for node in function.body if isinstance(node, ast.AnnAssign)]
    values = (64, 'Exp', -1.0, True) * 4
    payload = tuple(tl.constexpr(value) for value in values) if wrapped_members else values
    constants = [arg.arg for arg in function.args.args if arg.annotation]
    for group in constants:
        visitor.lscope[group] = tl.constexpr(payload)
    assert source_name
    for statement in members:
        visitor.visit(statement)
        result = visitor.lscope[statement.target.id]
        assert isinstance(result, tl.constexpr)
        expected = values[statement.value.slice.value]
        assert repr(result.value) == repr(expected)


@pytest.mark.parametrize('access', ['configuration', 'configuration.value'])
def test_reviewed_group_layout_keeps_historical_and_payload_bindings(access):
    """Recognize both forms without relaxing the full-source review gate."""
    from triton_eval.reviewed_kernels import parameter_groups

    source = 'def kernel(configuration):\n    first: tl.constexpr = ' + access + '[0]'
    function = ast.parse(source).body[0]
    assert parameter_groups(function) == {'configuration': ['first']}


@pytest.mark.parametrize('access', ['configuration.other', 'holder.configuration.value'])
def test_reviewed_group_layout_does_not_accept_unrecognized_attributes(access):
    """Only a direct group name and its explicit value attribute have a binding."""
    from triton_eval.reviewed_kernels import parameter_groups

    source = 'def kernel(configuration):\n    first: tl.constexpr = ' + access + '[0]'
    assert not parameter_groups(ast.parse(source).body[0])


@pytest.mark.parametrize('argument_count', [1, 3])
def test_cpu_kernel_simulation_does_not_drop_extra_or_missing_arguments(argument_count):
    """A simulation must reject arity differences instead of truncating the call."""
    from kernel_simulation import constant_arguments

    def kernel(pointer, configuration):
        """Provide the exact signature whose arity is under test."""
        return pointer, configuration

    with pytest.raises(ValueError):
        constant_arguments(kernel, tuple(range(argument_count)))


def test_cpu_kernel_simulation_keeps_runtime_tuples_and_wraps_only_annotated_tuples():
    """Numerical simulation keeps geometry runtime-bound and configuration constant."""
    from kernel_simulation import constant_arguments

    def kernel(geometry, configuration: 'tl.constexpr', scalar: 'tl.constexpr'):
        """Declare runtime and compile-time metadata without importing Triton."""
        return geometry, configuration, scalar

    geometry, configuration = (1, 2), (32, 'Exp')
    actual = constant_arguments(kernel, (geometry, configuration, 64))
    assert actual[0] is geometry and actual[1].value is configuration
    assert actual[2] == 64
