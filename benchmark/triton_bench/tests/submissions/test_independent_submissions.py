# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU interpretation of new source; no Triton compilation or NPU certification."""

from __future__ import annotations

import ast
import hashlib
import inspect
import itertools
import math
from types import SimpleNamespace

import pytest
import torch

from triton_eval.comparison import output_matches
from triton_eval.corpusmod import load_manifest
from triton_eval.independent_submissions import KERNEL_SOURCE, definitions
from triton_eval.official_reference import make_inputs, reference
from triton_eval.package import ingest_zip
from triton_eval.specload import load_tolerance
from triton_eval.submission_preflight import submission_readiness
from triton_eval.submission_simulation import DEFAULT_CORPUS, DEFAULT_SOURCE, _entries, build_suite, scenarios


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor = tensor
        self.offset = offset

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)


def storage(pointer):
    tensor = pointer.tensor
    return torch.as_strided(tensor, (tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,), 0)


def load(pointer, mask=True, other=0):
    offsets = torch.as_tensor(pointer.offset) + pointer.tensor.storage_offset()
    selected = torch.broadcast_to(torch.as_tensor(mask), offsets.shape)
    output = torch.full(offsets.shape, other, dtype=pointer.tensor.dtype)
    output[selected] = storage(pointer)[offsets[selected]]
    return output


def store(pointer, value, mask=True):
    offsets = torch.as_tensor(pointer.offset) + pointer.tensor.storage_offset()
    selected = torch.broadcast_to(torch.as_tensor(mask), offsets.shape)
    data = torch.broadcast_to(torch.as_tensor(value), offsets.shape).to(pointer.tensor.dtype)
    storage(pointer)[offsets[selected]] = data[selected]


@pytest.fixture
def interpreted():
    program = [0, 0]

    class Kernel:
        def __init__(self, function):
            self.function = function

        def __call__(self, *arguments):
            return self.function(*arguments)

        def __getitem__(self, grid):
            def launch(*arguments):
                converted = [Pointer(value) if isinstance(value, torch.Tensor) else value for value in arguments]
                for coordinates in itertools.product(*(range(count) for count in grid)):
                    program[:] = list(coordinates) + [0] * (2 - len(coordinates))
                    self.function(*converted)
            return launch

    def scan(values, axis, combine):
        assert axis == 0
        output = tuple(value.clone() for value in values)
        distance = 1
        while distance < output[0].numel():
            combined = combine(*(value[:-distance] for value in output),
                               *(value[distance:] for value in output))
            output = tuple(torch.cat((value[:distance], update)) for value, update in zip(output, combined))
            distance *= 2
        return output

    language = SimpleNamespace(
        constexpr=object(), program_id=lambda axis: program[axis], arange=torch.arange,
        float32=torch.float32, int64=torch.int64, load=load, store=store,
        abs=torch.abs, exp=torch.exp, where=torch.where,
        sum=lambda value, axis: torch.sum(value, dim=axis),
        min=lambda value, axis: torch.min(value, dim=axis).values,
        max=lambda value, axis: torch.max(value, dim=axis).values,
        full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
        minimum=lambda first, second: torch.minimum(torch.as_tensor(first), torch.as_tensor(second)),
        dot=lambda first, second: first.float() @ second.float(), associative_scan=scan)
    compiler = SimpleNamespace(jit=Kernel, cdiv=lambda size, tile: (size + tile - 1) // tile,
                               next_power_of_2=lambda size: 1 << (size - 1).bit_length())
    tree = ast.parse(KERNEL_SOURCE.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(torch=torch, triton=compiler, tl=language, math=math)
    exec(compile(tree, str(KERNEL_SOURCE), 'exec'), namespace)
    return namespace


MANIFEST = load_manifest(DEFAULT_CORPUS / 'manifest.json')
SMALL_OPERATORS = {'Abs', 'Neg', 'Exp', 'Add', 'Softmax', 'ReduceSum', 'Transpose', 'Matmul'}
SMALL_CASES = [case for case in MANIFEST.cases if case.operator in SMALL_OPERATORS and math.prod(case.shape) <= 1024]


@pytest.mark.parametrize('entry', ['absolute', 'neg', 'exp', 'add'])
def test_pointwise_launch_uses_scalar_runtime_count(interpreted, monkeypatch, entry):
    kernel = interpreted['_pointwise_kernel']
    signature = inspect.signature(kernel.function)
    launches = []

    class CheckedKernel:
        @staticmethod
        def __getitem__(grid):
            def launch(*arguments):
                bound = signature.bind(*arguments)
                for name, value in bound.arguments.items():
                    assert not isinstance(value, tuple), f'Grouped kernel argument: {name}'
                assert type(bound.arguments['count']) is int
                assert bound.arguments['count'] == 257
                assert len(bound.arguments) == 10
                launches.append(grid)
                return kernel[grid](*arguments)
            return launch

    monkeypatch.setitem(interpreted, '_pointwise_kernel', CheckedKernel())
    first = torch.linspace(-1, 1, 514)[1::2]
    second = torch.linspace(0, 1, 771)[2::3]
    original = first.clone()
    if entry == 'add':
        actual = interpreted[entry](first, second)
        expected = first + second
    else:
        actual = interpreted[entry](first)
        expected = {'absolute': torch.abs, 'neg': torch.neg, 'exp': torch.exp}[entry](first)
    assert launches == [(2,)]
    assert torch.equal(first, original)
    assert torch.allclose(actual, expected)


@pytest.mark.parametrize('entry,attributes', [
    ('absolute', {}), ('neg', {}), ('add', {}), ('exp', {}),
    ('exp', dict(base=2.0, scale=0.5, shift=-0.25)),
    ('exp', dict(base=10.0, scale=-0.5, shift=0.125)),
    ('exp', dict(base=0.5, scale=0.0, shift=1.0)),
])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
@pytest.mark.parametrize('count', [1, 255, 256, 257, 513])
def test_pointwise_uses_only_scalar_launch_arguments(interpreted, monkeypatch, entry, attributes, dtype, count):
    kernel = interpreted['_pointwise_kernel']
    signature = inspect.signature(kernel.function)
    launches = []

    class CheckedKernel:
        @staticmethod
        def __getitem__(grid):
            def launch(*arguments):
                bound = signature.bind(*arguments)
                for name, value in bound.arguments.items():
                    assert isinstance(value, torch.Tensor) or type(value) in (int, float), name
                assert bound.arguments['count'] == count
                assert bound.arguments['first_step'] == 2
                assert bound.arguments['second_step'] == (3 if entry == 'add' else 2)
                expected_operation = {'absolute': 0, 'neg': 1, 'add': 2, 'exp': 3}[entry]
                assert bound.arguments['operation'] == expected_operation
                launches.append(grid)
                return kernel[grid](*arguments)
            return launch

    monkeypatch.setitem(interpreted, '_pointwise_kernel', CheckedKernel())
    first = torch.linspace(-1, 1, count * 2 + 1, dtype=dtype)[1::2]
    second = torch.linspace(0, 1, count * 3 + 2, dtype=dtype)[2::3]
    originals = first.clone(), second.clone()
    if entry == 'add':
        actual = interpreted[entry](first, second)
        expected = (first.float() + second.float()).to(dtype)
    else:
        actual = interpreted[entry](first, **attributes)
        if entry == 'exp':
            exponent = first.float() * attributes.get('scale', 1.0) + attributes.get('shift', 0.0)
            base = attributes.get('base', -1.0)
            expected = torch.exp(exponent * (1.0 if base == -1.0 else math.log(base))).to(dtype)
        else:
            expected = {'absolute': torch.abs, 'neg': torch.neg}[entry](first)
    assert launches == [((count + 255) // 256,)]
    torch.testing.assert_close(actual, expected)
    assert actual.shape == first.shape and actual.dtype == dtype and actual.is_contiguous()
    assert torch.equal(first, originals[0]) and torch.equal(second, originals[1])


def test_pointwise_scalar_signature_respects_quality_limit_and_fixed_tile():
    tree = ast.parse(KERNEL_SOURCE.read_text())
    kernel = next(node for node in tree.body
                  if isinstance(node, ast.FunctionDef) and node.name == '_pointwise_kernel')
    fields = ['first_step', 'second_step', 'operation', 'scale', 'shift', 'logarithm']
    assert [argument.arg for argument in kernel.args.args] == ['first', 'second', 'output', 'count', *fields]
    assert not kernel.args.posonlyargs and not kernel.args.kwonlyargs
    assert kernel.args.vararg is None and kernel.args.kwarg is None
    for argument in kernel.args.args[4:]:
        assert isinstance(argument.annotation, ast.Attribute)
        assert isinstance(argument.annotation.value, ast.Name) and argument.annotation.value.id == 'tl'
        assert argument.annotation.attr == 'constexpr'
    tile = kernel.body[0]
    assert isinstance(tile, ast.AnnAssign)
    assert isinstance(tile.target, ast.Name) and tile.target.id == 'tile'
    assert isinstance(tile.annotation, ast.Attribute)
    assert isinstance(tile.annotation.value, ast.Name) and tile.annotation.value.id == 'tl'
    assert tile.annotation.attr == 'constexpr'
    assert isinstance(tile.value, ast.Constant) and tile.value.value == 256


@pytest.mark.parametrize('case', SMALL_CASES, ids=lambda case: case.code)
def test_new_source_matches_existing_frozen_reference_on_cpu(interpreted, case):
    from triton_eval.official_catalog import submission_entry_name
    from triton_eval.operator_registry import by_name

    if case.operator == 'Transpose' and case.dtype == 'int16':
        from triton_eval.coverage_reference import make_inputs as coverage_inputs
        inputs = coverage_inputs('Transpose', seed=case.seed, dtype=torch.int16, shape=case.shape,
                                 device='cpu', attrs=case.attrs)
        expected = inputs[0].permute(*case.attrs['perm']).contiguous()
    else:
        inputs = make_inputs(case.operator, seed=case.seed, dtype=getattr(torch, case.dtype),
                             shape=case.shape, device='cpu', attrs=case.attrs, value_range=case.value_range)
        expected = reference(case.operator, *inputs, **case.attrs)
    originals = [value.clone() for value in inputs]
    entry = submission_entry_name(case.operator, by_name(case.operator).entry)
    actual = interpreted[entry](*inputs, **case.attrs)
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assert actual.device == inputs[0].device and actual.is_contiguous()
    assert output_matches(actual, expected, load_tolerance().for_dtype(case.dtype))
    assert all(torch.equal(before, after) for before, after in zip(originals, inputs))


@pytest.mark.parametrize('shape', [(257,), (3, 7, 17), (2, 3, 5, 17), (2, 4097)])
@pytest.mark.parametrize('dtype', [torch.float16, torch.float32])
def test_flat_strided_input_with_storage_offset_and_distinct_auxiliary_stride(interpreted, shape, dtype):
    count = math.prod(shape)
    backing = torch.randn(count * 2 + 4).to(dtype)
    first = backing[3:3 + count * 2:2].view(shape)
    second = torch.randn(shape).to(dtype)
    original = backing.clone()
    actual = interpreted['add'](first, second)
    torch.testing.assert_close(actual, (first.float() + second.float()).to(dtype), rtol=0, atol=0)
    assert torch.equal(backing, original)


@pytest.mark.parametrize('length', [1, 7, 1024, 1025, 2053])
def test_independent_scan_cross_block_ties_and_tail(interpreted, length):
    inputs = torch.randint(-5, 6, (1, length)).float()
    inputs[0, 0] = -8
    if length > 1024:
        inputs[0, 1023:1025] = -8
    original = inputs.clone()
    values, indices = interpreted['cummin'](inputs, -1)
    expected = torch.cummin(inputs, -1)
    assert torch.equal(values, expected.values)
    assert torch.equal(indices, expected.indices)
    assert indices.dtype == torch.int64 and values.is_contiguous() and indices.is_contiguous()
    assert torch.equal(inputs, original)


@pytest.mark.parametrize('axis', [0, 1, 2, -1])
def test_independent_scan_nonlast_axes_preserve_output_order(interpreted, axis):
    inputs = torch.randint(-3, 4, (3, 5, 17)).float()
    values, indices = interpreted['cummin'](inputs, axis)
    expected = torch.cummin(inputs, axis)
    assert torch.equal(values, expected.values) and torch.equal(indices, expected.indices)


@pytest.mark.parametrize('shape', [(1, 1), (3, 7), (65, 513)])
def test_layout_view_is_legal_even_when_output_is_not_contiguous(interpreted, shape):
    inputs = torch.arange(math.prod(shape), dtype=torch.int16).view(shape)
    actual = interpreted['transpose_view'](inputs, [1, 0])
    assert torch.equal(actual, inputs.T)
    assert actual.untyped_storage().data_ptr() == inputs.untyped_storage().data_ptr()
    if min(shape) > 1:
        assert not actual.is_contiguous()


@pytest.fixture(scope='module')
def built(tmp_path_factory):
    path = tmp_path_factory.mktemp('independent-submissions')
    return path, build_suite(path)


def test_new_implementations_do_not_copy_historical_source_or_import_evaluator(built, tmp_path):
    directory, report = built
    records = [item for item in report['scenarios'] if item['origin'] == 'independent']
    assert len(records) == len(definitions()) == 20
    assert report['independent_source_sha256'] == hashlib.sha256(KERNEL_SOURCE.read_bytes()).hexdigest()
    historical = {hashlib.sha256(path.read_bytes()).hexdigest() for path in DEFAULT_SOURCE.rglob('*.py')}
    for record in records:
        package = ingest_zip(directory / record['package'], tmp_path / record['code'])
        readiness = submission_readiness(package.dest, required_operators=record['operators'])
        assert readiness['ok'], readiness
        requested = record['request_operators'] or record['operators']
        assert record['case_codes'] == [case.code for case in MANIFEST.cases if case.operator in requested]
        for source in package.dest.rglob('*.py'):
            assert hashlib.sha256(source.read_bytes()).hexdigest() not in historical
            assert 'triton_eval' not in source.read_text()
            assert 'coverage-v18' not in source.read_text()
        assert record['expected_score'] == dict(coverage_ops_core=100.0, coverage_strict=100.0)


def test_repeat_is_identical_zip_but_scenarios_remain_distinct(built):
    directory, report = built
    records = {record['code']: record for record in report['scenarios']}
    assert records['NEW-REPEAT-A']['sha256'] == records['NEW-REPEAT-B']['sha256']
    assert (directory / 'NEW-REPEAT-A.zip').read_bytes() == (directory / 'NEW-REPEAT-B.zip').read_bytes()


def test_fixed_negatives_isolate_math_error_and_actual_input_change():
    for code in ['EXEC-WRONG', 'EXEC-INPUT']:
        scenario = next(item for item in scenarios() if item.code == code)
        source = _entries(DEFAULT_SOURCE, scenario)['operators/simulation.py']
        namespace = {}
        exec(source, namespace)
        try:
            exp = namespace['exp']
        except KeyError as exc:
            raise AssertionError(f"Missing exp entry for negative scenario {code}") from exc
        backing = torch.zeros(8, 14)
        inputs = backing[:, ::2]
        original = inputs.clone()
        actual = exp(inputs)
        if code == 'EXEC-WRONG':
            assert actual.is_contiguous() and torch.equal(inputs, original)
            assert not torch.equal(actual, inputs.exp())
        else:
            assert not torch.equal(inputs, original)


def test_all_new_entries_load_with_declared_signatures_using_fake_triton(built, tmp_path, interpreted, monkeypatch):
    import sys

    compiler = interpreted['triton']
    compiler.language = interpreted['tl']
    monkeypatch.setitem(sys.modules, 'triton', compiler)
    monkeypatch.setitem(sys.modules, 'triton.language', interpreted['tl'])
    directory, report = built
    for record in report['scenarios']:
        if record['origin'] != 'independent':
            continue
        package = ingest_zip(directory / record['package'], tmp_path / record['code'])
        readiness = submission_readiness(package.dest, load=True)
        assert readiness['ok'] and readiness['load_state'] == 'PASS', readiness


@pytest.mark.parametrize('depth', [65, 2048, 65600])
def test_matrix_dynamic_reduction_loop_and_tail_use_full_input(interpreted, depth):
    generator = torch.Generator().manual_seed(20261028)
    first = torch.randn((3, depth), generator=generator).half()
    second = torch.randn((depth, 7), generator=generator).half()
    actual = interpreted['matmul'](first, second)
    expected = (first.float() @ second.float()).half()
    assert output_matches(actual, expected, load_tolerance().for_dtype('float16'))
