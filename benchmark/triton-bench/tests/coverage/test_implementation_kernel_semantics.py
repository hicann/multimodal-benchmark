# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU tensor simulation of the actual new kernel source's indexing and arithmetic.

This verifies semantic plans without importing Triton, compiling IR, or claiming
hardware coverage. Device tests exercise compilation and real tensor bindings.
"""
import ast
import itertools
from pathlib import Path
from types import SimpleNamespace

from kernel_simulation import constant_arguments

import pytest
import torch
from triton_eval.fixture_sources import fixture_path

SOURCE = fixture_path(Path(__file__).resolve().parents[2] / 'scripts/templates/implementation_witness_kernels.py')


class Pointer:
    def __init__(self, tensor, offset=0):
        self.tensor, self.offset = tensor, offset

    def __add__(self, offset):
        return Pointer(self.tensor, self.offset + offset)


def flat_storage(tensor):
    return torch.as_strided(tensor, (tensor.untyped_storage().nbytes() // tensor.element_size(),), (1,), 0)


def load(pointer, valid=True, other=0):
    offset = torch.as_tensor(pointer.offset)
    valid = torch.broadcast_to(torch.as_tensor(valid), offset.shape)
    result = torch.full(offset.shape, other, dtype=pointer.tensor.dtype)
    result[valid] = flat_storage(pointer.tensor)[offset[valid]]
    return result


def store(pointer, value, valid=True):
    offset = torch.as_tensor(pointer.offset)
    valid = torch.broadcast_to(torch.as_tensor(valid), offset.shape)
    value = torch.broadcast_to(value, offset.shape).to(pointer.tensor.dtype)
    flat_storage(pointer.tensor)[offset[valid]] = value[valid]


@pytest.fixture
def cpu_kernels(request):
    pid = [0, 0, 0]

    class Kernel:
        def __init__(self, fn):
            self.fn = fn

        def __call__(self, *args, **kwargs):
            return self.fn(*constant_arguments(self.fn, args), **kwargs)

        def __getitem__(self, grid):
            def launch(*args, **kwargs):
                converted = [Pointer(a) if isinstance(a, torch.Tensor) else a for a in args]
                for coords in itertools.product(*(range(n) for n in grid)):
                    pid[:] = [*coords, *([0] * (3 - len(coords)))]
                    self(*converted)
            return launch
    triton = SimpleNamespace(jit=Kernel, cdiv=lambda a, b: (a + b - 1) // b,
                             next_power_of_2=lambda n: 1 << (n - 1).bit_length())
    tl = SimpleNamespace(constexpr=object(), program_id=lambda axis: pid[axis], arange=torch.arange,
                         load=load, store=store, float32=torch.float32, where=torch.where, exp=torch.exp, log=torch.log,
                         max=lambda x, axis: torch.max(x, dim=axis).values, sum=lambda x, axis: torch.sum(x, dim=axis),
                         full=lambda shape, value, dtype: torch.full(shape, value, dtype=dtype),
                         dot=lambda first, second: first.to(torch.float32) @ second.to(torch.float32),
                         int32=torch.int32, static_range=range,
                         minimum=lambda first, second: torch.minimum(first, torch.as_tensor(second, dtype=first.dtype)),
                         maximum=lambda first, second: torch.maximum(first, torch.as_tensor(second, dtype=first.dtype)))
    source = getattr(request, 'param', SOURCE)
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(torch=torch, triton=triton, tl=tl)
    exec(compile(tree, str(source), 'exec'), namespace)
    return namespace


@pytest.mark.parametrize('shape', [(2, 4097),
                          (2, 4096),
                          (3, 7, 17),
                          (2, 3, 513),
                          (1, 1, 1),
                          (2, 3, 5, 17),
                          (3, 2, 7, 513)])
@pytest.mark.parametrize('strided', [False, True])
def test_neg_actual_source_covers_elements_once_with_correct_offsets(cpu_kernels, shape, strided):
    shape = list(shape)
    physical = shape[:-1] + [shape[-1] * (2 if strided else 1)]
    x = torch.randn(physical)
    if strided:
        x = x[..., ::2]
    result = cpu_kernels['neg'](x)
    assert torch.equal(result, -x)


@pytest.mark.parametrize('dim', [0, 1, 2, -1])
@pytest.mark.parametrize('strided', [False, True])
@pytest.mark.parametrize('mask_dtype', [torch.bool, torch.int8, torch.uint8])
def test_single_kernel_masked_softmax_matches_independent_torch(cpu_kernels, dim, strided, mask_dtype):
    x = torch.randn(3, 5, 34 if strided else 17)
    if strided:
        x = x[..., ::2]
    mask = torch.randint(0, 2, x.shape, dtype=mask_dtype)
    # Ensure a nonempty reduction line without relying on input data values.
    axis = dim % x.ndim
    index = [slice(None)] * 3
    index[axis] = 0
    mask[tuple(index)] = 1
    result = cpu_kernels['masked_softmax'](x, mask, dim)
    expected = torch.softmax(x.masked_fill(mask == 0, -float('inf')), dim=dim)
    torch.testing.assert_close(result, expected, rtol=1e-6, atol=1e-7)


COMPATIBILITY = SOURCE.with_name('compatibility_kernels.py')


@pytest.mark.parametrize('cpu_kernels', [COMPATIBILITY], indirect=True)
@pytest.mark.parametrize('groups', [(3, 5, 8), (8, 8), (1, 17, 3), (19,), (2, 1, 5, 3)])
@pytest.mark.parametrize('dimensions', [(32, 16), (33, 19)])
@pytest.mark.parametrize('dtype', [torch.float32, torch.float16, torch.bfloat16])
def test_grouped_matmul_scalar_metadata_matches_each_independent_group(cpu_kernels, groups, dimensions, dtype):
    reduction, columns = dimensions
    generator = torch.Generator().manual_seed(20261026)
    first = torch.randn((sum(groups), reduction), generator=generator).to(dtype)
    weights = torch.randn((len(groups), reduction, columns), generator=generator).to(dtype)
    bias = torch.randn((len(groups), columns), generator=generator).to(dtype)
    originals = [value.clone() for value in (first, weights, bias)]
    actual = cpu_kernels['grouped_matmul'](first, weights, bias, list(groups))
    expected = torch.cat([(chunk.float() @ weight.float() + offset.float()).to(dtype)
                          for chunk, weight, offset in zip(first.split(groups), weights, bias)])
    from triton_eval.comparison import output_matches
    from triton_eval.specload import load_tolerance
    tolerance = load_tolerance().for_dtype(str(dtype).removeprefix('torch.'))
    assert output_matches(actual, expected, tolerance)
    assert all(torch.equal(before, after) for before, after in zip(originals, (first, weights, bias)))


@pytest.mark.parametrize('cpu_kernels', [COMPATIBILITY], indirect=True)
@pytest.mark.parametrize('strides', [(1, 1), (2, 1), (1, 2), (2, 3)])
def test_signed_floor_div_wrapper_tail_and_independent_input_strides(cpu_kernels, strides):
    first_stride, second_stride = strides
    first = torch.randint(-2147483648, 2147483647, (3, 257 * first_stride), dtype=torch.int32)[:, ::first_stride]
    second = torch.randint(-16, 17, (3, 257 * second_stride), dtype=torch.int32)[:, ::second_stride]
    second[second == 0] = 1
    first[0, 0] = -2147483648
    second[0, 0] = 1
    actual = cpu_kernels['floor_div'](first, second)
    expected = torch.div(first.long(), second.long(), rounding_mode='floor').int()
    assert torch.equal(actual, expected)


def test_new_jit_source_avoids_failed_unsigned_cast_and_constexpr_tuple_paths():
    tree = ast.parse(COMPATIBILITY.read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    division = ast.unparse(functions['_floor_divide_int32'])
    assert all(word not in division for word in ('uint', 'int64', 'float', '/', '**'))
    grouped = functions['_grouped']
    assert not any(isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)
                   and node.value.id in ('GROUPS', 'group_list') for node in ast.walk(grouped))
    assert all(argument.arg not in ('GROUPS', 'group_list') for argument in grouped.args.args)
    assert 'tl.static_range(0, reduction, bk)' in ast.unparse(grouped)


@pytest.mark.parametrize('cpu_kernels', [SOURCE.with_name('stability_kernels.py')], indirect=True)
@pytest.mark.parametrize('reduction', ['mean', 'sum'])
@pytest.mark.parametrize('offset', [0, 100, 1000000])
def test_stable_cross_entropy_actual_source_preserves_reference_and_inputs(cpu_kernels, reduction, offset):
    from triton_eval.comparison import output_matches
    from triton_eval.specload import load_tolerance

    generator = torch.Generator().manual_seed(20261026)
    data = torch.randn((7, 33), generator=generator) + offset
    target = torch.randint(0, 33, (7,), generator=generator)
    target[0] = -100
    original = data.clone()
    actual = cpu_kernels['cross_entropy_loss'](data, target, reduction, -100)
    expected = torch.nn.functional.cross_entropy(data, target, reduction=reduction, ignore_index=-100)
    assert output_matches(actual, expected, load_tolerance().for_dtype('float32'))
    assert torch.equal(data, original)
