# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Check the actual Triton integer algorithm against independent CPU oracles.

The NumPy adapter executes the helper's source with fixed-width vector values;
it does not compile Triton or establish NPU correctness/performance.
"""

import ast
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_eval.fixture_sources import fixture_path

ROOT = Path(__file__).resolve().parents[2]


def _floor_division_function(path):
    definitions = [node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef)]
    return next(node for node in definitions if node.name == "_floor_divide_int32")


@pytest.fixture(params=['official', 'compatibility'])
def integer_division(request):
    np = pytest.importorskip("numpy")

    class Tensor(np.ndarray):
        def to(self, dtype):
            return self.astype(dtype)

    def tensor(value):
        return np.asarray(value).view(Tensor)

    language = SimpleNamespace(
        uint32=np.uint32, int32=np.int32, static_range=range,
        full=lambda shape, value, dtype: tensor(np.full(shape, value, dtype=dtype)),
        where=lambda condition, yes, no: tensor(np.where(condition, yes, no)),
        minimum=lambda first, second: tensor(np.minimum(first, second)),
        maximum=lambda first, second: tensor(np.maximum(first, second)),
    )
    # Execute the same helper that is shipped to the NPU, rather than a
    # separately maintained Python reimplementation of the division algorithm.
    sources = [fixture_path(ROOT / "scripts/templates/official_kernels.py"),
               fixture_path(ROOT / "examples/official-ascend-submission/operators/kernels.py")]
    if request.param == 'compatibility':
        sources = [fixture_path(ROOT / 'scripts/templates/compatibility_kernels.py'),
                   fixture_path(ROOT / 'examples/coverage-v12-ascend-submission/operators/compatibility_kernels.py')]
    functions = [_floor_division_function(path) for path in sources]
    assert ast.dump(functions[0]) == ast.dump(functions[1])
    function = functions[0]
    function.decorator_list = []
    module = ast.fix_missing_locations(ast.Module(body=[function], type_ignores=[]))
    namespace = {"tl": language}
    exec(compile(module, str(sources[0]), "exec"), namespace)
    try:
        floor_divide = namespace["_floor_divide_int32"]
    except KeyError as exc:
        raise AssertionError(f"Missing floor division helper in {sources[0]}") from exc

    def check(first, second):
        first, second = tensor(first), tensor(second)
        actual = floor_divide(first, second)
        # Widen before division so the independent oracle never loses integer
        # precision or overflows while computing a representable result.
        expected = np.floor_divide(first.astype(np.int64), second.astype(np.int64))
        np.testing.assert_array_equal(actual, expected)
        assert actual.dtype == np.int32

    return check


def test_full_int32_boundaries_and_both_divisor_signs(integer_division):
    np = pytest.importorskip("numpy")
    values = np.array([-2147483648, -2147483647, -16777217, -1025, -128,
                       -7, -1, 0, 1, 7, 127, 1024, 16777217,
                       2147483646, 2147483647], dtype=np.int32)
    divisors = np.array([-2147483648, -2147483647, -127, -16, -3, -2, -1,
                         1, 2, 3, 16, 127, 2147483647], dtype=np.int32)
    first, second = np.meshgrid(values, divisors)
    legal = ~((first == -2147483648) & (second == -1))
    integer_division(first[legal], second[legal])


def test_random_full_width_and_small_divisors(integer_division):
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(20261024)
    first = rng.integers(-2147483648, 2147483648, 30000, dtype=np.int32)
    second = rng.integers(-2147483648, 2147483648, 30000, dtype=np.int32)
    second[:10000] = rng.integers(-16, 17, 10000, dtype=np.int32)
    second[second == 0] = 1
    second[(first == -2147483648) & (second == -1)] = 1
    integer_division(first, second)


def test_exhaustive_small_signed_operands(integer_division):
    np = pytest.importorskip('numpy')
    values = np.arange(-257, 258, dtype=np.int32)
    divisors = values[values != 0]
    first, second = np.meshgrid(values, divisors)
    integer_division(first, second)


@pytest.mark.torch
def test_all_frozen_floor_division_inputs(integer_division):
    from triton_eval.corpusmod import load_manifest
    from triton_eval.inputs import _dtype_of, make_inputs_for
    from triton_eval.official_catalog import VERSION
    from triton_eval.official_reference import bind

    _, factory = bind("FloorDiv")
    cases = load_manifest(ROOT / "corpus" / VERSION / "manifest.json").cases
    checked = 0
    for case in cases:
        if case.operator != "FloorDiv":
            continue
        inputs = make_inputs_for(case, _dtype_of(case.dtype), "cpu", case.attrs,
                                 make_inputs=factory)
        integer_division(*(value.numpy() for value in inputs))
        checked += 1
    assert checked == 20


@pytest.mark.npu
@pytest.mark.parametrize("stride", [1, 2])
def test_npu_full_int32_boundaries_and_signed_divisors(stride):
    import torch
    import torch_npu  # noqa: F401 — registers the NPU device
    from triton_eval.official_reference_cpu import reference as independent
    from triton_eval.submission import discover, load_all

    values = [-2147483648, -2147483647, -16777217, -7, -1, 0, 1, 7,
              16777217, 2147483646, 2147483647]
    divisors = [-2147483648, -127, -16, -3, -2, -1, 1, 2, 3, 16, 127, 2147483647]
    from itertools import product

    pairs = product(values, divisors)
    pairs = [pair for pair in pairs if pair != (-2147483648, -1)]
    first_cpu = torch.tensor([[a for a, _ in pairs]], dtype=torch.int32)
    second_cpu = torch.tensor([[b for _, b in pairs]], dtype=torch.int32)
    first = torch.empty((1, len(pairs) * stride), dtype=torch.int32,
                        device="npu:0")[:, ::stride]
    first.copy_(first_cpu.to("npu:0"))
    second = second_cpu.to("npu:0")
    submission = load_all(discover(fixture_path(ROOT / "examples/coverage-v12-ascend-submission")))
    assert "FloorDiv" in submission.loaded, submission.failures
    actual = submission.loaded["FloorDiv"].func(first, second)
    expected = independent("FloorDiv", first_cpu, second_cpu)
    assert actual.dtype == torch.int32
    assert torch.equal(actual.cpu(), expected)
