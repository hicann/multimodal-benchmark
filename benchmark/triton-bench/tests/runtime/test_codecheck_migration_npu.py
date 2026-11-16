# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Real device regression of all six renamed APIs; no synthetic device results."""

import functools
import hashlib
import importlib.util
import sys
from pathlib import Path

import pytest

from triton_eval.fixture_sources import fixture_path
from triton_eval.operator_registry import by_name

ROOT = Path(__file__).resolve().parents[2]
pytestmark = pytest.mark.npu
ENTRIES = [
    ('SwiGlu', 'examples/demo-submission/operators/swi_glu/kernel.py'),
    ('ArgMax', 'examples/adversarial-submissions/ADV-C1-fullhouse/operators/scalar_kernels.py'),
    ('SwiGlu', 'examples/adversarial-submissions/ADV-C1-fullhouse/operators/scalar_kernels.py'),
    ('Cummin', 'scripts/templates/coverage_kernels.py'),
    ('CrossEntropyLoss', 'scripts/templates/coverage_kernels.py'),
    ('Cummin', 'scripts/templates/independent_submission_kernels.py'),
]


@functools.lru_cache(maxsize=None)
def implementation(relative):
    source = fixture_path(ROOT / relative)
    name = 'quality_' + hashlib.sha256(relative.encode()).hexdigest()[:12]
    spec = importlib.util.spec_from_file_location(name, source)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def swiglu_reference(value, variant):
    import torch.nn.functional as functional

    attributes = dict(dim=-1 if variant != 1 else 1)
    first, second = value.chunk(2, dim=attributes['dim'])
    return attributes, functional.silu(first) * second


def argmax_reference(value, variant):
    import torch

    attributes = dict(dim=(None, -1, 0)[variant], keepdim=variant == 2)
    return attributes, torch.argmax(value, **attributes)


def cummin_reference(value, variant):
    import torch

    attributes = dict(dim=(-1, 0, 1)[variant])
    return attributes, torch.cummin(value, **attributes)


def cross_entropy_reference(value, variant):
    import torch
    import torch.nn.functional as functional

    target = torch.tensor([0, -100, 6], dtype=torch.int64, device=value.device)
    attributes = dict(target=target, reduction=('none', 'sum', 'mean')[variant], ignore_index=-100)
    return attributes, functional.cross_entropy(value, **attributes)


REFERENCES = dict(SwiGlu=swiglu_reference, ArgMax=argmax_reference, Cummin=cummin_reference,
                  CrossEntropyLoss=cross_entropy_reference)


def assert_device_outputs(actual, expected, dtype_name):
    import torch

    actual_parts = actual if isinstance(actual, (tuple, list)) else (actual,)
    expected_parts = expected if isinstance(expected, (tuple, list)) else (expected,)
    assert len(actual_parts) == len(expected_parts)
    tolerance = dict(bfloat16=2e-2, float16=2e-3, float32=1e-5)[dtype_name]
    for output, reference in zip(actual_parts, expected_parts):
        assert output.device.type == 'npu' and output.dtype == reference.dtype
        assert output.shape == reference.shape
        if reference.is_floating_point():
            torch.testing.assert_close(output.cpu(), reference.cpu(), rtol=tolerance, atol=tolerance)
        else:
            assert torch.equal(output.cpu(), reference.cpu())


@pytest.mark.parametrize('operator,relative', ENTRIES)
@pytest.mark.parametrize('dtype_name', ['float16', 'bfloat16', 'float32'])
@pytest.mark.parametrize('variant', [0, 1, 2])
def test_renamed_api_matches_device_reference(operator, relative, dtype_name, variant):
    import torch

    torch.manual_seed(20261109 + variant)
    dtype = getattr(torch, dtype_name)
    function = getattr(implementation(relative), by_name(operator).entry)
    shape = (3, 14) if operator == 'SwiGlu' else (3, 7)
    value = (torch.arange(shape[0] * shape[1], device='npu:0') % 11 - 5).reshape(shape).to(dtype) / 8
    attributes, expected = REFERENCES[operator](value, variant)
    # Check the public keyword name and the positional call used by the evaluator.
    for actual in (function(input_value=value, **attributes), function(value, **attributes)):
        torch.npu.synchronize()
        assert_device_outputs(actual, expected, dtype_name)
