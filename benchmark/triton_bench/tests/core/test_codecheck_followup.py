# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Follow-up checks for optimized worker guards and the device-test refactoring."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
PROBE = ROOT / 'tests/submissions/worker_restart_probe.py'
PROBE_PROGRAM = """
import json
import runpy
import sys
namespace = runpy.run_path(sys.argv[1])
namespace['main'].__globals__['EvaluationService'] = lambda *args, **kwargs: None
current, phase = json.loads(sys.argv[2])
def intercept(patch, service, callback):
    callback(current, phase, [])
namespace['main'].__globals__['intercept_engine_processes'] = intercept
namespace['main'](['unused', 'unused', 'expected', 'unused'])
"""


@pytest.mark.parametrize('current,phase,message', [
    ({'id': 'other', 'status': 'RUNNING'}, 'run', 'unexpected task or phase'),
    ({'id': 'expected', 'status': 'RUNNING'}, 'compile', 'unexpected task or phase'),
    ({'id': 'expected', 'status': 'QUEUED'}, 'run', 'must be RUNNING'),
])
def test_worker_guards_survive_optimized_python(current, phase, message):
    environment = dict(os.environ)
    environment['PYTHONPATH'] = os.pathsep.join([str(ROOT / 'src'), str(PROBE.parent)])
    result = subprocess.run([sys.executable, '-O', '-c', PROBE_PROGRAM, str(PROBE),
                             json.dumps([current, phase])], env=environment,
                            capture_output=True, text=True, timeout=30)
    assert result.returncode != 0
    assert 'AssertionError: ' in result.stderr and message in result.stderr


def load_device_tests():
    path = ROOT / 'tests/runtime/test_codecheck_migration_npu.py'
    spec = importlib.util.spec_from_file_location('quality_device_helpers', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.torch
# Local Torch does not implement CPU float16 cross entropy; device coverage stays 54 cases.
@pytest.mark.parametrize('operator,dtype_name', [
    ('SwiGlu', 'float16'), ('SwiGlu', 'bfloat16'), ('SwiGlu', 'float32'),
    ('ArgMax', 'float16'), ('ArgMax', 'bfloat16'), ('ArgMax', 'float32'),
    ('Cummin', 'float16'), ('Cummin', 'bfloat16'), ('Cummin', 'float32'),
    ('CrossEntropyLoss', 'float32'),
])
@pytest.mark.parametrize('variant', [0, 1, 2])
def test_split_reference_builders_preserve_attributes_and_math(operator, variant, dtype_name):
    torch = pytest.importorskip('torch')
    helpers = load_device_tests()
    shape = (3, 14) if operator == 'SwiGlu' else (3, 7)
    value = (torch.arange(shape[0] * shape[1]) % 11 - 5).reshape(shape).to(getattr(torch, dtype_name)) / 8
    attributes, actual = helpers.REFERENCES[operator](value, variant)
    if operator == 'SwiGlu':
        assert attributes == dict(dim=-1 if variant != 1 else 1)
        first, second = value.chunk(2, dim=attributes['dim'])
        expected = torch.nn.functional.silu(first) * second
    elif operator == 'ArgMax':
        assert attributes == dict(dim=None if variant == 0 else (-1 if variant == 1 else 0),
                                  keepdim=variant == 2)
        expected = torch.argmax(value, **attributes)
    elif operator == 'Cummin':
        assert attributes == dict(dim=-1 if variant == 0 else variant - 1)
        expected = torch.cummin(value, **attributes)
    else:
        assert torch.equal(attributes['target'], torch.tensor([0, -100, 6]))
        assert attributes['reduction'] == ('none', 'sum', 'mean')[variant]
        assert attributes['ignore_index'] == -100
        expected = torch.nn.functional.cross_entropy(value, **attributes)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
