# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""The renamed API remains strict for maintained and exported submissions."""

import ast
import copy
import inspect
from pathlib import Path

import pytest

from triton_eval.fixture_sources import fixture_path
from triton_eval.operator_registry import by_name
from triton_eval.submission_preflight import source_callable
from triton_eval.validator import signature_errors

ROOT = Path(__file__).resolve().parents[2]
ENTRIES = [
    ('SwiGlu', 'examples/demo-submission/operators/swi_glu/kernel.py'),
    ('ArgMax', 'examples/adversarial-submissions/ADV-C1-fullhouse/operators/scalar_kernels.py'),
    ('SwiGlu', 'examples/adversarial-submissions/ADV-C1-fullhouse/operators/scalar_kernels.py'),
    ('Cummin', 'scripts/templates/coverage_kernels.py'),
    ('CrossEntropyLoss', 'scripts/templates/coverage_kernels.py'),
    ('Cummin', 'scripts/templates/independent_submission_kernels.py'),
]


@pytest.mark.parametrize('operator,relative', ENTRIES)
def test_renamed_contract_accepts_keyword_tensor_and_rejects_other_names(operator, relative):
    spec = by_name(operator)
    tree = ast.parse(fixture_path(ROOT / relative).read_text())
    node = next(part for part in tree.body if isinstance(part, ast.FunctionDef) and part.name == spec.entry)
    function = source_callable(node)
    assert not signature_errors(function, spec)
    assert spec.inputs[0].name == spec.params[0] == 'input_value'
    arguments = {name: None for name in spec.params}
    inspect.signature(function).bind(**arguments)
    for wrong_name in ('input', 'unregistered_tensor'):
        invalid = copy.deepcopy(node)
        invalid.args.args[0].arg = wrong_name
        assert [error.rule_id for error in signature_errors(source_callable(invalid), spec)] == ['SUB-LOD-007']
