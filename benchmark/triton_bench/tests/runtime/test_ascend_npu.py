# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""修订示范实现的真实 NPU 契约回归；完整集由 suite 入口执行。"""

from math import prod
from pathlib import Path

import pytest
from triton_eval.fixture_sources import fixture_path

from triton_eval.corpusmod import load_manifest

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus/v0.7-ascend-contract"
MANIFEST = load_manifest(CORPUS / "manifest.json")
OPERATORS = sorted({case.operator for case in MANIFEST.cases})


def _regression_cases():
    cases = []
    for operator in OPERATORS:
        available = [item for item in MANIFEST.cases if item.operator == operator]
        chosen = available if operator == "CausalConv1d" else [min(available, key=lambda item: prod(item.shape))]
        cases.extend(chosen)
    return cases


REGRESSION_CASES = _regression_cases()
pytestmark = pytest.mark.npu


@pytest.mark.parametrize("case", REGRESSION_CASES, ids=lambda case: case.code)
def test_adapted_operator_case(case):
    from triton_eval.harness.npu_exec import execute_case
    from triton_eval.judge import decide, tolerance_for_case
    from triton_eval.models import CaseStatus
    from triton_eval.oracle import active_corpus, configure_corpus
    from triton_eval.runner import execute_kwargs
    from triton_eval.specload import load_tolerance
    from triton_eval.submission import discover, load_all

    previous = active_corpus()
    configure_corpus(CORPUS)
    try:
        submission = load_all(discover(fixture_path(ROOT / "examples/ascend-ops-submission")))
        assert case.operator in submission.loaded, submission.failures
        measurement = execute_case(case, **execute_kwargs(case, submission.loaded[case.operator].func))
        status = decide(measurement, tolerance_for_case(load_tolerance(), case, measurement))
        assert status == CaseStatus.PASS, (case.code, status, measurement.compile_err,
                                           measurement.max_abs_err, measurement.max_rel_err)
    finally:
        configure_corpus(previous)
