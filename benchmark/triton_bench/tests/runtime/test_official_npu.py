# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Run the five operators affected by the first official NPU evaluation in full."""

from pathlib import Path

import pytest
from triton_eval.fixture_sources import fixture_path

from triton_eval.corpusmod import load_manifest
from triton_eval.official_catalog import VERSION

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "corpus" / VERSION
AFFECTED = {"ArgMax", "FloorDiv", "Signbit", "Pow", "TopKValues"}
CASES = [case for case in load_manifest(CORPUS / "manifest.json").cases if case.operator in AFFECTED]
pytestmark = pytest.mark.npu


@pytest.mark.parametrize("case", CASES, ids=lambda case: case.code)
def test_official_repaired_case(case):
    from triton_eval.harness.npu_exec import execute_case
    from triton_eval.judge import decide, tolerance_for_case
    from triton_eval.models import CaseStatus
    from triton_eval.oracle import active_corpus, configure_corpus
    from triton_eval.refcheck import check_case
    from triton_eval.runner import execute_kwargs
    from triton_eval.specload import load_tolerance
    from triton_eval.submission import discover, load_all

    previous = active_corpus()
    configure_corpus(CORPUS)
    try:
        assert check_case(case, "npu:0") is None, case.code
        submission = load_all(discover(fixture_path(ROOT / "examples/official-ascend-submission")))
        assert case.operator in submission.loaded, submission.failures
        measurement = execute_case(case, **execute_kwargs(case, submission.loaded[case.operator].func))
        status = decide(measurement, tolerance_for_case(load_tolerance(), case, measurement))
        assert status == CaseStatus.PASS, (case.code, status, measurement.compile_err,
                                           measurement.max_abs_err, measurement.max_rel_err)
    finally:
        configure_corpus(previous)
