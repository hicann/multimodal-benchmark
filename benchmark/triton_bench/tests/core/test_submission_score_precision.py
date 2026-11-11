# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Score verification keeps independent aggregation and the public rounding contract."""

import pytest

from triton_eval.models import CaseResultRecord, CaseStatus
from triton_eval.scorer import compute_run_score
from triton_eval.submission_simulation import _check_simulation_score


def score_fixture(weights, statuses):
    """Create real scorer output alongside independently verified JSON rows."""
    records = []
    rows = []
    for index, (operator, status) in enumerate(statuses):
        records.append(CaseResultRecord(run_id=1, case_code=str(index), operator=operator, status=CaseStatus(status)))
        rows.append(dict(case_code=str(index), operator=operator, status=status))
    score = compute_run_score(1, records, weights, [1.0, 1.2, 2.0], 0.03, 'test')
    return dict(operator_weights=weights), dict(score=score.model_dump(), cases=rows)


@pytest.mark.parametrize('weights,statuses,expected', [
    ({}, [('A', 'COMPILE_FAIL'), ('B', 'COMPILE_FAIL'), ('C', 'PASS')], (33.3333, 33.3333)),
    ({}, [('A', 'PASS'), ('A', 'COMPILE_FAIL'), ('A', 'RUNTIME_FAIL')], (33.3333, 0.0)),
    ({'A': 1, 'B': 6}, [('A', 'PASS'), ('B', 'WRONG_RESULT')], (14.2857, 14.2857)),
    ({'A': 1, 'B': 2}, [('A', 'PASS'), ('B', 'SHAPE_MISMATCH'), ('B', 'SUBMISSION_INVALID')],
     (33.3333, 33.3333)),
])
def test_independent_score_recalculation_matches_real_scorer_precision(weights, statuses, expected):
    """Fractional, weighted, partial and non-counted cases honor four decimals."""
    scenario, results = score_fixture(weights, statuses)
    failures = []
    _check_simulation_score(scenario, results, results['cases'], failures)
    assert not failures
    assert results['score']['coverage_ops_core'] == pytest.approx(expected[0])
    assert results['score']['coverage_strict'] == pytest.approx(expected[1])


@pytest.mark.parametrize('field', ['coverage_ops_core', 'coverage_strict'])
@pytest.mark.parametrize('wrong_value', [33.3334, 33.3332, 100.0, float('nan'), float('inf'), -float('inf')])
def test_rounding_contract_does_not_hide_wrong_or_nonfinite_scores(field, wrong_value):
    """Even a one-unit change in the last public decimal must fail verification."""
    scenario, results = score_fixture({}, [('A', 'PASS'), ('B', 'RUNTIME_FAIL'), ('C', 'COMPILE_FAIL')])
    results['score'][field] = wrong_value
    failures = []
    _check_simulation_score(scenario, results, results['cases'], failures)
    assert failures


@pytest.mark.parametrize('change', ['weights', 'missing_case', 'missing_score', 'strict_only'])
def test_score_recalculation_still_rejects_scope_weight_and_strict_errors(change):
    """Independent aggregation cannot be replaced with trusting scorer output."""
    scenario, results = score_fixture({'A': 1, 'B': 2}, [('A', 'PASS'), ('B', 'RUNTIME_FAIL')])
    if change == 'weights':
        scenario['operator_weights'] = {'A': 2, 'B': 1}
    elif change == 'missing_case':
        results['cases'].pop()
    elif change == 'missing_score':
        results['score'] = None
    else:
        results['score']['coverage_strict'] = 0.0
    failures = []
    _check_simulation_score(scenario, results, results['cases'], failures)
    assert failures


@pytest.mark.parametrize('reported', [None, dict(coverage_ops_core=0.0, coverage_strict=0.0)])
def test_no_counted_cases_require_no_score(reported):
    """A numerical zero is not a substitute for unavailable scoring."""
    rows = [dict(operator='A', status='SUBMISSION_INVALID')]
    failures = []
    _check_simulation_score({}, dict(score=reported), rows, failures)
    assert bool(failures) is (reported is not None)


@pytest.mark.parametrize('reported,valid', [(100.0, True), (99.9999, False), (float('nan'), False)])
def test_independent_expected_score_uses_the_same_strict_precision(reported, valid):
    """An independent correct fixture still has to achieve the required full score."""
    scenario = dict(expected_score={'coverage_ops_core': 100.0})
    results = dict(score=dict(coverage_ops_core=reported, coverage_strict=100.0))
    failures = []
    _check_simulation_score(scenario, results, [dict(operator='A', status='PASS')], failures)
    assert bool(failures) is not valid
