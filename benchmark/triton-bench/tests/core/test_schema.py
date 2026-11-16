# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""用例与算子 schema 的单元测试（L0，无需 NPU）。"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from triton_eval.models import (
    NON_COUNTED_STATUSES,
    CaseResultRecord,
    CaseSource,
    CaseSpec,
    CaseStatus,
    OperatorSpec,
    is_counted,
    is_severe,
)
from triton_eval.specload import load_feature_atoms, load_spec, load_tolerance


def _valid_case(**overrides: object) -> dict:
    base = {
        "code": "add-f32-1024-0001",
        "operator": "add",
        "source": "vendor_selftest",
        "dtype": "float32",
        "shape": [1024],
        "seed": 20261002,
        "launch": {"grid": [4], "BLOCK": 256, "num_warps": 4, "num_stages": 3},
        "feature_atoms": ["FA-MATH-ADD", "FA-LOAD-CONTIG"],
    }
    base.update(overrides)
    return base


# ---------- 容差表（任务 2.1） ----------


def test_tolerance_table_covers_all_target_dtypes() -> None:
    table = load_tolerance()
    for dtype in ["float32", "float16", "bfloat16", "float64",
                  "int8", "int16", "int32", "int64", "uint8", "bool"]:
        assert table.has(dtype), f"容差表缺少 {dtype}"


def test_reference_dtype_is_float32_because_ascend_lacks_fp64() -> None:
    """昇腾不支持 float64，参考精度只能取 float32。

    实测：`torch.zeros(4, dtype=torch.float64, device="npu:0")` 会返回成功，
    但运行时打印 "Device do not support double dtype now, dtype cast
    replace with float" —— 即被静默降级为 float32。
    若把参考精度设成 float64，得到的是一个假的高精度锚点。
    """
    assert load_tolerance().reference_dtype == "float32"


def test_integer_tolerances_require_exact_match() -> None:
    table = load_tolerance()
    for dtype in ["int8", "int16", "int32", "int64", "uint8", "bool"]:
        assert table.for_dtype(dtype).is_exact, f"{dtype} 必须精确相等"


def test_float_tolerances_are_not_zero() -> None:
    table = load_tolerance()
    for dtype in ["float32", "float16", "bfloat16"]:
        assert not table.for_dtype(dtype).is_exact


def test_float32_tolerance_accommodates_reduction_error() -> None:
    """float32 档必须容得下归约类算子的浮点重排误差。

    实测 layer_norm 主 oracle 与 kernel 在 float32 下相差 2.4e-5。
    用 1e-5 会把它误判为静默错误；同时真正的静默错误量级在 1e-1 以上，
    1e-4 仍能捕获。
    """
    tol = load_tolerance().for_dtype("float32")
    assert tol.rel >= 1e-4, "float32 容差过严，会把正常浮点重排误判为静默错误"
    assert tol.rel <= 1e-1, "float32 容差过松，会放过真正的静默错误"


def test_tolerance_lookup_rejects_unknown_dtype() -> None:
    table = load_tolerance()
    with pytest.raises(KeyError):
        table.for_dtype("float8_e4m3")


# ---------- 用例 schema（任务 2.2） ----------


def test_valid_case_passes() -> None:
    case = CaseSpec(**_valid_case())
    assert case.code == "add-f32-1024-0001"
    assert case.source is CaseSource.VENDOR_SELFTEST
    assert case.timeout_s == 60


def test_case_missing_required_field_is_rejected() -> None:
    payload = _valid_case()
    del payload["seed"]
    with pytest.raises(ValidationError):
        CaseSpec(**payload)


def test_case_tolerance_override_is_rejected() -> None:
    """case 级容差覆盖必须在 schema 层面就无法表达。"""
    payload = _valid_case(tolerance_rel=1.0, tolerance_abs=1.0)
    with pytest.raises(ValidationError):
        CaseSpec(**payload)


def test_case_requires_real_kernel_file_path() -> None:
    """kernel 必须指向文件路径，不能是内联代码。"""
    payload = _valid_case(kernel_source="inline python code")
    with pytest.raises(ValidationError):
        CaseSpec(**payload)


# ---------- 算子 schema（任务 2.3） ----------


def test_operator_requires_weight() -> None:
    with pytest.raises(ValidationError):
        OperatorSpec(name="add", source="vendor_selftest", ref_path="operators/add/ref.py")


def test_valid_operator_passes() -> None:
    op = OperatorSpec(
        name="add", source="vendor_selftest", weight=1.0, ref_path="operators/add/ref.py"
    )
    assert op.weight == 1.0


# ---------- 终态派生规则 ----------


def test_terminal_states_are_mece() -> None:
    """终态互斥且完备。提交不合规加入后共 9 个。"""
    values = [s.value for s in CaseStatus]
    assert len(values) == len(set(values)) == 9
    assert CaseStatus.SUBMISSION_INVALID.value in values


def test_severe_only_for_wrong_result() -> None:
    for status in CaseStatus:
        assert is_severe(status) is (status is CaseStatus.WRONG_RESULT)


def test_non_counted_statuses() -> None:
    """不计入分母的终态：corpus / 评测方 / 提交合规性问题，
    被测方不为其背分。
    """
    assert NON_COUNTED_STATUSES == {
        CaseStatus.SPEC_ERROR,
        CaseStatus.ORACLE_FAIL,
        CaseStatus.ENV_ABORT,
        CaseStatus.SUBMISSION_INVALID,
    }
    for status in CaseStatus:
        assert is_counted(status) is (status not in NON_COUNTED_STATUSES)


def test_result_record_derives_severe_and_counted() -> None:
    rec = CaseResultRecord(run_id=1, case_code="c1", operator="add",
                           status=CaseStatus.WRONG_RESULT)
    assert rec.severe is True
    assert rec.counted is True

    rec2 = CaseResultRecord(run_id=1, case_code="c2", operator="add",
                            status=CaseStatus.ORACLE_FAIL)
    assert rec2.severe is False
    assert rec2.counted is False


# ---------- 规范加载 ----------


def test_spec_loads_profiles_and_performance_config() -> None:
    spec = load_spec()
    assert spec.spec_version
    assert spec.scorer_version
    assert {"smoke", "standard", "official"} <= set(spec.profile_names)
    assert spec.compose_total is False, "规范禁止合成跨赛道总分"
    assert spec.profile("smoke").measure_perf is False
    assert spec.profile("standard").measure_perf is True


def test_feature_atoms_have_unique_ids() -> None:
    atoms = load_feature_atoms()["atoms"]
    ids = [a["id"] for a in atoms]
    assert len(ids) == len(set(ids)), "FA id 必须唯一"
