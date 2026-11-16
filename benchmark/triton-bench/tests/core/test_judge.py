# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""判定层单元测试（L0，无需 NPU）。

判定逻辑是最容易出 bug 的地方，也是「采集 / 解析分离」最大的收益点：
全部用构造的标量输入驱动，不碰任何张量与硬件。
"""

from __future__ import annotations

import pytest

from triton_eval.bench.stats import summarize
from triton_eval.judge import CaseMeasurement, build_record, decide, within_tolerance
from triton_eval.models import CaseStatus
from triton_eval.specload import load_tolerance

F32 = load_tolerance().for_dtype("float32")
INT32 = load_tolerance().for_dtype("int32")


def _m(**overrides: object) -> CaseMeasurement:
    base = {
        "case_code": "add-f32-0001",
        "operator": "add",
        "max_rel_err": 1e-7,
        "max_abs_err": 1e-9,
    }
    base.update(overrides)
    return CaseMeasurement(**base)  # type: ignore[arg-type]


# ---------- 容差判定（任务 6.1） ----------


def test_float_within_tolerance() -> None:
    # 或语义：任一口径达标即过。原合取版本在消零/近零输出上不可达
    # （rel 1e-3 但 abs 1e-8：参考值 ~1e-5，实现精度极高却被判错——
    # ADV-C1 实测的 gelu 负尾区冤案）。
    assert within_tolerance(1e-6, 1e-8, F32) is True
    assert within_tolerance(1e-3, 1e-8, F32) is True, "abs 达标即过（近零参考）"
    assert within_tolerance(1e-7, 1e-3, F32) is True, "rel 达标即过（大参考）"
    assert within_tolerance(1e-3, 1e-3, F32) is False, "两口径都超才算错"


def test_integer_requires_exact_match() -> None:
    assert within_tolerance(0.0, 0.0, INT32) is True
    assert within_tolerance(1e-18, 0.0, INT32) is False, "整型任何偏差都算错"
    assert within_tolerance(0.0, 1e-18, INT32) is False


def test_missing_error_values_are_not_within_tolerance() -> None:
    assert within_tolerance(None, 0.0, F32) is False
    assert within_tolerance(0.0, None, F32) is False


def test_allclose_rescues_near_zero_grouped_matmul() -> None:
    """grouped_matmul 近零输出的回归钉（2026-10-09 容差修复）。

    真实场景：近零输出上 oracle 自身 fp32 累加误差 1.3e-5 → rel 4.3e-4。
    f32 档 rel=1e-4 / abs=1e-5，聚合口径两条都超——但逐元素 allclose
    （|impl-ref| ≤ atol + rtol·|ref|）对近零参考有正确的渐近行为，
    判定通过。修复前（合取语义、无 allclose 通道）grouped_matmul 的
    f32-00 / f16-01 两例被冤判 WRONG_RESULT，且错记在被测方头上。
    """
    m = _m(
        case_code="grouped_matmul-f32-00", operator="grouped_matmul",
        max_rel_err=4.3e-4, max_abs_err=1.3e-5, allclose_ok=True,
    )
    assert within_tolerance(4.3e-4, 1.3e-5, F32) is False, "聚合口径两条都超"
    assert decide(m, F32) is CaseStatus.PASS, "allclose 通道必须救回近零冤案"


def test_allclose_is_not_unconditional_pass() -> None:
    """双通道都不过才算错——allclose 不是无条件放行。"""
    m = _m(max_rel_err=1e-1, max_abs_err=1e-1, allclose_ok=False)
    assert decide(m, F32) is CaseStatus.WRONG_RESULT
    # allclose_ok=None（旧数据未计算）不得当 True 用
    m2 = _m(max_rel_err=1e-1, max_abs_err=1e-1)
    assert decide(m2, F32) is CaseStatus.WRONG_RESULT


# ---------- 容差标定 1.1（2026-10-11 判卷口径修正） ----------


def test_bf16_tolerance_accommodates_one_ulp() -> None:
    """bf16 rel 容差必须容纳 1 ULP（理论区间 (2^-8, 2^-7]）。

    实测冤案：swi_glu-bf16-M40 参考量级 ~336 处 1 ULP = 2.0
    （rel 5.95e-3），被旧容差 5e-3 错杀。fp16 的 1e-3 本就容纳 1 ULP。
    """
    bf16 = load_tolerance().for_dtype("bfloat16")
    assert bf16.rel == pytest.approx(7.8e-3), "bf16 rel 必须 ≥ 2^-7"
    assert within_tolerance(5.952e-3, 2.0, bf16) is True, "1 ULP 的合法舍入"
    assert within_tolerance(1e-1, 2.0, bf16) is False, "真错（rel 10%）仍拦"


def _gelu_f32_case():
    from triton_eval.models import CaseSource, CaseSpec

    return CaseSpec(
        code="gelu-f32-S120", operator="gelu", source=CaseSource.VENDOR_SELFTEST,
        dtype="float32", shape=[2, 4, 4, 4, 4], seed=1, launch={}, feature_atoms=[],
        attrs={},
    )


def test_operator_override_covers_reference_self_noise() -> None:
    """算子级覆盖：参考自身的硬件偏差超过默认档时，覆盖优先于 dtype 表。

    归因证据（2026-10-11，详见 spec/tolerance.yaml 注释）：昇腾 NPU 上
    torch 的 f32 gelu 默认模式（要求 erf 精确式）实际返回 tanh 近似精度
    ——与 erf 的差至 4.73e-4，比 f32 默认 abs=1e-5 大两个数量级。
    更精确的 erf 实现反被判错（ADV-C1 的 gelu-f32-S120/M48 冤案）。
    """
    from triton_eval.judge import tolerance_for_case

    table = load_tolerance()
    case = _gelu_f32_case()

    # gelu 的 f32 走覆盖档（abs=1e-3），其他算子仍走默认档（abs=1e-5）
    gelu_tol = tolerance_for_case(table, case, _m(case_code="gelu-f32-S120",
                                                  operator="gelu"))
    assert gelu_tol.abs == pytest.approx(1e-3)
    exp_tol = tolerance_for_case(table, case, _m(case_code="exp-f32-00",
                                                 operator="exp"))
    assert exp_tol.abs == pytest.approx(1e-5), "未登记证据的算子不得放宽"

    # gelu-S120 冤案的现场值：rel 3.6e-2（聚合口径超）+ abs 4.6e-4
    # （旧 abs=1e-5 错杀）→ 新覆盖档下 abs 达标即过
    m = _m(case_code="gelu-f32-S120", operator="gelu",
           max_rel_err=3.619e-2, max_abs_err=4.589e-4)
    assert within_tolerance(3.619e-2, 4.589e-4, gelu_tol) is True
    assert decide(m, gelu_tol) is CaseStatus.PASS, "两种合法写法（精确式/近似式）都过"

    # 真错不放过：误差比参考自身噪声大两个数量级以上
    m_bad = _m(case_code="gelu-f32-S120", operator="gelu",
               max_rel_err=1e-1, max_abs_err=1e-1)
    assert decide(m_bad, gelu_tol) is CaseStatus.WRONG_RESULT


def test_grouped_matmul_override_covers_f32_accumulation_noise() -> None:
    """grouped_matmul 的 f32 覆盖：extreme 输入下参考自身累加噪声 ~6e-4。

    证据：grouped_matmul-f32-M18 冤案（本地复现参考 vs f64 真值
    max_abs 6.0e-4，比默认 abs=1e-5 紧两个数量级——判据比参考自己还严）。
    """
    from triton_eval.judge import tolerance_for_case

    table = load_tolerance()
    case = _gelu_f32_case()  # 任意 case；容差由 measurement.operator 决定
    m = _m(case_code="grouped_matmul-f32-M18", operator="grouped_matmul",
           max_rel_err=5.372e-3, max_abs_err=2.441e-4)
    tol = tolerance_for_case(table, case, m)
    assert tol.abs == pytest.approx(1e-3)
    assert decide(m, tol) is CaseStatus.PASS


def test_operator_override_matches_class_name() -> None:
    """覆盖档查找必须认类名：measurement.operator 是注册表类名
    （GroupedMatmul），yaml 覆盖键是算子 ID（grouped_matmul）。

    2026-10-12 真机复核实测穿帮：旧实现只做 lower()，"Gelu"碰巧命中
    而"GroupedMatmul"变 groupedmatmul 错失覆盖档——tolerance 1.1 落地
    后 M18 冤案复发（gelu 修复生效、grouped_matmul 失灵的分裂现象）。
    本单测当时用 slug 喂入所以全绿；本钉用真机观测的原始数值 +
    类名形式重放。
    """
    from triton_eval.judge import tolerance_for_case

    table = load_tolerance()
    case = _gelu_f32_case()  # 任意 case；容差由 measurement.operator 决定

    # 类名形式必须命中覆盖档（bug 现场：GroupedMatmul → groupedmatmul ≠ grouped_matmul）
    m = _m(case_code="grouped_matmul-f32-M18", operator="GroupedMatmul",
           max_rel_err=5.371803117433414, max_abs_err=2.44140625e-4)
    tol = tolerance_for_case(table, case, m)
    assert tol.abs == pytest.approx(1e-3), "类名 GroupedMatmul 必须命中覆盖档"
    assert decide(m, tol) is CaseStatus.PASS, "真机 M18 冤案的原始数值重放"

    # gelu 的类名形式（此前靠 lower() 碰巧命中，钉住防回退）
    m_gelu = _m(case_code="gelu-f32-S120", operator="Gelu",
                max_rel_err=3.619e-2, max_abs_err=4.589e-4)
    assert tolerance_for_case(table, case, m_gelu).abs == pytest.approx(1e-3)

    # 未登记覆盖的算子，类名形式仍走默认档（不得因归一化而误放宽）
    m_exp = _m(case_code="exp-f32-00", operator="Exp")
    assert tolerance_for_case(table, case, m_exp).abs == pytest.approx(1e-5)


# ---------- 终态判定（任务 6.2） ----------


def test_pass() -> None:
    assert decide(_m(), F32) is CaseStatus.PASS


def test_wrong_result_is_silent_error() -> None:
    """编译运行都通过，但结果错——最高优先级风险信号。

    或语义下必须**两个口径都超**才判错（abs 用默认 1e-9 会走 abs 达标）。
    """
    assert decide(_m(max_rel_err=1e-2, max_abs_err=1e-2), F32) \
        is CaseStatus.WRONG_RESULT


def test_compile_fail() -> None:
    assert decide(_m(compiled=False, compile_err="syntax error"), F32) is CaseStatus.COMPILE_FAIL


def test_runtime_fail() -> None:
    assert decide(_m(executed=False, runtime_err="illegal memory access"), F32) is CaseStatus.RUNTIME_FAIL


def test_shape_mismatch() -> None:
    assert decide(_m(shape_matches=False), F32) is CaseStatus.SHAPE_MISMATCH


def test_spec_error() -> None:
    assert decide(_m(spec_ok=False), F32) is CaseStatus.SPEC_ERROR


def test_env_abort() -> None:
    assert decide(_m(env_ok=False), F32) is CaseStatus.ENV_ABORT


def test_oracle_fail_when_reference_broken() -> None:
    """参考实现自己崩了：不能判 PASS 也不能判 WRONG_RESULT。"""
    assert decide(_m(oracle_ok=False), F32) is CaseStatus.ORACLE_FAIL


def test_submission_invalid_when_no_implementation() -> None:
    """提交里没有该算子的实现：不是算错，也不是编译失败。

    回归教训：decide 曾不看 submission_ok，导致「没提交」落到容差分支
    被判成 WRONG_RESULT——既冤枉被测方，又污染静默错误计数。
    """
    assert decide(_m(submission_ok=False), F32) is CaseStatus.SUBMISSION_INVALID
    from triton_eval.models import is_counted

    assert is_counted(CaseStatus.SUBMISSION_INVALID) is False


def test_submission_invalid_takes_precedence_over_wrong_result() -> None:
    """没有被测对象时，误差信号（此处为空）不能反推出静默错误。"""
    m = _m(submission_ok=False, max_rel_err=None, max_abs_err=None)
    assert decide(m, F32) is CaseStatus.SUBMISSION_INVALID


def test_env_abort_takes_precedence() -> None:
    """环境异常时，其他信号都不可信。"""
    assert decide(_m(env_ok=False, compiled=False, oracle_ok=False), F32) is CaseStatus.ENV_ABORT


def test_oracle_fail_takes_precedence_over_compile_fail() -> None:
    """参考都跑不出来，后面的对比没有意义——先判参考失效。"""
    assert decide(_m(oracle_ok=False, compiled=False), F32) is CaseStatus.ORACLE_FAIL


# ---------- 第二参考交叉校验（任务 6.3） ----------


def test_cross_check_disagreement_is_oracle_fail() -> None:
    """两个参考实现互相不一致 → 评测方自己的问题，不计入分母。"""
    status = decide(_m(cross_check_ok=False), F32)
    assert status is CaseStatus.ORACLE_FAIL
    from triton_eval.models import is_counted

    assert is_counted(status) is False


# ---------- 记录组装（任务 6.4） ----------


def test_build_record_carries_failure_evidence() -> None:
    m = _m(compiled=False, compile_err="triton.CompilationError: ...", duration_ms=12.5,
           log_path="runs/1/logs/add-f32-0001.log")
    rec = build_record(run_id=1, measurement=m, status=CaseStatus.COMPILE_FAIL)
    assert rec.compile_err == "triton.CompilationError: ..."
    assert rec.log_path == "runs/1/logs/add-f32-0001.log", "失败必须留下可归因的线索"


def test_build_record_computes_speedup_from_samples() -> None:
    samples = [0.412, 0.409, 0.415, 0.410, 0.411]
    m = _m(samples_ms=samples, baseline_median_ms=1.204)
    summary = summarize(samples)
    rec = build_record(run_id=1, measurement=m, status=CaseStatus.PASS, summary=summary)
    assert rec.perf_median_ms == pytest.approx(0.411)
    assert rec.samples_ms == samples, "原始样本必须全留，不只存中位数"
    assert rec.speedup == pytest.approx(1.204 / 0.411, rel=1e-3)


def test_build_record_without_perf_data() -> None:
    rec = build_record(run_id=1, measurement=_m(), status=CaseStatus.PASS)
    assert rec.perf_median_ms is None
    assert rec.speedup is None
    assert rec.samples_ms == []


def test_fusion_f32_order_noise_overrides_replay_real_machine() -> None:
    """四条 B 类覆盖（tolerance 1.2）用真机实测原始数值重放。

    2026-10-13 三轮真机定案（归因证据见 spec-conflicts C2 与
    robustness-findings）：dot 归因探针实测 f32 分块 tl.dot 与 torch
    matmul 在 K≤1024 逐位一致、K≥2048 出现求和顺序噪声且与分块无关
    ——实现侧无抓手，按既有机制登记算子级 abs 覆盖。本钉重放三个
    f32 大 K 用例 + 段和的实测值：登记前全部 WRONG（第二轮、第三轮
    逐位复现），登记后按证据口径通过；10× 超出实测值的真错仍拦。
    """
    from triton_eval.judge import tolerance_for_case

    table = load_tolerance()
    case = _gelu_f32_case()  # 任意 case；容差由 measurement.operator 决定

    # (算子类名, 用例, 真机实测 rel, 真机实测 abs, 覆盖 abs 档)
    replays = [
        ("TransposeLinear", "transpose_linear-f32-X48",
         2.0947176684881602e-1, 2.593994140625e-4, 1e-3),
        ("RmsNormLinear", "rms_norm_linear-f32-X56",
         2.008159052227792e-3, 5.92041015625e-3, 1e-2),
        ("GroupedMatmulSwiGlu", "grouped_matmul_swiglu-f32-X48",
         1.97721664704354528e-2, 2.44140625e-2, 5e-2),
        ("UnsortedSegmentSum", "unsorted_segment_sum-f32-X92",
         3.6811197533216693e-4, 2.74658203125e-4, 1e-3),
    ]
    for op, code, rel, ab, want_abs in replays:
        m = _m(case_code=code, operator=op, max_rel_err=rel, max_abs_err=ab)
        tol = tolerance_for_case(table, case, m)
        assert tol.abs == pytest.approx(want_abs), f"{op} 必须命中覆盖档"
        assert decide(m, tol) is CaseStatus.PASS, f"{code} 真机实测值重放"

    # 真错控制组：10× 于实测噪声量级的偏差必须仍判错——覆盖不是后门
    m = _m(case_code="grouped_matmul_swiglu-f32-X48",
           operator="GroupedMatmulSwiGlu",
           max_rel_err=0.2, max_abs_err=0.2)
    tol = tolerance_for_case(table, case, m)
    assert decide(m, tol) is CaseStatus.WRONG_RESULT, "覆盖档内也须拦真错"

    # 半精度不受 f32 覆盖影响：仍走默认档（半精度用例按段语义舍回后
    # 在默认档内通过，无需也不应放宽）
    m = _m(case_code="rms_norm_linear-bf16-S08", operator="RmsNormLinear",
           max_rel_err=1e-2, max_abs_err=1e-2)
    assert tolerance_for_case(table, case, m).abs == pytest.approx(1e-2), \
        "bf16 默认档（1e-2）不应被 f32 覆盖影响"
