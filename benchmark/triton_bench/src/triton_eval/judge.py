# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""结果判定（L0，纯逻辑，无 NPU 依赖）。

「采集 / 解析分离」的落点：
    L1 的 npu_exec 只负责把张量算成几个标量（误差、shape 是否一致、参考是否可用），
    本模块拿这些标量做终态判定。易错的判定逻辑因此可在 macOS 上完整测试。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .bench.stats import PerfSummary, speedup
from .models import CaseResultRecord, CaseStatus
from .specload import Tolerance


@dataclass
class CaseMeasurement:
    """一次用例执行的观测量。全部是标量或布尔，不含张量。"""

    case_code: str
    operator: str

    # 前置条件
    env_ok: bool = True              # 环境是否正常（False → ENV_ABORT）
    submission_ok: bool = True       # 提交是否提供了该算子的实现（False → SUBMISSION_INVALID）
    output_ok: bool = True           # 运行期输出是否合规（伪张量 / 非连续 → SUBMISSION_INVALID）
    spec_ok: bool = True             # case 定义是否合法（False → SPEC_ERROR）
    oracle_ok: bool = True           # 主参考实现是否执行成功
    cross_check_ok: bool = True      # 第二参考与主参考是否一致

    # 运行期输出不合规的原因（规则编号 + 说明）。仅诊断用。
    output_err: str | None = None

    # 参考输出的实际 dtype。容差必须按**输出** dtype 查表——见下方说明。
    output_dtype: str | None = None

    # 被测实现实际耗时（毫秒）。用于发现「吞掉超时异常」——
    # 见 harness.npu_exec 中对 ADV-F4 的说明。
    impl_ms: float | None = None

    # 交叉校验失败的具体原因。仅诊断用，不进入结果记录。
    # 曾经这里只置 cross_check_ok=False 而不记原因，导致 ORACLE_FAIL
    # 无法定位——误差显示 0 却判失败，只能靠猜。
    cross_check_err: str | None = None

    # 逐元素 allclose 判定（|impl-ref| ≤ atol + rtol·|ref|，PyTorch 口径）。
    # 聚合的 rel/abs 是诊断量，判定还补这条**行业标准**的逐元素形式：
    # 它对「近零参考值」的误差有正确的渐进行为——聚合 rel 在近零区失真
    # （实测：grouped_matmul 近零输出上 oracle 自身 fp32 累加误差 1.3e-5
    # → rel 4.3e-4，聚合口径冤枉更精确的实现）。None = 未计算（旧数据）。
    allclose_ok: bool | None = None

    outputs_match: bool | None = None

    # 编译与执行
    compiled: bool = True
    compile_err: str | None = None   # 超时也走这个字段，细节写在这里
    executed: bool = True
    runtime_err: str | None = None

    # 比对
    shape_matches: bool = True
    max_rel_err: float | None = None
    max_abs_err: float | None = None

    # 计时
    samples_ms: list[float] = field(default_factory=list)
    baseline_median_ms: float | None = None

    duration_ms: float | None = None
    log_path: str | None = None


# 超时事后核对的余量：允许被测实现比硬超时多跑一点点
# （计时本身有抖动、同步也需要时间），但绝不允许显著超出。
# 放在 L0 而非 npu_exec：这条契约的测试不应依赖昇腾运行时。
TIMEOUT_SLACK = 1.05


def exceeded_timeout(impl_ms: float | None, timeout_s: int) -> bool:
    """判断被测实现的实际耗时是否越过硬超时。

    不在「能否中断被测代码」上赌——SIGALRM 注入的是异常，
    `try: ... except BaseException: 继续赖着` 就能吞掉它并在超时
    窗口之外继续运行（ADV-F4 实测：吞掉后再赖 5s 仍被记 PASS）。
    可靠做法是不依赖中断，改为**事后核对实际耗时**。
    """
    if timeout_s <= 0 or impl_ms is None:
        return False
    return impl_ms > timeout_s * 1000 * TIMEOUT_SLACK


def is_compile_error(exc: BaseException) -> bool:
    """识别 Triton 编译阶段异常，不据此断言缺陷责任。

    只认异常类型，绝不看 message——文本由提交方书写。
    实测（ADV-F5）：`raise RuntimeError("triton compile failed: ...")`
    就把提交方自己的运行期缺陷判成了 COMPILE_FAIL。

    COMPILE_FAIL 记录失败阶段；提交语法、非法算子组合或编译器能力缺口
    都可能触发它。责任归因还需要合法输入、最小复现及能力探针证据。

    放在 L0（而非 npu_exec）的原因：这条归因规则的测试不需要 NPU，
    留在 npu_exec 里会因后者依赖昇腾运行时，只能在昇腾机上验证。
    """
    t = type(exc)
    if t.__name__.lower() not in ("compilationerror", "compileerror", "mlircompilationerror"):
        return False
    module = (getattr(t, "__module__", "") or "").lower()
    return "triton" in module


def exception_detail(exc: BaseException, limit: int = 2000) -> str:
    """Keep the root diagnostic at the tail of a nested compiler exception."""
    text = f"{type(exc).__name__}: {exc}"
    if len(text) <= limit:
        return text
    marker = "\n... [truncated] ...\n"
    head = (limit - len(marker)) // 2
    return text[:head] + marker + text[-(limit - len(marker) - head):]


def within_tolerance(
    max_rel_err: float | None, max_abs_err: float | None, tol: Tolerance
) -> bool:
    """容差判定（或语义）：相对**或**绝对任一口径达标即通过。

    整型（rel=0 且 abs=0）仍要求精确相等。

    为什么是「或」不是「且」（实测教训，ADV-C1 对抗样例钉死）：
    合取在消零/近零输出上**不可达**——gelu 负尾区参考值 ~2.4e-4，
    实现 abs 误差 2.4e-7 完全达标，但 rel = 2.2e-4 超限；
    一份比 oracle 更精确的实现反被判错。正确语义是：
    误差落在任一口径的界内即可，与 PyTorch allclose 的
    `atol + rtol·|ref|`（本模块另有逐元素判定）一致。
    """
    if max_rel_err is None or max_abs_err is None:
        return False
    if tol.is_exact:
        return max_rel_err == 0.0 and max_abs_err == 0.0
    return max_rel_err <= tol.rel or max_abs_err <= tol.abs


def tolerance_for_case(
    table: Any, case: Any, measurement: CaseMeasurement
) -> Tolerance:
    """取一个用例的容差档位。

    容差按**参考输出**的 dtype 查表，不是 case.dtype。
    实测（ADV-C1）：cross_entropy_loss 的 case.dtype 是 int32/int64
    （那是 target 的类型），而输出是 fp32 标量。按 case.dtype
    取到「精确相等」档，于是 rel=7.18e-08 也被判 WRONG——
    更精确的实现反而拿不到分。判据必须对齐被比对对象的精度。

    算子级覆盖优先于 dtype 默认档（登记须附参考自身偏差的归因证据，
    见 spec/tolerance.yaml 的 operator_overrides 注释）。

    放在 judge（L0）：CLI 与评测子进程都要用，分头写必然分叉。
    """
    dtype = measurement.output_dtype or case.dtype
    if table.has(dtype, measurement.operator):
        return table.for_dtype(dtype, measurement.operator)
    return table.for_dtype(dtype)


def decide(measurement: CaseMeasurement, tol: Tolerance) -> CaseStatus:
    """把一次执行观测量判定为唯一终态。

    判定顺序是有意为之：
      环境 → 提交 → 规范 → 参考 → 编译 → 运行 → 形状 → 容差
    两条顺序约束都有实测教训：
      - 提交不合规必须先于容差判定。曾经 decide 不看 submission_ok，
        「提交里没有该算子」会落到容差分支被判成 WRONG_RESULT——
        既冤枉了被测方（没提交被当成算错），又污染了静默错误计数。
      - 参考失效要先于编译失败判定——参考都跑不出来，后面的对比没有意义。
    """
    if not measurement.env_ok:
        return CaseStatus.ENV_ABORT
    if not measurement.submission_ok:
        return CaseStatus.SUBMISSION_INVALID
    # 运行期兜底紧随其后：静态扫描看不到运行期行为（伪张量、非连续输出），
    # 这类不合规只有真跑一遍才暴露，且必须优先于精度判定——
    # 拿一个非真实的张量去比误差，得出的「通过」毫无意义。
    if not measurement.output_ok:
        return CaseStatus.SUBMISSION_INVALID
    if not measurement.spec_ok:
        return CaseStatus.SPEC_ERROR
    if not measurement.oracle_ok or not measurement.cross_check_ok:
        return CaseStatus.ORACLE_FAIL
    if not measurement.compiled:
        return CaseStatus.COMPILE_FAIL
    if not measurement.executed:
        return CaseStatus.RUNTIME_FAIL
    if not measurement.shape_matches:
        return CaseStatus.SHAPE_MISMATCH
    if measurement.outputs_match is not None:
        return CaseStatus.PASS if measurement.outputs_match else CaseStatus.WRONG_RESULT
    # 判定 = 聚合口径（或语义）∪ 逐元素 allclose（行业标准口径）。
    # 两者覆盖不同的误差形态：前者用报告里的聚合量，后者对近零参考值
    # 有正确的渐近行为——见 CaseMeasurement.allclose_ok 的说明。
    if within_tolerance(measurement.max_rel_err, measurement.max_abs_err, tol) \
            or measurement.allclose_ok is True:
        return CaseStatus.PASS
    return CaseStatus.WRONG_RESULT


def build_record(
    run_id: int,
    measurement: CaseMeasurement,
    status: CaseStatus,
    summary: PerfSummary | None = None,
) -> CaseResultRecord:
    """把观测量 + 终态组装成一条结果记录。"""
    sp = speedup(measurement.baseline_median_ms, summary.median_ms) if summary else None
    return CaseResultRecord(
        run_id=run_id,
        case_code=measurement.case_code,
        operator=measurement.operator,
        status=status,
        duration_ms=measurement.duration_ms,
        compile_err=measurement.compile_err or None,
        max_rel_err=measurement.max_rel_err,
        max_abs_err=measurement.max_abs_err,
        samples_ms=list(summary.samples) if summary else list(measurement.samples_ms),
        perf_median_ms=summary.median_ms if summary else None,
        perf_cv=summary.cv if summary else None,
        baseline_median_ms=measurement.baseline_median_ms,
        speedup=sp,
        log_path=measurement.log_path,
    )
