# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""用例执行器（L1）——全仓库唯一 import triton 的模块。

职责边界：把张量算成标量观测量（误差、shape 是否一致、参考是否可用、耗时样本），
然后交给 L0 的 judge 做终态判定。任何判定逻辑都不应放进本模块。

kernel 文件契约（corpus/operators/<op>/kernel.py 必须提供）：
    make_inputs(seed, dtype, shape, device) -> list[Tensor]   造输入
    reference(inputs) -> Tensor                               主 oracle（torch eager，同设备）
    launch(inputs, out, cfg) -> None                          触发 kernel 启动（内部做 kernel[grid](...)）
以及被测的 @triton.jit kernel 本体。

第二参考实现（corpus/operators/<op>/ref.py）：
    reference(inputs) -> Tensor                               CPU 路径，用于交叉校验

关于 triton 的导入位置（实测踩坑，勿改回函数内导入）：
    `@triton.jit` 的 kernel 注解（如 `BLOCK: tl.constexpr`）由 Triton 在解析
    源码时求值，它只查找 `fn.__globals__`。若 `import triton.language as tl`
    写在函数体内部，`tl` 只是局部变量，Triton 看不到，会报
    `NameError('tl is not defined')`。
    因此 triton / triton.language 必须在**模块级**导入。
    同理，本模块不能用 `from __future__ import annotations`——那会把注解变成
    字符串，Triton 拿不到真实的 constexpr 标记。
"""

import importlib.util
import signal
import sys
import time
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import ModuleType
from typing import Any, Callable, Iterator

import torch

# 副作用导入，必须保留：torch_npu 会在 import 时把 `torch.npu` 这个属性
# 注册到 torch 上。不导入的话，本模块里任何 torch.npu.xxx 调用都会报
#     AttributeError: module 'torch' has no attribute 'npu'
# 这种隐式依赖极易漏——曾经漏在 build_corpus.py --screen 和各测试文件里，
# 表现为「CLI 能跑但测试全崩」。集中在本模块导入一次，彻底消除该隐患。
import torch_npu  # noqa: F401
import triton
import triton.language as tl

from ..bench.measure import measure, measure_baseline
from ..bench.stats import summarize
from ..comparison import allclose as _allclose, compare, output_matches
from ..inputs import _as_tensor, _dtype_of, as_outputs, make_inputs_for, tensor_fp  # noqa: F401
from ..judge import (
    CaseMeasurement,
    exceeded_timeout,
    is_compile_error,
    exception_detail,
)
from ..models import CaseSpec
from ..specload import load_tolerance


class EvaluationTimeoutError(Exception):
    pass




@contextmanager
def deadline(seconds: int) -> Iterator[None]:
    """尽力而为的超时保护。JIT 编译可能无法被中断，故为 best-effort。"""

    def _handler(signum: int, frame: Any) -> None:
        raise EvaluationTimeoutError(f"超过 {seconds}s 硬超时")

    if seconds <= 0:
        yield
        return
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def load_module(path: Path, name: str) -> ModuleType:
    """从真实文件路径加载模块。

    @triton.jit 依赖 inspect.getsourcelines 反查源码，
    所以 kernel 必须来自真实文件——这也是 corpus 不能用数据库存代码的原因。
    """
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法从 {path} 加载模块")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def _attrs_for_call(attrs: dict[str, Any] | None) -> dict[str, Any]:
    """把用例属性展开为调用参数。None 视为「不给，用实现自己的默认值」。"""
    if not attrs:
        return {}
    return {k: v for k, v in attrs.items() if v is not None}


def _device() -> str:
    return f"npu:{torch.npu.current_device()}"


def execute_case(
    case: CaseSpec, *,
    impl: Callable[..., Any] | None = None,
    reference: Callable[..., Any] | None = None,
    cross_reference: Callable[..., Any] | None = None,
    attrs: dict[str, Any] | None = None,
    make_inputs: Callable[..., Any] | None = None,
    category: str | None = None,
    measure_perf: bool = False, repeats: int = 1,
    warmup_ms: int = 25, rep_ms: int = 100,
    log_path: str | None = None,
    tamper_check: Callable[[], list[str]] | None = None,
    compilation_dir: str | None = None, evidence_identity: dict | None = None,
) -> CaseMeasurement:
    """执行一个用例并返回观测量，不在这里判定终态。

    被测实现 impl 由提交方提供，reference 与 cross_reference 来自评测方，二者来源必须分离。
    impl 为 None 表示提交中没有实现，此时置 submission_ok=False，由 judge 判为 SUBMISSION_INVALID。
    tamper_check 在实现返回后、比对开始前调用；返回被篡改接口名，空列表表示干净。
    比对链路不允许在污染状态下运行，必须先验身份，再做比对。
    """
    from types import SimpleNamespace

    started = time.time()
    m = CaseMeasurement(case_code=case.code, operator=case.operator, log_path=log_path)
    options = SimpleNamespace(impl=impl, reference=reference, cross_reference=cross_reference, attrs=attrs,
                              make_inputs=make_inputs, category=category, tamper_check=tamper_check,
                              compilation_dir=compilation_dir, evidence_identity=evidence_identity)
    result = _execute_functionality(case, options, m, started)
    if result is None:
        return m
    inputs, impl_ms = result

    if measure_perf and m.compiled and m.executed:
        from ..reference_execution import reference_execution

        def kernel_fn() -> None:
            impl(*inputs, **_attrs_for_call(attrs))

        def baseline_fn() -> None:
            with reference_execution(case.operator, inputs):
                reference(*inputs, **_attrs_for_call(attrs))

        try:
            with deadline(case.timeout_s):
                m.samples_ms = measure(kernel_fn, repeats=repeats,
                                       warmup_ms=warmup_ms, rep_ms=rep_ms)
                m.baseline_median_ms = measure_baseline(
                    baseline_fn, repeats=repeats, warmup_ms=warmup_ms, rep_ms=rep_ms
                )
        except EvaluationTimeoutError:
            m.samples_ms = []

    m.duration_ms = (time.time() - started) * 1000
    m.impl_ms = impl_ms
    return m


# ---------- 健康探针 ----------


@triton.jit
def _probe_add_kernel(x_ptr, y_ptr, o_ptr, n, block: tl.constexpr):
    """探针 kernel 定义在模块级。

    Triton 会在 `fn.__globals__` 中解析注解，模块级定义能保证
    `tl` 可见，且 `inspect.getsourcelines` 取到的源码上下文完整。
    """
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    tl.store(o_ptr + offs,
             tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
             mask=mask)


@torch.no_grad()
def health_probe() -> tuple[bool, str]:
    """运行前探针：编译并执行一个最小 kernel，比对参考。

    这是防御「被测编译器被评测工具链弄坏」的唯一手段。
    triton 与 triton-ascend 目录同名，后装 triton 会静默破坏被测对象，
    症状是 kernel 大面积失败——探针失败时整轮必须判 INVALID。
    """
    try:
        dev = _device()
        n = 1024
        x = torch.randn(n, device=dev)
        y = torch.randn(n, device=dev)
        o = torch.empty_like(x)
        _probe_add_kernel[(n // 256,)](x, y, o, n, block=256)
        torch.npu.synchronize()
        diff = float((o - (x + y)).abs().max().item())
        if diff != 0.0:
            return False, f"探针结果与参考不一致，最大差 {diff}"
        return True, ""
    except Exception as exc:  # noqa: BLE001
        return False, f"{type(exc).__name__}: {exc}"[:500]


def _execute_functionality(case, options, m, started):
    if not _check_execution_references(case, options, m, started):
        return None
    inputs = _make_case_inputs(case, options, m, started)
    if inputs is None:
        return None
    expected = _run_case_reference(case, options, inputs, m, started)
    if expected is None:
        return None
    if not _check_case_cross_reference(options, inputs, expected, m, started):
        return None
    result = _call_case_implementation(case, options, inputs, m, started)
    if result is None:
        return None
    actual, impl_ms, input_fp = result
    if not _check_execution_policy(case, options, impl_ms, m, started):
        return None
    if not _check_case_output_contract(case, options, inputs, input_fp, (actual, expected), m, started):
        return None
    _compare_case_outputs(case, actual, expected, m)
    return inputs, impl_ms


def _check_execution_references(case, options, m, started):
    if options.impl is None:
        m.submission_ok = False
        m.compile_err = f"提交中没有算子 {case.operator} 的实现"
        m.duration_ms = (time.time() - started) * 1000
        return False
    if options.reference is None:
        from ..oracle import oracle_load_error

        m.oracle_ok = False
        # 带上真实原因：只报「缺少参考实现」无法区分是文件缺失、教程 import
        # 失败、还是 ref.py 引用了不存在的函数，实测靠猜完全定位不了。
        _why = oracle_load_error(case.operator)
        m.compile_err = (
            f"算子 {case.operator} 缺少参考实现（评测方资产缺失）"
            + (f"：{_why}" if _why else "")
        )
        m.duration_ms = (time.time() - started) * 1000
        return False

    return True


def _make_case_inputs(case, options, m, started):
    dev = _device()
    dtype = _dtype_of(case.dtype)

    try:
        inputs = make_inputs_for(case, dtype, dev, options.attrs, options.make_inputs)
        from ..ascend_contracts import validate_inputs

        validate_inputs(case.operator, inputs, options.attrs or {})
    except Exception as exc:  # noqa: BLE001
        # 归因有区别：
        #   - 算子自定义的 make_inputs 来自评测方资产（ref.py），
        #     它构造失败是「评测方没准备好」，记 oracle 侧，不落到被测方头上
        #   - 默认构造失败才是用例定义本身有问题（如 shape 非法）
        if options.make_inputs is not None:
            m.oracle_ok = False
        else:
            m.spec_ok = False
        m.compile_err = f"构造输入失败: {type(exc).__name__}: {exc}"[:2000]
        m.duration_ms = (time.time() - started) * 1000
        return None

    return inputs


def _run_case_reference(case, options, inputs, m, started):
    try:
        reference_inputs = [tensor.clone() for tensor in inputs]
        from ..reference_execution import reference_execution
        with reference_execution(case.operator, reference_inputs):
            expected = options.reference(*reference_inputs, **_attrs_for_call(options.attrs))
        if any(not torch.equal(before, after)
               for before, after in zip(inputs, reference_inputs)):
            raise ValueError("参考实现修改了输入")
        expected = [tensor.clone() for tensor in as_outputs(expected)]
        _validate_reference_outputs(case, expected, inputs)
    except Exception as exc:  # noqa: BLE001
        m.oracle_ok = False
        m.compile_err = f"参考实现失败: {type(exc).__name__}: {exc}"[:2000]
        m.duration_ms = (time.time() - started) * 1000
        return None

    return expected


def _validate_reference_outputs(case, expected, inputs):
    from ..operator_registry import by_name

    registered = by_name(case.operator)
    if registered is not None:
        if len(expected) != len(registered.outputs):
            raise ValueError("参考输出元数不符合注册规格")
        for tensor, declared in zip(expected, registered.outputs):
            if str(tensor.dtype).replace("torch.", "") not in declared.dtype:
                raise ValueError("参考输出 dtype 不符合注册规格")
            if tensor.device != inputs[0].device:
                raise ValueError("参考输出设备不符合输入设备")


def _check_case_cross_reference(options, inputs, expected, m, started):
    if options.cross_reference is not None:
        try:
            cpu_inputs = [t.detach().to("cpu").clone() for t in inputs]
            cpu_before = [t.clone() for t in cpu_inputs]
            cpu_expected = options.cross_reference(
                *cpu_inputs, **_attrs_for_call(options.attrs)
            )
            cpu_expected = as_outputs(cpu_expected)
            if any(not torch.equal(before, after)
                   for before, after in zip(cpu_before, cpu_inputs)):
                raise ValueError("第二参考修改了输入")
            if len(expected) == len(cpu_expected) and all(
                first.dtype == second.dtype and output_matches(first, second, load_tolerance().for_dtype(
                    str(first.dtype).replace("torch.", "")))
                for first, second in zip(expected, cpu_expected)
            ):
                m.cross_check_ok = True
            else:
                m.cross_check_ok = False
                m.cross_check_err = (
                    "两参考的输出元数或逐输出数值不一致"
                )
        except Exception as exc:  # noqa: BLE001
            # 必须记录原因。曾经这里静默吞异常，导致 ORACLE_FAIL 无从定位。
            m.cross_check_ok = False
            m.cross_check_err = f"交叉校验异常: {type(exc).__name__}: {exc}"[:500]

        # 交叉校验的失败原因必须落到 compile_err，
        # 否则 ORACLE_FAIL 在 diagnose 里只显示空线索，无从定位。
        if not m.cross_check_ok and not m.compile_err:
            m.compile_err = m.cross_check_err
        if not m.cross_check_ok:
            m.duration_ms = (time.time() - started) * 1000
            return False

    return True


def _call_case_implementation(case, options, inputs, m, started):
    input_fp = [tensor_fp(t) for t in inputs]

    # ---------- 执行被测实现 ----------
    # 契约：实现返回输出张量。只调用一次——既用于比对也用于计时，
    # 分两次调用会让计时样本与实际比对的对象不一致。
    impl_started = time.time()
    capture = nullcontext()
    if options.compilation_dir is not None:
        from ..compilation_evidence import CompilationCapture
        capture = CompilationCapture(triton, options.compilation_dir, dict(
            options.evidence_identity or {}, case_code=case.code, operator=case.operator,
            case_hash=case.content_hash, device=str(inputs[0].device),
            main_input_index=getattr(options.make_inputs, 'main_input_index', 0),
            input_metadata=[dict(dtype=str(t.dtype), shape=list(t.shape), stride=list(t.stride())) for t in inputs]))
    try:
        with capture as observation:
            if observation is not None:
                observation.bind_inputs(inputs)
            with deadline(case.timeout_s):
                actual = options.impl(*inputs, **_attrs_for_call(options.attrs))
                torch.npu.synchronize()
                _bind_completed_observation(observation, actual)
    except EvaluationTimeoutError as exc:
        m.compiled = False
        m.compile_err = str(exc)
        m.duration_ms = (time.time() - started) * 1000
        return None
    except Exception as exc:  # noqa: BLE001
        msg = exception_detail(exc)
        if is_compile_error(exc):
            m.compiled = False
        else:
            m.executed = False
        m.compile_err = msg
        m.duration_ms = (time.time() - started) * 1000
        return None

    impl_ms = (time.time() - impl_started) * 1000
    m.impl_ms = impl_ms

    return actual, impl_ms, input_fp


def _bind_completed_observation(observation, actual):
    if observation is None:
        return
    observation.payload['synchronized'] = True
    try:
        outputs = as_outputs(actual)
    except TypeError as error:
        # Instrumentation must not turn a contract violation into RUNTIME_FAIL.
        observation.payload['diagnostics'].append(f'output binding unavailable: {error}')
        return
    observation.bind_outputs(outputs)


def _check_execution_policy(case, options, impl_ms, m, started):
    if options.tamper_check is not None:
        tampered = options.tamper_check()
        if tampered:
            m.output_ok = False
            m.output_err = (
                f"SUB-EXE-005: 关键接口在实现运行后被替换："
                f"{', '.join(tampered[:8])}"
            )
            m.compile_err = m.output_err
            m.duration_ms = (time.time() - started) * 1000
            return False

    # ---------- 超时事后核对 ----------
    # 不能在「能否中断被测代码」上赌：SIGALRM 注入的是异常，
    # 而 `try: ... except BaseException: 继续赖着` 就能吞掉它
    # 并在超时窗口之外继续运行（ADV-F4 实测：吞掉后再赖 5s 仍被记 PASS）。
    #
    # 可靠的做法是不依赖中断，而是**事后核对实际耗时**：
    # 只要被测实现跑过了硬超时，无论它是否吞掉异常，都判失败。
    if exceeded_timeout(impl_ms, case.timeout_s):
        m.executed = False
        m.compile_err = (
            f"超过 {case.timeout_s}s 硬超时（实际 {impl_ms / 1000:.1f}s）；"
            "被测实现不得吞掉超时控制"
        )[:2000]
        m.duration_ms = (time.time() - started) * 1000
        return False

    return True


def _check_case_output_contract(case, options, inputs, input_fp, outputs, m, started):
    actual, expected = outputs
    now_fp = [tensor_fp(t) for t in inputs]
    if now_fp != input_fp:
        m.output_ok = False
        m.output_err = (
            "SUB-EXE-004: 实现就地修改了输入张量；"
            "被测实现只有计算并返回的权利，不得改动输入"
        )
        m.compile_err = m.output_err
        m.duration_ms = (time.time() - started) * 1000
        return False

    # ---------- 运行期兜底（静态扫描看不到）----------
    # 必须在 _as_tensor 之前检查原始返回值：元组输出（如 cummin 的
    # values/indices）经归一化后只剩第一个元素，包装器与张量子类
    # 也只有在原始对象上才认得出来。
    if options.category is not None:
        from ..validator import check_output

        violations = check_output(case.operator, actual, options.category,
                                  expected_device=inputs[0].device, expected_count=len(as_outputs(expected)))
        if violations:
            m.output_ok = False
            m.output_err = f"{violations[0].rule_id}: {violations[0].detail}"
            if not m.compile_err:
                m.compile_err = m.output_err
            m.duration_ms = (time.time() - started) * 1000
            return False

    return True


def _compare_case_outputs(case, actual, expected, m):
    actual = as_outputs(actual)
    expected = as_outputs(expected)
    # 记录参考输出的实际 dtype：容差必须按它查表，而不是 case.dtype
    if expected:
        m.output_dtype = str(expected[0].dtype).replace("torch.", "")

    # 输出元数必须一致：少交一个输出等价于「没算」，
    # 曾经只取第一个元素，于是少交输出照样 PASS（ADV-F2 实测）。
    if len(actual) != len(expected):
        m.shape_matches = False
        m.compile_err = (
            f"输出元数不符：期望 {len(expected)} 个，实际 {len(actual)} 个"
        )[:2000]
        return

    # ---------- 逐个输出比对，取最差者 ----------
    # 多输出算子的每个输出都必须对。曾经只比第一个，
    # 于是 cummin 的 indices 填垃圾也能全 PASS（ADV-F1 实测）。
    m.max_rel_err = 0.0
    m.max_abs_err = 0.0
    # 逐元素 allclose 判定（|impl-ref| ≤ atol + rtol·|ref|）。
    # 容差按参考输出 dtype 查表——与 judge 的聚合判定同一张表。
    from ..judge import tolerance_for_case

    allclose_all: list[bool] = []
    for a, e in zip(actual, expected):
        output_measurement = CaseMeasurement(
            case_code=case.code, operator=case.operator,
            output_dtype=str(e.dtype).replace("torch.", ""))
        _tol = tolerance_for_case(load_tolerance(), case, output_measurement)
        if a.shape != e.shape:
            m.shape_matches = False
            allclose_all.append(False)
            # 记下两侧形状：只报 SHAPE_MISMATCH 而不给具体形状，排查时无从下手
            # （PaddedGatherScatter 13 例即如此，结果里 compile_err 是空的）。
            m.compile_err = (
                f"输出 shape 不符：期望 {tuple(e.shape)}，实际 {tuple(a.shape)}"
            )[:2000]
            continue
        rel, ab, nan_ok = compare(a, e)
        m.max_rel_err = max(m.max_rel_err or 0.0, rel or 0.0)
        m.max_abs_err = max(m.max_abs_err or 0.0, ab or 0.0)
        if not nan_ok:
            allclose_all.append(False)
            m.compile_err = (
                "输出含 NaN/Inf：与参考逐元素比较时出现非有限值"
                f"（期望 {tuple(e.shape)}，实际 {tuple(a.shape)}）"
            )[:2000]
            continue
        allclose_all.append(output_matches(a, e, _tol))
    if allclose_all:
        m.allclose_ok = all(allclose_all)
        m.outputs_match = len(actual) == len(expected) and all(allclose_all)

    # 输出 dtype 必须等于目标 dtype。
    # 曾经完全不查，于是返回 float64 也能过（ADV-F3 实测）——
    # 用高精度算再交上来，等于绕开了低精度这条考题本身。
    _check_output_dtypes(actual, expected, m)


def _check_output_dtypes(actual, expected, m):
    if len(actual) == len(expected):
        for i, (a, e) in enumerate(zip(actual, expected)):
            if a.dtype != e.dtype:
                m.output_ok = False
                m.output_err = (
                    f"SUB-EXE-003: 第 {i} 个输出 dtype 为 {a.dtype}，"
                    f"期望 {e.dtype}——不得以更高精度绕过目标精度的考题"
                )
                m.compile_err = m.output_err
                break


def __getattr__(name):
    """Preserve the historical exception import without a nonstandard class alias."""
    if name == 'TimeoutError_':
        return EvaluationTimeoutError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
