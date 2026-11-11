# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""提交的三阶段校验（L0，无 NPU 依赖）。

    静态 AST  →  加载  →  执行

设计要点：
    1. 前一阶段失败即**短路**，不进入后续阶段。
       省一次冷编译（实测昇腾上冷编译约 22 秒）。
    2. 每条规则有**唯一编号**与**声明的失败结果**，可被引用与测试。
    3. 任何未预期异常都被归入阶段失败，**绝不终止整体评测**。
    4. 校验**确定性**：同一输入两次结果必须一致。
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass, field
from pathlib import Path

from .anticheat import scan_source
from .fixture_sources import fixture_path
from .operator_registry import by_name
from .operatorspec import OperatorSpec
from .submission import Submission, load_all, load_manifest

# ---------------- 阶段 ----------------

STAGE_STATIC = "static"
STAGE_LOAD = "load"
STAGE_EXEC = "exec"

# ---------------- 规则编号与失败结果 ----------------

RULES: dict[str, str] = {
    # 静态
    "SUB-STR-001": "静态阶段：将核心计算交给框架内置接口",
    "SUB-STR-002": "静态阶段：主机侧往返计算",
    "SUB-STR-003": "静态阶段：以输入地址为键的缓存",
    "SUB-STR-004": "静态阶段：布局变换越权（非布局类算子）",
    "SUB-STR-005": "静态阶段：计时或同步接口被干预",
    "SUB-STR-006": "静态阶段：源码无法解析",
    "SUB-STR-007": "静态阶段：自行搜索启动配置（autotune）",
    # 加载
    "SUB-LOD-001": "加载阶段：清单文件缺失",
    "SUB-LOD-002": "加载阶段：清单声明了未登记的算子",
    "SUB-LOD-003": "加载阶段：实现文件不存在",
    "SUB-LOD-004": "加载阶段：入口函数缺失",
    "SUB-LOD-005": "加载阶段：入口函数不可调用",
    "SUB-LOD-006": "加载阶段：入口函数参数个数与规格不符",
    "SUB-LOD-007": "加载阶段：入口函数参数名与规格不符",
    "SUB-LOD-008": "加载阶段：实现路径逃出提交目录",
    "SUB-LOD-009": "加载阶段：入口含仅限位置参数，无法按约定传入属性",
    # 执行
    "SUB-EXE-001": "执行阶段：输出未通过 Tensor 实例检查",
    "SUB-EXE-002": "执行阶段：输出非连续且该算子不属于布局变换类",
    "SUB-EXE-003": "执行阶段：输出 dtype 与目标 dtype 不符",
    "SUB-EXE-004": "执行阶段：实现就地修改了输入张量",
    "SUB-EXE-005": "执行阶段：关键接口在提交代码运行期间被替换（运行时篡改校验）",
    "SUB-EXE-006": "执行阶段：输出设备与输入设备不符",
    # 包级（zip 摄取；规则详情见 package.py——独立成层是刻意的：
    # 包规则与内容规则分属「收包方」与「运行器」两个角色）
    "SUB-PKG-001": "包级：不是有效的 zip 文件（损坏 / 加密 / 头部与内容不一致）",
    "SUB-PKG-002": "包级：条目数超过上限",
    "SUB-PKG-003": "包级：解压后总大小超过上限",
    "SUB-PKG-004": "包级：单个文件解压后超过上限",
    "SUB-PKG-005": "包级：压缩比异常，疑似压缩炸弹",
    "SUB-PKG-006": "包级：条目路径不安全（绝对路径 / 上跳 / 反斜杠 / 盘符）",
    "SUB-PKG-007": "包级：包含符号链接或其它非常规文件条目",
    "SUB-PKG-008": "包级：扩展名不在白名单（仅允许纯源码与文档）",
    "SUB-PKG-009": "包级：解压后找不到 manifest.yaml（提交清单缺失）",
}


# 反作弊行为规则 → 静态阶段规则的映射。
# 行为规则沿用参考仓库的 SUB-BEH-* 编号（语义一致，便于对照），
# 但对外统一以 SUB-STR-*（静态阶段）报出，因为我们在执行前就拦截。
BEHAVIOR_TO_STATIC_RULE: dict[str, str] = {
    "SUB-BEH-001": "SUB-STR-001",  # 内置接口代算
    "SUB-BEH-002": "SUB-STR-002",  # 主机侧往返
    "SUB-BEH-003": "SUB-STR-003",  # 地址键缓存
    "SUB-BEH-006": "SUB-STR-005",  # 计时/同步干预
    "SUB-BEH-007": "SUB-STR-004",  # 布局变换越权
    "SUB-BEH-008": "SUB-STR-006",  # 源码无法解析
    "SUB-BEH-009": "SUB-STR-007",  # 自行搜索启动配置
}


@dataclass
class SubmissionValidationError:
    """一条校验失败。"""

    rule_id: str
    stage: str
    operator: str
    detail: str
    lineno: int | None = None

    @property
    def description(self) -> str:
        return RULES.get(self.rule_id, self.rule_id)

    def render(self) -> str:
        loc = f" 位置：第 {self.lineno} 行" if self.lineno else ""
        return (
            f"[{self.rule_id}] {self.description}\n"
            f"  算子：{self.operator}  阶段：{self.stage}\n"
            f"  详情：{self.detail}{loc}"
        )




@dataclass
class ValidationResult:
    """对一次提交的校验结果。"""

    submit_id: str
    errors: list[SubmissionValidationError] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.errors

    def errors_for(self, operator: str) -> list[SubmissionValidationError]:
        return [e for e in self.errors if e.operator == operator]

    def render(self) -> str:
        if self.ok:
            return f"提交 {self.submit_id} 校验通过"
        lines = [f"提交 {self.submit_id} 校验失败，共 {len(self.errors)} 项：", ""]
        for e in self.errors:
            lines.append(e.render())
            lines.append("")
        return "\n".join(lines).rstrip()


# ---------------- 阶段一：静态 ----------------


def _add_error(result, rule, stage, operator, detail):
    result.errors.append(SubmissionValidationError(rule, stage, operator, detail))


def validate_static(sub_dir: Path, result: ValidationResult) -> None:
    """静态 AST 扫描。不加载、不执行。"""
    sub_dir = fixture_path(Path(sub_dir))
    try:
        manifest = load_manifest(sub_dir)
    except FileNotFoundError as exc:
        _add_error(result, "SUB-LOD-001", STAGE_LOAD, "-", str(exc))
        return
    except Exception as exc:  # noqa: BLE001
        _add_error(result, "SUB-LOD-001", STAGE_LOAD, "-", f"{type(exc).__name__}: {exc}")
        return

    for entry in manifest.operators:
        # 路径逃逸必须先判：提交目录是唯一的信任边界。
        # 曾经直接 `sub_dir / entry.path` 拼接，于是
        # `path: ../../../etc/xxx` 能读到提交目录之外的文件并被执行，
        # 而快照只扫 root 内部，逃逸文件完全不可见（ADV-E1 实测 3×PASS）。
        try:
            resolved = (sub_dir / entry.path).resolve()
            inside = resolved.is_relative_to(sub_dir.resolve())
        except Exception:  # noqa: BLE001
            inside = False
        if not inside:
            _add_error(result, "SUB-LOD-008", STAGE_STATIC, entry.name, f"实现路径逃出提交目录：{entry.path}")
            continue

        path = sub_dir / entry.path
        if not path.is_file():
            _add_error(result, "SUB-LOD-003", STAGE_STATIC, entry.name, f"实现文件不存在：{entry.path}")
            continue

        spec = by_name(entry.name)
        category = spec.category.value if spec else ""
        if spec is None:
            _add_error(result, "SUB-LOD-002", STAGE_STATIC, entry.name, f"清单声明了未登记的算子：{entry.name}")
            continue

        try:
            source = path.read_text(encoding="utf-8")
        except Exception as exc:  # noqa: BLE001
            _add_error(result, "SUB-LOD-003", STAGE_STATIC, entry.name,
                       f"无法读取实现文件：{type(exc).__name__}: {exc}")
            continue

        for finding in scan_source(source, category=category, entry=entry.entry):
            rule = BEHAVIOR_TO_STATIC_RULE.get(finding.rule_id, "SUB-STR-001")
            result.errors.append(
                SubmissionValidationError(
                    rule, STAGE_STATIC, entry.name, finding.message, finding.lineno
                )
            )


# ---------------- 阶段二：加载 ----------------


def _param_names(func: object) -> list[str]:
    try:
        params = inspect.signature(func).parameters
    except (TypeError, ValueError):
        return []
    # 属性按 **kwargs 传入（见 operator_registry_ascend_ops 契约），
    # VAR_KEYWORD 不计入参数个数/名称比对，否则带 **attrs 的入口会被误判。
    return [n for n, p in params.items()
            if p.kind != inspect.Parameter.VAR_KEYWORD]


def _positional_only_names(func: Any) -> list[str]:
    """返回仅限位置参数名（`def f(x, base, /)` 中 `/` 之前的那些）。

    参数名与个数都对，但属性是按关键字传入的，这类签名调用必然 TypeError。
    曾经只看名字与个数，于是它通过校验、执行时炸掉，
    而失败记成 RUNTIME_FAIL 落在提交方头上（ADV-E4 实测）——
    那是校验器漏检，不是提交方的运行期错误。
    """
    try:
        sig = inspect.signature(func)
    except (TypeError, ValueError):
        return []
    return [
        p.name for p in sig.parameters.values()
        if p.kind is inspect.Parameter.POSITIONAL_ONLY
    ]


def signature_errors(func: object, spec: OperatorSpec) -> list[SubmissionValidationError]:
    """Apply the same entry contract to imported functions and source signatures."""
    errors: list[SubmissionValidationError] = []
    # 仅限位置参数（def f(x, base, /)）必须拒绝：
    # 参数名和个数都对，但属性是按**关键字**传入的，
    # 调用必然 TypeError。曾经只看名字与个数，于是这类签名通过校验、
    # 执行时炸掉，而失败被记成 RUNTIME_FAIL 落在提交方头上（ADV-E4 实测）。
    # 那是校验器的漏检，不是提交方的运行期错误。
    posonly = _positional_only_names(func)
    if posonly:
        errors.append(
            SubmissionValidationError(
                "SUB-LOD-009", STAGE_LOAD, spec.name,
                f"入口含仅限位置参数 {posonly}；"
                "属性以关键字传入，仅限位置参数无法接收",
            )
        )
        return errors

    actual = _param_names(func)
    expected = spec.params
    if len(actual) != len(expected):
        errors.append(
            SubmissionValidationError(
                "SUB-LOD-006", STAGE_LOAD, spec.name,
                f"参数个数不符：期望 {len(expected)} 个 {expected}，"
                f"实际 {len(actual)} 个 {actual}",
            )
        )
        return errors
    if actual != expected:
        errors.append(
            SubmissionValidationError(
                "SUB-LOD-007", STAGE_LOAD, spec.name,
                f"参数名不符：期望 {expected}，实际 {actual}",
            )
        )
    return errors


def validate_load(sub: Submission, result: ValidationResult) -> None:
    """加载与签名校验。静态已失败的算子不再重复加载。"""
    statically_failed = {e.operator for e in result.errors if e.stage == STAGE_STATIC}

    for entry in sub.manifest.operators:
        if entry.name in statically_failed:
            continue  # 短路：静态失败不进入加载

        spec = by_name(entry.name)
        if spec is None:
            result.errors.append(
                SubmissionValidationError(
                    "SUB-LOD-002", STAGE_LOAD, entry.name,
                    f"未登记的算子：{entry.name}",
                )
            )
            continue

        try:
            from .safeexec import guard_call
            from .submission import load_operator

            # guard_call 把 SystemExit/GeneratorExit 转成普通 Exception：
            # 提交代码的 import 期可以随便调用 sys.exit()，那会穿透下面
            # 所有的 except，让整个进程静默消失（实测：无输出、退出码 0）。
            loaded = guard_call(load_operator, sub, entry)
        except FileNotFoundError as exc:
            _add_error(result, "SUB-LOD-003", STAGE_LOAD, entry.name, str(exc))
            continue
        except AttributeError as exc:
            _add_error(result, "SUB-LOD-004", STAGE_LOAD, entry.name, str(exc))
            continue
        except TypeError as exc:
            _add_error(result, "SUB-LOD-005", STAGE_LOAD, entry.name, str(exc))
            continue
        except Exception as exc:  # noqa: BLE001
            result.errors.append(
                SubmissionValidationError(
                    "SUB-LOD-005", STAGE_LOAD, entry.name,
                    f"{type(exc).__name__}: {exc}",
                )
            )
            continue
        # 兜底：任何 BaseException（除操作者中断）都必须转成违规记录。
        # 漏掉这一层，提交方就能用一份文件把评测进程整个带走。
        except BaseException as exc:  # noqa: BLE001
            if isinstance(exc, KeyboardInterrupt):
                raise
            result.errors.append(
                SubmissionValidationError(
                    "SUB-LOD-005", STAGE_LOAD, entry.name,
                    f"提交试图终止进程：{type(exc).__name__}: {exc}",
                )
            )
            continue

        result.errors.extend(signature_errors(loaded.func, spec))


# ---------------- 阶段三：执行（运行期兜底） ----------------


def check_output(
    entry_name: str, output: object, category: str, expected_device=None, *, expected_count=None
) -> list[SubmissionValidationError]:
    """执行后的输出判定：返回违规列表，不修改任何状态。

    纯判定而非「追加到 result」的原因：
        执行期兜底必须能被 L1 的执行器**逐用例**调用，而执行器手里没有
        ValidationResult——它只需要一个「这次输出合不合规」的答案。
        曾经只有追加式版本，于是这条规则只在测试里跑过，
        真实评测中从不生效：上传一份返回包装器或非连续视图的实现，
        照样能通过校验并拿 PASS。
    """
    from .anticheat import looks_like_fake_tensor

    sequence = output if isinstance(output, (list, tuple)) else (output,)
    nested = any(isinstance(item, (list, tuple)) for item in sequence)
    invalid_nesting = expected_count is not None and nested
    if not sequence or invalid_nesting or looks_like_fake_tensor(output):
        return [SubmissionValidationError(
            "SUB-EXE-001", STAGE_EXEC, entry_name,
            "输出必须是 Tensor 或非空的扁平 Tensor 序列（不接受包装器或嵌套容器）",
        )]
    tensors = flatten_output_tensors(output)
    return check_output_tensors(entry_name, tensors, category, expected_device, expected_count)


def flatten_output_tensors(output):
    """Preserve recursive standalone validation for callers without an arity contract."""
    if not isinstance(output, (list, tuple)):
        return [output]
    return [tensor for item in output for tensor in flatten_output_tensors(item)]


def check_output_tensors(entry_name, tensors, category, expected_device, expected_count):
    """Apply device, arity and layout precedence to a validated flat output sequence."""
    from .anticheat import is_contiguous_output

    if expected_device is not None:
        for tensor in tensors:
            if tensor.device != expected_device:
                return [SubmissionValidationError(
                    "SUB-EXE-006", STAGE_EXEC, entry_name,
                    f"输出设备 {tensor.device} 与输入设备 {expected_device} 不符",
                )]
    # Arity is diagnosed by the comparator as SHAPE_MISMATCH, regardless of
    # incidental strides. Invalid types/devices must still be rejected first.
    if expected_count is not None and len(tensors) != expected_count:
        return []
    if not category or category == "LayoutTransform":
        return []
    for tensor in tensors:
        if not is_contiguous_output(tensor):
            return [SubmissionValidationError(
                "SUB-EXE-002", STAGE_EXEC, entry_name,
                f"输出非连续，而 {category} 类算子不属于布局变换类；"
                "这属结构性失败而非精度不达标",
            )]
    return []


def validate_exec_output(
    entry_name: str, output: object, spec: OperatorSpec | None,
    result: ValidationResult,
) -> None:
    """执行后的输出校验。静态检测的兜底。"""
    category = spec.category.value if spec is not None else ""
    result.errors.extend(check_output(entry_name, output, category))


# ---------------- 入口 ----------------


def validate_submission(sub_dir: Path) -> ValidationResult:
    """完整校验一次提交。返回结构化结果，不抛异常。"""
    sub_dir = fixture_path(Path(sub_dir))
    result = ValidationResult(submit_id=sub_dir.name)

    try:
        validate_static(sub_dir, result)
        # 静态失败短路：不进入加载
        if any(e.stage == STAGE_STATIC for e in result.errors):
            return result

        # 加载阶段需要 torch（实现会 import torch）。
        # 纯 L0 环境（本地开发机）没有 torch，此时不应进入加载——
        # 否则会把「环境缺依赖」误报成「提交不可调用」。
        import importlib.util

        if importlib.util.find_spec("torch") is None:
            return result

        sub = Submission(sub_dir, load_manifest(sub_dir))
        validate_load(sub, result)
    except Exception as exc:  # noqa: BLE001
        # 任何未预期异常都归入失败，绝不终止整体评测
        result.errors.append(
            SubmissionValidationError(
                "SUB-LOD-001", STAGE_LOAD, "-",
                f"校验过程出现未预期异常：{type(exc).__name__}: {exc}",
            )
        )
    return result


def list_rules() -> list[tuple[str, str]]:
    return sorted(RULES.items())


def __getattr__(name):
    """Preserve the historical exception import without a nonstandard class alias."""
    if name == 'ValidationError_':
        return SubmissionValidationError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
