# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反作弊检测（L0，静态分析部分无 NPU 依赖）。

我们的独特优势 vs 参考仓库：
    参考仓库需支持 AscendC（C++ 编译产物），**看不到源码**，
    因此反作弊只能靠运行时 guard，且布局变换类违规**只能人工审查**
    ——因为它怕误伤 Transpose / Gather 这类以数据搬运为核心语义的算子。

    我们只支持 triton（纯 Python，源码可见），因此可以：
      1. 静态 AST 分析，执行前就拒绝（比运行时更快、能定位到行号）
      2. 结合 category 做差异化白名单（我们知道算子语义族）

    把参考仓库的「人工审查项」自动化了。
"""

from __future__ import annotations

import ast
from dataclasses import dataclass
from pathlib import Path
from typing import Final

# ---------------- 规则编号 ----------------

RULE_DELEGATION: Final[str] = "SUB-BEH-001"
RULE_DEVICE_FALLBACK: Final[str] = "SUB-BEH-002"
RULE_CACHING: Final[str] = "SUB-BEH-003"
RULE_FAKE_TENSOR: Final[str] = "SUB-BEH-005"
RULE_TIMING_TAMPER: Final[str] = "SUB-BEH-006"
RULE_LAYOUT_MOVE: Final[str] = "SUB-BEH-007"
RULE_UNPARSEABLE: Final[str] = "SUB-BEH-008"
RULE_LAUNCH_CONFIG: Final[str] = "SUB-BEH-009"

# 把目标算子的核心计算交给框架内置接口。
# 覆盖直接调用与等价写法（运算符重载、算子变体）。
BUILTIN_COMPUTE_CALLS: Final[set[str]] = {
    # 归一化 / 激活
    "softmax", "log_softmax", "layer_norm", "group_norm", "rms_norm",
    "batch_norm", "normalize",
    # 逐元素数学：缺了它们，Elementwise 类算子最直白的作弊反而畅通
    # （实测：torch.exp 曾不在名单里，Exp 算子里直接调 torch.exp 判不出来）
    "exp", "pow", "sqrt", "rsqrt", "sigmoid", "silu", "relu", "gelu", "tanh",
    "maximum", "minimum", "clamp", "sin", "cos",
    # 归约：方法形态（x.sum()）是最高频的绕过写法
    "sum", "mean", "max", "min", "argmax", "argmin", "std", "var",
    # 张量构造与线性代数
    "matmul", "bmm", "mm", "dot", "addmm", "einsum", "conv1d", "conv2d",
    # 矩阵 / 卷积
    "matmul", "bmm", "mm", "conv1d", "conv2d", "conv3d",
    "conv_transpose2d", "linear",
    # 归约 / 选择
    "sum", "mean", "max", "min", "argmax", "argmin", "topk", "sort", "unique",
    "cumsum", "cummin", "cummax", "prod", "std", "var",
    # 损失 / 其他
    "cross_entropy", "nll_loss", "mse_loss", "gather", "scatter",
    "scatter_add", "index_select", "embedding",
}

# 主机侧往返：把张量搬回 CPU 计算再搬回加速器
DEVICE_FALLBACK_CALLS: Final[set[str]] = {"cpu", "to_cpu", "numpy", "tolist"}

# 计时 / 同步接口篡改
TIMING_TAMPER_CALLS: Final[set[str]] = {
    "synchronize", "npu_synchronize", "cuda_synchronize", "event_elapsed_time",
}

# 自动调优：自行搜索启动配置（BLOCK / num_warps / num_stages）。
# 启动配置应当是**评测方控制的变量**，由被测方自选等于把考题改成开卷；
# 且 autotune 的搜索开销不计入被测实现耗时，性能数据会失真。
LAUNCH_CONFIG_CALLS: Final[set[str]] = {"autotune"}

# 布局变换类操作：仅在 LayoutTransform 类别下允许
LAYOUT_MOVE_CALLS: Final[set[str]] = {
    "transpose", "permute", "contiguous", "reshape", "view", "flatten", "moveaxis",
}

# 以布局变换为核心语义的类别 —— 白名单
LAYOUT_ALLOWED_CATEGORIES: Final[set[str]] = {"LayoutTransform"}


@dataclass(frozen=True)
class Finding:
    """一条违规发现。"""

    rule_id: str
    message: str
    lineno: int | None = None

    def render(self) -> str:
        loc = f"（第 {self.lineno} 行）" if self.lineno else ""
        return f"[{self.rule_id}] {self.message}{loc}"


# triton 语言内置命名空间。tl.xxx 是真实的 triton 语义，
# 绝不能当成「交给框架内置接口」——曾经把 tl.max 误判为 torch.max，
# 导致所有合法的归约算子被误杀。
TRITON_NAMESPACES: Final[set[str]] = {"tl", "triton", "language"}


# 框架侧模块前缀。只有带这些前缀的调用才算「交给框架内置接口」。
# 裸名字（如 max(1, n)、sum([...])）是 Python 内置函数，不是代算——
# 曾经把 `max(1, triton.next_power_of_2(n))` 误判为 torch.max。
FRAMEWORK_PREFIXES: Final[set[str]] = {
    "torch", "F", "nn", "functional", "tf", "np", "numpy", "mindspore",
}

# Python 标准库：用它算标量（math.exp / math.sqrt）不是「交给框架」，
# 合法实现里拿它算个常量是常态。曾经把 math.sqrt 判成张量方法代算，
# 于是合法正控制（ADV-C1）被误杀——误杀比漏放更伤，
# 因为它会让提交方放弃本来合规的写法。
STDLIB_MODULES: Final[set[str]] = {
    "math", "cmath", "os", "sys", "json", "time", "random", "itertools",
    "functools", "re", "struct", "array", "collections",
}


class _ImportIndex:
    """把文件里的局部名字映射回真实模块根。

    为什么不直接比对源码文本：
        反作弊不能只认 `torch.xxx` 这一种写法。实测（ADV-D2 / ADV-D3）：
            import torch as _t                    → _t.exp(x)          绕过
            from torch import exp                 → exp(x)             绕过
            import torch.nn.functional as F       → F.softmax(x)       绕过
            torch.nn.functional.softmax(x)        → 属性链，只认最后一段  绕过
        四者与 `torch.exp()` 语义完全等价，扫描器曾经全部放行——
        于是「反作弊」对换写法零抵抗，只惩罚老实人。
    """

    def __init__(self, tree: ast.AST) -> None:
        self._alias: dict[str, str] = {}
        self._from_names: dict[str, str] = {}
        for node in ast.walk(tree):
            self._record_import(node)

    def resolve(self, name: str) -> str | None:
        """返回该名字对应的**顶级模块**；不是模块则返回 None。"""
        return self._alias.get(name) or self._from_names.get(name)

    def is_framework(self, name: str) -> bool:
        root = self.resolve(name)
        return root is not None and root in FRAMEWORK_PREFIXES

    def _record_import(self, node: ast.AST) -> None:
        if isinstance(node, ast.Import):
            for alias in node.names:
                key = alias.asname or alias.name.split(".")[0]
                self._alias[key] = alias.name.split(".")[0]
        elif isinstance(node, ast.ImportFrom) and node.module is not None:
            root = node.module.split(".")[0]
            for alias in node.names:
                if alias.name != "*":
                    self._from_names[alias.asname or alias.name] = root


class _CallTarget:
    """一次调用的解析结果。"""

    __slots__ = ("name", "root", "is_triton_builtin", "is_method", "lineno")

    def __init__(self, name, root, is_triton_builtin, is_method, lineno):
        self.name = name
        self.root = root
        self.is_triton_builtin = is_triton_builtin
        # 「方法形态」：接收者不是已知模块，且不是 tl/triton ——
        # 那它就是某个对象（绝大多数情况是张量）的方法调用。
        self.is_method = is_method
        self.lineno = lineno


def _resolve_call(node: ast.Call, index: _ImportIndex) -> _CallTarget | None:
    """解析一次调用：名字、所属模块根、是否 triton 内置、是否方法形态。"""
    func = node.func

    if isinstance(func, ast.Name):
        return _CallTarget(
            name=func.id,
            root=index.resolve(func.id),
            is_triton_builtin=False,
            is_method=False,
            lineno=node.lineno,
        )

    if isinstance(func, ast.Attribute):
        # 回溯属性链取最左端的名字：torch.nn.functional.softmax → torch
        base = func.value
        while isinstance(base, ast.Attribute):
            base = base.value
        base_name = base.id if isinstance(base, ast.Name) else ""

        root = index.resolve(base_name) if base_name else None
        # 方法形态的三种接收者，都必须认：
        #   x.exp()            —— 最左端是变量名
        #   (x * s).exp()      —— 最左端是表达式（实测 ADV-D1 就是这种）
        #   self._f(x).exp()   —— 链式调用
        # 曾经要求接收者必须是 Name，于是 `(x*a+b).exp()` 直接漏掉。
        #
        # 排除项：triton 命名空间（真实 kernel 语义）与标准库（算标量常量）。
        base_is_module = root is not None
        base_is_exempt = (base_name in TRITON_NAMESPACES
                          or base_name in STDLIB_MODULES
                          or (base_name in STDLIB_MODULES if base_name else False))
        is_method = not base_is_module and not base_is_exempt

        return _CallTarget(
            name=func.attr,
            root=root,
            is_triton_builtin=base_name in TRITON_NAMESPACES,
            is_method=is_method,
            lineno=node.lineno,
        )
    return None


def _call_name(node: ast.Call) -> tuple[str, str, bool]:
    """兼容旧接口：返回 (名字, 模块前缀, 是否 triton 内置)。

    只做单层解析，**不解析 import 别名**——保留给只需要粗略判定的调用方。
    真正的反作弊判定一律走 _resolve_call。
    """
    func = node.func
    if isinstance(func, ast.Attribute):
        base = func.value.id if isinstance(func.value, ast.Name) else ""
        return func.attr, base, base in TRITON_NAMESPACES
    if isinstance(func, ast.Name):
        return func.id, "", False
    return "", "", False


def _is_triton_kernel(func: ast.AST) -> bool:
    """判断一个函数定义是否被 @triton.jit 装饰。"""
    if not isinstance(func, (ast.FunctionDef, ast.AsyncFunctionDef)):
        return False
    for dec in func.decorator_list:
        if isinstance(dec, ast.Attribute) and dec.attr == "jit":
            return True
        if isinstance(dec, ast.Name) and dec.id == "jit":
            return True
    return False


def scan_source(
    source: str,
    *,
    category: str,
    entry: str,
) -> list[Finding]:
    """静态扫描单个实现文件。

    category 决定布局变换是否放行——这是按语义族差异化拦截的关键。
    """
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        # 不可解析 ≠ 代算。曾经一律报成 SUB-STR-001，
        # 于是提交方被告知「不要把计算交给框架」，而实际问题只是语法错了。
        return [Finding(RULE_UNPARSEABLE, f"源码无法解析：{exc.msg}", exc.lineno)]

    layout_allowed = category in LAYOUT_ALLOWED_CATEGORIES
    findings: list[Finding] = []
    index = _ImportIndex(tree)
    kernel_ranges = _kernel_ranges(tree)
    _scan_caching(tree, kernel_ranges, findings)
    _scan_timing_assignments(tree, kernel_ranges, findings)
    _scan_calls(tree, kernel_ranges, index, category, layout_allowed, findings)
    return findings


def _kernel_ranges(tree: ast.AST) -> list[tuple[int, int]]:

    # 收集被 @triton.jit 装饰的函数：其内部不算「交给框架」。
    # 记录行范围而非节点集合，避免对每个调用点做子树遍历。
    # 必须最先收集：后面的所有检测都要按行范围跳过 kernel 内部。
    kernel_ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if _is_triton_kernel(node):
            start = getattr(node, "lineno", 0)
            end = max(
                (getattr(n, "lineno", start) for n in ast.walk(node)),
                default=start,
            )
            kernel_ranges.append((start, end))
    return kernel_ranges


def _scan_caching(tree, kernel_ranges, findings):

    # 地址键缓存走 **AST** 检测：
    # 曾经用文本匹配 `data_ptr`，于是注释或 docstring 里提到这个词
    # 也会被判违规（ADV-E5 实测误拒）。AST 不含注释，天然免疫。
    # 另一个入口是 id(x)：对象地址复用同样会让陈旧结果被命中（ADV-D5）。
    for node in ast.walk(tree):
        if _in_ranges(getattr(node, "lineno", 0), kernel_ranges):
            continue
        hit = None
        if isinstance(node, ast.Attribute) and node.attr == "data_ptr":
            hit = "data_ptr()"
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) \
                and node.func.id == "id":
            hit = "id()"
        if hit:
            findings.append(
                Finding(
                    RULE_CACHING,
                    f"检测到以对象地址为依据的缓存键（{hit}）："
                    "输入轮换后陈旧结果会被复用，等价于根本没有计算",
                    node.lineno,
                )
            )


def _scan_timing_assignments(tree, kernel_ranges, findings):
    # 赋值型篡改必须**单独扫一遍**：
    # `torch.npu.synchronize = lambda: None` 里根本没有 Call 节点，
    # 只在下面的调用循环里检查会完全漏掉。
    for node in ast.walk(tree):
        if _in_ranges(getattr(node, "lineno", 0), kernel_ranges):
            continue
        if _is_timing_tamper_assign(node):
            findings.append(
                Finding(
                    RULE_TIMING_TAMPER,
                    "检测到计时/同步接口被重新赋值；"
                    "不得替换、伪装或删除评测的计时与同步",
                    node.lineno,
                )
            )


def _scan_calls(tree, kernel_ranges, index, category, layout_allowed, findings):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        # kernel 内部的调用不检查：那是真实的 triton 语义
        if _in_ranges(node.lineno, kernel_ranges):
            continue

        target = _resolve_call(node, index)
        if target is None or not target.name:
            continue
        # 自动调优要**先于** triton 内置豁免判定：
        # 它挂在 triton 命名空间下，若先被当成真实 kernel 语义放行就永远抓不到。
        if target.root == "triton" and target.name in LAUNCH_CONFIG_CALLS:
            findings.append(
                Finding(
                    RULE_LAUNCH_CONFIG,
                    "检测到自动调优（autotune）：启动配置应由评测方给定，"
                    "由被测方自行搜索等于把考题改成开卷，且其搜索开销不计入耗时",
                    node.lineno,
                )
            )
            continue

        if target.is_triton_builtin:
            # tl.xxx / triton.xxx 是真实实现，不是代算
            continue
        finding = _call_finding(node, target, category, layout_allowed)
        if finding is not None:
            findings.append(finding)


def _call_finding(node, target, category, layout_allowed):
    name = target.name
    is_framework_call = target.root in FRAMEWORK_PREFIXES if target.root else False
    is_tensor_method = target.root is None and (target.is_method or _is_tensor_method(node))
    if _is_setattr_tamper(node):
        return Finding(
            RULE_TIMING_TAMPER,
            "检测到通过 setattr 替换计时/同步接口；"
            "不得替换、伪装或删除评测的计时与同步",
            node.lineno,
        )
    if name in BUILTIN_COMPUTE_CALLS and (is_framework_call or is_tensor_method):
        how = "方法调用" if (is_tensor_method and not is_framework_call) else "内置接口"
        return Finding(
            RULE_DELEGATION,
            f"将核心计算交给框架{how} {name}()；"
            f"{category} 类算子必须自行实现",
            node.lineno,
        )
    if _is_device_fallback(node, target, name):
        return Finding(
            RULE_DEVICE_FALLBACK,
            f"检测到主机侧往返 {name}()；不得把输入搬回主机计算",
            node.lineno,
        )
    if name in TIMING_TAMPER_CALLS and (is_framework_call or is_tensor_method):
        return Finding(
            RULE_TIMING_TAMPER,
            f"检测到计时/同步接口 {name}() 的调用；"
            "不得替换、伪装或删除评测的计时与同步",
            node.lineno,
        )
    if name in LAYOUT_MOVE_CALLS and not layout_allowed:
        return Finding(
            RULE_LAYOUT_MOVE,
            f"检测到布局变换 {name}()；{category} 类算子不属于布局变换，"
            "实质性数据搬运不被允许",
            node.lineno,
        )
    return None


def _is_timing_tamper_assign(node: ast.AST) -> bool:
    """判定是否为「给计时/同步接口赋值」（把 synchronize 替换成空操作）。"""
    if not isinstance(node, (ast.Assign, ast.AugAssign, ast.AnnAssign)):
        return False
    targets = node.targets if isinstance(node, ast.Assign) else [node.target]
    for target in targets:
        if not (isinstance(target, ast.Attribute) and target.attr in TIMING_TAMPER_CALLS):
            continue
        base = target.value
        while isinstance(base, ast.Attribute):
            base = base.value  # type: ignore[assignment]
        if isinstance(base, ast.Name) and base.id in TRITON_NAMESPACES:
            continue
        return True
    return False


def _is_device_fallback(node: ast.Call, target: _CallTarget, name: str) -> bool:
    """设备回退判定。

    `x.cpu()` 的名字在名单里，好抓；但 `x.to("cpu")` 的方法名是 `to`——
    只看名字永远抓不到，必须看**字符串参数**（ADV-D4 实测绕过）。
    """
    if name in DEVICE_FALLBACK_CALLS:
        return target.is_method or _is_tensor_method(node)
    if name == "to":
        for arg in node.args:
            if not (isinstance(arg, ast.Constant) and isinstance(arg.value, str)):
                continue
            dev = arg.value.lower()
            if dev.startswith("cpu"):
                return True
    return False


def _is_setattr_tamper(node: ast.Call) -> bool:
    """判定 setattr 型篡改：setattr(torch.npu, "synchronize", fake)。

    与赋值形态同源，但走函数调用，扫描器按调用名查表自然漏掉。
    """
    if not isinstance(node.func, ast.Name) or node.func.id != "setattr":
        return False
    if len(node.args) < 2:
        return False
    attr = node.args[1]
    return isinstance(attr, ast.Constant) and attr.value in TIMING_TAMPER_CALLS


def _line_of(source: str, needle: str) -> int | None:
    """返回 needle 首次出现的行号（1-based）。找不到返回 None。"""
    for i, line in enumerate(source.splitlines(), start=1):
        if needle in line:
            return i
    return None


def _in_ranges(lineno: int, ranges: list[tuple[int, int]]) -> bool:
    """判定行号是否落在任一 kernel 的行范围内。"""
    return any(start <= lineno <= end for start, end in ranges)


def _is_tensor_method(node: ast.Call) -> bool:
    """判断是否为「某对象的方法调用」（如 x.cpu()、torch.npu.synchronize()）。

    用于设备回退与计时检测：这类调用的接收者是变量或模块属性，
    不能只按模块前缀判定——曾经漏检 x.cpu() 与 torch.npu.synchronize()。
    """
    func = node.func
    if not isinstance(func, ast.Attribute):
        return False
    # x.cpu()：接收者是 Name
    if isinstance(func.value, ast.Name):
        return func.value.id not in TRITON_NAMESPACES
    # torch.npu.synchronize()：接收者是 Attribute
    if isinstance(func.value, ast.Attribute):
        base = func.value
        while isinstance(base, ast.Attribute):
            base = base.value  # type: ignore[assignment]
        if isinstance(base, ast.Name):
            return base.id not in TRITON_NAMESPACES
    return False


def scan_file(path: Path, *, category: str, entry: str) -> list[Finding]:
    return scan_source(path.read_text(encoding="utf-8"), category=category, entry=entry)


# ---------------- 运行期检测（兜底） ----------------


def _try_import_torch():
    """运行期检测需要 torch；纯 L0 环境（无 torch）下应优雅降级。

    这些检测只在昇腾执行机上生效；本地无 torch 时不能让校验器崩溃——
    那会让整套 L0 测试在本地跑不起来。
    """
    try:
        import torch

        return torch
    except ModuleNotFoundError:
        return None


def looks_like_fake_tensor(obj: object) -> bool:
    """检测非 Tensor 输出，接纳 isinstance 认可的 Tensor 及其子类。

    类型身份不再要求精确等于 torch.Tensor；设备、布局、精度等检查独立执行。
    无 torch 时返回 False，交由实际执行阶段判定。
    """
    torch = _try_import_torch()
    if torch is None:
        return False

    if isinstance(obj, (list, tuple)):
        return any(looks_like_fake_tensor(x) for x in obj)
    return not isinstance(obj, torch.Tensor)


def detect_address_keyed_cache(source: str) -> bool:
    """检测以输入地址/指针为键的缓存（SUB-BEH-003）。"""
    return "data_ptr" in source or "data_ptr()" in source


def is_contiguous_output(output: object) -> bool:
    """输出是否连续。非连续在布局类之外属结构性失败。

    无 torch 时返回 True（无法判定，交由实际执行阶段）。
    """
    torch = _try_import_torch()
    if torch is None:
        return True
    if isinstance(output, torch.Tensor):
        return bool(output.is_contiguous())
    return True
