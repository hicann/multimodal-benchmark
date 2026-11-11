# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""规范资产加载（L0）。

只做加载与校验，不含任何 NPU 依赖，可在 macOS 上完整测试。
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from .models import CaseStatus

PROJECT_ROOT = Path(__file__).resolve().parents[2]
SPEC_DIR = PROJECT_ROOT / "spec"


@dataclass(frozen=True)
class Tolerance:
    rel: float
    abs: float

    @property
    def is_exact(self) -> bool:
        """整型要求精确相等。"""
        return self.rel == 0.0 and self.abs == 0.0


class ToleranceTable:
    """按 dtype 分级的容差表。禁止 case 级覆盖。

    算子级覆盖（operator_overrides）：仅当参考实现在该硬件上的**自身**
    偏差超过默认档时登记——每条必须在 yaml 里附归因证据（见
    docs/spec-conflicts.md C2）。查表顺序：算子+dtype 覆盖 → dtype 默认档。
    """

    def __init__(self, mapping: dict[str, Tolerance], reference_dtype: str,
                 overrides: dict[str, dict[str, Tolerance]] | None = None) -> None:
        self._mapping = {k.lower(): v for k, v in mapping.items()}
        # 覆盖键按候选形式展开注册（yaml 用算子 ID，如 grouped_matmul；
        # 查询端来的是注册表类名，如 GroupedMatmul——两边都要能命中）
        self._overrides: dict[str, dict[str, Tolerance]] = {}
        for op, per in (overrides or {}).items():
            for key in self._op_keys(op):
                self._overrides[key] = {d.lower(): t for d, t in per.items()}
        self.reference_dtype = reference_dtype

    @staticmethod
    def _op_keys(operator: str) -> list[str]:
        """算子名的候选键，兼容两种写法。

        注册表里 measurement.operator 是类名（GroupedMatmul），yaml 覆盖键
        是算子 ID（grouped_matmul）。只做 lower() 的话 Gelu 碰巧命中、
        GroupedMatmul 变 groupedmatmul 而错失覆盖档——2026-10-12 真机
        复核实测穿帮（grouped_matmul-f32-M18 冤案复发、gelu 却正常），
        单测当时用 slug 喂入所以全绿。三种候选：全小写、camel→snake、
        去下划线，任意组合的注册/查询都能相交。
        """
        keys: list[str] = []
        for k in (operator.lower(),
                  re.sub(r"(?<!^)(?=[A-Z])", "_", operator).lower(),
                  operator.lower().replace("_", "")):
            if k not in keys:
                keys.append(k)
        return keys

    def for_dtype(self, dtype: str, operator: str | None = None) -> Tolerance:
        key = dtype.lower()
        if operator:
            tol = self._override_for(key, operator)
            if tol is not None:
                return tol
        if key not in self._mapping:
            raise KeyError(f"容差表未覆盖 dtype={dtype!r}；请先在 spec/tolerance.yaml 中定义")
        return self._mapping[key]

    def has(self, dtype: str, operator: str | None = None) -> bool:
        key = dtype.lower()
        if operator and self._override_for(key, operator) is not None:
            return True
        return key in self._mapping

    def _override_for(self, dtype_key: str, operator: str) -> Tolerance | None:
        for key in self._op_keys(operator):
            per = self._overrides.get(key)
            if per and dtype_key in per:
                return per[dtype_key]
        return None


@dataclass(frozen=True)
class Profile:
    name: str
    repeats: int
    warmup_ms: int
    rep_ms: int
    measure_perf: bool
    cv_gate: float


class Spec:
    """评测规范。"""

    def __init__(self, raw: dict[str, Any]) -> None:
        self.raw = raw
        self.spec_version: str = raw["spec_version"]
        self.scorer_version: str = raw["scorer_version"]
        self.case_timeout_s: int = raw.get("case_timeout_s", 60)
        self.compose_total: bool = raw.get("compose_total", False)
        self.fast_thresholds: list[float] = list(
            raw.get("performance", {}).get("fast_thresholds", [1.0, 1.2, 2.0])
        )
        # 两轮对比口径：加速比显著变化的门槛（compare 命令用）
        self.speedup_change_gate: float = float(
            raw.get("compare", {}).get("speedup_change_gate", 0.10)
        )
        self._profiles = {
            name: Profile(
                name=name,
                repeats=cfg.get("repeats", 1),
                warmup_ms=cfg.get("warmup_ms", 25),
                rep_ms=cfg.get("rep_ms", 100),
                measure_perf=cfg.get("measure_perf", False),
                cv_gate=cfg.get("cv_gate", 0.05),
            )
            for name, cfg in (raw.get("profiles") or {}).items()
        }

    @property
    def profile_names(self) -> list[str]:
        return sorted(self._profiles)

    def profile(self, name: str) -> Profile:
        if name not in self._profiles:
            raise KeyError(f"未定义的运行档位 {name!r}；可选：{sorted(self._profiles)}")
        return self._profiles[name]


def load_tolerance(path: Path | None = None) -> ToleranceTable:
    path = path or (SPEC_DIR / "tolerance.yaml")
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    mapping = {
        dtype: Tolerance(rel=float(v["rel"]), abs=float(v["abs"]))
        for dtype, v in (raw.get("default") or {}).items()
    }
    overrides: dict[str, dict[str, Tolerance]] = {}
    for op, per in (raw.get("operator_overrides") or {}).items():
        overrides[op] = {
            dtype: Tolerance(rel=float(v["rel"]), abs=float(v["abs"]))
            for dtype, v in (per or {}).items()
        }
    return ToleranceTable(mapping, reference_dtype=raw.get("reference_dtype", "float64"),
                          overrides=overrides)


def load_spec(path: Path | None = None) -> Spec:
    path = path or (SPEC_DIR / "spec.yaml")
    return Spec(yaml.safe_load(path.read_text(encoding="utf-8")))


def load_feature_atoms(path: Path | None = None) -> dict[str, Any]:
    path = path or (SPEC_DIR / "feature_atoms.yaml")
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def known_statuses() -> list[str]:
    return [s.value for s in CaseStatus]
