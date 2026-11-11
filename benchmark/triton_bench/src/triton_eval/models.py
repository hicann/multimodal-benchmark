# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""数据模型与终态定义。

这里的定义是「无歧义」的落点：
  - 终态枚举互斥且完备（MECE），每个 case 结果只能落其中之一
  - severe 与 counted 由 status 派生，不落库列
  - CaseSpec 用 extra="forbid" 从 schema 层面禁止 case 级容差覆盖
"""

from __future__ import annotations

from enum import Enum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class CaseSource(str, Enum):
    """用例来源。用于隔离「被测方自带用例」——那是刷分后门，必须可识别。"""

    ATEN_ALIGNED = "aten_aligned"
    REALWORLD = "realworld"
    PARAM_VARIANT = "param_variant"
    VENDOR_SELFTEST = "vendor_selftest"


class CaseStatus(str, Enum):
    """用例终态：8 个值，互斥且完备。

    超时不单列——合并进 COMPILE_FAIL / RUNTIME_FAIL，细节留在 compile_err。
    这样状态机不会因为「卡死」和「报错」多出两个分支而让库表与报告变复杂。
    """

    SPEC_ERROR = "SPEC_ERROR"            # case 定义非法（schema 不过）
    COMPILE_FAIL = "COMPILE_FAIL"        # 编译失败或编译超时
    RUNTIME_FAIL = "RUNTIME_FAIL"        # 运行失败或运行超时
    PASS = "PASS"                        # 误差在容差内
    WRONG_RESULT = "WRONG_RESULT"        # 静默错误：编译运行均通过，结果错
    SHAPE_MISMATCH = "SHAPE_MISMATCH"    # 输出 shape/dtype 与参考不符
    ORACLE_FAIL = "ORACLE_FAIL"          # 参考实现自身失败或交叉校验不一致
    ENV_ABORT = "ENV_ABORT"              # 环境异常导致无法判定
    SUBMISSION_INVALID = "SUBMISSION_INVALID"  # 提交不合规：校验未过或无对应实现


# 不计入评分分母的终态：这些都是 corpus / 评测方 / 提交合规性的问题，
# 被测方不为其背分，但必须落库并告警。
NON_COUNTED_STATUSES = frozenset(
    {
        CaseStatus.SPEC_ERROR,
        CaseStatus.ORACLE_FAIL,
        CaseStatus.ENV_ABORT,
        CaseStatus.SUBMISSION_INVALID,
    }
)


def is_severe(status: CaseStatus) -> bool:
    """静默错误是最高优先级风险信号：编译运行都通过，但结果是错的。"""
    return status is CaseStatus.WRONG_RESULT


def is_counted(status: CaseStatus) -> bool:
    """是否计入评分分母。"""
    return status not in NON_COUNTED_STATUSES


class OperatorSpec(BaseModel):
    """算子定义。评分以算子为聚合单位。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="算子名，如 add / rms_norm")
    source: CaseSource = Field(description="算子来源")
    weight: float = Field(description="权重 ω_o，随 corpus 冻结，不随 run 变化")
    ref_path: str = Field(description="第二参考实现（CPU 路径）相对路径")
    desc: str = Field(default="", description="算子简述")


class CaseSpec(BaseModel):
    """一个算子变体 = 一个评测最小单元。

    extra="forbid" 有两重作用：
      1. 使「case 级容差覆盖」在 schema 层面无法表达（不是靠文档约定）
      2. 使「case 绑定 kernel」在 schema 层面无法表达

    第 2 点是支持用户上传的前提：用例只描述**输入数据与判定参数**，
    被测实现由提交方提供。若用例自带 kernel，评测方就在替被测方写实现。
    """

    model_config = ConfigDict(extra="forbid")

    code: str = Field(description="业务唯一键，如 exp-f32-00，报告与缺陷清单用")
    operator: str = Field(description="所属算子名，用于按算子聚合")
    source: CaseSource = Field(description="用例来源")
    dtype: str = Field(description="目标数据类型，容差按此查表")
    shape: list[int] = Field(description="主输入形状")
    attrs: dict[str, Any] = Field(
        default_factory=dict,
        description="算子属性（dim / perm / group_list / reduction ...）。"
        "必填属性必须在此给出——单个 shape 推不出「沿哪一维归约」这类语义，"
        "缺失时用例无法执行；未给出的可选项由算子规格默认值补齐",
    )
    # 数值范围维度（normal / zeros / mixed / extreme）。
    # 曾经只在派生层存在、冻结时丢弃——「五维派生」实际只有三维进了题库。
    # extreme 限 fp32 / bf16：fp16 的动态范围装不下 ±20σ 的中间结果。
    value_range: dict[str, Any] = Field(
        default_factory=lambda: {"kind": "normal"},
        description="输入数值分布：kind ∈ normal / zeros / mixed / extreme",
    )
    # 布局维度：非连续输入考验实现处理 stride 的能力（或显式聚合）
    contiguous: bool = Field(
        default=True,
        description="主输入是否连续；False 为跨步构造（逻辑 shape 不变）",
    )
    seed: int = Field(description="输入数据随机种子，保证可复现")
    launch: dict[str, Any] = Field(
        default_factory=dict,
        description="固定启动配置 {grid, BLOCK_M, BLOCK_N, num_warps, num_stages}。"
        "性能主口径禁 autotune，配置必须写死",
    )
    timeout_s: int = Field(default=60, description="单 case 硬超时（秒）")
    feature_atoms: list[str] = Field(
        default_factory=list, description="触及的 FA id 列表，覆盖度证据"
    )
    content_hash: str = Field(
        default="", description="kernel 文件 + 本行配置的 sha256，用于防篡改校验"
    )


class CorpusManifest(BaseModel):
    """冻结测试集清单。"""

    model_config = ConfigDict(extra="forbid")

    version: str = Field(description="版本标识，如 v0.3-multi-input-frozen")
    spec_version: str = Field(description="评测规范版本")
    feature_atoms_version: str = Field(default="", description="FA 清单版本")
    frozen_at: str = Field(description="冻结时间，ISO8601 UTC")
    operators: list[OperatorSpec]
    cases: list[CaseSpec]
    manifest_hash: str = Field(
        default="", description="本清单规范化后的 sha256（不含本字段）"
    )
    excluded: list[dict[str, str]] = Field(
        default_factory=list,
        description="确定性筛查中被剔除的用例及其原因",
    )


class CaseResultRecord(BaseModel):
    """一条用例结果。对应 results.jsonl 的一行。"""

    model_config = ConfigDict(extra="forbid")

    run_id: int = Field(description="所属运行（自增主键）")
    case_code: str = Field(description="用例业务键")
    operator: str = Field(description="所属算子名，冗余便于聚合")
    status: CaseStatus = Field(description="终态")

    duration_ms: float | None = Field(default=None, description="该用例总耗时（毫秒）")
    compile_err: str | None = Field(default=None, description="截断的异常摘要")

    max_rel_err: float | None = Field(default=None, description="最大相对误差")
    max_abs_err: float | None = Field(default=None, description="最大绝对误差")

    samples_ms: list[float] = Field(
        default_factory=list, description="性能原始计时样本，全留不只存中位数"
    )
    perf_median_ms: float | None = Field(default=None, description="样本中位数")
    perf_cv: float | None = Field(default=None, description="变异系数")
    baseline_median_ms: float | None = Field(default=None, description="同会话基线中位数")
    speedup: float | None = Field(default=None, description="加速比")

    log_path: str | None = Field(default=None, description="日志文件路径")

    @property
    def severe(self) -> bool:
        return is_severe(self.status)

    @property
    def counted(self) -> bool:
        return is_counted(self.status)


class RunScoreRecord(BaseModel):
    """一次运行的聚合分数。不合成跨赛道总分（规范 §3.3）。"""

    model_config = ConfigDict(extra="forbid")

    run_id: int
    coverage_ops_core: float = Field(description="功能主得分 OpScore（百分制）")
    coverage_strict: float = Field(description="严格通过率 OpStrictPass（百分制）")
    severe_count: int = Field(description="静默错误用例数")

    perf_geo: float | None = Field(default=None, description="几何平均加速比（裸值，仅统计判对用例）")
    perf_score: float | None = Field(
        default=None, description="性能得分：有分算子按权重加权平均（按算子锚点口径）"
    )
    fast_1: float | None = Field(default=None, description="加速比 ≥1.0 且正确的占比")
    fast_1_2: float | None = Field(default=None, description="加速比 ≥1.2 且正确的占比")
    fast_2: float | None = Field(default=None, description="加速比 ≥2.0 且正确的占比")
    unstable_rate: float = Field(default=0.0, description="CV 超标的性能用例占比")

    baseline_anchor_ref: str = Field(
        description="锚点口径：PER-OP = 按算子锚点（各算子首次全对轮次）"
    )
    scorer_version: str = Field(description="评分器版本")
    computed_at: str = Field(description="计算时间，ISO8601 UTC")

    # 按 category / level 的分组通过率。
    # None 表示该组无数据（无人提交 / 无计入分母的用例），
    # 必须与「0 分（全部失败）」区分——否则会被误读成全部失败。
    by_category: dict[str, float | None] | None = Field(
        default=None, description="按算子语义族的加权通过率"
    )
    by_level: dict[str, float | None] | None = Field(
        default=None, description="按难度等级的加权通过率"
    )
