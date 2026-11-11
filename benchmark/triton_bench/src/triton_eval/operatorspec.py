# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""算子规格（L0，无 NPU 依赖）。

规格是「用例生成」与「提交校验」的共同依据：
算子语义不依赖任何具体实现代码即可被判定——这正是支持外部提交的前提。
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .taxonomy import Category, Level


class AttrSpec(BaseModel):
    """算子属性定义。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    type: str = Field(description="声明类型，如 float / int64 / Int / String")
    default: Any = Field(default=None, description="默认值；必填属性为 None")
    required: bool = Field(default=False)

    @field_validator("name")
    @classmethod
    def name_not_empty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("属性名不得为空")
        return v


class IOSpec(BaseModel):
    """一个输入或输出张量的声明。"""

    model_config = ConfigDict(extra="forbid")

    name: str
    dtype: list[str] = Field(description="支持的数据类型集合")
    optional: bool = Field(default=False)

    @field_validator("dtype")
    @classmethod
    def dtype_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("dtype 集合不得为空")
        return v


class ShapeSupport(BaseModel):
    """形状约束。"""

    model_config = ConfigDict(extra="forbid")

    min_rank: int = Field(default=1, ge=0)
    max_rank: int = Field(default=8, ge=1)
    notes: str = Field(default="", description="维度间关系的自然语言约束")


class OperatorSpec(BaseModel):
    """算子规格。对应参考仓库的 `proto.yaml`。"""

    model_config = ConfigDict(extra="forbid")

    name: str = Field(description="算子名，如 Exp / RmsNorm")
    category: Category = Field(description="语义族，主分类维度")
    level: Level = Field(description="难度等级，与 category 独立")
    # 字段名不用 schema：Pydantic 的 BaseModel 已有同名属性，遮蔽会告警
    signature: str = Field(description="签名，如 exp(Tensor x, float base) -> Tensor y")
    inputs: list[IOSpec]
    outputs: list[IOSpec]
    case_dtypes: list[str] | None = Field(default=None, min_length=1)
    attrs: list[AttrSpec] = Field(default_factory=list)
    shape_support: ShapeSupport = Field(default_factory=ShapeSupport)

    # 来源追溯（对应 reference/cann-bench 的目录）
    source_name: str = Field(default="", description="参考仓库中的算子名")
    source_dir: str = Field(default="", description="参考仓库中的目录，如 level1/exp")

    # 用例绑定的入口签名，用于校验提交
    entry: str = Field(description="提交侧入口函数名，通常与算子同名小写")
    params: list[str] = Field(description="入口函数参数名列表，顺序敏感")

    @field_validator("params")
    @classmethod
    def params_not_empty(cls, v: list[str]) -> list[str]:
        if not v:
            raise ValueError("入口参数列表不得为空")
        return v

    def dtype_of(self, io_name: str) -> list[str]:
        for item in self.inputs + self.outputs:
            if item.name == io_name:
                return item.dtype
        raise KeyError(f"算子 {self.name} 未声明名为 {io_name} 的输入输出")

    def attr(self, name: str) -> AttrSpec | None:
        for a in self.attrs:
            if a.name == name:
                return a
        return None

    def required_attrs(self) -> list[str]:
        return [a.name for a in self.attrs if a.required]

    def validate_param_order(self) -> None:
        """校验参数序：张量参数必须全部排在属性参数之前。

        这是调用约定的硬约束——执行器按 make_inputs 产出的顺序位置传张量，
        属性一律走关键字。一旦属性夹在张量中间，第 k 个位置实参就会绑到
        属性形参上，随后同名关键字传入必然 `got multiple values for argument`。

        曾无任何检查，于是 Scatter 的 (data, dim, indices, updates, reduce)
        安静通过了所有测试，直到有真实提交照它写实现才发现：
        该算子**没有可行签名**（实测 5/5 RUNTIME_FAIL）。
        """
        tensor_names = {i.name for i in self.inputs}
        seen_attr = False
        for p in self.params:
            if p in tensor_names:
                if seen_attr:
                    raise ValueError(
                        f"{self.name} 的参数序违反调用约定：张量参数 {p!r} "
                        f"出现在属性参数之后（params={self.params}）"
                    )
            else:
                seen_attr = True

    def validate_case_params(
        self, dtype: str, shape: list[int], attrs: dict[str, Any]
    ) -> None:
        """校验一个派生用例是否落在规格范围内。三处越界都必须明确报错。"""
        allowed = (set(self.case_dtypes) if self.case_dtypes is not None else
                   {dtype for item in self.inputs + self.outputs for dtype in item.dtype})
        if dtype not in allowed:
            raise ValueError(
                f"算子 {self.name} 不支持 dtype={dtype}；"
                f"合法取值：{sorted(allowed)}"
            )

        rank = len(shape)
        if not (self.shape_support.min_rank <= rank <= self.shape_support.max_rank):
            raise ValueError(
                f"算子 {self.name} 的 shape 维度数 {rank} 超出范围 "
                f"[{self.shape_support.min_rank}, {self.shape_support.max_rank}]"
            )

        for name in self.required_attrs():
            if name not in attrs:
                raise ValueError(f"算子 {self.name} 缺少必填属性：{name}")

        for name, value in attrs.items():
            spec = self.attr(name)
            if spec is None:
                raise ValueError(f"算子 {self.name} 未声明属性：{name}")
            expected = spec.type.lower()
            actual = type(value).__name__

            # 列表类型（int[] / List[int]）必须先判断：
            # 否则 "int[]".startswith("int") 会把它当标量整型，误报类型不符
            if "[]" in expected or expected.startswith("list"):
                if not isinstance(value, (list, tuple)):
                    raise ValueError(
                        f"算子 {self.name} 属性 {name} 类型不符：期望列表，实际 {actual}"
                    )
                continue

            if expected.startswith("int") and not isinstance(value, int):
                raise ValueError(
                    f"算子 {self.name} 属性 {name} 类型不符：期望整型，实际 {actual}"
                )
            if expected.startswith("float") and not isinstance(value, (int, float)):
                raise ValueError(
                    f"算子 {self.name} 属性 {name} 类型不符：期望浮点，实际 {actual}"
                )
            if expected.startswith("str") and not isinstance(value, str):
                raise ValueError(
                    f"算子 {self.name} 属性 {name} 类型不符：期望字符串，实际 {actual}"
                )

    def resolve_attrs(self, attrs: dict[str, Any] | None) -> dict[str, Any]:
        """补齐默认值。必填缺失由 validate_case_params 负责报错。"""
        resolved: dict[str, Any] = {}
        for a in self.attrs:
            if attrs and a.name in attrs:
                resolved[a.name] = attrs[a.name]
            elif not a.required:
                resolved[a.name] = a.default
        if attrs:
            for name, value in attrs.items():
                resolved.setdefault(name, value)
        return resolved
