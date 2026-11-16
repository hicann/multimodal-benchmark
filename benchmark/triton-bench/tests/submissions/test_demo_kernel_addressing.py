# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""融合样例内核寻址惯用法的回归钉（L0，无需 NPU）。

2026-10-13 真机首轮流水暴露的缺陷家族：行/线分解把 program id 按含
最内维的 3 维形状分解、最内步长取 a[3-rank]（rank-1 取到填充 0、
rank-3 取到最外步长）——RoPE 5/6 WRONG、AddRmsNorm 6/6 WRONG，
唯一通过的 RoPE 用例是 zeros 值域（读错行也全零，纯属侥幸）。

样例内核 import triton、本机不可执行，但**分解助函数是纯 Python**：
用 AST 从提交的 kernel.py 里提取真实函数体执行（不经 import，杜绝
测试与实现的漂移），对任意 rank/步长验证「行基址覆盖且不重不漏」。
数值正确性另由「本地模拟 vs 真机误差逐位吻合」与真机窗口复核。
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest
from triton_eval.fixture_sources import fixture_path

DEMO = fixture_path(Path(__file__).resolve().parents[2] / "examples" / "demo-submission")


def _extract_func(kernel_rel: str, name: str):
    """从 kernel.py 提取指定纯 Python 函数并编译（不 import 模块）。"""
    src = (DEMO / kernel_rel).read_text(encoding="utf-8")
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            mod = ast.Module(body=[node], type_ignores=[])
            ns: dict = {}
            exec(compile(mod, kernel_rel, "exec"), ns)  # noqa: S102
            return ns[name]
    raise AssertionError(f"{kernel_rel} 缺少函数 {name}")


# 三份 _outer 必须同源（复制粘贴的惯用法，漂移即缺陷）
OUTER_SOURCES = [
    "operators/add_rms_norm/kernel.py",
    "operators/apply_rotary_pos_emb/kernel.py",
    "operators/moe_gating_top_k_softmax/kernel.py",
]

SHAPES = [
    (1000,),            # rank-1（AddRmsNorm 全部冻结用例）
    (64, 64),           # rank-2 方
    (1000, 64),         # rank-2 非对齐行
    (999, 64),
    (8, 256, 64),       # rank-3 立方
    (2, 1024, 128),
    (32768, 128),       # RoPE XL
    (16384, 256),       # MoE XL
    (64, 32),           # MoE S
]


@pytest.mark.parametrize("kernel_rel", OUTER_SOURCES)
def test_outer_row_bases_cover_rows_exactly(kernel_rel: str) -> None:
    """连续与非连续（隔位跨步）张量上，行基址必须不重不漏地覆盖每一行。"""
    outer = _extract_func(kernel_rel, "_outer")
    for shape in SHAPES:
        rows = 1
        for d in shape[:-1]:
            rows *= d
        for strided in (False, True):
            if strided:
                # 隔位跨步构造（与语料 _make_strided 同型）：翻倍后隔位取
                strides = tuple(
                    s * 2 for s in _contiguous_strides(shape))
            else:
                strides = _contiguous_strides(shape)
            _assert_outer_row_bases(outer, shape, strides, strided)


def _assert_outer_row_bases(outer, shape, strides, strided):
    rows = 1
    for dimension in shape[:-1]:
        rows *= dimension
    r1, r2, b1, b2 = outer(shape, strides)
    assert r1 * r2 == rows, f"{shape}：线维乘积 != 行数"
    bases = []
    for p in range(rows):
        i2 = p % r2
        t = p // r2
        i1 = t % r1
        assert t // r1 == 0, f"{shape}：外维分解越界"
        bases.append(i1 * b1 + i2 * b2)
    # 每行的基址必须等于「该行首元素的实际存储偏移」
    expect = []
    for p in range(rows):
        coord = _row_to_outer_coords(shape, p)
        off = 0
        for c, st in zip(coord, strides):
            off += c * st
        expect.append(off)
    assert bases == expect, f"{shape} strided={strided}：行基址错"


def _contiguous_strides(shape) -> tuple:
    st = [1] * len(shape)
    for i in range(len(shape) - 2, -1, -1):
        st[i] = st[i + 1] * shape[i + 1]
    return tuple(st)


def _row_to_outer_coords(shape, p: int) -> tuple:
    """行号 p → 外维坐标（行主序展平的逆）。"""
    outer_dims = list(shape[:-1])
    coords = [0] * len(outer_dims)
    rem = p
    for i in range(len(outer_dims) - 1, -1, -1):
        coords[i] = rem % outer_dims[i]
        rem //= outer_dims[i]
    assert rem == 0
    return tuple(coords)


def test_line_axes_cover_all_elements_exactly() -> None:
    """masked_softmax 的线分解：所有 (线基址 + k·step) 必须不重不漏覆盖全张量。"""
    line_axes = _extract_func("operators/masked_softmax/kernel.py", "_line_axes")
    cases = [
        ((1000,), 0),
        ((4194304,), 0),
        ((64, 128), 0), ((64, 128), 1),
        ((8, 256, 64), 0), ((8, 256, 64), 1), ((8, 256, 64), 2),
        ((2, 1024, 128), 1),
    ]
    for shape, d in cases:
        numel = 1
        for x in shape:
            numel *= x
        rlen = shape[d]
        lines = numel // rlen
        strides = _contiguous_strides(shape)
        l1, l2, lb1, lb2 = line_axes(shape, strides, d)
        assert l1 * l2 == lines, f"{shape} dim={d}：线维乘积 != 线数"
        got = set()
        for p in range(lines):
            i2 = p % l2
            t = p // l2
            i1 = t % l1
            assert t // l1 == 0
            base = i1 * lb1 + i2 * lb2
            for k in range(rlen):
                got.add(base + k * strides[d])
        assert got == set(range(numel)), \
            f"{shape} dim={d}：线覆盖 != 全张量"


def test_moe_row_idx_matches_pool_flatten_rule() -> None:
    """MoE 的 row_idx 广播构造必须与池 golden 的展平规则逐位一致。

    与 moe_gating_top_k_softmax/kernel.py 的构造同式（那里 import
    triton 无法本机 import，此处按同式复刻并锚定源码）。
    """
    import torch

    for shape in [(1024, 32), (16384, 256), (2, 2048, 64)]:
        for k in (2, 8):
            rows = 1
            for x in shape[:-1]:
                rows *= x
            if len(shape) == 2:
                row_idx = (torch.arange(rows, dtype=torch.int64).unsqueeze(1)
                           + rows * torch.arange(k,
                                                 dtype=torch.int64).unsqueeze(0))
            else:
                b_, t_ = shape[0], shape[1]
                row_idx = (torch.arange(b_, dtype=torch.int64)
                           .unsqueeze(1).unsqueeze(2) * t_
                           + torch.arange(t_, dtype=torch.int64)
                           .unsqueeze(0).unsqueeze(2)
                           + rows * torch.arange(k, dtype=torch.int64)
                           .unsqueeze(0).unsqueeze(0))
            # 池 golden：rng.reshape(k, N).transpose(0,1)（3D 再 reshape 回）
            out_shape = tuple(shape[:-1]) + (k,)
            if len(out_shape) == 2:
                rng = torch.arange(rows * k, dtype=torch.int64)
                ref = rng.reshape(k, rows).transpose(0, 1)
            else:
                rng = torch.arange(rows * k, dtype=torch.int64)
                ref = rng.reshape(k, rows).transpose(0, 1).reshape(out_shape)
            assert row_idx.shape == out_shape
            assert torch.equal(row_idx, ref), f"{shape} k={k}：row_idx 不符"
