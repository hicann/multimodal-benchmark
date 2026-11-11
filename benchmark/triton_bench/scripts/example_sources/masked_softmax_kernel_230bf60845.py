# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""MaskedSoftmax 的 triton 实现（合法提交样例）。

y = softmax(x masked_fill(mask==0, -inf), dim)：mask=0 的位置不参与归一化。

要点：
  - 融合算子的样例内核是**锚点载体**（要求全部用例判对），按 stride
    寻址实现——归约线内核对任意 dim、任意 rank≤3、非连续 x 都成立，
    不以守卫挡用例
  - 每个 program 处理一条**归约线**（沿 dim 的一列元素）：线基址由
    **非归约维**坐标合成（宿主侧已把归约维剔除），线内按 dim 方向
    的 stride 步进。三遍循环：取 max（数值稳定）→ sum(exp) → 归一化
    写回。归约线可以很长（rank-1 dim=0 时整条 4M 元素是一线）——
    分块循环扛住，慢但正确
  - BLOCK=256：真机 910B4 实测 BLOCK=1024 的 f32/i8 瓦片超统一缓冲区
    （BiShengHIR 报 "ub overflow, requires 2097152 bits while 1572864
    bits available"，f16/bf16 塞得下所以侥幸通过）；256 留 3× 余量
  - mask 由评测方构造为连续、与 x 同形：按其自身 strides 寻址
  - 不用 permute/transpose 绕道维度搬运（Normalization 类白名单外）
"""

from __future__ import annotations


from scripts.example_sources.device_dependencies import tl as tl, torch as torch, triton as triton





@triton.jit
def _masked_softmax_kernel(x_ptr, m_ptr, o_ptr, geometry, configuration: tl.constexpr):
    lines, rlen, lb1, lb2, xstep, mb1, mb2, mstep, ob1, ob2, ostep, l1, l2 = geometry
    block: tl.constexpr = configuration.value[0]
    p = tl.program_id(0)
    if p >= lines:
        return
    i2 = p % l2
    t = p // l2
    i1 = t % l1
    x_base = i1 * lb1 + i2 * lb2
    m_base = i1 * mb1 + i2 * mb2
    o_base = i1 * ob1 + i2 * ob2
    m_max = float("-inf")
    for k0 in range(0, rlen, block):
        offs = k0 + tl.arange(0, block)
        valid = offs < rlen
        xv = tl.load(x_ptr + x_base + offs * xstep, mask=valid, other=float("-inf")).to(tl.float32)
        mv = tl.load(m_ptr + m_base + offs * mstep, mask=valid, other=0).to(tl.float32)
        xv = tl.where(mv != 0, xv, float("-inf"))
        m_max = tl.maximum(m_max, tl.max(xv, axis=0))
    z = 0.0
    for k0 in range(0, rlen, block):
        offs = k0 + tl.arange(0, block)
        valid = offs < rlen
        xv = tl.load(x_ptr + x_base + offs * xstep, mask=valid, other=float("-inf")).to(tl.float32)
        mv = tl.load(m_ptr + m_base + offs * mstep, mask=valid, other=0).to(tl.float32)
        xv = tl.where(mv != 0, xv, float("-inf"))
        z += tl.sum(tl.exp(xv - m_max), axis=0)
    for k0 in range(0, rlen, block):
        offs = k0 + tl.arange(0, block)
        valid = offs < rlen
        xv = tl.load(x_ptr + x_base + offs * xstep, mask=valid, other=float("-inf")).to(tl.float32)
        mv = tl.load(m_ptr + m_base + offs * mstep, mask=valid, other=0).to(tl.float32)
        xv = tl.where(mv != 0, xv, float("-inf"))
        y = tl.exp(xv - m_max) / z
        tl.store(o_ptr + o_base + offs * ostep, y.to(o_ptr.dtype.element_ty), mask=valid)


def _line_axes(shape, strides, d):
    """归约维 d 以外的轴（重新打包、前补 1 到 2 个）：返回 (l1, l2), (步长1, 步长2)。

    rank-1 dim=0 → (1,1)/(0,0)（单线、基址 0）；其余按非归约维的先后
    排到最后两槽（rank≤3 时非归约维至多 2 个）。
    """
    dims = tuple(shape) + (1,) * (3 - len(shape))
    sts = tuple(strides) + (0,) * (3 - len(strides))
    nd = [dims[j] for j in range(3) if j != d]
    nb = [sts[j] for j in range(3) if j != d]
    nd = [1] * (2 - len(nd)) + nd
    nb = [0] * (2 - len(nb)) + nb
    return nd[0], nd[1], nb[0], nb[1]


def masked_softmax(x, mask, dim):
    if x.dim() > 3:
        raise NotImplementedError(f"样例只实现 rank≤3，实际 rank={x.dim()}")
    d = dim if dim >= 0 else dim + x.dim()
    if not (0 <= d < x.dim()):
        raise NotImplementedError(f"dim={dim} 超出 rank={x.dim()}")
    if x.shape != mask.shape:
        raise NotImplementedError("样例要求 mask 与 x 同形")

    out = torch.empty(x.shape, dtype=x.dtype, device=x.device)
    rlen = x.shape[d]
    lines = x.numel() // rlen
    block = 256
    l1x, l2x, xlb1, xlb2 = _line_axes(x.shape, x.stride(), d)
    l1m, l2m, mlb1, mlb2 = _line_axes(mask.shape, mask.stride(), d)
    l1o, l2o, olb1, olb2 = _line_axes(out.shape, out.stride(), d)
    _masked_softmax_kernel[lines,](
        x,
        mask,
        out,
        (lines, rlen, xlb1, xlb2, x.stride(d), mlb1, mlb2, mask.stride(d), olb1, olb2, out.stride(d), l1x, l2x),
        (block,),
    )
    return out
