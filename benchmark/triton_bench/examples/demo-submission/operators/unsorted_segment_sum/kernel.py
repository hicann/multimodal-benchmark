# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""UnsortedSegmentSum 的 triton 实现（合法提交样例）。

y[i] = Σ data[j] where segment_ids[j] == i：散射与段归约一次完成。

要点：
  - 每段独立归约、各写各的行——写入顺序天然无关，这正是
    FA-CONCUR-ACCUM 考察的「顺序无关累加」
  - data 按 stride 寻址（冻结用例含非连续 data）；ids 连续（评测方构造）
  - 浮点在 fp32 域累加后转回目标 dtype；整型用 int64 累加（防 int32
    溢出）再转回——与参考实现的精度口径一致
  - 输出形状对齐参考：rank-1 data → (num_segments,)，rank-2 →
    (num_segments, cols)（2026-10-13 真机前自查发现旧版 rank-1 时
    返回 (num_segments, 1)，形状比对必错——冻结用例 9/10 是 rank-1）
  - rank-1 整型累加器用 **1 元素块 + 块指针存回**：标量累加器初始
    是 int32、循环内升级 int64 会报 loop-carried 类型不一致（第三轮
    真机实测）；1 元素块存进标量指针又报 block/value 不匹配（第二轮
    真机实测）——块值配块指针两头都对
  - rank-2 路径 = rank-1 结构逐列复制（第四轮真机定案）：每
    (段, 列) 一个 program、[BR] 车道一维行循环——该结构已在
    1000~104 万元素、五种 dtype 上全量真机验证。此前三种 2D 设计
    ([128,32]/[256,64] 瓦片行循环、[512,16] 无循环树归约) 均在
    [4096,64] 上炸统一缓冲区（6.4M~8.4M 位，BiShengHIR 对 2D 瓦片
    循环的多缓冲预算不可控），而同负载的一维 [1024] 车道循环始终
    在预算内——用被验证的结构，不与后端的预算黑箱搏斗
  - 朴素全扫实现：每段扫一遍全部 id（num_segments ≤ 64、N ≤ 1M，
    有界可跑）。排序/直方图等更快做法留给真正的参赛实现——性能
    差距正是这个维度要测量的内容
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _segsum_line_kernel(d_ptr, i_ptr, o_ptr,
                        n, cols,
                        ds0, ds1,
                        is_float: tl.constexpr, br: tl.constexpr):
    """每 (段, 列) 一个 program：一维行循环累加该列上命中本段的元素。

    cols=1（rank-1）时 o_ptr + p_seg*cols + col 即 o_ptr + p_seg，
    与 rank-1 语义完全一致——两种 rank 共用同一结构。
    """
    p_seg = tl.program_id(0)
    col = tl.program_id(1)
    if is_float:
        total = 0.0
    else:
        total = tl.zeros((1,), dtype=tl.int64)
    for k0 in range(0, n, br):
        k = k0 + tl.arange(0, br)
        kv = k < n
        ids = tl.load(i_ptr + k, mask=kv, other=-1).to(tl.int64)
        hit = (ids == p_seg) & kv
        d = tl.load(d_ptr + k * ds0 + col * ds1, mask=hit, other=0)
        if is_float:
            total += tl.sum(d.to(tl.float32), axis=0)
        else:
            total += tl.sum(d.to(tl.int64), axis=0)
    if is_float:
        tl.store(o_ptr + p_seg * cols + col,
                 total.to(o_ptr.dtype.element_ty))
    else:
        # 块值配块指针：1 元素块不能存进标量指针（第二轮真机实测）
        tl.store(o_ptr + p_seg * cols + col + tl.arange(0, 1),
                 total.to(o_ptr.dtype.element_ty))


def unsorted_segment_sum(data, segment_ids, num_segments):
    num_segments = int(num_segments)
    if data.dim() > 2:
        raise NotImplementedError(f"样例只实现 rank≤2，实际 rank={data.dim()}")
    if not segment_ids.is_contiguous():
        raise NotImplementedError("segment_ids 预期为连续布局")
    if segment_ids.shape[0] != data.shape[0]:
        raise NotImplementedError("segment_ids 长度须等于 data 第 0 维")

    n = data.shape[0]
    cols = data.shape[1] if data.dim() == 2 else 1
    # 输出形状与参考一致：rank-1 → (num_segments,)；rank-2 → (nseg, cols)
    out_shape = (num_segments, cols) if data.dim() == 2 else (num_segments,)
    out = torch.empty(out_shape, dtype=data.dtype, device=data.device)
    ds = (data.stride() + (0,) * (2 - data.dim()))
    is_float = data.dtype in (torch.float16, torch.bfloat16, torch.float32)
    _segsum_line_kernel[(num_segments, cols)](
        data, segment_ids, out, n, cols,
        ds[0], ds[1],
        is_float=is_float, br=1024)
    return out
