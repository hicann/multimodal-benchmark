# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""UnsortedSegmentSum 的参考实现（评测方资产）。

来源池 tasks/level2/unsorted_segment_sum：y[i] = sum(data[j] where ids[j]==i)。

确定性实现（与池 golden 的关键差异，必须说明）：
    池 golden 用 index_add_ 累加——在加速卡上它可能走原子加法，累加顺序
    不确定，两次执行可能出现 ULP 级漂移，过不了「参考两遍一致」门禁。
    本参考改为**逐段掩码求和**：每段一个布尔掩码、段内一次 sum——
    求和顺序由算法固定，任何设备上两遍一致。段数由生成侧约束在 ≤ 64
    （见 casegen），逐段循环的开销有界。
    浮点在 fp32 域累加（与池 golden 的 fp16/bf16 升精度约定一致）；
    整数逐段精确求和（整型和与顺序无关，天然确定）。
"""

from __future__ import annotations

import torch


def _rand(seed: int, n: int, dtype):
    g = torch.Generator(device="cpu").manual_seed(seed)
    if dtype in (torch.int32, torch.int64):
        return torch.randint(-100, 100, (n,), generator=g,
                             dtype=torch.int64).to(dtype)
    return torch.randn(n, generator=g).to(dtype)


def make_inputs(*, seed, dtype, shape, device, attrs):
    num_segments = int(attrs["num_segments"])
    n = shape[0]
    g = torch.Generator(device="cpu").manual_seed(seed + 1)
    # 段 id 均匀覆盖 [0, num_segments)：重复是本算子的语义常态（段长不均）
    segment_ids = torch.randint(0, num_segments, (n,), generator=g,
                                dtype=torch.int64)
    rest = 1
    for d in shape[1:]:
        rest *= d
    data = _rand(seed, n * rest, dtype)
    return [data.reshape(shape).to(device), segment_ids.to(device)]


def reference(data, segment_ids, num_segments):
    num_segments = int(num_segments)
    ids = segment_ids.to(torch.int64)
    trailing = data.shape[1:]
    if data.dtype in (torch.int32, torch.int64):
        out = torch.zeros((num_segments, *trailing), dtype=data.dtype,
                          device=data.device)
        for s in range(num_segments):
            out[s] = data[ids == s].sum(dim=0)
        return out
    d32 = data.to(torch.float32)
    out32 = torch.zeros((num_segments, *trailing), dtype=torch.float32,
                        device=data.device)
    for s in range(num_segments):
        out32[s] = d32[ids == s].sum(dim=0)
    return out32.to(data.dtype)
