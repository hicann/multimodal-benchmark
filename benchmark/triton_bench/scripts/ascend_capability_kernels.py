# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""合法、固定规模的编译器能力探针；仅由独立子进程载入。"""

import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def _gather(x, indices, out, m: tl.constexpr, n: tl.constexpr):
    source = tl.load(x + tl.arange(0, m))
    indices_value = tl.load(indices + tl.arange(0, n))
    values = tl.gather(source, indices_value, 0)
    tl.store(out + tl.arange(0, n), values)


def run(name):
    if name == "health":
        from triton_eval.harness.npu_exec import health_probe

        passed, reason = health_probe()
        if not passed:
            raise RuntimeError(reason)
        return
    dtype = torch.int32 if name == "gather_int32" else torch.float32
    values = torch.arange(256, device="npu:0", dtype=torch.int32) + 2 ** 25
    values = values.to(dtype)
    indices = torch.arange(127, -1, -1, device="npu:0", dtype=torch.int32)
    output = torch.empty(128, dtype=dtype, device="npu:0")
    _gather[(1,)](values, indices, output, 256, 128)
    torch.npu.synchronize()
    expected = values.cpu().index_select(0, indices.cpu().long())
    if not torch.equal(output.cpu(), expected):
        raise AssertionError("tl.gather output differs from exact CPU index_select")
