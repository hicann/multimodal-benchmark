# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""计时测试用的最小 kernel 夹具。

存在原因：
    @triton.jit 依赖 inspect.getsourcelines 反查源码，kernel 必须来自
    真实文件且定义在模块级。计时测试曾在测试函数体里内联定义 kernel，
    在被测编译器上直接失败；也曾复用 corpus/operators/add/kernel.py，
    契约迁移（case 不再绑定 kernel）后该文件已随旧算子一起淘汰，
    表现为 FileNotFoundError。夹具因此固定在此处，不随 corpus 变动。

注意：本文件不在冻结 corpus 内，不参与评分。
"""

import triton
import triton.language as tl


@triton.jit
def _add_kernel(x_ptr, y_ptr, o_ptr, n, block: tl.constexpr):
    pid = tl.program_id(0)
    offs = pid * block + tl.arange(0, block)
    mask = offs < n
    tl.store(o_ptr + offs,
             tl.load(x_ptr + offs, mask=mask) + tl.load(y_ptr + offs, mask=mask),
             mask=mask)


def launch(inputs, out, cfg):
    """触发一次 kernel 启动。签名与计时测试约定一致。"""
    x, y = inputs[0], inputs[1]
    n = x.numel()
    block = int(cfg.get("BLOCK", 256))
    _add_kernel[(triton.cdiv(n, block),)](x, y, out, n, block=block)


def reference(inputs):
    """性能归因脚本用的 eager 基线（x + y）。"""
    return inputs[0] + inputs[1]
