# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反例 SUB-STR-002：主机侧往返——把张量搬回 CPU 算完再搬回加速器。

这类实现的耗时几乎全是拷贝，性能数据完全失真；
而它确实「算对了」，只有静态扫描能在执行前拦住。

刻意不调用 torch.exp：否则会同时命中 SUB-STR-001（代算）。
样例的原则是「一个目录只演示一条违规」。
"""


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    cpu = x.cpu()
    y = cpu * scale + shift
    return y.to(x.device)
