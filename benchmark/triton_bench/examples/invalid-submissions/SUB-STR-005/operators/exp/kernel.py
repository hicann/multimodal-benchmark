# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""反例 SUB-STR-005：干预计时/同步接口。

把 synchronize 替换成空操作后，计时测到的只是「发起调用」的时间，
kernel 实际还没跑完——性能数据会漂亮得毫无意义。

注意这里**故意不调用** torch.exp：
否则会先被 SUB-STR-001（代算）拦下，报出的编号就不是这一条了。
样例必须能单独命中它对应的规则，否则失去定位价值。
"""

import torch


def _skip_synchronization(*arguments, **options):
    return None


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    # 伪装成同步，实际什么也不做
    torch.npu.synchronize = _skip_synchronization
    return x
