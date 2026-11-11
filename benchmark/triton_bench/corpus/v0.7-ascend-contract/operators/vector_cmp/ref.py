# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from triton_eval.inputs import vector_cmp_inputs
import torch


def make_inputs(seed, dtype, shape, device, attrs):
    return vector_cmp_inputs(seed, shape, device)


def reference(x, **attrs):
    # 口径必须与教程 kernel 对齐，否则整算子 20 例 WRONG_RESULT：
    #   1) 方差用**总体**方差（/N），不是 torch.std 默认的样本方差（/(N-1)）；
    #   2) 公式是 out = (x - mean) * rstd，其中 rstd = 1/sqrt(var + eps)，
    #      即 eps 加在方差上、再开方；不是 (x - mean) / (std + eps)。
    eps = float(attrs.get("eps", 1e-6))
    mean = x.mean(dim=-1, keepdim=True)
    var = x.var(dim=-1, keepdim=True, unbiased=False)
    rstd = 1.0 / torch.sqrt(var + eps)
    out = (x.to(torch.float32) - mean) * rstd
    return out, mean.squeeze(-1), rstd.squeeze(-1)
