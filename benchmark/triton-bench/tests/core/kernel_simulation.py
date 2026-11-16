# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Model grouped JIT constant boundaries in CPU-only numerical simulations.

This only wraps annotated tuple arguments; it does not implement a compiler.
"""

import inspect
from types import SimpleNamespace


def constant_arguments(function, arguments):
    """Wrap annotated tuples while keeping runtime tuples and pointer values intact."""
    parameters = list(inspect.signature(function).parameters.values())
    converted = []
    for parameter, argument in zip(parameters, arguments, strict=True):
        if parameter.annotation is not inspect.Parameter.empty and isinstance(argument, tuple):
            converted.append(SimpleNamespace(value=argument))
        else:
            converted.append(argument)
    return converted
