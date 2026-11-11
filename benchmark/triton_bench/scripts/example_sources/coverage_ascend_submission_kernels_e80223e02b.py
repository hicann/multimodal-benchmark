# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Real Triton kernels. Host code only allocates outputs and selects launch metadata."""
__all__ = [
    '_dispatch_output_metadata',
    '_elementwise',
    '_elementwise_binary_0',
    '_elementwise_binary_1',
    '_elementwise_binary_2',
    '_elementwise_binary_3',
    '_elementwise_binary_4',
    '_elementwise_unary_0',
    '_elementwise_unary_1',
    '_elementwise_unary_2',
    '_elementwise_unary_3',
    '_elementwise_unary_4',
    '_elementwise_unary_5',
    '_elementwise_unary_6',
    '_floor_divide_int32',
    '_gather',
    '_launch_dispatch',
    '_launch_elementwise',
    '_layout',
    '_matmul',
    '_rows',
    '_sort',
    '_validate_dispatch_inputs',
    'arg_max',
    'dispatch',
    'extension',
    'libdevice',
    'math',
    'swi_glu',
    'tl',
    'torch',
    'triton',
    'validate_elementwise_layout',
]

import math
import torch
import triton
import triton.language as tl
import triton.language.extra.cann.libdevice as libdevice
import triton.language.extra.cann.extension as extension
from src.triton_eval.inputs import validate_elementwise_layout
from scripts.templates.official_kernels import _floor_divide_int32
from scripts.templates.official_kernels import _elementwise_binary_4
from scripts.templates.official_kernels import _elementwise_binary_3
from scripts.templates.official_kernels import _elementwise_binary_2
from scripts.templates.official_kernels import _elementwise_binary_1
from scripts.templates.official_kernels import _elementwise_binary_0
from scripts.templates.official_kernels import _elementwise_unary_6
from scripts.templates.official_kernels import _elementwise_unary_5
from scripts.templates.official_kernels import _elementwise_unary_4
from scripts.templates.official_kernels import _elementwise_unary_3
from scripts.templates.official_kernels import _elementwise_unary_2
from scripts.templates.official_kernels import _elementwise_unary_1
from scripts.templates.official_kernels import _elementwise_unary_0
from scripts.templates.official_kernels import _elementwise
from scripts.templates.official_kernels import _rows
from scripts.templates.official_kernels import _layout
from scripts.templates.official_kernels import _gather
from scripts.templates.official_kernels import _sort
from scripts.templates.official_kernels import _matmul
from scripts.templates.official_kernels import _validate_dispatch_inputs
from scripts.templates.official_kernels import _dispatch_output_metadata
from scripts.templates.official_kernels import dispatch
from scripts.templates.official_kernels import _launch_dispatch
from scripts.templates.official_kernels import _launch_elementwise
from scripts.example_sources.adversarial_submissions_scalar_kernels_340c775b28 import arg_max
from scripts.example_sources.adversarial_submissions_scalar_kernels_340c775b28 import swi_glu






















































FAMILIES = {
    'Abs': 'unary',
    'Neg': 'unary',
    'Exp': 'unary',
    'Exp2': 'unary',
    'Expm1': 'unary',
    'Log': 'unary',
    'Log2': 'unary',
    'Log10': 'unary',
    'Log1p': 'unary',
    'Sqrt': 'unary',
    'Rsqrt': 'unary',
    'Sin': 'unary',
    'Cos': 'unary',
    'Tan': 'unary',
    'Tanh': 'unary',
    'Sinh': 'unary',
    'Cosh': 'unary',
    'Floor': 'unary',
    'Ceil': 'unary',
    'Trunc': 'unary',
    'Relu': 'unary',
    'Sigmoid': 'unary',
    'Signbit': 'unary',
    'Add': 'binary',
    'Sub': 'binary',
    'Mul': 'binary',
    'Div': 'binary',
    'FloorDiv': 'binary',
    'Mod': 'binary',
    'Pow': 'binary',
    'Maximum': 'binary',
    'Minimum': 'binary',
    'Equal': 'comparison',
    'NotEqual': 'comparison',
    'Greater': 'comparison',
    'GreaterEqual': 'comparison',
    'Less': 'comparison',
    'LessEqual': 'comparison',
    'BitwiseAnd': 'bitwise',
    'BitwiseOr': 'bitwise',
    'BitwiseXor': 'bitwise',
    'BitwiseNot': 'bitwise_unary',
    'LogicalAnd': 'logical',
    'LogicalOr': 'logical',
    'Clamp': 'unary',
    'Where': 'where',
    'ReduceSum': 'reduce',
    'ReduceMean': 'reduce',
    'ReduceMax': 'reduce',
    'ReduceMin': 'reduce',
    'ArgMax': 'arg',
    'ArgMin': 'arg',
    'XorSum': 'reduce',
    'CumSum': 'scan',
    'CumProd': 'scan',
    'Transpose': 'transpose',
    'Concat': 'concat',
    'Flip': 'flip',
    'Gather': 'gather',
    'SortValues': 'sort',
    'TopKValues': 'topk',
    'Softmax': 'softmax',
    'RmsNorm': 'rms',
    'LayerNorm': 'layer',
    'Matmul': 'matmul',
    'Mish': 'unary',
    'Gelu': 'unary',
    'SwiGlu': 'swiglu'}


def absolute(x):
    return dispatch('Abs', (x,), {})


def neg(x):
    return dispatch('Neg', (x,), {})


def exp(x, base=-1.0, scale=1.0, shift=0.0):
    return dispatch('Exp', (x,), {'base': base, 'scale': scale, 'shift': shift})


def exp2(x):
    return dispatch('Exp2', (x,), {})


def expm1(x):
    return dispatch('Expm1', (x,), {})


def log(x):
    return dispatch('Log', (x,), {})


def log2(x):
    return dispatch('Log2', (x,), {})


def log10(x):
    return dispatch('Log10', (x,), {})


def log1p(x):
    return dispatch('Log1p', (x,), {})


def sqrt(x):
    return dispatch('Sqrt', (x,), {})


def rsqrt(x):
    return dispatch('Rsqrt', (x,), {})


def sin(x):
    return dispatch('Sin', (x,), {})


def cos(x):
    return dispatch('Cos', (x,), {})


def tan(x):
    return dispatch('Tan', (x,), {})


def tanh(x):
    return dispatch('Tanh', (x,), {})


def sinh(x):
    return dispatch('Sinh', (x,), {})


def cosh(x):
    return dispatch('Cosh', (x,), {})


def floor(x):
    return dispatch('Floor', (x,), {})


def ceil(x):
    return dispatch('Ceil', (x,), {})


def trunc(x):
    return dispatch('Trunc', (x,), {})


def relu(x):
    return dispatch('Relu', (x,), {})


def sigmoid(x):
    return dispatch('Sigmoid', (x,), {})


def signbit(x):
    return dispatch('Signbit', (x,), {})


def add(x, y):
    return dispatch('Add', (x, y), {})


def sub(x, y):
    return dispatch('Sub', (x, y), {})


def mul(x, y):
    return dispatch('Mul', (x, y), {})


def div(x, y):
    return dispatch('Div', (x, y), {})


def floor_div(x, y):
    return dispatch('FloorDiv', (x, y), {})


def mod(x, y):
    return dispatch('Mod', (x, y), {})


def power(x, y):
    return dispatch('Pow', (x, y), {})


def maximum(x1, x2):
    return dispatch('Maximum', (x1, x2), {})


def minimum(x, y):
    return dispatch('Minimum', (x, y), {})


def equal(x, y):
    return dispatch('Equal', (x, y), {})


def not_equal(x, y):
    return dispatch('NotEqual', (x, y), {})


def greater(x, y):
    return dispatch('Greater', (x, y), {})


def greater_equal(x, y):
    return dispatch('GreaterEqual', (x, y), {})


def less(x, y):
    return dispatch('Less', (x, y), {})


def less_equal(x, y):
    return dispatch('LessEqual', (x, y), {})


def bitwise_and(x, y):
    return dispatch('BitwiseAnd', (x, y), {})


def bitwise_or(x, y):
    return dispatch('BitwiseOr', (x, y), {})


def bitwise_xor(x, y):
    return dispatch('BitwiseXor', (x, y), {})


def bitwise_not(x):
    return dispatch('BitwiseNot', (x,), {})


def logical_and(x, y):
    return dispatch('LogicalAnd', (x, y), {})


def logical_or(x, y):
    return dispatch('LogicalOr', (x, y), {})


def clamp(x, lower=-0.5, upper=0.5):
    return dispatch('Clamp', (x,), {'lower': lower, 'upper': upper})


def where(condition, x, y):
    return dispatch('Where', (condition, x, y), {})


def reduce_sum(x, dim=-1, keepdim=False):
    return dispatch('ReduceSum', (x,), {'dim': dim, 'keepdim': keepdim})


def reduce_mean(x, dim=-1, keepdim=False):
    return dispatch('ReduceMean', (x,), {'dim': dim, 'keepdim': keepdim})


def reduce_max(x, dim=-1, keepdim=False):
    return dispatch('ReduceMax', (x,), {'dim': dim, 'keepdim': keepdim})


def reduce_min(x, dim=-1, keepdim=False):
    return dispatch('ReduceMin', (x,), {'dim': dim, 'keepdim': keepdim})


def _arg_max_implementation(input_value, dim=None, keepdim=False):
    return dispatch('ArgMax', (input_value,), {'dim': dim, 'keepdim': keepdim})




def arg_min(x, dim=-1, keepdim=False):
    return dispatch('ArgMin', (x,), {'dim': dim, 'keepdim': keepdim})


def xor_sum(x, dim=-1, keepdim=False):
    return dispatch('XorSum', (x,), {'dim': dim, 'keepdim': keepdim})


def cum_sum(x, dim=-1):
    return dispatch('CumSum', (x,), {'dim': dim})


def cum_prod(x, dim=-1):
    return dispatch('CumProd', (x,), {'dim': dim})


def transpose(x, perm=None):
    return dispatch('Transpose', (x,), {'perm': perm})


def concat(x, y, dim=-1):
    return dispatch('Concat', (x, y), {'dim': dim})


def flip(x, dim=-1):
    return dispatch('Flip', (x,), {'dim': dim})


def gather(x, index, dim=0):
    return dispatch('Gather', (x, index), {'dim': dim})


def sort_values(x, dim=-1, descending=False):
    return dispatch('SortValues', (x,), {'dim': dim, 'descending': descending})


def top_k_values(x, dim=-1, k=1):
    return dispatch('TopKValues', (x,), {'dim': dim, 'k': k})


def softmax(x, dim=-1):
    return dispatch('Softmax', (x,), {'dim': dim})


def rms_norm(x, gamma, epsilon=1e-06):
    return dispatch('RmsNorm', (x, gamma), {'epsilon': epsilon})


def layer_norm(x, weight, bias, epsilon=1e-06):
    return dispatch('LayerNorm', (x, weight, bias), {'epsilon': epsilon})


def matmul(x, weight):
    return dispatch('Matmul', (x, weight), {})


def mish(x):
    return dispatch('Mish', (x,), {})


def gelu(x, approximate='none'):
    return dispatch('Gelu', (x,), {'approximate': approximate})


def _swi_glu_implementation(input_value, dim=-1):
    return dispatch('SwiGlu', (input_value,), {'dim': dim})


