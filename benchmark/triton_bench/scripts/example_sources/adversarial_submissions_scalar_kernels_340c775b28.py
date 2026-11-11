# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Standalone scalar implementations for the fourteen positive controls."""

from __future__ import annotations
__all__ = [
    '_nest',
    '_walk',
    'cross_entropy_loss',
    'cummin',
    'cummin_values',
    'itertools',
    'math',
    'product',
    'softmax_values',
    'torch',
]

from itertools import product
import itertools
import math
import torch
from scripts.example_sources.scalar_support import cummin_values
from scripts.example_sources.scalar_support import softmax_values
from scripts.example_sources._shared_escapee_10c7261575 import _walk
from scripts.example_sources._shared_escapee_10c7261575 import _nest
from scripts.templates.coverage_kernels import cross_entropy_loss
from scripts.templates.coverage_kernels import cummin










def _build_arg_max(shape, values, dtype, device):
    if not shape:
        # 0 维输出（如 1-D 输入 dim=0 keepdim=False）：值列表只有一个元素
        return torch.tensor(values[0], dtype=dtype, device=device)
    return torch.tensor(_nest(shape, iter(values)), dtype=dtype, device=device)


def _build(shape, values, dtype, device):
    if not shape:
        return torch.tensor(values, dtype=dtype, device=device)
    return torch.tensor(_nest(shape, iter(values)), dtype=dtype, device=device)


def _like(x, values, shape=None):
    shape = list(x.shape) if shape is None else list(shape)
    if not shape:
        return torch.tensor(values, dtype=x.dtype, device=x.device)
    return torch.tensor(_nest(shape, iter(values)), dtype=x.dtype, device=x.device)


def _like2(x, m, n, values):
    nested = [[values[i * n + j] for j in range(n)] for i in range(m)]
    return torch.tensor(nested, dtype=x.dtype, device=x.device)


def _arg_max_implementation(input_value, dim, keepdim=False):
    shape = list(input_value.shape)
    d = dim if dim >= 0 else dim + len(shape)
    n = shape[d]

    xf = list(_walk(input_value))
    inner = 1
    for s in shape[d + 1:]:
        inner *= s
    outer = len(xf) // (n * inner)

    out_shape = list(shape)
    if bool(keepdim):
        out_shape[d] = 1
    else:
        out_shape.pop(d)

    res = []
    for o, k in product(range(outer), range(inner)):
        best_j = 0
        best_v = None
        for j in range(n):
            v = xf[(o * n + j) * inner + k]
            if best_v is None or v > best_v:  # 平手保留先出现者
                best_v, best_j = v, j
        res.append(float(best_j))
    return _build_arg_max(out_shape, res, torch.int64, input_value.device)


def arg_max(input_value, dim=None, keepdim=False):
    """Use the registered input_value contract at the API boundary."""
    return _arg_max_implementation(input_value, dim, keepdim)


def _cross_entropy_loss_implementation(input_value, target, reduction="mean", ignore_index=-100):
    in_shape = list(input_value.shape)
    if len(in_shape) < 2:
        in_shape = [1, in_shape[0]] if in_shape else [1, 1]
    n, c = in_shape[0], in_shape[1]
    spatial = in_shape[2:]  # (N, C, *spatial)；target 是 (N, *spatial)
    sp = 1
    for s in spatial:
        sp *= s

    xf = list(_walk(input_value))
    tf = [int(v) for v in _walk(target)]
    ig = int(ignore_index)

    losses = []
    for i in range(n):
        for p in range(sp):
            t = tf[i * sp + p] if i * sp + p < len(tf) else ig
            if t == ig:
                continue
            base = i * c * sp + p
            row = [xf[base + j * sp] for j in range(c)]
            m = max(row)
            exps = [math.exp(v - m) for v in row]
            logsum = m + math.log(sum(exps))
            losses.append(logsum - row[t])

    if reduction == "sum":
        v = sum(losses) if losses else 0.0
    elif reduction == "mean":
        v = (sum(losses) / len(losses)) if losses else float("nan")
    else:
        raise ValueError(f"本实现只支持 mean/sum，收到：{reduction}")
    return torch.tensor(v, dtype=input_value.dtype, device=input_value.device)




def _cummin_implementation(input_value, dim):
    shape, values, indices = cummin_values(input_value, dim)
    output = _build(shape, values, input_value.dtype, input_value.device)
    positions = _build(shape, indices,
                       torch.int64, input_value.device)
    return output, positions




def exp(x, base=-1.0, scale=1.0, shift=0.0):
    lb = 1.0 if base in (None, -1.0) else math.log(float(base))
    vals = [
        math.exp((v * float(scale) + float(shift)) * lb)
        for v in _walk(x)
    ]
    return _like(x, vals)


def gather(x, index, dim=0):
    shape = list(x.shape)
    d = dim if dim >= 0 else dim + len(shape)
    n = shape[d]

    xf = list(_walk(x))
    idxf = [int(v) for v in _walk(index)]

    inner = 1
    for s in shape[d + 1:]:
        inner *= s
    outer = len(idxf) // (n * inner)

    out = [0.0] * len(xf)
    for o in range(outer):
        for j in range(n):
            for k in range(inner):
                pos = (o * n + j) * inner + k
                src = (o * n + idxf[pos]) * inner + k
                out[pos] = xf[src]
    return _like(x, out)


def gelu(x, approximate="none"):
    out = []
    for v in _walk(x):
        v = float(v)
        if approximate == "tanh":
            inner = math.sqrt(2.0 / math.pi) * (v + 0.044715 * v ** 3)
            y = 0.5 * v * (1.0 + math.tanh(inner))
        else:
            y = 0.5 * v * (1.0 + math.erf(v / math.sqrt(2.0)))
        out.append(y)
    return _like(x, out)


def grouped_matmul(x, weight, bias, group_list):
    m = x.shape[0]
    k = x.shape[1] if x.dim() > 1 else 1
    n = weight.shape[2]

    xf = list(_walk(x))
    wf = list(_walk(weight))
    bf = list(_walk(bias))

    out = [0.0] * (m * n)
    start = 0
    for g, cnt in enumerate(int(v) for v in group_list):
        cnt = int(cnt)
        for i, j in product(range(cnt), range(n)):
            acc = bf[g * n + j]
            for c in range(k):
                acc += xf[(start + i) * k + c] * wf[(g * k + c) * n + j]
            out[(start + i) * n + j] = acc
        start += cnt
    return _like2(x, m, n, out)


def masked_scale(x, mask, scale=1.0):
    vals = [
        v * m * float(scale)
        for v, m in zip(_walk(x), _walk(mask))
    ]
    return _like(x, vals)


def maximum(x1, x2):
    vals = []
    for a, b in zip(_walk(x1), _walk(x2)):
        fa, fb = float(a), float(b)
        vals.append(fa if fa >= fb else fb)
    return _like(x1, vals)


def rms_norm(x, gamma, epsilon=1e-6):
    shape = list(x.shape)
    n = shape[-1]
    outer = 1
    for s in shape[:-1]:
        outer *= s

    xf = list(_walk(x))
    gf = list(_walk(gamma))
    out = [0.0] * len(xf)
    for o in range(outer):
        row = xf[o * n:(o + 1) * n]
        var = sum(v * v for v in row) / n
        inv = 1.0 / math.sqrt(var + float(epsilon))
        for i in range(n):
            out[o * n + i] = row[i] * inv * gf[i]
    return _like(x, out)


def scatter(data, indices, updates, dim, reduce=None):
    if reduce is not None and reduce != "none":
        raise ValueError(f"本实现只支持 reduce=None，收到：{reduce}")

    shape = list(data.shape)
    d = dim if dim >= 0 else dim + len(shape)
    n = shape[d]

    df = list(_walk(data))
    uf = list(_walk(updates))
    idxf = [int(v) for v in _walk(indices)]

    inner = 1
    for s in shape[d + 1:]:
        inner *= s
    outer = len(df) // (n * inner)

    k = len(idxf) // (outer * inner)  # index 沿 dim 的长度（make_inputs 取一半）
    out = list(df)
    for o in range(outer):
        for j in range(k):
            for m in range(inner):
                upos = (o * k + j) * inner + m
                tgt = (o * n + idxf[upos]) * inner + m
                out[tgt] = uf[upos]
    return _like(data, out)


def softmax(x, dim=-1):
    shape, values = softmax_values(x, dim)
    return torch.tensor(values[0] if not shape else _nest(shape, iter(values)),
                        dtype=x.dtype, device=x.device)


def _swi_glu_implementation(input_value, dim=-1):
    shape = list(input_value.shape)
    d = dim if dim >= 0 else dim + len(shape)
    n = shape[d]
    half = n // 2

    xf = list(_walk(input_value))
    inner = 1
    for s in shape[d + 1:]:
        inner *= s
    outer = len(xf) // (n * inner)

    out = [0.0] * (outer * half * inner)
    for o in range(outer):
        for j in range(half):
            for k in range(inner):
                a = xf[(o * n + j) * inner + k]
                b = xf[(o * n + half + j) * inner + k]
                silu = a / (1.0 + math.exp(-a))
                out[(o * half + j) * inner + k] = silu * b

    new_shape = list(shape)
    new_shape[d] = half
    return _build(new_shape, out, input_value.dtype, input_value.device)


def swi_glu(input_value, dim=-1):
    """Use the registered input_value contract at the API boundary."""
    return _swi_glu_implementation(input_value, dim)


def transpose(x, perm):
    shape = list(x.shape)
    perm = [int(p) for p in perm]
    new_shape = [shape[p] for p in perm]

    xf = list(_walk(x))
    out = []
    for idx in itertools.product(*[range(s) for s in new_shape]):
        src = [0] * len(shape)
        for i, p in enumerate(perm):
            src[p] = idx[i]
        f = 0
        for v, s in zip(src, shape):
            f = f * s + v
        out.append(xf[f])
    return _build(new_shape, out, x.dtype, x.device)
