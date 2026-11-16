# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Independent Torch and NumPy references for bounded coverage extensions.

No delegation between oracles. No submission imports. Inputs are generated on
CPU from the frozen seed before copying to the chosen device.
"""
from __future__ import annotations
import math


def _invalid_mask_layout(mask, tensor, dimension):
    if (mask.shape != tensor.shape):
        return True
    if (not 0 <= dimension < tensor.ndim):
        return True
    if (not bool(
                        ((mask == 0) | (mask == 1)).all())):
        return True
    return not bool((mask != 0).any(dim=dimension).all())


def _invalid_group_layout(groups, tensor):
    if (not groups):
        return True
    if (min(groups) < 1):
        return True
    if (sum(groups) != tensor.shape[0]):
        return True
    return bool(tensor.shape[1] != 32)


def _invalid_cummin_layout(tensor, attrs):
    if (tensor.ndim != 3):
        return True
    if (not 0 <= attrs['dim'] < 3):
        return True
    return bool(max(tensor.shape) > 32)


def _invalid_scatter_layout(dimension, tensor, index, update):
    if (not 0 <= dimension < tensor.ndim):
        return True
    if (index.shape != update.shape):
        return True
    return bool(update.dtype != tensor.dtype)


def make_inputs(name, *, seed, dtype, shape, device, attrs, value_range=None):
    import torch
    if name == "Transpose" and dtype != torch.int16:
        from .official_reference import make_inputs as official_inputs
        return official_inputs(name, seed=seed, dtype=dtype, shape=shape,
                               device=device, attrs=attrs, value_range=value_range)
    g = torch.Generator(device="cpu").manual_seed(seed)

    def rand(size):
        if dtype.is_floating_point:
            return torch.randn(size, generator=g).to(dtype).to(device)
        return torch.randint(1, 16, size, generator=g).to(dtype).to(device)
    x = (torch.randn(shape, generator=g).to(device)
         if name == "MaskedSoftmax" and not dtype.is_floating_point else rand(shape))
    if name == "Transpose":
        return [x]
    if name == "Scatter":
        dim = attrs["dim"]
        idx_shape = list(shape)
        idx_shape[dim] //= 2
        if attrs.get("reduce") == "add":
            # Guaranteed collisions on every lane, and untouched destinations.
            index = torch.zeros(idx_shape, dtype=torch.int64, device=device)
        else:
            index = torch.rand(idx_shape, generator=g).argsort(dim=dim).to(device)
        return [x, index, rand(idx_shape)]
    if name == "Cummin":
        # Include equal minima: indices must resolve ties to the last occurrence.
        x.select(attrs["dim"], 0).fill_(-2)
        x.select(attrs["dim"], 1).fill_(-2)
        return [x]
    if name == "GroupedMatmul":
        groups = attrs["group_list"]
        return [x, rand([len(groups), shape[1], 16]), rand([len(groups), 16])]
    if name == "CrossEntropyLoss":
        target = torch.randint(0, shape[1], (shape[0],), generator=g, dtype=torch.int64)
        target[0] = attrs["ignore_index"]
        # Large common offsets test the stable logsumexp path.
        return [x + 100, target.to(device)]
    if name == "MaskedSoftmax":
        mask = torch.randint(0, 2, shape, generator=g, dtype=torch.int64).to(
            dtype if not dtype.is_floating_point else torch.int8).to(device)
        mask.select(attrs["dim"], 0).fill_(1)
        return [x, mask]
    if name == "ApplyRotaryPosEmb":
        angles = torch.randn([*shape[:-1], shape[-1] // 2], generator=g)
        return [x, angles.cos().to(dtype).to(device), angles.sin().to(dtype).to(device)]
    raise ValueError(name)


def reference(name, *inputs, **attrs):
    import torch
    x = inputs[0]
    if name == "Transpose":
        return x.permute(*attrs["perm"]).contiguous()
    if name == "Scatter":
        out = x.clone()
        if attrs.get("reduce") == "add":
            return out.scatter_add_(attrs["dim"], inputs[1], inputs[2])
        return out.scatter_(attrs["dim"], inputs[1], inputs[2])
    if name == "Cummin":
        return tuple(torch.cummin(x, attrs["dim"]))
    if name == "GroupedMatmul":
        result, start = [], 0
        for i, size in enumerate(attrs["group_list"]):
            result.append(x[start:start + size].float() @ inputs[1][i].float() + inputs[2][i].float())
            start += size
        return torch.cat(result).to(x.dtype)
    if name == "CrossEntropyLoss":
        return torch.nn.functional.cross_entropy(x.float(), inputs[1], **attrs).to(x.dtype)
    if name == "MaskedSoftmax":
        return torch.softmax(x.float().masked_fill(inputs[1] == 0, -float("inf")), attrs["dim"]).to(x.dtype)
    if name == "ApplyRotaryPosEmb":
        if attrs.get("rotary_mode", "half") == "half":
            a, b = x.float().chunk(2, -1)
            rotated = torch.cat([-b, a], -1)
            cos, sin = [torch.cat([t.float(), t.float()], -1) for t in inputs[1:]]
        else:
            pairs = x.float().reshape(*x.shape[:-1], -1, 2)
            rotated = torch.stack([-pairs[..., 1], pairs[..., 0]], -1).reshape(x.shape)
            cos, sin = [t.float().repeat_interleave(2, -1) for t in inputs[1:]]
        return (x.float() * cos + rotated * sin).to(x.dtype)
    raise ValueError(name)


def cpu_reference(name, *inputs, **attrs):
    import numpy as np
    import torch
    arrays = [t.detach().float().numpy().astype(np.float64) if t.is_floating_point()
              else t.detach().numpy().copy() for t in inputs]
    x = arrays[0]

    def tensor(value, dtype=None):
        return torch.from_numpy(np.array(value, copy=True)).to(dtype or inputs[0].dtype)
    if name == "Transpose":
        return tensor(np.transpose(x, attrs["perm"]))
    if name == "Scatter":
        return _cpu_scatter(x, arrays, attrs, tensor)
    if name == "Cummin":
        return _cpu_cummin(x, arrays, attrs, tensor)
    if name == "GroupedMatmul":
        pieces, start = [], 0
        for i, size in enumerate(attrs["group_list"]):
            pieces.append(np.einsum("ik,kj->ij", x[start:start + size], arrays[1][i]) + arrays[2][i])
            start += size
        return tensor(np.concatenate(pieces))
    if name == "CrossEntropyLoss":
        losses = []
        for row, label in zip(x, arrays[1]):
            if label == attrs["ignore_index"]:
                continue
            maximum = max(row)
            losses.append(maximum + math.log(sum(math.exp(v - maximum) for v in row)) - row[int(label)])
        return tensor(sum(losses) if attrs["reduction"] == "sum" else sum(losses) / len(losses))
    if name == "MaskedSoftmax":
        masked = np.where(arrays[1] != 0, x, -np.inf)
        dim = attrs["dim"]
        exponent = np.exp(masked - np.max(masked, axis=dim, keepdims=True))
        return tensor(exponent / np.sum(exponent, axis=dim, keepdims=True))
    if name == "ApplyRotaryPosEmb":
        return _cpu_rotary(x, arrays, attrs, tensor)
    raise ValueError(name)


def bind(name):
    def main(*inputs, **attrs):
        return reference(name, *inputs, **attrs)

    def factory(*, seed, dtype, shape, device, attrs, value_range=None):
        return make_inputs(name, seed=seed, dtype=dtype, shape=shape,
                           device=device, attrs=attrs, value_range=value_range)
    return main, factory


def bind_cpu(name):
    def cross(*inputs, **attrs):
        return cpu_reference(name, *inputs, **attrs)
    return cross


def validate_inputs(name, inputs, attrs):
    import torch
    from .operator_registry import by_name
    spec = by_name(name)
    if len(inputs) != len(spec.inputs):
        raise ValueError('coverage input arity mismatch')
    for tensor, declared in zip(inputs, spec.inputs):
        if not isinstance(tensor, torch.Tensor) or str(tensor.dtype).replace('torch.', '') not in declared.dtype:
            raise ValueError(f'coverage input dtype mismatch: {declared.name}')
        if tensor.device != inputs[0].device or not tensor.is_contiguous():
            raise ValueError('coverage extensions require contiguous tensors on one device')
    validators = {
        'Transpose': _validate_transpose_inputs,
        'Scatter': _validate_scatter_inputs,
        'Cummin': _validate_cummin_inputs,
        'GroupedMatmul': _validate_grouped_inputs,
        'CrossEntropyLoss': _validate_cross_entropy_inputs,
        'MaskedSoftmax': _validate_masked_softmax_inputs,
        'ApplyRotaryPosEmb': _validate_rotary_inputs,
    }
    validator = validators.get(name)
    if validator is not None:
        validator(inputs, attrs)


def _validate_transpose_inputs(inputs, attrs):
    x = inputs[0]
    if x.ndim != 2 or attrs['perm'] != [1, 0]:
        raise ValueError('coverage transpose requires 2D perm=[1,0]')


def _validate_scatter_inputs(inputs, attrs):
    import torch

    x = inputs[0]
    index, update = inputs[1:]
    d = attrs['dim']
    if _invalid_scatter_layout(d, x, index, update):
        raise ValueError('invalid scatter shape/axis/dtype')
    if any(index.shape[j] != x.shape[j] for j in range(x.ndim) if j != d):
        raise ValueError('scatter non-axis dimensions must agree')
    if bool(((index < 0) | (index >= x.shape[d])).any()):
        raise ValueError('scatter index out of bounds')
    if attrs.get('reduce') == 'add':
        if x.dtype not in (torch.int32, torch.int64):
            raise ValueError('scatter add requires exact wide-integer accumulation')
    elif attrs.get('reduce') is not None:
        raise ValueError('unsupported scatter reduction')
    elif index.shape[d] > 1 and not bool((index.sort(dim=d).values.diff(dim=d) > 0).all()):
        raise ValueError('scatter overwrite requires unique indices')


def _validate_cummin_inputs(inputs, attrs):
    x = inputs[0]
    if _invalid_cummin_layout(x, attrs):
        raise ValueError('bounded cummin requires rank3 and axes <=32')


def _validate_grouped_inputs(inputs, attrs):
    x = inputs[0]
    groups = attrs['group_list']
    if _invalid_group_layout(groups, x):
        raise ValueError('invalid group partition/K')
    if inputs[1].shape != (len(groups), 32, 16) or inputs[2].shape != (len(groups), 16):
        raise ValueError('invalid grouped weight/bias shapes')


def _validate_cross_entropy_inputs(inputs, attrs):
    x = inputs[0]
    target = inputs[1]
    if x.ndim != 2 or target.shape != (x.shape[0],) or attrs['reduction'] not in ('mean', 'sum'):
        raise ValueError('invalid cross entropy contract')
    valid = target != attrs['ignore_index']
    if not bool(valid.any()) or bool(((target[valid] < 0) | (target[valid] >= x.shape[1])).any()):
        raise ValueError('invalid cross entropy targets')


def _validate_masked_softmax_inputs(inputs, attrs):
    x = inputs[0]
    mask = inputs[1]
    d = attrs['dim']
    if _invalid_mask_layout(mask, x, d):
        raise ValueError('invalid mask or fully masked softmax row')


def _validate_rotary_inputs(inputs, attrs):
    x = inputs[0]
    expected = (*x.shape[:-1], x.shape[-1] // 2)
    if x.shape[-1] % 2 or any(t.shape != expected for t in inputs[1:]
                              ) or attrs['rotary_mode'] not in ('half', 'interleaved'):
        raise ValueError('invalid rotary inputs')


def _cpu_scatter(value, arrays, attrs, tensor):
    import numpy as np

    out, index, updates = value.copy(), arrays[1], arrays[2]
    for position in np.ndindex(index.shape):
        dest = list(position)
        dest[attrs["dim"]] = int(index[position])
        if attrs.get("reduce") == "add":
            out[tuple(dest)] += updates[position]
        else:
            out[tuple(dest)] = updates[position]
    return tensor(out)


def _cpu_cummin(value, arrays, attrs, tensor):
    import numpy as np
    import torch

    dim = attrs["dim"]
    rows = np.moveaxis(value, dim, -1)
    values = np.minimum.accumulate(rows, axis=-1)
    indices = np.zeros(rows.shape, dtype=np.int64)
    for i in range(rows.shape[-1]):
        indices[..., i] = np.where(rows[..., i] == values[..., i], i,
                                   indices[..., i - 1] if i else 0)
    return tensor(np.moveaxis(values, -1, dim)), tensor(np.moveaxis(indices, -1, dim), torch.int64)


def _cpu_rotary(value, arrays, attrs, tensor):
    import numpy as np

    half = value.shape[-1] // 2
    out = np.empty_like(value)
    for j in range(value.shape[-1]):
        if attrs.get("rotary_mode", "half") == "half":
            pair, source, sign = j % half, (j + half) % (2 * half), -1 if j < half else 1
        else:
            pair, source, sign = j // 2, j ^ 1, -1 if j % 2 == 0 else 1
        out[..., j] = value[..., j] * arrays[1][..., pair] + sign * value[..., source] * arrays[2][..., pair]
    return tensor(out)
