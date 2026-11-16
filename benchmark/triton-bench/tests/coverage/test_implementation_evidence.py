# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Synthetic compiler fragments exercise recognition and rejection, never NPU credit."""
import ast
import json
import copy
from pathlib import Path
from types import SimpleNamespace

import pytest

from triton_eval.coverage_evidence import feature_witness
from triton_eval.implementation_evidence import IRGraph

ROOT = Path(__file__).resolve().parents[2]


def tensor(identity, shape, dtype='torch.float32'):
    stride, next_stride = [], 1
    for n in reversed(shape):
        stride.insert(0, next_stride)
        next_stride *= n
    return dict(tensor_id=identity, storage_id=identity, shape=shape, stride=stride, dtype=dtype)


def synthetic(name, shape, operator, ir, scalars, inputs=None, grid=None, lowered=None, attrs=None):
    # This synthetic IR uses historical argument roles, not current capture data.
    registry = json.loads((ROOT / 'assets/reviewed-kernels.json').read_text())
    node = ast.parse(registry['kernels'][name]['source']).body[0]
    names = [a.arg for a in node.args.args]
    ins = inputs or {'X': tensor(0, shape)}
    output = tensor(10, shape)
    bindings = {k: dict(kind='tensor', **v) for k, v in dict(ins, Out=output).items()}
    bindings.update({k: dict(kind='scalar', value=v, constexpr=True) for k, v in scalars.items()})
    launch = dict(name=name, source_text=ast.unparse(node), stage_text=dict(ttir=ir, **(lowered or {})),
                  arguments=bindings, runtime_arg_names=[n for n in names if n not in scalars], grid=grid)
    return dict(case=SimpleNamespace(shape=shape, operator=operator, contiguous=True, attrs=attrs or {}),
                data=dict(input_bindings=list(ins.values()), output_bindings=[output]),
                source=launch['source_text'], ir=ir, launches=[launch])


NEG = '''tt.func public @_witness_neg(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<f32>) {
 %pid = tt.get_program_id x : i32
 %r = tt.make_range {start = 0, end = 256} : tensor<256xi32>
 %base = arith.muli %pid, %c256 : i32
 %basev = tt.splat %base : i32 -> tensor<256xi32>
 %i = arith.addi %basev, %r : tensor<256xi32>
 %mask = arith.cmpi slt, %i, %n : tensor<256xi32>
 %xp = tt.splat %arg0 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
 %p = tt.addptr %xp, %i : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
 %x = tt.load %p, %mask, %zero : tensor<256x!tt.ptr<f32>>
 %y = arith.negf %x : tensor<256xf32>
 %op = tt.splat %arg1 : !tt.ptr<f32> -> tensor<256x!tt.ptr<f32>>
 %q = tt.addptr %op, %i : tensor<256x!tt.ptr<f32>>, tensor<256xi32>
 tt.store %q, %y, %mask : tensor<256x!tt.ptr<f32>>
 tt.return
}'''


def neg(n=8194, lowered=None):
    return synthetic('_witness_neg', [2, n // 2], 'Neg', NEG, dict(N=n, STRIDE=1, B=256),
                     grid=[(n + 255) // 256], lowered=lowered)


def verified(obs, atom):
    obs.pop('implementation_witnesses', None)
    return feature_witness(atom, obs)[0]


def test_live_masked_load_and_store_cover_nonempty_tail():
    obs = neg()
    assert verified(obs, 'FA-PAT-MASK-TAIL')
    obs['launches'][0]['stage_text']['ttir'] = NEG.replace('tt.store %q, %y, %mask', 'tt.store %q, %y, %always')
    assert not verified(obs, 'FA-PAT-MASK-TAIL')


@pytest.mark.parametrize('change', ['constant', 'aligned', 'dead', 'source', 'grid', 'binding', 'shape'])
def test_tail_rejects_decorative_keywords_or_inconsistent_binding(change):
    obs = neg()
    launch = obs['launches'][0]
    if change == 'constant':
        launch['stage_text']['ttir'] = NEG.replace('tt.load %p, %mask', 'tt.load %p, %always')
    elif change == 'aligned':
        obs = neg(8192)
    elif change == 'dead':
        launch['stage_text']['ttir'] = NEG.replace('arith.negf %x', 'arith.negf %zero')
    elif change == 'source':
        launch['source_text'] = launch['source_text'].replace('-x', 'x')
    elif change == 'grid':
        launch['grid'] = [1]
    elif change == 'binding':
        launch['arguments']['X']['tensor_id'] = 99
    else:
        launch['arguments']['N']['value'] = 1234
    assert not verified(obs, 'FA-PAT-MASK-TAIL')


def test_vector_load_requires_lowered_vector_memory_operation():
    obs = neg(8192)
    assert not verified(obs, 'FA-PAT-VEC-LOAD')
    obs['launches'][0]['stage_text']['llir'] = ('%v = load <8 x float>, ptr %input, align 32\n%neg = fneg <8 x float> '
                                                '%v\nstore <8 x float> %neg, ptr %out')
    assert verified(obs, 'FA-PAT-VEC-LOAD')
    obs['launches'][0]['stage_text']['llir'] = '%v = load float, ptr %input, align 4'
    assert not verified(obs, 'FA-PAT-VEC-LOAD')
    obs['launches'][0]['stage_text']['llir'] = ('%v = load <8 x float>, ptr %input, align 32\n%neg = fneg <8 x float> '
                                                '%v\nstore <8 x float> %neg, ptr %out')
    obs['case'].contiguous = False
    assert not verified(obs, 'FA-PAT-VEC-LOAD')


GRID = NEG.replace('@_witness_neg(', '@_witness_neg4(').replace(
    ' %base = arith.muli %pid, %c256 : i32',
    ''' %py = tt.get_program_id y : i32
 %pz = tt.get_program_id z : i32
 %outer = arith.muli %pz, %d2 : i32
 %row = arith.addi %outer, %py : i32
 %left = arith.muli %row, %d3 : i32
 %right = arith.muli %pid, %c256 : i32
 %base = arith.addi %left, %right : i32''')


def test_logical_4d_bijection_does_not_claim_native_four_axis_grid():
    obs = synthetic('_witness_neg4', [2, 3, 5, 17], 'Neg', GRID, dict(D0=2, D1=3, D2=5, D3=17, B=256), grid=[1, 5, 6])
    assert verified(obs, 'FA-PROG-GRID4D')
    proof = obs['implementation_witnesses']['FA-PROG-GRID4D']
    assert proof['native_grid_rank'] == 3 and proof['logical_shape'] == [2, 3, 5, 17]
    for grid in ([1, 5, 5], [1, 5, 2, 3], [1]):
        obs['launches'][0]['grid'] = grid
        assert not verified(obs, 'FA-PROG-GRID4D')
    obs['launches'][0]['grid'] = [1, 5, 6]
    obs['launches'][0]['stage_text']['ttir'] = GRID.replace('arith.addi %outer, %py', 'arith.addi %outer, %zero')
    assert not verified(obs, 'FA-PROG-GRID4D')


GATHER = '''tt.func public @_gather(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i32>, %arg2: !tt.ptr<f32>) {
 %r = tt.make_range {start = 0, end = 1024} : tensor<1024xi32>
 %p = tt.addptr %arg0, %r : !tt.ptr<f32>, tensor<1024xi32>
 %x = tt.load %p : tensor<1024x!tt.ptr<f32>>
 %j = tt.load %arg1 : !tt.ptr<i32>
 %g = tt.gather %x, %j {axis = 0} : (tensor<1024xf32>, i32) -> tensor<64xf32>
 tt.store %arg2, %g : !tt.ptr<f32>
 tt.return
}'''


def test_row_prefetch_and_index_input_feed_live_gather():
    obs = synthetic('_gather', [257, 1024], 'Gather', GATHER,
                    dict(COLS=1024, OUT_COLS=64, SOURCE_BLOCK=1024, INDEX_BLOCK=64),
                    inputs={'X': tensor(0, [257, 1024]), 'Index': tensor(1, [257, 64], 'torch.int32')}, grid=[257])
    assert verified(obs, 'FA-PAT-COALESCE-GATHER')
    for broken in (GATHER.replace('tt.store %arg2, %g', 'tt.store %arg2, %x'),
                   GATHER.replace('tt.gather %x, %j', 'tt.gather %x, %zero'),
                   GATHER.replace('tt.addptr %arg0, %r', 'tt.addptr %arg0, %constant')):
        obs['launches'][0]['stage_text']['ttir'] = broken
        assert not verified(obs, 'FA-PAT-COALESCE-GATHER')


FUSION = '''tt.func public @_witness_masked_softmax(%arg0: !tt.ptr<f32>, %arg1: !tt.ptr<i8>, %arg2: !tt.ptr<f32>) {
 %x = tt.load %arg0 : !tt.ptr<f32>
 %m = tt.load %arg1 : !tt.ptr<i8>
 %masked = arith.select %m, %x, %negative_inf : f32
 %maximum = tt.reduce %masked : f32
 %shifted = arith.subf %masked, %maximum : f32
 %e = math.exp %shifted : f32
 %sum = tt.reduce %e : f32
 %result = arith.divf %e, %sum : f32
 tt.store %arg2, %result : !tt.ptr<f32>
 tt.return
}'''


def fusion():
    return synthetic('_witness_masked_softmax',
                     [3,
                      5,
                      17],
                     'MaskedSoftmax',
                     FUSION,
                     dict(R=3,
                          L2=17,
                          XS1=17,
                          XS2=1,
                          XSR=85,
                          MS1=17,
                          MS2=1,
                          MSR=85,
                          OS1=17,
                          OS2=1,
                          OSR=85,
                          B=16),
                     inputs={'X': tensor(0,
                                         [3,
                                          5,
                                          17]),
                             'Mask': tensor(1,
                                            [3,
                                             5,
                                             17],
                                            'torch.int8')},
                     grid=[85],
                     attrs={'dim': 0})


def test_intermediate_elimination_requires_ssa_chain_and_no_scratch_or_output_alias():
    obs = fusion()
    assert verified(obs, 'FA-FUSE-INTERM-ELIDE')
    assert obs['implementation_witnesses']['FA-FUSE-INTERM-ELIDE']['global_store_roots'] == ['Out']
    changes = [
        lambda o:o['launches'].append(copy.deepcopy(o['launches'][0])),
        lambda o:o['launches'][0]['stage_text'].update(
            ttir=FUSION.replace(
                'tt.store %arg2, %result',
                'tt.store %arg0, %result')),
        lambda o:o['launches'][0]['stage_text'].update(
            ttir=FUSION.replace('%e = math.exp %shifted', '%e = math.exp %x')),
        lambda o:o['launches'][0]['stage_text'].update(ttir=FUSION.replace(
            'tt.return', 'tt.call @hidden_scratch()\n tt.return')),
        lambda o:o['data']['output_bindings'][0].update(storage_id=0),
    ]
    for change in changes:
        bad = fusion()
        change(bad)
        assert not verified(bad, 'FA-FUSE-INTERM-ELIDE')


MATMUL = '''tt.func public @_matmul(%arg0: !tt.ptr<f16>, %arg1: !tt.ptr<f16>, %arg2: !tt.ptr<f16>) {
 %acc = scf.for %iv = %zero to %k step %one iter_args(%prev = %zeros) -> (tensor<16x16xf32>) {
   %x = tt.load %arg0 : !tt.ptr<f16>
   %w = tt.load %arg1 : !tt.ptr<f16>
   %dot = tt.dot %x, %w, %prev : f16
   scf.yield %dot : tensor<16x16xf32>
 }
 tt.store %arg2, %acc : !tt.ptr<f16>
 tt.return
}'''


def test_tiled_dot_requires_loop_dataflow_and_explicit_shared_memory_lowering():
    obs = synthetic('_matmul', [129, 2048], 'Matmul', MATMUL,
                    dict(ROWS=129, COLS=32, INNER=2048, TILE=16, INNER_TILE=32),
                    inputs={'X': tensor(0, [129, 2048]), 'Weight': tensor(1, [2048, 32])}, grid=[9, 2])
    assert not verified(obs, 'FA-PAT-TILE-GEMM')
    obs['launches'][0]['stage_text'][
        'ttgir'] = '''%sa = ttg.local_alloc %x : (tensor<16x32xf16>) -> !ttg.memdesc<16x32xf16, #shared, #smem>
%sb = ttg.local_alloc %w : (tensor<32x16xf16>) -> !ttg.memdesc<32x16xf16, #shared, #smem>
%a = ttg.local_load %sa : !ttg.memdesc<16x32xf16, #shared, #smem> -> tensor<16x32xf16>
%b = ttg.local_load %sb : !ttg.memdesc<32x16xf16, #shared, #smem> -> tensor<32x16xf16>
%d = tt.dot %a, %b, %zero : tensor<16x16xf32>
tt.store %arg2, %d : tensor<16x16x!tt.ptr<f16>>'''
    assert verified(obs, 'FA-PAT-TILE-GEMM')
    obs['launches'][0]['stage_text']['ttir'] = MATMUL.replace('scf.yield %dot', 'scf.yield %zeros')
    assert not verified(obs, 'FA-PAT-TILE-GEMM')


def test_dead_shared_allocation_cannot_certify_tile_optimization():
    obs = synthetic('_matmul', [129, 2048], 'Matmul', MATMUL,
                    dict(ROWS=129, COLS=32, INNER=2048, TILE=16, INNER_TILE=32),
                    inputs={'X': tensor(0, [129, 2048]), 'Weight': tensor(1, [2048, 32])}, grid=[9, 2])
    obs['launches'][0]['stage_text'][
        'ttgir'] = """%sa = ttg.local_alloc %x : (tensor<16x32xf16>) -> !ttg.memdesc<16x32xf16, #shared, #smem>
%d = tt.dot %x, %w, %zero : tensor<16x16xf32>"""
    assert not verified(obs, 'FA-PAT-TILE-GEMM')


def test_nested_dead_copy_of_trusted_source_does_not_certify_wrapper():
    import textwrap
    obs = neg()
    original = obs['launches'][0]['source_text']
    obs['launches'][0]['source_text'] = 'def _witness_neg(X, Out):\n' + \
        textwrap.indent(original, '    ') + '\n    pass\n'
    assert not verified(obs, 'FA-PAT-MASK-TAIL')


def test_dead_lowered_vector_read_is_not_memory_evidence():
    obs = neg(8192, lowered={'llir': """%unused = load <8 x float>, ptr %input, align 32
%scalar = load float, ptr %input
%neg = fneg float %scalar
store float %neg, ptr %out"""})
    assert not verified(obs, 'FA-PAT-VEC-LOAD')


def test_dead_lowered_shared_dot_is_not_tile_evidence():
    obs = synthetic('_matmul', [129, 2048], 'Matmul', MATMUL,
                    dict(ROWS=129, COLS=32, INNER=2048, TILE=16, INNER_TILE=32),
                    inputs={'X': tensor(0, [129, 2048]), 'Weight': tensor(1, [2048, 32])}, grid=[9, 2])
    obs['launches'][0]['stage_text'][
        'ttgir'] = """%sa = ttg.local_alloc %x : (tensor<16x32xf16>) -> !ttg.memdesc<16x32xf16, #shared, #smem>
%sb = ttg.local_alloc %w : (tensor<32x16xf16>) -> !ttg.memdesc<32x16xf16, #shared, #smem>
%a = ttg.local_load %sa : !ttg.memdesc<16x32xf16, #shared, #smem> -> tensor<16x32xf16>
%b = ttg.local_load %sb : !ttg.memdesc<32x16xf16, #shared, #smem> -> tensor<32x16xf16>
%dead = tt.dot %a, %b, %zero : tensor<16x16xf32>
%actual = tt.dot %x, %w, %zero : tensor<16x16xf32>
tt.store %arg2, %actual : tensor<16x16x!tt.ptr<f16>>"""
    assert not verified(obs, 'FA-PAT-TILE-GEMM')
