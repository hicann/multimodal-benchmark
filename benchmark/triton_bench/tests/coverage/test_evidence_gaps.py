# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only checks of evidence rules, not fabricated compiler qualification."""
import json
import os
from types import SimpleNamespace

import pytest
from test_implementation_evidence import (
    FUSION,
    GRID,
    fusion,
    neg,
    synthetic,
    tensor,
    verified,
)
from triton_eval.compilation_evidence import compiler_passes
from triton_eval.implementation_evidence import IRGraph
from triton_eval.reference_execution import reference_execution

QUOTED_FUSION = (
    FUSION.replace(
        '%maximum = tt.reduce %masked : f32',
        '''%maximum = "tt.reduce"(%masked) <{axis = 0 : i32}> ({
^bb0(%left: f32, %right: f32):
  %temporary = arith.maxnumf %left, %right : f32
  tt.reduce.return %temporary : f32
}) : (tensor<16xf32>) -> f32''',
    )
    .replace(
        '%sum = tt.reduce %e : f32',
        '''%sum = "tt.reduce"(%e) <{axis = 0 : i32}> ({
^bb0(%left: f32, %right: f32):
  %temporary = arith.addf %left, %right : f32
  tt.reduce.return %temporary : f32
}) : (tensor<16xf32>) -> f32''',
    )
)


def test_quoted_reduction_regions_are_dependencies_without_leaking_region_ssa():
    graph = IRGraph(QUOTED_FUSION, ['X', 'Mask', 'Out'])
    assert graph.defs['%maximum'][0] == graph.defs['%sum'][0] == 'tt.reduce'
    assert '%temporary' not in graph.defs and '%left' not in graph.defs
    observation = fusion()
    observation['launches'][0]['stage_text']['ttir'] = QUOTED_FUSION
    assert verified(observation, 'FA-FUSE-INTERM-ELIDE')
    observation['launches'][0]['stage_text']['ttir'] = (
        QUOTED_FUSION.replace('arith.divf %e, %sum', 'arith.divf %x, %one'))
    assert not verified(observation, 'FA-FUSE-INTERM-ELIDE')


def test_unterminated_generic_reduction_fails_closed():
    with pytest.raises(ValueError, match='unterminated'):
        IRGraph('%sum = "tt.reduce"(%e) <{axis = 0 : i32}> ({\n', ['X'])


def grid3():
    return synthetic(
        '_witness_neg3', [3, 7, 17], 'Neg',
        GRID.replace('@_witness_neg4', '@_witness_neg3'),
        {'D0': 3, 'D1': 7, 'D2': 17, 'STRIDE': 1, 'BLOCK': 256},
        inputs={'Input': tensor(0, [3, 7, 17])}, grid=[1, 7, 3])


def test_three_dimensional_launch_requires_live_axes_actual_shape_and_stride():
    assert verified(grid3(), 'FA-PROG-GRID3D')
    for change in ('grid', 'shape', 'stride', 'axis'):
        observation = grid3()
        if change == 'grid':
            observation['launches'][0]['grid'] = [21]
        elif change == 'shape':
            observation['launches'][0]['arguments']['D1']['value'] = 6
        elif change == 'stride':
            observation['data']['input_bindings'][0]['stride'] = [119, 18, 1]
        else:
            observation['launches'][0]['stage_text']['ttir'] = (
                GRID.replace('tt.get_program_id z', 'tt.get_program_id y'))
        assert not verified(observation, 'FA-PROG-GRID3D')


ASCEND_VECTOR = (
    'module {\n'
    'func.func @_witness_neg(%input: memref<?xf16> {tt.tensor_kind = 0 : i32},'
    ' %output: memref<?xf16> {tt.tensor_kind = 1 : i32}) {\n'
    ' %buffer = memref.alloc() : memref<256xf16, #hivm.address_space<ub>>\n'
    ' %result = memref.alloc() : memref<256xf16, #hivm.address_space<ub>>\n'
    ' hivm.hir.load ins(%input : memref<?xf16>) outs(%buffer : memref<256xf16, #hivm.address_space<ub>>)\n'
    ' hivm.hir.vneg ins(%buffer : memref<256xf16>) outs(%result : memref<256xf16>)\n'
    ' hivm.hir.store ins(%result : memref<256xf16>) outs(%output : memref<?xf16>)\n'
    ' return\n'
    '}\n'
    '}'
)


def test_explicit_ascend_block_transfer_requires_live_output_and_annotated_memory():
    observation = neg(8192, lowered={'compiler-pass-0001': ASCEND_VECTOR})
    assert verified(observation, 'FA-PAT-VEC-LOAD')
    for lowered in (
            ASCEND_VECTOR.replace('hivm.hir.load', 'memref.copy'),
            ASCEND_VECTOR.replace('#hivm.address_space<ub>', '#hivm.address_space<unknown>'),
            ASCEND_VECTOR.replace('ins(%result :', 'ins(%unrelated :'),
            ASCEND_VECTOR.replace('hivm.hir.vneg ins(%buffer', 'hivm.hir.vneg ins(%unrelated'),
            ASCEND_VECTOR.replace('tt.tensor_kind = 0', 'other.attribute = 0')):
        observation = neg(8192, lowered={'compiler-pass-0001': lowered})
        assert not verified(observation, 'FA-PAT-VEC-LOAD')


def test_plain_adapter_copy_and_truncated_or_unscoped_dumps_never_certify_vectorization():
    for stage, text in [('ttadapter', ASCEND_VECTOR.replace('hivm.hir.', 'adapter.')),
                        ('compiler_stderr', ASCEND_VECTOR), ('compiler_stderr_truncated', ASCEND_VECTOR)]:
        assert not verified(neg(8192, lowered={stage: text}), 'FA-PAT-VEC-LOAD')


ASCEND_TILE = (
    'module {\n'
    'func.func @_matmul(%first: memref<?xf16> {tt.tensor_kind = 0 : i32},'
    ' %second: memref<?xf16> {tt.tensor_kind = 0 : i32}, %output: memref<?xf16> {tt.tensor_kind = 1 : i32}) {\n'
    ' %left_l1 = memref.alloc() : memref<16x32xf16, #hivm.address_space<l1>>\n'
    ' %right_l1 = memref.alloc() : memref<32x16xf16, #hivm.address_space<l1>>\n'
    ' %left = memref.alloc() : memref<16x32xf16, #hivm.address_space<l0a>>\n'
    ' %right = memref.alloc() : memref<32x16xf16, #hivm.address_space<l0b>>\n'
    ' %accumulator = memref.alloc() : memref<16x16xf32, #hivm.address_space<l0c>>\n'
    ' hivm.hir.load ins(%first : memref<?xf16>) outs(%left_l1 : memref<16x32xf16>)\n'
    ' hivm.hir.load ins(%second : memref<?xf16>) outs(%right_l1 : memref<32x16xf16>)\n'
    ' memref.copy %left_l1, %left : memref<16x32xf16> to memref<16x32xf16>\n'
    ' memref.copy %right_l1, %right : memref<32x16xf16> to memref<32x16xf16>\n'
    ' hivm.hir.mmad ins(%left, %right : memref<16x32xf16>,'
    ' memref<32x16xf16>) outs(%accumulator : memref<16x16xf32>)\n'
    ' hivm.hir.store ins(%accumulator : memref<16x16xf32>) outs(%output : memref<?xf16>)\n'
    ' return\n'
    '}\n'
    '}'
)


def test_ascend_matrix_staging_requires_both_global_operands_and_live_matrix_result():
    from test_implementation_evidence import MATMUL

    def matrix(text):
        return synthetic('_matmul', [129, 2048], 'Matmul', MATMUL,
                         {'ROWS': 129, 'COLS': 32, 'INNER': 2048, 'TILE': 16, 'INNER_TILE': 32},
                         inputs={'X': tensor(0, [129, 2048]), 'Weight': tensor(1, [2048, 32])}, grid=[9, 2],
                         lowered={'compiler-pass-0001': text})
    assert verified(matrix(ASCEND_TILE), 'FA-PAT-TILE-GEMM')
    for text in (ASCEND_TILE.replace('ins(%second :', 'ins(%first :'),
                 ASCEND_TILE.replace('#hivm.address_space<l0b>', '#hivm.address_space<ub>'),
                 ASCEND_TILE.replace('ins(%accumulator :', 'ins(%unrelated :'),
                 ASCEND_TILE.replace('hivm.hir.mmad', 'linalg.matmul')):
        assert not verified(matrix(text), 'FA-PAT-TILE-GEMM')


def test_compiler_dump_modules_are_separate_complete_after_passes_for_the_same_kernel():
    text = '// -----// IR Dump Before SomePass //----- //\n' + ASCEND_VECTOR
    text += '\n// -----// IR Dump After SomePass //----- //\n' + ASCEND_VECTOR
    text += '\n// -----// IR Dump After OtherPass //----- //\n' + ASCEND_VECTOR.replace('@_witness_neg', '@unrelated')
    sections = compiler_passes(text, '_witness_neg')
    assert len(sections) == 1 and 'hivm.hir.load' in next(iter(sections.values()))
    assert not compiler_passes('// IR Dump After Pass\n' + ASCEND_VECTOR[:-2], '_witness_neg')


@pytest.mark.parametrize('failure', [False, True])
def test_strict_reference_execution_restores_previous_mode_even_on_failure(monkeypatch, failure):
    import sys

    state = {'enabled': False, 'warning': True}
    calls = []

    def set_mode(enabled, warn_only=False):
        calls.append((enabled, warn_only))
        state.update(enabled=enabled, warning=warn_only)
    fake = SimpleNamespace(are_deterministic_algorithms_enabled=lambda: state['enabled'],
                           is_deterministic_algorithms_warn_only_enabled=lambda: state['warning'],
                           use_deterministic_algorithms=set_mode,
                           npu=SimpleNamespace(synchronize=lambda: calls.append('sync')))
    monkeypatch.setitem(sys.modules, 'torch', fake)

    def execute():
        with reference_execution('CrossEntropyLoss', [SimpleNamespace(device='npu:0')]):
            assert state == {'enabled': True, 'warning': False}
            if failure:
                raise RuntimeError('unsupported deterministic algorithm')
    if failure:
        with pytest.raises(RuntimeError, match='unsupported'):
            execute()
    else:
        execute()
        assert 'sync' in calls
    assert state == {'enabled': False, 'warning': True}
    assert calls[0] == (True, False) and calls[-1] == (False, True)


def test_cpu_and_other_operator_references_do_not_change_execution_policy(monkeypatch):
    import torch

    monkeypatch.setattr(torch, 'use_deterministic_algorithms', lambda *args, **kwargs: pytest.fail('unexpected policy'))
    for operator, device in [('CrossEntropyLoss', 'cpu'), ('Matmul', 'npu:0')]:
        with reference_execution(operator, [SimpleNamespace(device=device)]):
            pass


def test_capture_saves_native_compiler_pass_dump_and_restores_stderr(tmp_path, monkeypatch):
    from test_coverage_evidence import SyntheticJIT, triton_fixture
    from triton_eval.compilation_evidence import CompilationCapture, compiler_stderr

    monkeypatch.setenv('TRITON_EVAL_CAPTURE_PASSES', '1')

    class DumpingJIT(SyntheticJIT):
        def run(self, *args, **kwargs):
            os.write(2, ('// IR Dump After Pass\n' + ASCEND_VECTOR).encode())
            return super().run(*args, **kwargs)
    directory = tmp_path / 'compilation/case'
    with CompilationCapture(triton_fixture(DumpingJIT), directory, {}) as capture:
        DumpingJIT().run(grid=(1,))
    artifacts = capture.payload['launches'][0]['artifacts']
    assert any(entry['stage'] == 'compiler_stderr' for entry in artifacts)
    with pytest.raises(RuntimeError), compiler_stderr():
        raise RuntimeError('compile failed')
    with compiler_stderr() as stream:
        os.write(2, b'restored')
        stream.seek(0)
        assert stream.read() == b'restored'
    assert not json.loads((directory / 'observation.json').read_text())['diagnostics']


def test_failed_jit_preserves_native_diagnostic_without_granting_any_launch(tmp_path, monkeypatch):
    from test_coverage_evidence import SyntheticJIT, triton_fixture
    from triton_eval.compilation_evidence import CompilationCapture
    from triton_eval.coverage_evidence import read_blob

    monkeypatch.setenv('TRITON_EVAL_CAPTURE_PASSES', '1')

    class FailingJIT(SyntheticJIT):
        def run(self, *args, **kwargs):
            os.write(2, b'backend diagnostic')
            raise RuntimeError('compile failure')
    directory = tmp_path / 'compilation/case'
    with pytest.raises(RuntimeError, match='compile failure'):
        with CompilationCapture(triton_fixture(FailingJIT), directory, {}) as capture:
            FailingJIT().run(grid=(1,))
    assert not capture.payload['launches']
    entry = capture.payload['failed_compilation_artifacts'][0]
    assert read_blob(directory, entry) == b'backend diagnostic'
    assert capture.payload['diagnostics']
