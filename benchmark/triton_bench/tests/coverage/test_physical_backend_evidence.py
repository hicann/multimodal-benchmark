# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Archived IR tests recognition only; these tests never grant device coverage."""
import hashlib
from pathlib import Path

import pytest

from triton_eval.ascend_lowering import PhysicalMemoryFlow, ascend_memory_witness
from triton_eval.compilation_evidence import compiler_passes, function_snapshot
from test_implementation_evidence import MATMUL, neg, synthetic, tensor, verified


FIXTURES = Path(__file__).parents[1] / 'fixtures/ascend_backend'
HASHES = {
    'Neg': '09b77a5def80e4a05ba142fdb5f8820693f08cd2f48cefffa32423e3f12679bc',
    'Matmul': 'dccaad7cef930e608216b562564094fc80a58f28f861d3ca67e32b2dba7a702f',
}
NAMES = {'Neg': '_witness_neg', 'Matmul': '_matmul'}
ATOMS = {'Neg': 'FA-PAT-VEC-LOAD', 'Matmul': 'FA-PAT-TILE-GEMM'}


def archived(operator):
    raw = (FIXTURES / (operator + '.txt')).read_bytes()
    assert hashlib.sha256(raw).hexdigest() == HASHES[operator]
    return raw.decode()


def snapshots(operator):
    return compiler_passes(archived(operator), NAMES[operator], include_before=True, allow_function=True)


def after(operator):
    return next(text for stage, text in snapshots(operator).items() if '-after-' in stage)


def observed(operator, lowered):
    if operator == 'Neg':
        return neg(524288, lowered=lowered)
    return synthetic('_matmul', [129, 2048], 'Matmul', MATMUL,
                     dict(ROWS=129, COLS=132, INNER=2048, TILE=16, INNER_TILE=32),
                     inputs={'X': tensor(0, [129, 2048]), 'Weight': tensor(1, [2048, 132])},
                     grid=[9, 9], lowered=lowered)


def line_containing(text, marker):
    return next(line for line in text.splitlines() if marker in line)


@pytest.mark.parametrize('operator', NAMES)
def test_archived_function_scoped_snapshots_preserve_phase_and_pass(operator):
    sections = snapshots(operator)
    assert len(sections) == 2
    assert any('-before-' in stage for stage in sections)
    assert any('-after-' in stage for stage in sections)
    assert all(stage.endswith('-hivm-graph-sync-solver') for stage in sections)
    assert all('infer_task_type_function' not in text and 'warning:' not in text for text in sections.values())
    assert not compiler_passes(archived(operator), NAMES[operator])
    for stage, text in sections.items():
        observation = observed(operator, {stage: text})
        assert verified(observation, ATOMS[operator])


@pytest.mark.parametrize('operator', NAMES)
@pytest.mark.parametrize('damage', ['truncated', 'foreign', 'host', 'duplicate', 'tail', 'return', 'header'])
def test_function_boundaries_reject_truncation_foreign_functions_and_trailing_ir(operator, damage):
    text = after(operator)
    if damage == 'truncated':
        text = text[:-1]
    elif damage == 'foreign':
        text = text.replace('@' + NAMES[operator] + '(', '@other(')
    elif damage == 'host':
        text = text.replace('hacc.function_kind<DEVICE>', 'hacc.function_kind<HOST>')
    elif damage == 'duplicate':
        text += '\n' + text
    elif damage == 'tail':
        text += '\nfunc.func @other() { return }'
    elif damage == 'return':
        text = text.replace('  return\n', '')
    else:
        text = text.replace('attributes {SyncBlockLockArgIdx', 'attributes {{SyncBlockLockArgIdx')
    assert function_snapshot(text, NAMES[operator]) is None
    assert not compiler_passes('// IR Dump After GraphSyncSolver\n' + text,
                               NAMES[operator], allow_function=True)


@pytest.mark.parametrize('operator', NAMES)
def test_dump_sections_cannot_complete_each_others_truncated_body(operator):
    text = after(operator)
    raw = '// IR Dump After First\n' + text[:-1] + '\n// IR Dump After Second\n' + text
    result = compiler_passes(raw, NAMES[operator], allow_function=True)
    assert len(result) == 1 and next(iter(result)).startswith('compiler-pass-0001')


def test_same_physical_ub_address_is_one_mutable_buffer_not_three_allocations():
    text = after('Neg')
    launch = observed('Neg', {})['launches'][0]
    flow = PhysicalMemoryFlow(text, launch)
    assert flow.supported
    assert flow.root('%9') == flow.root('%14') == flow.root('%15')
    assert len(flow.buffers) == 1
    assert len([node for node in flow.nodes if node['operation'] == 'hivm.hir.load']) == 1


@pytest.mark.parametrize('damage', [
    'dead_load', 'conditional_load', 'overwrite_alias', 'overlap', 'dynamic_pointer',
    'nonzero_padding', 'else', 'unknown_write', 'shifted_view', 'strided_view', 'direction',
])
def test_physical_vector_rejects_unproven_or_overwritten_dataflow(damage):
    text = after('Neg')
    load = line_containing(text, 'hivm.hir.load ins(')
    if damage == 'dead_load':
        text = text.replace(load, '')
    elif damage == 'conditional_load':
        text = text.replace(load, '  scf.if %13 {\n' + load + '\n  }')
    elif damage == 'overwrite_alias':
        multiply = line_containing(text, 'hivm.hir.vmul')
        text = text.replace(
            multiply,
            '  hivm.hir.vbrc ins(%cst_0 : f16) outs(%14 : memref<256xf16, #hivm.address_space<ub>>)\n' +
            multiply)
    elif damage == 'overlap':
        text = text.replace('%14 = hivm.hir.pointer_cast(%c0_i64)', '%14 = hivm.hir.pointer_cast(%c256_i32)')
    elif damage == 'dynamic_pointer':
        text = text.replace('pointer_cast(%c0_i64)', 'pointer_cast(%arg5)')
    elif damage == 'nonzero_padding':
        text = text.replace('vbrc ins(%cst_0', 'vbrc ins(%cst')
    elif damage == 'else':
        text = text.replace('} {hivm.unlikely_condition}', '} else {\n  }')
    elif damage == 'unknown_write':
        text = text.replace('hivm.hir.vmul', 'hivm.hir.unknown_write')
    elif damage == 'shifted_view':
        text = text.replace('memref.subview %9[0]', 'memref.subview %9[128]')
    elif damage == 'strided_view':
        text = text.replace('%9[0] [%12] [1]', '%9[0] [%12] [2]')
    else:
        text = text.replace('tt.tensor_kind = 0', 'tt.tensor_kind = 1')
    assert not verified(observed('Neg', {'compiler-pass-after-test': text}), ATOMS['Neg'])


@pytest.mark.parametrize('damage', [
    'left_only', 'dead_input', 'overlap', 'self_overlap', 'missing_annotation', 'empty_loop',
    'zero_step', 'dynamic_loop', 'conditional_transfer', 'swapped_operands', 'unrelated_output',
    'overwritten_input', 'missing_output', 'unknown_matrix', 'conditional_matrix', 'nested_loop',
])
def test_physical_matrix_requires_both_nonaliased_live_inputs_and_nonempty_loop(damage):
    text = after('Matmul')
    left = line_containing(text, 'hivm.hir.nd2nz {dst_continuous} ins(%subview_6')
    matrix = line_containing(text, 'hivm.hir.mmadL1')
    if damage == 'left_only':
        text = text.replace('ins(%subview_11 :', 'ins(%subview_6 :')
    elif damage == 'dead_input':
        text = text.replace(left, '')
    elif damage == 'overlap':
        text = text.replace('pointer_cast(%c1024_i64, %c3072_i64)', 'pointer_cast(%c1024_i64, %c2048_i64)')
    elif damage == 'self_overlap':
        text = text.replace('pointer_cast(%c1024_i64, %c3072_i64)', 'pointer_cast(%c1024_i64, %c1024_i64)')
    elif damage == 'missing_annotation':
        text = text.replace('hivm.multi_buffer = 2', 'hivm.multi_buffer = 3')
    elif damage == 'empty_loop':
        text = text.replace('to %c64_i32 step', 'to %c0_i32 step')
    elif damage == 'zero_step':
        text = text.replace('step %c1_i32', 'step %c0_i32')
    elif damage == 'dynamic_loop':
        text = text.replace('to %c64_i32 step', 'to %arg6 step')
    elif damage == 'conditional_transfer':
        text = text.replace(left, '    scf.if %49 {\n' + left + '\n    }')
    elif damage == 'swapped_operands':
        text = text.replace('ins(%cast_5, %cast_10,', 'ins(%cast_10, %cast_5,')
    elif damage == 'unrelated_output':
        text = text.replace('ins(%cast_1 :', 'ins(%cast_8 :')
    elif damage == 'overwritten_input':
        text = text.replace(
            matrix,
            '    hivm.hir.vbrc ins(%cst : f16) outs(%41 : memref<2x1x16x16xf16, #hivm.address_space<cbuf>>)\n' +
            matrix)
    elif damage == 'missing_output':
        text = text.replace(line_containing(text, 'hivm.hir.fixpipe {'), '')
    elif damage == 'unknown_matrix':
        text = text.replace('hivm.hir.mmadL1', 'hivm.hir.unproven_mmad')
    elif damage == 'conditional_matrix':
        text = text.replace(matrix, '    scf.if %55 {\n' + matrix + '\n    }')
    else:
        text = text.replace(
            matrix,
            '    scf.for %nested = %c0_i32 to %c64_i32 step %c1_i32 : i32 {\n' +
            matrix +
            '\n    }')
    assert not verified(observed('Matmul', {'compiler-pass-after-test': text}), ATOMS['Matmul'])


@pytest.mark.parametrize('operator', NAMES)
def test_raw_diagnostic_stderr_is_not_a_coverage_evidence_stage(operator):
    for stage in ('probe-stderr', 'compiler_stderr', 'compiler_stderr_truncated'):
        launch = observed(operator, {stage: archived(operator)})['launches'][0]
        assert ascend_memory_witness(launch, 'vector' if operator == 'Neg' else 'tile') is None
