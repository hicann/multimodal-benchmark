# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""CPU-only pattern planning, actual-source simulation and adversarial IR tests.

Synthetic compiler fixtures and NumPy simulation never certify NPU coverage.
This module is also runnable with unittest when the full CLI dependencies are absent.
"""

from __future__ import annotations

import ast
import copy
import itertools
import json
import math
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from kernel_simulation import constant_arguments

from triton_eval.pattern_generation import TARGETS, candidate_plans, device_bytes, select_patterns, target_features
from triton_eval.pattern_evidence import pattern_witnesses
from triton_eval.pattern_release import pattern_gate, preview, submission_source
from triton_eval.anticheat import scan_source

try:
    import numpy as np
except ImportError:
    np = None

ROOT = Path(__file__).resolve().parents[2]


def candidate(operator, dtype, shape, attrs, index=0):
    case = SimpleNamespace(
        code=f"candidate-{index}",
        operator=operator,
        dtype=dtype,
        shape=shape,
        attrs=attrs,
        contiguous=True,
        value_range={"kind": "normal"},
        seed=index,
    )
    case.feature_atoms = sorted(target_features(case))
    return case


def _pattern_statement(node, helpers):
    return isinstance(node, ast.Assign) or isinstance(node, ast.FunctionDef) and node.name in helpers


class PatternSelectionTests(unittest.TestCase):
    def setUp(self):
        self.pool = [candidate(*plan, index=index) for index, plan in enumerate(candidate_plans())]

    def test_all_explicit_targets_have_two_structural_witnesses(self):
        selected, report = select_patterns([], self.pool, TARGETS)
        self.assertEqual(len(selected), 8)
        self.assertEqual(report["target_witnesses"], {atom: 2 for atom in TARGETS})
        self.assertTrue(all(device_bytes(case) < 64 * 1024**2 for case in selected))
        reverse, reverse_report = select_patterns([], list(reversed(self.pool)), TARGETS)
        self.assertEqual([case.code for case in selected], [case.code for case in reverse])
        self.assertEqual(report, reverse_report)
        self.assertEqual(report["compilation_state"], "NOT_RUN")

    def test_explicit_missing_target_is_not_hidden_by_ninety_percent(self):
        baseline = candidate("Neg", "float32", [8], {}, 100)
        baseline.feature_atoms = [f"base-{index}" for index in range(67)]
        active = baseline.feature_atoms + list(TARGETS)
        with self.assertRaisesRegex(ValueError, "unreachable"):
            select_patterns([baseline], self.pool[:-2], active)
        self.assertEqual(len(baseline.feature_atoms), 67)

    def test_budget_and_memory_fail_closed(self):
        for kwargs in ({"budget": 7}, {"per_operator": 1}, {"max_device_bytes": 1024}):
            with self.subTest(kwargs=kwargs), self.assertRaisesRegex(ValueError, "within budget"):
                select_patterns([], self.pool, TARGETS, **kwargs)

    def test_seed_only_variation_cannot_supply_second_witness(self):
        pool = [copy.deepcopy(case) for case in self.pool[::2]]
        for case in list(pool):
            duplicate = copy.deepcopy(case)
            duplicate.code += "-seed"
            duplicate.seed += 1000
            pool.append(duplicate)
        with self.assertRaisesRegex(ValueError, "within budget"):
            select_patterns([], pool, TARGETS)

    def test_total_elements_do_not_substitute_for_scan_axis(self):
        case = candidate("Cummin", "float32", [2**20, 1], {"dim": 1})
        self.assertFalse(target_features(case))

    def test_plain_matmul_cannot_claim_bias_reuse(self):
        case = candidate("Matmul", "float16", [2048, 32], {})
        self.assertNotIn("FA-PAT-VEC-BROADCAST", target_features(case))

    def test_preview_remains_unfrozen(self):
        report = preview()
        self.assertEqual(report["retained_parent_cases"], 1377)
        self.assertEqual(report["projected_cases"], 1385)
        self.assertEqual(report["state"], "DRAFT_NOT_FROZEN")
        self.assertEqual(report["static_validation"], "NOT_RUN")
        self.assertEqual(len(report["active_atoms"]), 71)

    def test_entry_signatures_and_legacy_kernel_sources_are_preserved(self):
        tree = ast.parse(submission_source())
        definitions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
        for name, parameters in dict(
            reduce_max=["x", "dim", "keepdim"],
            matmul=["x", "weight"],
            cummin=["input_value", "dim"],
            grouped_matmul=["x", "weight", "bias", "group_list"],
        ).items():
            self.assertEqual([arg.arg for arg in definitions[name].args.args], parameters)
        self.assertTrue({"_matmul", "_cummin", "_grouped"} <= set(definitions))

    def test_host_source_passes_existing_anticheat_scanner(self):
        for entry, category in [
            ("reduce_max", "Reduction"),
            ("cummin", "Reduction"),
            ("matmul", "Contraction"),
            ("grouped_matmul", "Contraction"),
        ]:
            self.assertEqual(scan_source(submission_source(), category=category, entry=entry), [])

    def test_new_candidate_targets_are_derived_by_existing_static_predicates(self):
        source = ast.parse((ROOT / "src/triton_eval/coverage.py").read_text())
        pattern_helpers = {
            "_pattern_atoms", "_contraction_pattern_atoms", "_reduction_pattern_atoms", "_has_static_group_loop"}
        future = ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0)
        predicates = [node for node in source.body if _pattern_statement(node, pattern_helpers)]
        source.body = [future] + predicates
        namespace = {}
        exec(compile(ast.fix_missing_locations(source), "<actual-static-predicates>", "exec"), namespace)
        try:
            pattern_atoms = namespace["_pattern_atoms"]
        except KeyError as exc:
            raise AssertionError("Static predicate extraction omitted _pattern_atoms") from exc
        for case in self.pool:
            spec = SimpleNamespace(
                category=SimpleNamespace(
                    value="Contraction" if case.operator in ("Matmul", "GroupedMatmul") else "Reduction"
                )
            )
            self.assertTrue(target_features(case) <= set(pattern_atoms(case, spec)))


def binding(identity, shape, dtype="torch.float32"):
    stride, strides = 1, []
    for size in reversed(shape):
        strides.insert(0, stride)
        stride *= size
    return dict(tensor_id=identity, storage_id=identity, shape=list(shape), stride=strides, dtype=dtype)


def reduction(value, incoming, operation="maximumf"):
    return f""" {value} = "tt.reduce"({incoming}) ({{
 ^bb0(%left: f32, %right: f32):
  %combined = arith.{operation} %left, %right : f32
  tt.reduce.return %combined : f32
 }}) : (tensor<1024xf32>) -> f32"""


def scan(value, incoming, indices):
    return f""" {value}:2 = "tt.associative_scan"({incoming}, {indices}) <{{axis = 0 : i32, reverse = false}}> ({{
 ^bb0(%left: f32, %left_index: i32, %right: f32, %right_index: i32):
  %less = arith.cmpf olt, %right, %left : f32
  %equal = arith.cmpf oeq, %right, %left : f32
  %later = arith.cmpi sgt, %right_index, %left_index : i32
  %tie = arith.andi %equal, %later : i1
  %take = arith.ori %less, %tie : i1
  %minimum = arith.select %take, %right, %left : f32
  %index = arith.select %take, %right_index, %left_index : i32
  tt.scan.return %minimum, %index : f32, i32
 }}) : (tensor<512xf32>, tensor<512xi32>) -> (tensor<512xf32>, tensor<512xi32>)"""


def launch(name, tensors, scalars, grid, body):
    # Synthetic tensor/scalar bindings below use the historical argument roles.
    registry = json.loads((ROOT / "assets/reviewed-kernels.json").read_text())
    source = ast.parse(registry["kernels"][name]["source"]).body[0]
    names = list(tensors)
    signature = ", ".join(f"%arg{index}: !tt.ptr<f32>" for index in range(len(names)))
    text = f"module {{\n tt.func public @{name}({signature}) {{\n{body}\n tt.return\n }}\n}}"
    return dict(
        name=name,
        source_text=ast.unparse(source),
        grid=grid,
        compiler_specialization_hash="a" * 64,
        runtime_arg_names=names,
        arguments={
            **{key: dict(kind="tensor", **value) for key, value in tensors.items()},
            **{key: dict(kind="scalar", constexpr=True, value=value) for key, value in scalars.items()},
        },
        stage_text={"ttir": text},
    )


def max_observation():
    case = candidate("ReduceMax", "float32", [1, 2**20], {"dim": -1, "keepdim": False})
    source, partial, output = binding(0, case.shape), binding(1, [1, 1024]), binding(2, [1])
    first = launch(
        "_pattern_max_partial",
        dict(X=source, Partial=partial),
        dict(COLUMNS=2**20, CHUNKS=1024, BLOCK=1024),
        [1024, 1],
        " %data = tt.load %arg0 : tensor<1024x!tt.ptr<f32>>\n"
        + reduction("%max", "%data")
        + "\n tt.store %arg1, %max : !tt.ptr<f32>",
    )
    final = launch(
        "_pattern_max_finish",
        dict(Partial=partial, Out=output),
        dict(CHUNKS=1024, BLOCK=1024),
        [1],
        " %data = tt.load %arg0 : tensor<1024x!tt.ptr<f32>>\n"
        + reduction("%max", "%data")
        + "\n tt.store %arg1, %max : !tt.ptr<f32>",
    )
    return dict(case=case, data=dict(input_bindings=[source], output_bindings=[output]), launches=[first, final])


def split_observation():
    case = candidate("Matmul", "float16", [16, 65536], {})
    source, weight = binding(0, case.shape, "torch.float16"), binding(1, [65536, 19], "torch.float16")
    partial, output = binding(2, [8, 16, 19]), binding(3, [16, 19], "torch.float16")
    body = """ %split = tt.get_program_id z : i32
 %segment = arith.constant 8192 : i32
 %weight_segment = arith.constant 155648 : i32
 %output_segment = arith.constant 304 : i32
 %x_offset = arith.muli %split, %segment : i32
 %w_offset = arith.muli %split, %weight_segment : i32
 %o_offset = arith.muli %split, %output_segment : i32
 %px = tt.addptr %arg0, %x_offset : !tt.ptr<f16>, i32
 %pw = tt.addptr %arg1, %w_offset : !tt.ptr<f16>, i32
 %first = tt.load %px : !tt.ptr<f16>
 %second = tt.load %pw : !tt.ptr<f16>
 %result = tt.dot %first, %second, %zero : f32
 %po = tt.addptr %arg2, %o_offset : !tt.ptr<f32>, i32
 tt.store %po, %result : !tt.ptr<f32>"""
    first = launch(
        "_pattern_split_partial",
        dict(X=source, Weight=weight, Partial=partial),
        dict(ROWS=16, COLUMNS=19, INNER=65536, SEGMENT=8192, TILE=16, INNER_TILE=32),
        [1, 2, 8],
        body,
    )
    final = launch(
        "_pattern_split_finish",
        dict(Partial=partial, Out=output),
        dict(ELEMENTS=304, PARTS=8, BLOCK=256),
        [2],
        " %data = tt.load %arg0 : tensor<8x256x!tt.ptr<f32>>\n"
        + reduction("%sum", "%data", "addf")
        + "\n tt.store %arg1, %sum : !tt.ptr<f16>",
    )
    return dict(
        case=case, data=dict(input_bindings=[source, weight], output_bindings=[output]), launches=[first, final]
    )


def scan_observation():
    case = candidate("Cummin", "float32", [1, 2**20], {"dim": 1})
    source, values, indices = binding(0, case.shape), binding(1, case.shape), binding(2, case.shape, "torch.int64")
    local_values, local_indices = binding(3, case.shape), binding(4, case.shape, "torch.int64")
    totals, total_indices = binding(5, [1, 2048]), binding(6, [1, 2048], "torch.int64")
    carry, carry_indices = binding(7, [1, 2048]), binding(8, [1, 2048], "torch.int64")
    local = launch(
        "_pattern_scan_local",
        dict(X=source, LocalValues=local_values, LocalIndices=local_indices, Totals=totals, TotalIndices=total_indices),
        dict(AXIS=2**20, INNER=1, CHUNKS=2048, BLOCK=512),
        [2048, 1],
        " %data = tt.load %arg0 : tensor<512x!tt.ptr<f32>>\n" + scan("%scan", "%data", "%positions") + "\n"
        " tt.store %arg1, %scan#0 : !tt.ptr<f32>\n tt.store %arg2, %scan#1 : !tt.ptr<i64>\n"
        " %tail_value = arith.select %is_last, %scan#0, %zero : tensor<512xf32>\n"
        " %tail_index = arith.select %is_last, %scan#1, %negative_one : tensor<512xi32>\n"
        + reduction("%total", "%tail_value", "addf")
        + "\n"
        + reduction("%total_index", "%tail_index", "maxsi")
        + "\n"
        " tt.store %arg3, %total : !tt.ptr<f32>\n tt.store %arg4, %total_index : !tt.ptr<i64>",
    )
    carrying = launch(
        "_pattern_scan_carry",
        dict(Totals=totals, TotalIndices=total_indices, Carry=carry, CarryIndices=carry_indices),
        dict(CHUNKS=2048, BLOCK=2048),
        [1],
        " %data = tt.load %arg0 : tensor<2048x!tt.ptr<f32>>\n %index = tt.load %arg1 : tensor<2048x!tt.ptr<i64>>\n"
        + scan("%scan", "%data", "%index")
        + "\n tt.store %arg2, %scan#0 : !tt.ptr<f32>\n tt.store %arg3, %scan#1 : !tt.ptr<i64>",
    )
    final = launch(
        "_pattern_scan_finish",
        dict(
            LocalValues=local_values,
            LocalIndices=local_indices,
            Carry=carry,
            CarryIndices=carry_indices,
            Values=values,
            Indices=indices,
        ),
        dict(AXIS=2**20, INNER=1, CHUNKS=2048, BLOCK=512),
        [2048, 1],
        """ %lv = tt.load %arg0 : !tt.ptr<f32>
 %li = tt.load %arg1 : !tt.ptr<i64>
 %cv = tt.load %arg2 : !tt.ptr<f32>
 %ci = tt.load %arg3 : !tt.ptr<i64>
 %pv = arith.cmpf olt, %cv, %lv : f32
 %equal = arith.cmpf oeq, %cv, %lv : f32
 %pi = arith.cmpi sgt, %ci, %li : i64
 %tie = arith.andi %equal, %pi : i1
 %predicate = arith.ori %pv, %tie : i1
 %value = arith.select %predicate, %cv, %lv : f32
 %index = arith.select %predicate, %ci, %li : i64
 tt.store %arg4, %value : !tt.ptr<f32>
 tt.store %arg5, %index : !tt.ptr<i64>""",
    )
    return dict(
        case=case,
        data=dict(input_bindings=[source], output_bindings=[values, indices]),
        launches=[local, carrying, final],
    )


def bias_observation():
    case = candidate("GroupedMatmul", "float16", [2048, 32], {"group_list": [2048]})
    source, weight = binding(0, case.shape, "torch.float16"), binding(1, [1, 32, 16], "torch.float16")
    bias, output = binding(2, [1, 16], "torch.float16"), binding(3, [2048, 16], "torch.float16")
    body = """ %row = tt.get_program_id x : i32
 %column = tt.get_program_id y : i32
 %px = tt.addptr %arg0, %row : !tt.ptr<f16>, i32
 %pw = tt.addptr %arg1, %column : !tt.ptr<f16>, i32
 %x = tt.load %px : tensor<16x!tt.ptr<f16>>
 %w = tt.load %pw : tensor<16x!tt.ptr<f16>>
 %dot = tt.dot %x, %w, %zero : f32
 %pb = tt.addptr %arg2, %column : !tt.ptr<f16>, i32
 %b = tt.load %pb : tensor<16x!tt.ptr<f16>>
 %expanded = tt.expand_dims %b {axis = 0} : tensor<16xf16> -> tensor<1x16xf16>
 %broadcast = tt.broadcast %expanded : tensor<1x16xf16> -> tensor<16x16xf16>
 %result = arith.addf %dot, %broadcast : tensor<16x16xf32>
 tt.store %arg3, %result : !tt.ptr<f16>"""
    compiled = launch(
        "_pattern_bias_tile",
        dict(X=source, Weight=weight, Bias=bias, Out=output),
        dict(ROWS=2048, START=0, GROUP=0, INNER=32, COLUMNS=16, TILE=16, INNER_TILE=16),
        [128, 1],
        body,
    )
    return dict(
        case=case, data=dict(input_bindings=[source, weight, bias], output_bindings=[output]), launches=[compiled]
    )


class PatternEvidenceTests(unittest.TestCase):
    def test_all_reviewed_synthetic_patterns(self):
        for atom, factory in zip(TARGETS, [max_observation, scan_observation, split_observation, bias_observation]):
            with self.subTest(atom=atom):
                proof = pattern_witnesses(factory())[atom]
                self.assertTrue(proof["verified"], proof["reason"])
                self.assertEqual(proof["performance_gain"], "NOT_ESTABLISHED")

    def test_cross_launch_foreign_buffer_and_alias_are_rejected(self):
        for atom, factory in zip(TARGETS, [max_observation, scan_observation, split_observation, bias_observation]):
            observed = factory()
            name = next(iter(observed["launches"][-1]["arguments"]))
            observed["launches"][-1]["arguments"][name]["tensor_id"] = 999
            with self.subTest(atom=atom):
                self.assertFalse(pattern_witnesses(observed)[atom]["verified"])
        observed = max_observation()
        observed["data"]["output_bindings"][0]["storage_id"] = 0
        self.assertFalse(pattern_witnesses(observed)["FA-PAT-MULTIPASS-REDUCE"]["verified"])

    def test_source_grid_specialization_and_truncation_fail_closed(self):
        for field, value in [("source_text", "def bad(): pass"), ("grid", [1]), ("compiler_specialization_hash", "")]:
            observed = max_observation()
            observed["launches"][0][field] = value
            self.assertFalse(pattern_witnesses(observed)["FA-PAT-MULTIPASS-REDUCE"]["verified"])
        observed = max_observation()
        observed["launches"][0]["stage_text"]["ttir"] = observed["launches"][0]["stage_text"]["ttir"][:-1]
        self.assertFalse(pattern_witnesses(observed)["FA-PAT-MULTIPASS-REDUCE"]["verified"])

    def test_dead_reduction_and_wrong_reduction_kind_are_rejected(self):
        for changed in [("tt.store %arg1, %max", "tt.store %arg1, %zero"), ("arith.maximumf", "arith.minimumf")]:
            observed = max_observation()
            text = observed["launches"][-1]["stage_text"]["ttir"]
            observed["launches"][-1]["stage_text"]["ttir"] = text.replace(*changed)
            self.assertFalse(pattern_witnesses(observed)["FA-PAT-MULTIPASS-REDUCE"]["verified"])

    def test_split_axis_must_reach_both_inputs_and_partial_address(self):
        for original, replacement in [
            ("%arg0, %x_offset", "%arg0, %zero"),
            ("%arg1, %w_offset", "%arg1, %zero"),
            ("%arg2, %o_offset", "%arg2, %zero"),
            ("tt.get_program_id z", "tt.get_program_id x"),
            ("constant 304", "constant 1"),
            ("constant 8192", "constant 1"),
        ]:
            observed = split_observation()
            observed["launches"][0]["stage_text"]["ttir"] = observed["launches"][0]["stage_text"]["ttir"].replace(
                original, replacement
            )
            self.assertFalse(pattern_witnesses(observed)["FA-PAT-SPLIT-K"]["verified"])

    def test_scan_requires_three_stages_both_outputs_and_latest_tie(self):
        for mutation in ("missing-stage", "index-output", "first-tie", "max-scan", "dead-carry"):
            observed = scan_observation()
            if mutation == "missing-stage":
                observed["launches"].pop(1)
            elif mutation == "index-output":
                observed["data"]["output_bindings"].pop()
            else:
                position = 2 if mutation == "dead-carry" else 0
                original, replacement = {
                    "first-tie": ("cmpi sgt", "cmpi slt"),
                    "max-scan": ("cmpf olt", "cmpf ogt"),
                    "dead-carry": ("%predicate, %cv, %lv", "%predicate, %lv, %lv"),
                }[mutation]
                observed["launches"][position]["stage_text"]["ttir"] = observed["launches"][position]["stage_text"][
                    "ttir"
                ].replace(original, replacement)
            with self.subTest(mutation=mutation):
                self.assertFalse(pattern_witnesses(observed)["FA-PAT-SCAN-BLOCKED"]["verified"])

    def test_bias_must_be_live_row_invariant_and_reused(self):
        for original, replacement in [
            ("%arg2, %column", "%arg2, %row"),
            ("%dot, %broadcast", "%dot, %dot"),
            ("tt.broadcast %expanded", "tt.reshape %expanded"),
        ]:
            observed = bias_observation()
            observed["launches"][0]["stage_text"]["ttir"] = observed["launches"][0]["stage_text"]["ttir"].replace(
                original, replacement
            )
            self.assertFalse(pattern_witnesses(observed)["FA-PAT-VEC-BROADCAST"]["verified"])

    def test_unknown_writes_and_control_flow_are_rejected(self):
        for extra in [
            "memref.store %data, %foreign[] : memref<1xf32>",
            "tt.call @hidden_write()",
            "scf.if %condition {\n }",
        ]:
            observed = max_observation()
            observed["launches"][0]["stage_text"]["ttir"] = observed["launches"][0]["stage_text"]["ttir"].replace(
                " tt.return", extra + "\n tt.return"
            )
            self.assertFalse(pattern_witnesses(observed)["FA-PAT-MULTIPASS-REDUCE"]["verified"])

    def test_swapped_matrix_inputs_and_intermediate_shapes_are_rejected(self):
        observed = split_observation()
        observed["launches"][0]["stage_text"]["ttir"] = observed["launches"][0]["stage_text"]["ttir"].replace(
            "tt.dot %first, %second", "tt.dot %second, %first"
        )
        self.assertFalse(pattern_witnesses(observed)["FA-PAT-SPLIT-K"]["verified"])
        observed = max_observation()
        observed["launches"][0]["arguments"]["Partial"]["shape"] = [1, 1]
        self.assertFalse(pattern_witnesses(observed)["FA-PAT-MULTIPASS-REDUCE"]["verified"])


class PatternGateTests(unittest.TestCase):
    def test_functionality_pass_does_not_replace_pattern_witnesses(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(dict(manifest_hash="test", cases=[])))
            (root / "coverage-validation.json").write_text(
                json.dumps(dict(selected_operator_validation_passed=True, manifest_hash="test"))
            )
            self.assertEqual(pattern_gate(root), 1)
            self.assertFalse(json.loads((root / "target-pattern-validation.json").read_text())["passed"])

    def test_gate_requires_two_structural_compile_and_execution_witnesses(self):
        pool = [candidate(*plan, index=index) for index, plan in enumerate(candidate_plans())]
        cases = [vars(case) for case in pool]
        witnesses = {atom: [case.code for case in pool if atom in case.feature_atoms] for atom in TARGETS}
        proof = dict(
            selected_operator_validation_passed=True,
            manifest_hash="test",
            selected_operators=sorted(TARGETS.values()),
            compilation_witnesses=copy.deepcopy(witnesses),
            execution_witnesses=copy.deepcopy(witnesses),
            compilation=dict(covered=sorted(TARGETS)),
            execution_validation=dict(covered=sorted(TARGETS)),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "manifest.json").write_text(json.dumps(dict(manifest_hash="test", cases=cases)))
            (root / "coverage-validation.json").write_text(json.dumps(proof))
            self.assertEqual(pattern_gate(root), 0)
            atom = next(iter(TARGETS))
            proof["execution_witnesses"][atom].pop()
            (root / "coverage-validation.json").write_text(json.dumps(proof))
            self.assertEqual(pattern_gate(root), 1)
            proof["execution_witnesses"][atom] = [*witnesses[atom], "foreign-case"]
            (root / "coverage-validation.json").write_text(json.dumps(proof))
            self.assertEqual(pattern_gate(root), 1)

    def test_shell_rejects_scope_expansion(self):
        result = subprocess.run(
            [str(ROOT / "scripts/coverage-patterns-verify.sh"), "--operators", "Neg"], capture_output=True
        )
        self.assertEqual(result.returncode, 2)

    def test_shell_has_preflight_and_exact_four_operator_scope(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            capture = root / "arguments"
            preflight = root / "preflight"
            preflight.write_text('#!/bin/sh\nprintf "preflight\\n" >> "$CAPTURE"\n')
            preflight.chmod(0o755)
            shell = root / "bash"
            shell.write_text('#!/bin/sh\nprintf "%s\\n" "$TRITON_EVAL_REQUIRE_TARGET_PATTERNS" "$@" >> "$CAPTURE"\n')
            shell.chmod(0o755)
            environment = dict(
                os.environ,
                PATH=f"{root}:{os.environ['PATH']}",
                CAPTURE=str(capture),
                TRITON_EVAL_PREFLIGHT_PYTHON=str(preflight),
            )
            result = subprocess.run(
                [str(ROOT / "scripts/coverage-patterns-verify.sh")],
                env=environment,
                capture_output=True,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            arguments = capture.read_text().splitlines()
            self.assertEqual(arguments[:2], ["preflight", "1"])
            self.assertIn("ReduceMax,Cummin,Matmul,GroupedMatmul", arguments)
            self.assertNotIn("Neg", arguments)


if np is not None:

    class Tensor(np.ndarray):
        @property
        def device(self):
            return "cpu"

        def to(self, dtype):
            return self.astype(dtype).view(Tensor)

        def numel(self):
            return self.size

        def is_contiguous(self):
            return self.flags.c_contiguous

    class Pointer:
        def __init__(self, tensor, offset=0):
            self.tensor, self.offset = tensor, offset

        def __add__(self, offset):
            return Pointer(self.tensor, self.offset + offset)


def _simulation_tensor(value):
    """Represent simulated kernel vectors as NumPy tensor views."""
    return np.asarray(value).view(Tensor)


def _simulation_load(pointer, valid=True, other=0):
    """Load only valid simulated vector addresses."""
    offsets = np.asarray(pointer.offset)
    valid = np.broadcast_to(valid, offsets.shape)
    result = np.full(offsets.shape, other, dtype=pointer.tensor.dtype)
    result[valid] = pointer.tensor.reshape(-1)[offsets[valid]]
    return _simulation_tensor(result)


def _simulation_store(pointer, value, valid=True):
    """Store simulated vectors without touching masked addresses."""
    offsets = np.asarray(pointer.offset)
    valid = np.broadcast_to(valid, offsets.shape)
    values = np.broadcast_to(value, offsets.shape)
    pointer.tensor.reshape(-1)[offsets[valid]] = values[valid]


def _simulation_scan(values, axis, combine):
    """Apply the associative scan used by the actual pattern source."""
    result = [_simulation_tensor(value.copy()) for value in values]
    distance = 1
    while distance < result[0].shape[axis]:
        left = [value[:-distance].copy() for value in result]
        right = [value[distance:].copy() for value in result]
        combined = combine(*left, *right)
        for index, output in enumerate(result):
            output[distance:] = combined[index]
        distance *= 2
    return tuple(result)


def _simulation_kernel(coordinates):
    """Build a launch recorder with JIT-like tuple constant boundaries."""
    class Kernel:
        """Execute only numerical simulation, never generate compiler evidence."""

        def __init__(self, function):
            self.function = function

        def __call__(self, *arguments):
            return self.function(*constant_arguments(self.function, arguments))

        def __getitem__(self, grid):
            def execute(*arguments, **kwargs):
                arguments = [Pointer(value) if isinstance(value, np.ndarray) else value for value in arguments]
                for point in itertools.product(*(range(size) for size in grid)):
                    coordinates[:] = [*point, *([0] * (3 - len(point)))]
                    self(*arguments)

            return execute

    return Kernel


def _simulation_language(triton, coordinates):
    """Provide the small numerical operation set used by pattern kernels."""
    return SimpleNamespace(
        constexpr=object(),
        float32=np.float32,
        int32=np.int32,
        int64=np.int64,
        cdiv=triton.cdiv,
        program_id=lambda axis: coordinates[axis],
        arange=lambda first, last: _simulation_tensor(np.arange(first, last)),
        load=_simulation_load,
        store=_simulation_store,
        associative_scan=_simulation_scan,
        where=lambda *args: _simulation_tensor(np.where(*args)),
        minimum=lambda first, second: _simulation_tensor(np.minimum(first, second)),
        min=lambda value, axis: _simulation_tensor(np.min(value, axis=axis)),
        max=lambda value, axis: _simulation_tensor(np.max(value, axis=axis)),
        sum=lambda value, axis: _simulation_tensor(np.sum(value, axis=axis)),
        dot=lambda first, second, accumulator=None: _simulation_tensor(
            first.astype(np.float32) @ second.astype(np.float32) + (0 if accumulator is None else accumulator)
        ),
        zeros=lambda shape, dtype: _simulation_tensor(np.zeros(shape, dtype=dtype)),
    )


def cpu_namespace():
    """Load unchanged kernel source into a CPU-only numerical simulation."""
    coordinates = [0, 0, 0]
    triton = SimpleNamespace(
        jit=_simulation_kernel(coordinates),
        cdiv=lambda first, second: (first + second - 1) // second,
        next_power_of_2=lambda value: 1 << (value - 1).bit_length(),
    )
    language = _simulation_language(triton, coordinates)
    torch = SimpleNamespace(
        float32=np.float32, int64=np.int64,
        empty=lambda shape, dtype, device: _simulation_tensor(np.empty(shape, dtype=dtype))
    )
    tree = ast.parse(submission_source())
    tree.body = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    namespace = dict(torch=torch, triton=triton, tl=language, math=math)
    exec(compile(tree, "<actual-pattern-kernel-source>", "exec"), namespace)
    return namespace, _simulation_tensor


@unittest.skipIf(np is None, "NumPy required for source simulation")
class PatternKernelTests(unittest.TestCase):
    def test_two_stage_max_handles_partial_block_and_keepdim(self):
        kernels, tensor = cpu_namespace()
        values = tensor(np.random.default_rng(17).normal(size=(3, 2051)).astype(np.float32))
        for keepdim in (False, True):
            result = kernels["reduce_max"](values, -1, keepdim)
            np.testing.assert_array_equal(result, np.max(values, axis=1, keepdims=keepdim))

    def test_split_k_actual_source_matches_independent_matmul(self):
        kernels, tensor = cpu_namespace()
        generator = np.random.default_rng(19)
        source = tensor(generator.normal(size=(16, 65536)).astype(np.float16) / 4)
        weight = tensor(generator.integers(-4, 5, size=(65536, 19)).astype(np.float16) / 8)
        result = kernels["matmul"](source, weight)
        expected = (source.astype(np.float64) @ weight.astype(np.float64)).astype(np.float16)
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=2e-3)

    def test_blocked_scan_actual_source_crosses_blocks_and_resolves_latest_ties(self):
        kernels, tensor = cpu_namespace()
        values = tensor(np.random.default_rng(23).normal(size=(1, 2**20 + 257)).astype(np.float32))
        values[:, [511, 512, 1023, 1024, values.shape[1] - 1]] = -20
        actual, indices = kernels["cummin"](values, -1)
        expected = np.minimum.accumulate(values, axis=1)
        positions = np.arange(values.shape[1])[None, :]
        expected_indices = np.maximum.accumulate(np.where(values == expected, positions, -1), axis=1)
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(indices, expected_indices)

    def test_bias_tile_source_handles_multiple_groups_and_tail_rows(self):
        kernels, tensor = cpu_namespace()
        generator = np.random.default_rng(29)
        source = tensor(generator.normal(size=(2064, 33)).astype(np.float16))
        weight = tensor(generator.normal(size=(2, 33, 19)).astype(np.float16))
        bias = tensor(generator.normal(size=(2, 19)).astype(np.float16))
        result = kernels["grouped_matmul"](source, weight, bias, [1023, 1041])
        expected = np.concatenate(
            [
                (source[:1023].astype(np.float64) @ weight[0].astype(np.float64) + bias[0]).astype(np.float16),
                (source[1023:].astype(np.float64) @ weight[1].astype(np.float64) + bias[1]).astype(np.float16),
            ]
        )
        np.testing.assert_allclose(result, expected, rtol=1e-3, atol=2e-3)

    def test_small_shape_fallbacks_keep_original_matmul_and_cummin_semantics(self):
        kernels, tensor = cpu_namespace()
        generator = np.random.default_rng(31)
        source = tensor(generator.normal(size=(7, 33)).astype(np.float16))
        weight = tensor(generator.normal(size=(33, 19)).astype(np.float16))
        np.testing.assert_allclose(
            kernels["matmul"](source, weight),
            (source.astype(np.float64) @ weight.astype(np.float64)).astype(np.float16),
            rtol=1e-3,
            atol=2e-3,
        )
        source = tensor(generator.normal(size=(3, 5, 17)).astype(np.float32))
        for axis in (0, 1, 2, -1):
            values, indices = kernels["cummin"](source, axis)
            expected = np.minimum.accumulate(source, axis=axis)
            shape = [1] * source.ndim
            shape[axis] = source.shape[axis]
            positions = np.arange(source.shape[axis]).reshape(shape)
            expected_indices = np.maximum.accumulate(np.where(source == expected, positions, -1), axis=axis)
            np.testing.assert_array_equal(values, expected)
            np.testing.assert_array_equal(indices, expected_indices)


if __name__ == "__main__":
    unittest.main()
