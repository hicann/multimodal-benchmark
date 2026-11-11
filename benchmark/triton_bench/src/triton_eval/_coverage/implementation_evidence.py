# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Audited source + launch bindings + live compiler structure witnesses.

This is a deliberately small recognizer for reviewed implementations, not a claim
that arbitrary MLIR can be proved correct. Unrecognized lowering stays unverified.
The frozen evaluator templates are the trust root; submission declarations are
never used as certificates. Functional PASS and performance gains remain separate.
"""

from __future__ import annotations
import math
import re

ATOMS = (
    "FA-PAT-COALESCE-GATHER",
    "FA-PAT-MASK-TAIL",
    "FA-PAT-TILE-GEMM",
    "FA-PAT-VEC-LOAD",
    "FA-PROG-GRID3D",
    "FA-PROG-GRID4D",
    "FA-FUSE-INTERM-ELIDE",
)
SSA = r"%[A-Za-z_0-9.$#]+"


def _valid_softmax_reduction(obs, graph, launch, case, dim, operations, live_loads):
    if len(obs["launches"]) != 1:
        return False
    if len(graph.stores) != 1:
        return False
    if _scalar(launch, "R") != case.shape[dim]:
        return False
    if not ({"arith.select", "math.exp", "tt.reduce", "arith.divf"}.issubset(operations)):
        return False
    return set().union(*(graph.roots(d[0]) for _, d, _ in live_loads)) == {"X", "Mask"}


def _valid_matmul_tile_part_4(tile, launch, dots, shared_dot, ascend_tile, case, cols):
    if (shared_dot):
        return True
    return bool(ascend_tile)


def _valid_matmul_tile(tile, launch, dots, shared_dot, ascend_tile, case, cols):
    if tile != 16:
        return False
    if _scalar(launch, "INNER_TILE") != 32:
        return False
    if not (dots):
        return False
    if not (_valid_matmul_tile_part_4(tile, launch, dots, shared_dot, ascend_tile, case, cols)):
        return False
    if _scalar(launch, "ROWS") != case.shape[0]:
        return False
    if _scalar(launch, "INNER") != case.shape[1]:
        return False
    if _scalar(launch, "COLS") != cols:
        return False
    return launch.get("grid") == [math.ceil(case.shape[0] / 16), math.ceil(cols / 16)]


def _is_rank_four_negation(name, case):
    if name != "_witness_neg4":
        return False
    if case.operator != "Neg":
        return False
    if len(case.shape) != 4:
        return False
    return bool(case.contiguous)


def _has_vectorized_load_part_3(case, stride, vector_width, ascend_vector):
    if (vector_width):
        return True
    return bool(ascend_vector)


def _has_vectorized_load(case, stride, vector_width, ascend_vector):
    if not (case.contiguous):
        return False
    if stride != 1:
        return False
    return bool(_has_vectorized_load_part_3(case, stride, vector_width, ascend_vector))


def _has_tail_mask(count, block, mask_ops, graph, mask):
    if not (count % block):
        return False
    if "arith.cmpi" not in mask_ops:
        return False
    if "tt.make_range" not in mask_ops:
        return False
    return bool(all(len(s) > 2 and s[2] == mask for s in graph.stores))


def _invalid_vector_launch(count, case, stride, metadata, block, launch):
    if (count != math.prod(case.shape)):
        return True
    if (stride != metadata["stride"][-1]):
        return True
    if (block != 256):
        return True
    return bool(launch.get("grid") != [math.ceil(count / block)])


class IRGraph:
    def __init__(self, text, names):
        self.text, self.defs, self.stores = text, {}, []
        signature = re.search(
            r"tt\.func\s+(?:public\s+)?@[^\s(]+\((.*?)\)\s*(?:attributes\s*\{[^}]*\}\s*)?\{", text, re.S
        )
        self.parameters = {}
        if signature:
            args = re.findall(r"(" + SSA + r")\s*:", signature[1])
            if len(args) == len(names):
                self.parameters = dict(zip(args, names))
        regions = []
        depth = 0
        reduction_depth = 0
        for line in text.splitlines():
            if reduction_depth:
                reduction_depth += line.count("{") - line.count("}")
                continue
            match = re.match(r"\s*(" + SSA + r')(?:\:\d+)?\s*=\s*(?:"([\w.]+)"|([\w.]+))\s*(.*)', line)
            if match:
                operation, operands = match[2] or match[3], match[4]
                self.defs[match[1]] = (operation, re.findall(SSA, operands.split(" : ")[0]), line)
                if operation == "tt.reduce" and "({" in operands:
                    reduction_depth = self._start_reduction(line)
                    continue
                if operation == "scf.for":
                    regions.append((depth, match[1]))
                    self._bind_loop_arguments(operands, line)
            elif re.match(r"\s*tt\.store\s", line):
                self.stores.append(re.findall(SSA, line.split(" : ")[0]))
            if regions and re.match(r"\s*scf.yield\b", line):
                self.defs[regions[-1][1]][1].extend(re.findall(SSA, line.split(" : ")[0]))
            depth += line.count("{") - line.count("}")
            while regions and depth <= regions[-1][0]:
                regions.pop()
        if reduction_depth:
            raise ValueError("unterminated reduction region")

    @staticmethod
    def _start_reduction(line):
        depth = line.count("{") - line.count("}")
        if depth <= 0:
            raise ValueError("unsupported inline reduction region")
        return depth

    def ancestors(self, value, seen=None):
        seen = set() if seen is None else seen
        if value in seen:
            return seen
        seen.add(value)
        for dep in self.defs.get(value, ("", [], ""))[1]:
            self.ancestors(dep, seen)
        return seen

    def roots(self, value, seen=None):
        seen = set() if seen is None else seen
        if value in seen:
            return set()
        if value in self.parameters:
            return {self.parameters[value]}
        seen.add(value)
        op, deps, _ = self.defs.get(value, ("", [], ""))
        if op not in (
            "tt.addptr",
            "tt.splat",
            "tt.expand_dims",
            "tt.broadcast",
            "tt.trans",
            "tt.reshape",
            "ttg.convert_layout",
        ):
            return set()
        deps = deps[:1] if op == "tt.addptr" else deps
        result = set()
        for dep in deps:
            result.update(self.roots(dep, seen))
        return result

    def loads(self):
        return [(value, deps, line) for value, (op, deps, line) in self.defs.items() if op == "tt.load"]

    def operations(self, values):
        operations = set()
        for value in values:
            for ancestor in self.ancestors(value):
                if ancestor in self.defs:
                    operations.add(self.defs[ancestor][0])
        return operations

    def _bind_loop_arguments(self, operands, line):
        for arg, initial in re.findall(r"(" + SSA + r")\s*=\s*(" + SSA + ")", operands):
            self.defs[arg] = ("region.arg", [initial], line)


def _live_vector_width(stages):
    """A lowered vector memory read must feed an actual store, not dead IR."""
    for stage, text in stages.items():
        if stage in ("ttir", "compiler_stderr", "compiler_stderr_truncated"):
            continue
        graph = IRGraph(text, [])
        outputs = _llvm_store_values(text)
        live = set().union(*(graph.ancestors(v) for v in outputs)) if outputs else set()
        for value, (op, _, line) in graph.defs.items():
            if value not in live or op not in ("load", "vector.load", "llvm.load"):
                continue
            match = re.search(r"\bload\s+(?:volatile\s+)?<\s*([2-9]\d*|1\d+)\s+x\s+[^>]+>", line)
            match = match or re.search(r"\b(?:vector\.load|llvm\.load)\b[^\n]*vector<([2-9]\d*|1\d+)x", line)
            if match:
                return int(match[1])
    return None


def _llvm_store_values(text):
    outputs = []
    for line in text.splitlines():
        if re.match(r"\s*(?:store\b|(?:vector|llvm)\.store\b)", line):
            values = re.findall(SSA, line.split(" : ")[0])
            if values:
                outputs.append(values[0])
    return outputs


def _trusted(source, name):
    from .reviewed_kernels import reviewed_source

    return reviewed_source(source, name)


def _scalar(launch, name):
    entry = launch.get("arguments", {}).get(name, {})
    return entry.get("value") if entry.get("kind") == "scalar" and entry.get("constexpr") else None


def _bindings(obs, launch, inputs, output="Out"):
    data, args = obs["data"], launch.get("arguments", {})
    incoming, outgoing = data.get("input_bindings", []), data.get("output_bindings", [])
    if len(incoming) != len(inputs) or len(outgoing) != 1:
        return False
    for name, actual in [*zip(inputs, incoming), (output, outgoing[0])]:
        bound = args.get(name, {})
        if bound.get("kind") != "tensor" or _binding_fields_missing(bound, actual):
            return False
    return outgoing[0]["storage_id"] not in {x["storage_id"] for x in incoming}


def implementation_witnesses(obs):
    reason = "no matching audited implementation and live compiler structure"
    results = {atom: dict(verified=False, reason=reason) for atom in ATOMS}
    case = obs.get("case")
    if case is None:
        return results
    for launch in obs["launches"]:
        name = launch.get("name")
        prepared = _prepare_implementation_launch(obs, launch)
        if prepared is None:
            continue
        launch, graph, loads, output_values, live, live_loads = prepared

        def reject(atom, reason, *, name=name):
            if not results[atom]["verified"]:
                results[atom].update(reason=reason, kernel=name)

        def accept(atom, reason, *, name=name, launch=launch, **details):
            results[atom] = dict(
                verified=True,
                reason=reason,
                kernel=name,
                compiler_specialization_hash=launch.get("compiler_specialization_hash"),
                **details,
            )

        if name == "_witness_neg" and case.operator == "Neg":
            _negation_witness(obs, launch, case, graph, live_loads, accept, reject)
            continue
        if name == "_witness_neg3" and case.operator == "Neg" and len(case.shape) == 3:
            _grid3_witness(obs, launch, case, graph, live_loads, accept)
            continue
        if _is_rank_four_negation(name, case):
            _grid4_witness(obs, launch, case, graph, live_loads, accept)
            continue
        if name == "_gather" and case.operator == "Gather":
            _gather_witness(launch, case, graph, live, live_loads, accept)
            continue
        if name == "_matmul" and case.operator == "Matmul":
            _matmul_witness(obs, launch, case, graph, live, accept, reject)
            continue
        if name == "_witness_masked_softmax" and case.operator == "MaskedSoftmax":
            _softmax_witness(obs, launch, case, graph, output_values, live_loads, loads, accept)
    return results


def _prepare_implementation_launch(obs, launch):
    name = launch.get("name")
    if name not in (
        "_witness_neg",
        "_witness_neg3",
        "_witness_neg4",
        "_witness_masked_softmax",
        "_gather",
        "_matmul",
    ):
        return None
    source = launch.get("source_text", "")
    if not _trusted(source, name):
        return None
    from .reviewed_kernels import canonical_launch

    try:
        launch = canonical_launch(launch)
    except (ValueError, KeyError, TypeError, SyntaxError):
        return None
    # TTIR preserves program-level pointer identities. Backend/other IR are
    # checked separately, never concatenated into a fictitious SSA graph.
    text = launch.get("stage_text", {}).get("ttir", "")
    graph = IRGraph(text, launch.get("runtime_arg_names", []))
    inputs = (
        ("X", "Mask")
        if name == "_witness_masked_softmax"
        else ("X", "Index")
        if name == "_gather"
        else ("X", "Weight")
        if name == "_matmul"
        else ("Input",)
        if name == "_witness_neg3"
        else ("X",)
    )
    if not graph.parameters or not _bindings(obs, launch, inputs):
        return None
    # Reviewed kernels have no device calls, atomics or hidden global writes.
    # Unknown/multi-kernel scratch paths cannot earn elimination credit.
    if re.search(r"\b(?:tt\.call|func\.call|llvm\.call|tt\.inline_asm|tt\.atomic|memref\.alloc|tt\.extern)", text):
        return None
    values = _live_launch_values(graph, inputs)
    if values is None:
        return None
    return launch, graph, *values


def _live_launch_values(graph, inputs):
    loads = graph.loads()
    if not loads or not graph.stores or any(len(s) < 2 or graph.roots(s[0]) != {"Out"} for s in graph.stores):
        return None
    if any(not graph.roots(d[0]) or not graph.roots(d[0]).issubset(inputs) for _, d, _ in loads if d):
        return None
    output_values = [s[1] for s in graph.stores]
    live = set().union(*(graph.ancestors(v) for v in output_values))
    live_loads = [(v, d, l) for v, d, l in loads if v in live]
    if not live_loads:
        return None
    return loads, output_values, live, live_loads


def _negation_witness(obs, launch, case, graph, live_loads, accept, reject):
    n, stride, block = (_scalar(launch, k) for k in ("N", "STRIDE", "B"))
    metadata = obs["data"]["input_bindings"][0]
    if (
        _invalid_vector_launch(n, case, stride, metadata, block, launch)
    ):
        return
    if len(live_loads) != 1 or graph.roots(live_loads[0][1][0]) != {"X"}:
        return
    _, deps, _ = live_loads[0]
    # A surviving nonconstant predicate must mask BOTH load and store.
    mask = deps[1] if len(deps) > 1 else None
    mask_ops = graph.operations([mask]) if mask else set()
    if (
        _has_tail_mask(n, block, mask_ops, graph, mask)
    ):
        accept(
            "FA-PAT-MASK-TAIL",
            "live boundary predicate masks input load and final store",
            elements=n,
            block=block,
            tail_elements=n % block,
        )
    # Tensor-shaped tt.load alone does NOT demonstrate vector hardware
    # access. Require a lowered vector memory operation too.
    vector_width = _live_vector_width(launch.get("stage_text", {}))
    from .ascend_lowering import ascend_memory_witness

    ascend_vector = ascend_memory_witness(launch, "vector")
    reject(
        "FA-PAT-VEC-LOAD",
        ("reviewed source and TTIR found; requires lowered vector memory load on contiguous input"),
    )
    if _has_vectorized_load(case, stride, vector_width, ascend_vector):
        accept(
            "FA-PAT-VEC-LOAD",
            "reviewed contiguous access plus lowered vector memory load",
            vector_width=vector_width or ascend_vector["vector_width"],
            physical_grid=launch["grid"],
            backend_evidence=ascend_vector,
        )


def _grid3_witness(obs, launch, case, graph, live_loads, accept):
    dims = [_scalar(launch, "D" + str(axis)) for axis in range(3)]
    block, stride = _scalar(launch, "BLOCK"), _scalar(launch, "STRIDE")
    metadata = obs["data"]["input_bindings"][0]
    if dims != case.shape or block != 256 or stride != metadata["stride"][-1]:
        return
    if metadata["stride"] != [dims[1] * dims[2] * stride, dims[2] * stride, stride]:
        return
    address_values = [deps[0] for _, deps, _ in live_loads] + [store[0] for store in graph.stores]
    address_nodes = set().union(*(graph.ancestors(value) for value in address_values))
    axes = set()
    for value in address_nodes:
        if not (value in graph.defs and graph.defs[value][0] == "tt.get_program_id"):
            continue
        axis = re.search(r"\b(?:x|y|z)\b", graph.defs[value][2])
        if axis:
            axes.add(axis[0])
    if launch.get("grid") == [math.ceil(dims[2] / block), dims[1], dims[0]] and axes == {"x", "y", "z"}:
        accept(
            "FA-PROG-GRID3D",
            "three physical grid axes feed live three-dimensional input and output addresses",
            logical_shape=dims,
            physical_grid=launch["grid"],
            linear_stride=stride,
        )


def _grid4_witness(obs, launch, case, graph, live_loads, accept):
    dims = [_scalar(launch, "D" + str(i)) for i in range(4)]
    block = _scalar(launch, "B")
    if dims != case.shape or block != 256:
        return
    expected = [math.ceil(dims[3] / block), dims[2], dims[0] * dims[1]]
    address_values = [d[0] for _, d, _ in live_loads] + [s[0] for s in graph.stores]
    address_nodes = set().union(*(graph.ancestors(v) for v in address_values))
    axes = set()
    for v in address_nodes:
        if not (v in graph.defs and graph.defs[v][0] == "tt.get_program_id"):
            continue
        axis = re.search(r"\b(?:x|y|z)\b", graph.defs[v][2])
        if axis:
            axes.add(axis[0])
    if launch.get("grid") == expected and axes == {"x", "y", "z"}:
        accept(
            "FA-PROG-GRID4D",
            "reviewed four-dimensional bijection bound to actual shape and live address calculation",
            logical_shape=dims,
            physical_grid=expected,
            mapping="z=i0*D1+i1; y=i2; x tiles i3",
            native_grid_rank=3,
        )


def _gather_witness(launch, case, graph, live, live_loads, accept):
    if _scalar(launch, "COLS") != case.shape[-1] or launch.get("grid") != [case.shape[0]]:
        return
    for value, (op, deps, _) in graph.defs.items():
        if not (op == "tt.gather" and value in live and len(deps) >= 2):
            continue
        table = _gather_input_addresses(live_loads, graph, deps[0], "X")
        index = _gather_input_addresses(live_loads, graph, deps[1], "Index")
        if table and index and "tt.make_range" in graph.operations([d[0] for d in table]):
            accept(
                "FA-PAT-COALESCE-GATHER",
                "contiguous row prefetch feeds live in-tile gather with actual index input",
                performance_gain="NOT_ESTABLISHED",
            )


def _matmul_witness(obs, launch, case, graph, live, accept, reject):
    tile = _scalar(launch, "TILE")
    # Require shared allocations to feed actual dot operands in the
    # lowered SSA graph; an unrelated/dead shared allocation is rejected.
    shared_dot = _has_shared_dot(launch)
    dots = [v for v, (op, _, _) in graph.defs.items() if op == "tt.dot" and v in live]
    from .ascend_lowering import ascend_memory_witness

    ascend_tile = ascend_memory_witness(launch, "tile")
    reject(
        "FA-PAT-TILE-GEMM",
        ("reviewed tiled dot found; requires live shared-memory lowering for both dot operands"),
    )
    weight = obs["data"]["input_bindings"][1]
    cols = weight["shape"][-1]
    if (
        _valid_matmul_tile(tile, launch, dots, shared_dot, ascend_tile, case, cols)
    ):
        accept(
            "FA-PAT-TILE-GEMM",
            "reviewed tiled dot with explicit shared-memory lowering",
            tile=tile,
            inner_tile=32,
            performance_gain="NOT_ESTABLISHED",
            backend_evidence=ascend_tile,
        )


def _softmax_witness(obs, launch, case, graph, output_values, live_loads, loads, accept):
    dim = case.attrs["dim"] % len(case.shape)
    operations = graph.operations(output_values)
    if (
        _valid_softmax_reduction(obs, graph, launch, case, dim, operations, live_loads)
    ):
        accept(
            "FA-FUSE-INTERM-ELIDE",
            (
                "masked intermediate feeds normalization in SSA; all global stores target returned output; no "
                "scratch pointer or second launch"
            ),
            global_load_roots=sorted(set().union(*(graph.roots(d[0]) for _, d, _ in loads))),
            global_store_roots=["Out"],
            launches=1,
        )


def _has_shared_dot(launch):
    shared_dot = False
    for stage, lowered in launch.get("stage_text", {}).items():
        if stage in ("ttir", "compiler_stderr", "compiler_stderr_truncated"):
            continue
        lg = IRGraph(lowered, launch.get("runtime_arg_names", []))
        shared_values = _shared_allocation_values(lg)
        lowered_live = set().union(*(lg.ancestors(s[1]) for s in lg.stores if len(s) > 1))
        for value, (op, deps, _) in lg.defs.items():
            if op == "tt.dot" and value in lowered_live and len(deps) >= 2:
                shared_dot |= all(bool(lg.ancestors(d) & shared_values) for d in deps[:2])
    return shared_dot


def _binding_fields_missing(bound, actual):
    for field in ("tensor_id", "storage_id", "shape", "stride", "dtype"):
        if bound.get(field) != actual.get(field) or field not in actual:
            return True
    return False


def _gather_input_addresses(live_loads, graph, operand, root):
    addresses = []
    for value, dependencies, _ in live_loads:
        if value in graph.ancestors(operand) and graph.roots(dependencies[0]) == {root}:
            addresses.append(dependencies)
    return addresses


def _shared_allocation_values(graph):
    values = set()
    for value, (operation, _, line) in graph.defs.items():
        if operation == "ttg.local_alloc" and re.search(r"!ttg\.memdesc<[^\n]*#\w*shared", line):
            values.add(value)
    return values
