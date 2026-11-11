# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Fail-closed, cross-launch compiled-dataflow witnesses for explicit patterns."""

from __future__ import annotations

from collections import namedtuple
import math
import re

from .compilation_evidence import complete_module
from .implementation_evidence import IRGraph, SSA
from .pattern_generation import TARGETS, target_features

TENSOR_KEYS = ("tensor_id", "storage_id", "shape", "stride", "dtype")


def _valid_scan_combine(names, calculated, body):
    if len(names) != 2:
        return False
    if not (calculated):
        return False
    if {calculated[3], calculated[4]} != set(names):
        return False
    return bool(re.search(r"tt.reduce.return\s+" + re.escape(calculated[1]) + r"\s*:", body))


class PatternGraph(IRGraph):
    def __init__(self, text, names):
        super().__init__(re.sub(r"\btt\.(?:associative_scan|scan)\b(?![\w.])", "tt.reduce", text), names)
        self.regions = {}
        lines = text.splitlines()
        for position, line in enumerate(lines):
            match = re.match(
                r"\s*(" +
                SSA +
                (r')(?::(\d+))?\s*=\s*(?:"(tt.reduce|tt.associative_scan|tt.scan)"'
                 + r'|(tt.reduce|tt.associative_scan|tt.scan))'
                 r'(?![\w.])'),
                line,
            )
            if not match:
                continue
            value, count = match[1], int(match[2] or 1)
            operation = match[3] or match[4]
            if operation == "tt.scan":
                operation = "tt.associative_scan"
            if operation == "tt.associative_scan" and count != 2:
                raise ValueError("scan must return both values and indices")
            if value not in self.defs or "({" not in line:
                raise ValueError("unsupported reduction/scan region spelling")
            depth = line.count("{") - line.count("}")
            region = []
            for tail in lines[position + 1:]:
                region.append(tail)
                depth += tail.count("{") - tail.count("}")
                if depth == 0:
                    break
            if depth:
                raise ValueError("incomplete reduction/scan region")
            self.regions[value] = "\n".join(region)
            _, dependencies, _ = self.defs[value]
            self.defs[value] = operation, dependencies, line
            if not (count > 1):
                continue
            for index in range(count):
                self.defs[f"{value}#{index}"] = ("region.result", [value], line)


def _tensor(launch, name):
    value = launch.get("arguments", {}).get(name, {})
    if value.get("kind") != "tensor" or any(key not in value for key in TENSOR_KEYS):
        raise ValueError("missing actual tensor binding: " + name)
    return {key: value[key] for key in TENSOR_KEYS}


def _same(first, second):
    return all(key in first and key in second and first[key] == second[key] for key in TENSOR_KEYS)


def _contiguous(tensor, shape, dtype):
    if tensor["shape"] != list(shape) or tensor["dtype"] != dtype:
        raise ValueError("intermediate/output tensor shape or dtype differs")
    stride = 1
    for size, actual in zip(reversed(shape), reversed(tensor["stride"])):
        if size > 1 and actual != stride:
            raise ValueError("noncontiguous intermediate/output")
        stride *= size
    if len(tensor["stride"]) != len(shape):
        raise ValueError("missing intermediate/output strides")


def _distinct(tensors):
    if len({tensor["storage_id"] for tensor in tensors}) != len(tensors):
        raise ValueError("input/intermediate/output storage alias")


def _scalar(launch, name):
    value = launch.get("arguments", {}).get(name, {})
    if value.get("kind") != "scalar" or value.get("constexpr") is not True:
        raise ValueError("missing bound compile-time argument: " + name)
    return value.get("value")


def _metadata(launch, grid, **scalars):
    if launch.get("grid") != grid:
        raise ValueError("actual launch grid differs from reviewed partition")
    if any(_scalar(launch, name) != value for name, value in scalars.items()):
        raise ValueError("bound dimensions/partition differ from workload")


def _graph(launch):
    name = launch.get("name")
    if not re.fullmatch(r"[0-9a-f]{64}", launch.get("compiler_specialization_hash", "")):
        raise ValueError("missing compiler specialization identity")
    from .reviewed_kernels import reviewed_source

    if not reviewed_source(launch.get("source_text", ""), name):
        raise ValueError("unreviewed pattern kernel source")
    text = launch.get("stage_text", {}).get("ttir", "")
    if (
        not complete_module(text, name)
        or len(re.findall(r"\btt.func\b", text)) != 1
        or re.search(
            (r"\b(?:tt.call|func.call|llvm.call|tt.inline_asm|tt.extern\w*|tt.atomic\w*|memref.store|memref.copy|"
             r"llvm.store|scf.if|scf.while|cf.\w+)"
             r"\b"),
            text,
        )
    ):
        raise ValueError("incomplete, foreign or unsupported compiler structure")
    graph = PatternGraph(text, launch.get("runtime_arg_names", []))
    if not graph.parameters or not graph.stores:
        raise ValueError("missing compiler input/output identities")
    if len([value for value in graph.defs.values() if value[0] == "scf.for"]) > 1:
        raise ValueError("unsupported loop structure")
    return graph


def _axes(graph, value):
    axes = set()
    for ancestor in graph.ancestors(value):
        if not (ancestor in graph.defs and graph.defs[ancestor][0] == "tt.get_program_id"):
            continue
        axis = re.search(r"\b(x|y|z)\b", graph.defs[ancestor][2])
        if axis:
            axes.add(axis[1])
    return axes


def _coefficient(graph, value, axis="z", seen=None):
    seen = set() if seen is None else set(seen)
    if value in seen:
        return None
    seen.add(value)
    operation, dependencies, line = graph.defs.get(value, ("", [], ""))
    if operation == "tt.get_program_id":
        return int(bool(re.search(r"\b" + axis + r"\b", line)))
    if axis not in _axes(graph, value):
        return 0
    if (
        operation
        in (
            "tt.splat",
            "tt.expand_dims",
            "tt.broadcast",
            "tt.reshape",
            "tt.trans",
            "ttg.convert_layout",
            "arith.extsi",
            "arith.extui",
        )
        and len(dependencies) == 1
    ):
        return _coefficient(graph, dependencies[0], axis, seen)
    if operation in ("tt.addptr", "arith.addi", "arith.subi") and len(dependencies) == 2:
        first = _coefficient(graph, dependencies[0], axis, seen)
        second = _coefficient(graph, dependencies[1], axis, seen)
        if first is not None and second is not None:
            return first - second if operation == "arith.subi" else first + second
    if operation == "arith.muli" and len(dependencies) == 2:
        for varying, constant in (dependencies, list(reversed(dependencies))):
            constant_line = graph.defs.get(constant, ("", [], ""))[2]
            match = re.search(r"arith.constant\s+(?:dense<)?(-?\d+)\b", constant_line)
            coefficient = _coefficient(graph, varying, axis, seen)
            if match and coefficient is not None:
                return coefficient * int(match[1])
    return None


def _flow(graph, output, inputs, operation=None):
    stores = [store for store in graph.stores if graph.roots(store[0]) == {output}]
    if not stores or any(len(store) < 2 for store in stores):
        raise ValueError("missing real final store")
    live = set().union(*(graph.ancestors(store[1]) for store in stores))
    loads = [(value, dependencies, line) for value, dependencies, line in graph.loads() if value in live]
    roots = set().union(*(graph.roots(dependencies[0]) for _, dependencies, _ in loads)) if loads else set()
    if roots != set(inputs) or any(not graph.roots(dependencies[0]) for _, dependencies, _ in loads):
        raise ValueError("live store does not depend on the required inputs")
    if operation and not any(graph.defs.get(value, ("", [], ""))[0] == operation for value in live):
        raise ValueError("required operation is dead or missing")
    return stores, live, loads


def _dot_inputs(graph, live):
    dots = [entry for value, entry in graph.defs.items() if value in live and entry[0] == "tt.dot"]
    if not dots:
        raise ValueError("no live matrix operation")
    for _, operands, _ in dots:
        if len(operands) < 2:
            raise ValueError("incomplete matrix operands")
        for operand, expected in zip(operands[:2], ["X", "Weight"]):
            roots = set()
            ancestors = graph.ancestors(operand)
            dependencies = (entry[1] for entry in graph.loads() if entry[0] in ancestors)
            for dependency in dependencies:
                roots.update(graph.roots(dependency[0]))
            if roots != {expected}:
                raise ValueError("matrix operands do not independently come from X and Weight")


def _stores(graph, outputs):
    if set().union(*(graph.roots(store[0]) for store in graph.stores)) != set(outputs):
        raise ValueError("unknown write or missing stage output")
    if any(len(graph.roots(store[0])) != 1 for store in graph.stores):
        raise ValueError("unknown destination pointer")


def _reduction(graph, live, operation):
    for value in live:
        body = graph.regions.get(value, "")
        arguments = re.search(r"\^bb\d+\((.*?)\):", body)
        if not arguments:
            continue
        names = re.findall("(" + SSA + r")\s*:", arguments[1])
        calculated = re.search(
            r"(" + SSA + r")\s*=\s*arith\.(" + operation + r")\s+(" + SSA + r"),\s*(" + SSA + ")", body
        )
        if (
            _valid_scan_combine(names, calculated, body)
        ):
            return True
    return False


def _scan(graph, live):
    for value in live:
        selection = _scan_selection(graph, value)
        if selection is None:
            continue
        left, left_index, right, right_index, definitions, predicate = selection

        def compare(value, operation, predicate_name, operands, *, definitions=definitions):
            entry = definitions.get(value)
            return (
                entry and entry[0] == operation and entry[1] == operands and entry[2].startswith(predicate_name + ",")
            )

        combined = definitions.get(predicate)
        if not combined or combined[0] != "ori":
            continue
        for less, tie in (combined[1], list(reversed(combined[1]))):
            tied = definitions.get(tie)
            if not tied or tied[0] != "andi" or not compare(less, "cmpf", "olt", [right, left]):
                continue
            if _scan_tie_matches(tied, compare, (right, left, right_index, left_index)):
                return True
    return False


def _scan_tie_matches(tied, compare, values):
    right, left, right_index, left_index = values
    for equal, later in (tied[1], list(reversed(tied[1]))):
        if compare(equal, "cmpf", "oeq", [right, left]) and compare(
            later, "cmpi", "sgt", [right_index, left_index]
        ):
            return True
    return False


def _bound(launch, name, actual):
    if not _same(_tensor(launch, name), actual):
        raise ValueError("stage tensor is not the observed input/output/intermediate: " + name)


def _multipass(obs, launches, graphs):
    if [launch["name"] for launch in launches] != ["_pattern_max_partial", "_pattern_max_finish"]:
        raise ValueError("not the reviewed two-stage reduction")
    first, last = launches
    rows, columns = obs["case"].shape
    chunks = math.ceil(columns / 1024)
    source, output = obs["data"]["input_bindings"][0], obs["data"]["output_bindings"][0]
    partial = _tensor(first, "Partial")
    _distinct([source, partial, output])
    _contiguous(partial, [rows, chunks], "torch.float32")
    _contiguous(output, [rows, 1] if obs["case"].attrs.get("keepdim") else [rows], source["dtype"])
    _bound(first, "X", source)
    _bound(last, "Partial", partial)
    _bound(last, "Out", output)
    _metadata(first, [chunks, rows], COLUMNS=columns, CHUNKS=chunks, BLOCK=1024)
    _metadata(last, [rows], CHUNKS=chunks, BLOCK=1 << (chunks - 1).bit_length())
    for graph, destination, incoming in ((graphs[0], "Partial", ["X"]), (graphs[1], "Out", ["Partial"])):
        _stores(graph, [destination])
        _, live, _ = _flow(graph, destination, incoming, "tt.reduce")
        if not _reduction(graph, live, "maximumf|maxnumf|maxf"):
            raise ValueError("live reduction is not a verified maximum")
    return dict(stages=2, chunks=chunks, intermediate="bound float32 partial maxima")


def _splitk(obs, launches, graphs):
    if [launch["name"] for launch in launches] != ["_pattern_split_partial", "_pattern_split_finish"]:
        raise ValueError("not the reviewed split-K and final reduction")
    first, last = launches
    source, weight = obs["data"]["input_bindings"]
    output = obs["data"]["output_bindings"][0]
    rows, inner = obs["case"].shape
    columns = weight["shape"][1]
    segment = math.ceil(math.ceil(inner / 8) / 32) * 32
    partial = _tensor(first, "Partial")
    _distinct([source, weight, partial, output])
    _contiguous(weight, [inner, columns], source["dtype"])
    _contiguous(partial, [8, rows, columns], "torch.float32")
    _contiguous(output, [rows, columns], source["dtype"])
    for name, tensor in [("X", source), ("Weight", weight)]:
        _bound(first, name, tensor)
    _bound(last, "Partial", partial)
    _bound(last, "Out", output)
    _metadata(
        first,
        [math.ceil(rows / 16), math.ceil(columns / 16), 8],
        ROWS=rows,
        COLUMNS=columns,
        INNER=inner,
        SEGMENT=segment,
        TILE=16,
        INNER_TILE=32,
    )
    _metadata(last, [math.ceil(rows * columns / 256)], ELEMENTS=rows * columns, PARTS=8, BLOCK=256)
    _stores(graphs[0], ["Partial"])
    stores, live, loads = _flow(graphs[0], "Partial", ["X", "Weight"], "tt.dot")
    _dot_inputs(graphs[0], live)
    if _invalid_split_offsets(graphs[0], loads, segment, columns):
        raise ValueError("parallel split axis does not partition both matrix inputs")
    if any(_coefficient(graphs[0], store[0]) != rows * columns for store in stores):
        raise ValueError("parallel splits do not have disjoint partial destinations")
    _stores(graphs[1], ["Out"])
    _, live, _ = _flow(graphs[1], "Out", ["Partial"], "tt.reduce")
    if not _reduction(graphs[1], live, "addf"):
        raise ValueError("partial products are not summed into actual output")
    return dict(stages=2, partitions=8, segment=segment, deterministic_merge=True)


def _blocked_scan(obs, launches, graphs):
    if [launch["name"] for launch in launches] != [
        "_pattern_scan_local",
        "_pattern_scan_carry",
        "_pattern_scan_finish",
    ]:
        raise ValueError("not the reviewed local/carry/fixup scan")
    local, carry, final = launches
    source = obs["data"]["input_bindings"][0]
    outputs = obs["data"]["output_bindings"]
    if len(outputs) != 2:
        raise ValueError("both scan values and indices must be bound")
    shape = obs["case"].shape
    axis_index = obs["case"].attrs["dim"] % len(shape)
    axis = shape[axis_index]
    inner = math.prod(shape[axis_index + 1:])
    chunks, lines = math.ceil(axis / 512), math.prod(shape) // axis
    intermediate = {}
    for launch, names in ((local, ["LocalValues", "LocalIndices", "Totals", "TotalIndices"]),
                          (carry, ["Carry", "CarryIndices"])):
        for name in names:
            intermediate[name] = _tensor(launch, name)
    _distinct([source, *outputs, *intermediate.values()])
    for name, tensor in intermediate.items():
        _contiguous(
            tensor,
            shape if name.startswith("Local") else [lines, chunks],
            "torch.int64" if name.endswith("Indices") else "torch.float32",
        )
    _contiguous(outputs[0], shape, source["dtype"])
    _contiguous(outputs[1], shape, "torch.int64")
    _bound(local, "X", source)
    for name in ["Totals", "TotalIndices"]:
        _bound(carry, name, intermediate[name])
    for name in ["LocalValues", "LocalIndices", "Carry", "CarryIndices"]:
        _bound(final, name, intermediate[name])
    _bound(final, "Values", outputs[0])
    _bound(final, "Indices", outputs[1])
    for launch in [local, final]:
        _metadata(launch, [chunks, lines], AXIS=axis, INNER=inner, CHUNKS=chunks, BLOCK=512)
    _metadata(carry, [lines], CHUNKS=chunks, BLOCK=1 << (chunks - 1).bit_length())
    _check_blocked_scan_dataflow(graphs)
    return dict(stages=3, chunks=chunks, tie_rule="last occurrence", outputs=["values", "indices"])


def _check_blocked_scan_dataflow(graphs):
    _stores(graphs[0], ["LocalValues", "LocalIndices", "Totals", "TotalIndices"])
    for destination in ["LocalValues", "LocalIndices"]:
        _, live, _ = _flow(graphs[0], destination, ["X"], "tt.associative_scan")
        if not _scan(graphs[0], live):
            raise ValueError("local scan does not prove minimum/latest-index pair semantics")
    for destination in ["Totals", "TotalIndices"]:
        _, live, _ = _flow(graphs[0], destination, ["X"], "tt.associative_scan")
        if not _scan(graphs[0], live):
            raise ValueError("block summary does not derive from the verified local scan")
    _stores(graphs[1], ["Carry", "CarryIndices"])
    for destination in ["Carry", "CarryIndices"]:
        _, live, _ = _flow(graphs[1], destination, ["Totals", "TotalIndices"], "tt.associative_scan")
        if not _scan(graphs[1], live):
            raise ValueError("block carry does not prove minimum/latest-index pair semantics")
    _stores(graphs[2], ["Values", "Indices"])
    for destination in ["Values", "Indices"]:
        _flow(graphs[2], destination, ["LocalValues", "LocalIndices", "Carry", "CarryIndices"], "arith.select")
    _scan_fixup(graphs[2])


def _scan_fixup_selections(graph):
    def unwrap(value):
        while graph.defs.get(value, ("", [], ""))[0] in ("arith.extf", "arith.truncf", "arith.extsi"):
            value = graph.defs[value][1][0]
        return value

    selections = {}
    for store in graph.stores:
        destination = next(iter(graph.roots(store[0])))
        selection = graph.defs.get(unwrap(store[1]), ("", [], ""))
        if selection[0] != "arith.select" or len(selection[1]) != 3:
            raise ValueError("scan fixup output is not a conditional pair merge")
        selections[destination] = selection[1]
    values, indices = selections.get("Values"), selections.get("Indices")
    if values is None or indices is None:
        raise ValueError("scan fixup is missing values or indices output")
    return values, indices


def _scan_fixup(graph):
    loads = {}
    for value, dependencies, _ in graph.loads():
        if len(graph.roots(dependencies[0])) == 1:
            loads[next(iter(graph.roots(dependencies[0])))] = value

    def matches_load(value, name):
        loaded = loads.get(name)
        if loaded is None:
            return False
        if value == loaded:
            return True
        operation, dependencies, line = graph.defs.get(value, ("", [], ""))
        dtype = {"Carry": "f32", "CarryIndices": "i64"}.get(name)
        return (
            dtype is not None
            and operation == "tt.splat"
            and dependencies == [loaded]
            and bool(
                re.search(
                    r"\btt\.splat\s+"
                    + re.escape(loaded)
                    + r"\s*:\s*"
                    + dtype
                    + r"\s*->\s*tensor<512x"
                    + dtype
                    + ">",
                    line,
                )
            )
        )

    values, indices = _scan_fixup_selections(graph)
    if (
        values[0] != indices[0]
        or not all(matches_load(value, name) for value, name in zip(values[1:], ["Carry", "LocalValues"]))
        or not all(matches_load(value, name) for value, name in zip(indices[1:], ["CarryIndices", "LocalIndices"]))
    ):
        raise ValueError("scan fixup does not select matching value/index pairs")
    predicate = graph.defs.get(values[0], ("", [], ""))
    if predicate[0] != "arith.ori" or len(predicate[1]) != 2:
        raise ValueError("missing scan fixup minimum/tie predicate")

    def compare(value, operation, condition, operands):
        entry = graph.defs.get(value, ("", [], ""))
        return (
            entry[0] == operation
            and len(entry[1]) == len(operands)
            and all(matches_load(actual, name) for actual, name in zip(entry[1], operands))
            and re.search(r"\b" + condition + r"\s*,", entry[2])
        )

    for less, tie in (predicate[1], list(reversed(predicate[1]))):
        conjunction = graph.defs.get(tie, ("", [], ""))
        if conjunction[0] != "arith.andi" or len(conjunction[1]) != 2:
            continue
        for equal, later in (conjunction[1], list(reversed(conjunction[1]))):
            if (
                compare(less, "arith.cmpf", "olt", ["Carry", "LocalValues"])
                and compare(equal, "arith.cmpf", "oeq", ["Carry", "LocalValues"])
                and compare(later, "arith.cmpi", "sgt", ["CarryIndices", "LocalIndices"])
            ):
                return
    raise ValueError("scan fixup does not preserve latest-index minimum semantics")


def _broadcast(obs, launches, graphs):
    source, weight, bias = obs["data"]["input_bindings"]
    output = obs["data"]["output_bindings"][0]
    groups = obs["case"].attrs["group_list"]
    inner, columns = source["shape"][1], weight["shape"][2]
    if len(launches) != len(groups) or any(launch["name"] != "_pattern_bias_tile" for launch in launches):
        raise ValueError("missing or extra group launches")
    _distinct([source, weight, bias, output])
    _contiguous(weight, [len(groups), inner, columns], source["dtype"])
    _contiguous(bias, [len(groups), columns], source["dtype"])
    _contiguous(output, [sum(groups), columns], source["dtype"])
    start = 0
    for group, (rows, launch, graph) in enumerate(zip(groups, launches, graphs)):
        if rows < 2:
            raise ValueError("no cross-row bias reuse")
        for name, tensor in [("X", source), ("Weight", weight), ("Bias", bias), ("Out", output)]:
            _bound(launch, name, tensor)
        _metadata(
            launch,
            [math.ceil(rows / 16), math.ceil(columns / 16)],
            ROWS=rows,
            START=start,
            GROUP=group,
            INNER=inner,
            COLUMNS=columns,
            TILE=16,
            INNER_TILE=16,
        )
        _stores(graph, ["Out"])
        _, live, loads = _flow(graph, "Out", ["X", "Weight", "Bias"], "tt.dot")
        _dot_inputs(graph, live)
        bias_loads = []
        for value, dependencies, line in loads:
            if graph.roots(dependencies[0]) == {"Bias"}:
                bias_loads.append((value, dependencies, line))
        if len(bias_loads) != 1 or "x" in _axes(graph, bias_loads[0][1][0]):
            raise ValueError("bias is not a single row-invariant vector load")
        value, _, line = bias_loads[0]
        if not re.search(r"tensor<16x(?:!tt.ptr<)?(?:f16|bf16|f32)", line):
            raise ValueError("bias is not the reviewed one-dimensional tile vector")
        if not _live_bias_broadcast(graph, live, value):
            raise ValueError("loaded bias is not broadcast into live multi-row output")
        start += rows
    return dict(
        tile_rows=16,
        groups=len(groups),
        evidence_level="compiled tile-local bias vector reuse",
        hardware_cache_residency="NOT_ESTABLISHED",
    )


def pattern_witnesses(obs):
    proofs = {}
    for atom in TARGETS:
        proofs[atom] = dict(verified=False, reason="no reviewed pattern workload and compiled dataflow")
    case = obs.get("case")
    if case is None:
        return proofs
    handlers = dict(zip(TARGETS, [_multipass, _blocked_scan, _splitk, _broadcast]))
    for atom in target_features(case):
        try:
            from .reviewed_kernels import canonical_launch

            launches = [canonical_launch(launch) for launch in obs["launches"]]
            graphs = [_graph(launch) for launch in launches]
            _contiguous(obs["data"]["input_bindings"][0], case.shape, "torch." + case.dtype)
            details = handlers[atom](obs, launches, graphs)
            proofs[atom] = dict(
                verified=True,
                reason="reviewed source, actual tensor chain and live compiled pattern",
                kernels=[launch["name"] for launch in launches],
                compiler_specialization_hashes=[launch.get("compiler_specialization_hash") for launch in launches],
                compiler_stage="ttir",
                performance_gain="NOT_ESTABLISHED",
                **details,
            )
        except (ValueError, KeyError, IndexError, TypeError, SyntaxError) as exc:
            proofs[atom] = dict(verified=False, reason=str(exc))
    return proofs


PairScan = namedtuple('PairScan', 'left left_index right right_index definitions predicate')


def _scan_selection(graph, value):
    if graph.defs.get(value, ("", [], ""))[0] != "tt.associative_scan":
        return None
    header_line = graph.defs[value][2]
    if not re.search(r"\baxis\s*=\s*0\s*:\s*i32\b", header_line) or not re.search(
        r"\breverse\s*=\s*false\b", header_line
    ):
        return None
    body = graph.regions.get(value, "")
    header = re.search(r"\^bb\d+\((.*?)\):", body)
    names = re.findall("(" + SSA + r")\s*:", header[1]) if header else []
    if len(names) != 4:
        return None
    left, left_index, right, right_index = names
    definitions = {}
    for match in re.finditer(r"(" + SSA + r")\s*=\s*arith\.(\w+)\s+([^\n]+)", body):
        definitions[match[1]] = match[2], re.findall(SSA, match[3].split(" : ")[0]), match[3]
    returned = re.search(r"tt.scan.return\s+(" + SSA + r"),\s*(" + SSA + ")", body)
    if not returned:
        return None
    selections = [definitions.get(returned[index]) for index in (1, 2)]
    if not all(entry and entry[0] == "select" and len(entry[1]) == 3 for entry in selections):
        return None
    if selections[0][1][1:] != [right, left] or selections[1][1][1:] != [right_index, left_index]:
        return None
    predicate = selections[0][1][0]
    if predicate != selections[1][1][0]:
        return None

    return PairScan(left, left_index, right, right_index, definitions, predicate)


def _live_bias_broadcast(graph, live, value):
    for ancestor in live:
        if (ancestor in graph.defs and graph.defs[ancestor][0] == 'tt.broadcast'
                and value in graph.ancestors(ancestor)):
            return True
    return False


def _invalid_split_offsets(graph, loads, segment, columns):
    for _, dependencies, _ in loads:
        coefficient = _coefficient(graph, dependencies[0])
        expected = segment if graph.roots(dependencies[0]) == {"X"} else segment * columns
        if coefficient != expected:
            return True
    return False
