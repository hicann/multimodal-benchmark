# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Recognize explicit Ascend on-chip memory dataflow, never generic memref copies."""

from __future__ import annotations

import math
import re


VALUE = r"%[\w.$#-]+"


def _live_fixpipe_output(proven, output, node, index):
    if not (all(proven)):
        return False
    if output["operation"] != "hivm.hir.fixpipe":
        return False
    if output["sources"] != node["destinations"]:
        return False
    return not (index not in output["parents"])


def _input_copy_to_ub(node, flow):
    if node["sources"] != ["X"]:
        return False
    if len(node["destinations"]) != 1:
        return False
    if flow.spaces.get(node["destinations"][0]) != "ub":
        return False
    return bool(flow.widths.get(node["destinations"][0], 0) >= 2)


def _unsupported_loop_bounds(self, bounds):
    if (self.regions):
        return True
    if (self.loop_count):
        return True
    if (any(type(value) is not int for value in bounds)):
        return True
    if (bounds[2] <= 0):
        return True
    return bool(bounds[1] <= bounds[0])


def _invalid_parameter_contract(parameters, names, kinds):
    if (not parameters):
        return True
    if (len(parameters) != len(names)):
        return True
    if (len(set(parameters)) != len(parameters)):
        return True
    return bool(kinds != [int(name == "Out") for name in names])


def _split_arguments(text):
    depth, start = 0, 0
    result = []
    for index, character in enumerate(text):
        if character in "(<[{":
            depth += 1
        elif character in ")>]}":
            depth -= 1
        elif character == "," and depth == 0:
            result.append(text[start:index])
            start = index + 1
    return [*result, text[start:]]


def _parameters(text, launch):
    header = re.search(
        r"func\.func\s+(?:public\s+)?@" + re.escape(launch["name"]) + r"\((.*)\)\s*(?:attributes\s*\{.*\}\s*)?\{", text
    )
    if not header:
        return {}
    parameters, kinds = [], []
    for argument in _split_arguments(re.sub(r'"(?:\\.|[^"\\])*"', '""', header[1])):
        name = re.match(r"\s*(" + VALUE + r")\s*:", argument)
        if name and re.search(r"\btt\.tensor_kind\s*=\s*[01]\b", argument):
            parameters.append(name[1])
            kinds.append(int(re.search(r"\btt\.tensor_kind\s*=\s*([01])\b", argument)[1]))
    names = [
        name
        for name in launch.get("runtime_arg_names", [])
        if launch.get("arguments", {}).get(name, {}).get("kind") == "tensor"
    ]
    if (
        _invalid_parameter_contract(parameters, names, kinds)
    ):
        return {}
    return dict(zip(parameters, names))


def _function_body(text, name):
    header = re.search(
        r"func\.func\s+(?:public\s+)?@" + re.escape(name) + r"\(.*\)\s*(?:attributes\s*\{.*\}\s*)?\{", text
    )
    if not header:
        return ""
    masked = re.sub(r'"(?:\\.|[^"\\])*"', lambda match: " " * len(match[0]), text)
    depth = 0
    for index in range(header.end() - 1, len(masked)):
        if masked[index] == "{":
            depth += 1
        elif masked[index] == "}":
            depth -= 1
            if depth == 0:
                return text[header.start(): index + 1]
    return ""


class MemoryFlow:
    def __init__(self, text, launch):
        text = _function_body(text, launch["name"])
        self.aliases = _parameters(text, launch)
        self.spaces, self.widths, self.latest, self.nodes = {}, {}, {}, []
        for root in self.aliases.values():
            self.spaces[root] = "gm"
        self._parse(text)

    def root(self, value):
        return self.aliases.get(value, value)

    def write(self, operation, incoming, outgoing, line):
        sources, destinations = list(map(self.root, incoming)), list(map(self.root, outgoing))
        operands = sources + (destinations if operation in ("hivm.hir.mmad", "linalg.matmul") else [])
        parents = {self.latest[root] for root in operands if root in self.latest}
        index = len(self.nodes)
        self.nodes.append(
            dict(operation=operation, sources=sources, destinations=destinations, parents=parents, line=line)
        )
        for root in destinations:
            self.latest[root] = index

    def ancestors(self, index):
        seen, pending = set(), [index]
        while pending:
            current = pending.pop()
            if current not in seen:
                seen.add(current)
                pending.extend(self.nodes[current]["parents"])
        return seen

    def live(self):
        return self.ancestors(self.latest["Out"]) if self.supported and "Out" in self.latest else set()

    def _parse(self, text):
        self.supported = bool(text) and not re.search(r"\b(?:scf\.(?:if|while)|cf\.[\w.]+)\b", text)
        for line in text.splitlines():
            self._consume_line(line)

    def _consume_line(self, line):
        definition = re.match(r"\s*(" + VALUE + r")\s*=\s*([\w.]+)\s*(.*)", line)
        if definition:
            value, operation, operands = definition.groups()
            references = re.findall(VALUE, operands.split(" : ")[0])
            if operation.startswith("memref.") and operation in (
                "memref.subview",
                "memref.reinterpret_cast",
                "memref.cast",
                "memref.collapse_shape",
                "memref.expand_shape",
            ):
                if references:
                    self.aliases[value] = self.root(references[0])
            elif operation in ("memref.alloc", "memref.alloca"):
                self.aliases[value] = value
                space = re.search(r"#hivm\.(?:address_space|addr_space)<(ub|l1|l0a|l0b|l0c)>", line)
                if space:
                    self.spaces[value] = space[1]
                shape = re.search(r"memref<((?:\d+x)+)", line)
                if shape:
                    self.widths[value] = math.prod(map(int, shape[1].rstrip("x").split("x")))
            if operation.startswith("memref."):
                return
        copy = re.match(r"\s*memref\.copy\s+(" + VALUE + r"),\s*(" + VALUE + r")", line)
        if copy:
            self.write("memref.copy", [copy[1]], [copy[2]], line)
            return
        operation = re.match(r"\s*(?:" + VALUE + r"\s*=\s*)?(hivm\.hir\.[\w]+|linalg\.[\w]+)\b", line)
        incoming = re.search(r"\bins\((.*?)\)", line)
        outgoing = re.search(r"\bouts\((.*?)\)", line)
        if operation and incoming and outgoing:
            self.write(
                operation[1],
                re.findall(VALUE, incoming[1].split(" : ")[0]),
                re.findall(VALUE, outgoing[1].split(" : ")[0]),
                line,
            )


class PhysicalMemoryFlow(MemoryFlow):
    @staticmethod
    def _has_multibuffer_annotation(value, addresses, text):
        if len(addresses) > 1:
            annotation = (
                r"(?m)^\s*annotation\.mark\s+"
                + re.escape(value)
                + r"\s*\{hivm\.multi_buffer = "
                + str(len(addresses))
                + r" : i32\}"
            )
            if not re.search(annotation, text):
                return False
        return True

    def consume(self, line, text):
        if self._consume_region(line):
            return
        conditional = any(region[0] == "if" for region in self.regions)
        definition = re.fullmatch(r"(" + VALUE + r")\s*=\s*([\w.]+)\s*(.*)", line)
        if definition:
            value, operation, operands = definition.groups()
            self._consume_definition(value, operation, operands, conditional, text)
            return
        if re.fullmatch(r"hivm\.hir\.(?:set_flag|wait_flag|pipe_barrier)\[.*\]", line):
            return
        if not conditional and (line == "hivm.hir.set_mask_norm" or line.startswith("annotation.mark ")):
            return
        self._consume_memory_operation(line, conditional)

    def pointer(self, value, operands, text):
        pointer = re.fullmatch(
            r"\(([^)]*)\)\s*:\s*memref<((?:\d+x)+)(f16|bf16|f32), "
            r"#hivm\.address_space<(ub|cbuf|cc)>>",
            operands,
        )
        if not pointer:
            self.supported = False
            return
        arguments = [argument.strip() for argument in pointer[1].split(",")]
        addresses = tuple(self.constants.get(argument) for argument in arguments)
        if not addresses or any(type(address) is not int or address < 0 for address in addresses):
            self.supported = False
            return
        if not self._has_multibuffer_annotation(value, addresses, text):
            self.supported = False
            return
        width = math.prod(map(int, pointer[2].rstrip("x").split("x")))
        dtype, space = pointer[3], pointer[4]
        size = width * (4 if dtype == "f32" else 2)
        if width < 2:
            self.supported = False
            return
        spans = [(address, address + size) for address in addresses]
        if any(
            left[0] < right[1] and right[0] < left[1]
            for index, left in enumerate(spans)
            for right in spans[index + 1:]
        ):
            self.supported = False
            return
        descriptor = (space, addresses, size, dtype)
        for other, root in self.buffers.items():
            if descriptor == other:
                self.aliases[value] = root
                return
            if space == other[0] and any(
                start < offset + other[2] and offset < end for start, end in spans for offset in other[1]
            ):
                self.supported = False
                return
        self.buffers[descriptor] = value
        self.aliases[value] = value
        self.spaces[value], self.widths[value], self.dtypes[value] = space, width, dtype

    def _parse(self, text):
        self.constants, self.regions, self.buffers, self.dtypes = {}, [], {}, {}
        lines = text.splitlines()
        self.supported = bool(self.aliases) and len(lines) >= 3 and lines[-2].strip() == "return"
        self.loop_count = 0
        for line in lines[1:-2]:
            if not self.supported:
                break
            self.consume(line.strip(), text)
        self.supported = self.supported and not self.regions

    def _consume_definition(self, value, operation, operands, conditional, text):
        references = re.findall(VALUE, operands.split(" : ")[0])
        if operation == "hivm.hir.pointer_cast" and not conditional:
            self.pointer(value, operands, text)
            return
        if operation in (
            "memref.subview",
            "memref.reinterpret_cast",
            "memref.cast",
            "memref.collapse_shape",
            "memref.expand_shape",
        ):
            self._consume_view(value, operation, operands, references)
            return
        if not conditional and (
            operation.startswith(("arith.", "affine.")) or operation == "hivm.hir.get_block_idx"
        ):
            constant = re.fullmatch(r"(-?\d+(?:\.\d+(?:e[+-]\d+)?)?)\s*:\s*(?:i\d+|index|f\d+)", operands)
            if operation == "arith.constant" and constant:
                self.constants[value] = float(constant[1]) if "." in constant[1] else int(constant[1])
            return
        self.supported = False
        return

    def _consume_view(self, value, operation, operands, references):
        if not references or references[0] not in self.aliases:
            self.supported = False
            return
        root = self.root(references[0])
        if operation == "memref.reinterpret_cast" and self.spaces[root] != "gm":
            self.supported = False
        if operation == "memref.subview":
            view = re.match(VALUE + r"\[([^]]*)\]\s*\[[^]]*\]\s*\[([^]]*)\]", operands)
            if (
                not view
                or any(offset.strip() != "0" for offset in view[1].split(","))
                or any(stride.strip() != "1" for stride in view[2].split(","))
            ):
                self.supported = False
        spaces = re.findall(r"#hivm\.address_space<(\w+)>", operands)
        if not spaces or any(space != self.spaces[root] for space in spaces):
            self.supported = False
        self.aliases[value] = root
        return

    def _consume_region(self, line):
        if not line:
            return True
        if re.fullmatch(r"}(?:\s*\{hivm\.unlikely_condition\})?", line):
            if not self.regions:
                self.supported = False
            else:
                self.regions.pop()
            return True
        conditional = any(region[0] == "if" for region in self.regions)
        loop = re.fullmatch(
            r"scf\.for\s+"
            + VALUE
            + r"\s*=\s*("
            + VALUE
            + r")\s+to\s+("
            + VALUE
            + r")\s+step\s+("
            + VALUE
            + r")\s*(?::\s*(?:i32|index))?\s*{",
            line,
        )
        if loop:
            bounds = [self.constants.get(value) for value in loop.groups()]
            if (
                _unsupported_loop_bounds(self, bounds)
            ):
                self.supported = False
                return True
            self.loop_count += 1
            self.regions.append(("for", self.loop_count))
            return True
        if re.fullmatch(r"scf\.if\s+" + VALUE + r"\s*{", line) and not conditional:
            self.regions.append(("if", None))
            return True
        return False

    def _consume_memory_operation(self, line, conditional):
        operation = re.match(r"(hivm\.hir\.[\w]+)\b", line)
        incoming, outgoing = re.search(r"\bins\((.*?)\)", line), re.search(r"\bouts\((.*?)\)", line)
        if not operation or not incoming or not outgoing:
            self.supported = False
            return
        sources = re.findall(VALUE, incoming[1].split(" : ")[0])
        destinations = re.findall(VALUE, outgoing[1].split(" : ")[0])
        roots, targets = list(map(self.root, sources)), list(map(self.root, destinations))
        name = operation[1]
        if len(targets) != 1 or targets[0] not in self.spaces:
            self.supported = False
            return
        source_spaces = [self.spaces.get(root) for root in roots]
        target_space = self.spaces[targets[0]]
        valid = self._valid_memory_operation(
            name, sources, roots, targets, source_spaces, target_space, conditional, line)
        if not valid:
            self.supported = False
            return
        self.write(name, sources, destinations, line)
        self.nodes[-1]["loop"] = next((region[1] for region in self.regions if region[0] == "for"), None)

    def _valid_memory_operation(self, name, sources, roots, targets, source_spaces, target_space, conditional, line):
        if name == "hivm.hir.vbrc":
            return len(sources) == 1 and self.constants.get(sources[0]) == 0 and target_space in ("ub", "cbuf")
        if conditional:
            return False
        if name in ("hivm.hir.load", "hivm.hir.nd2nz"):
            return (source_spaces == ["gm"] and roots[0] in ("X", "Weight")
                    and target_space == ("ub" if name.endswith(".load") else "cbuf"))
        if name in ("hivm.hir.vmul", "hivm.hir.vadd", "hivm.hir.vneg", "hivm.hir.mmadL1"):
            return self._valid_compute_operation(name, sources, roots, targets, source_spaces, target_space)
        if name in ("hivm.hir.store", "hivm.hir.fixpipe"):
            valid = (len(roots) == 1 and targets == ["Out"]
                     and source_spaces == [("ub" if name.endswith(".store") else "cc")])
            if name.endswith(".fixpipe"):
                valid = valid and "dma_mode = #hivm.dma_mode<nz2nd>" in line
            return valid
        return False

    def _valid_compute_operation(self, name, sources, roots, targets, source_spaces, target_space):
        valid = False
        if name in ('hivm.hir.vmul', 'hivm.hir.vadd', 'hivm.hir.vneg'):
            valid = (
                target_space == "ub"
                and "ub" in source_spaces
                and all(space == "ub" or source in self.constants for space, source in zip(source_spaces, sources))
            )
        elif name == 'hivm.hir.mmadL1':
            valid = (
                len(roots) == 6
                and source_spaces[:2] == ["cbuf", "cbuf"]
                and target_space == "cc"
                and self.dtypes.get(targets[0]) == "f32"
                and any(region[0] == "for" for region in self.regions)
            )
        return valid


def ascend_memory_witness(launch, kind):
    for stage, text in launch.get("stage_text", {}).items():
        if not stage.startswith(("compiler-pass-", "backend-")) or "hivm.hir." not in text:
            continue
        flow = PhysicalMemoryFlow(text, launch) if "hivm.hir.pointer_cast" in text else MemoryFlow(text, launch)
        result = _flow_memory_witness(flow, kind, stage)
        if result is not None:
            return result
    return None


def _flow_memory_witness(flow, kind, stage):
    for index in flow.live():
        node = flow.nodes[index]
        if kind == "vector" and node["operation"] == "hivm.hir.load":
            result = _vector_memory_witness(flow, index, node, stage)
            if result is not None:
                return result
        if kind == "tile" and node["operation"] == "hivm.hir.mmadL1":
            result = _l1_memory_witness(flow, index, node, stage)
            if result is not None:
                return result
        if not (kind == "tile" and node["operation"] == "hivm.hir.mmad" and len(node["sources"]) >= 2):
            continue
        result = _l0_memory_witness(flow, index, node, stage)
        if result is not None:
            return result
    return None


def _vector_memory_witness(flow, index, node, stage):
    if (
        _input_copy_to_ub(node, flow)
    ):
        output = flow.nodes[flow.latest["Out"]]
        if output["operation"] == "hivm.hir.store":
            return dict(
                stage=stage,
                operation=node["operation"],
                vector_width=flow.widths[node["destinations"][0]],
                memory_space="ub",
                evidence_level="explicit Ascend block transfer",
            )

    return None


def _l1_memory_witness(flow, index, node, stage):
    proven = []
    for operand, expected in zip(node["sources"][:2], ("X", "Weight")):
        writes = [
            flow.nodes[parent]
            for parent in node["parents"]
            if operand in flow.nodes[parent]["destinations"]
        ]
        proven.append(
            len(writes) == 1
            and writes[0]["operation"] == "hivm.hir.nd2nz"
            and writes[0]["sources"] == [expected]
            and writes[0]["loop"] == node["loop"]
        )
    output = flow.nodes[flow.latest["Out"]]
    if (
        _live_fixpipe_output(proven, output, node, index)
    ):
        return dict(
            stage=stage,
            operation=node["operation"],
            memory_spaces=["cbuf", "cc"],
            evidence_level="explicit Ascend L1 matrix operand staging and fixpipe output",
        )

    return None


def _l0_memory_witness(flow, index, node, stage):
    operands = node["sources"][:2]
    if [flow.spaces.get(root) for root in operands] != ["l0a", "l0b"]:
        return None
    proven = []
    for operand, expected in zip(operands, ("X", "Weight")):
        writes = [parent for parent in node["parents"] if operand in flow.nodes[parent]["destinations"]]
        ancestors = set().union(*(flow.ancestors(parent) for parent in writes)) if writes else set()
        proven.append(
            any(
                flow.nodes[parent]["operation"] == "hivm.hir.load"
                and flow.nodes[parent]["sources"] == [expected]
                and any(flow.spaces.get(root) == "l1" for root in flow.nodes[parent]["destinations"])
                for parent in ancestors
            )
        )
    if all(proven):
        if flow.nodes[flow.latest["Out"]]["operation"] != "hivm.hir.store":
            return None
        return dict(
            stage=stage,
            operation=node["operation"],
            memory_spaces=["l1", "l0a", "l0b"],
            evidence_level="explicit Ascend on-chip matrix operand staging",
        )

    return None
