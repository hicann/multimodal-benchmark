#!/usr/bin/env python3
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""构建独立的 v0.7 修订语料；默认参考门禁通过后才写入清单。"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
import logging

from itertools import product


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


from triton_eval.coverage import guided_selection, label_case
from triton_eval.corpusmod import build_manifest, write_manifest
from triton_eval.models import CaseSource, CaseSpec, OperatorSpec
from triton_eval.operator_registry import by_name
from triton_eval.specload import load_spec
from triton_eval.console import emit

VERSION = "v0.7-ascend-contract"
CORPUS_DIR = PROJECT_ROOT / "corpus" / VERSION

ASCEND_OP_NAMES = [
    "VectorAdd", "VectorCmp", "DiscreteMemoryAccess", "InsertSlice",
    "ExtractSlice", "AssignReqToTokenPool", "DecodeGroupedAttention",
    "CausalConv1d", "GatherScatter", "BinnedGatherScatter", "PaddedGatherScatter",
]


def _plan_vector_add():
    parameters = product(
        [512, 1024, 4096, 16384, 65536, 262144, 1048576, 4194304],
        ["float32", "int32", "int64"],
    )
    variants = []
    for length, dtype in parameters:
        variants.append(
            (
                (length,),
                dtype,
                {},
                'normal',
                True,
            )
        )
    return variants


def _plan_vector_cmp():
    parameters = product(
        [32, 64, 128, 256],
        [16, 64, 256],
        ["normal", "extreme"],
    )
    variants = []
    for columns, rows, value_range in parameters:
        variants.append(
            (
                (rows, columns),
                'float32',
                {'eps': 1e-06},
                value_range,
                True,
            )
        )
    return variants


def _plan_discrete_memory_access():
    parameters = product(
        [128, 256, 512, 1024],
        [0.5, 0.25, 0.75],
        ["float32", "int32"],
    )
    variants = []
    for length, ratio, dtype in parameters:
        variants.append(
            (
                (length,),
                dtype,
                {'m': max(1, int(length * ratio)), 'n': length},
                'normal',
                True,
            )
        )
    return variants


def _slice_variants():
    parameters = product([65, 256, 1024, 4096, 8192], [32, 64, 128, 256])
    return [((rows, dimension), "float32", {}, "normal", True) for rows, dimension in parameters]


def _routing_variants(top_k_values):
    parameters = product([256, 1024, 4096], [128, 512, 2048], [8, 32], top_k_values)
    variants = []
    for length, hidden, experts, top_k in parameters:
        variants.append(((length, hidden, experts), "float16", {"top_k": top_k}, "normal", True))
    return variants


def _plan_insert_slice():
    return _slice_variants()


def _plan_extract_slice():
    return _slice_variants()


def _plan_assign_req_to_token_pool():
    parameters = product(
        [1024, 2048, 4096, 8192],
        [4, 8, 16, 32, 64],
    )
    variants = []
    for pool_length, batch in parameters:
        variants.append(
            (
                (batch,),
                'int32',
                {'pool_len': pool_length, 'bs_upper': batch},
                'normal',
                True,
            )
        )
    return variants


def _plan_decode_grouped_attention():
    parameters = product(
        [1, 4, 8],
        [16, 32],
        [64, 128],
        [4, 8],
    )
    variants = []
    for batch, heads, dimension, max_splits in parameters:
        variants.append(
            (
                (batch, heads, dimension),
                'bfloat16',
                {'max_kv_splits': max_splits, 'sm_scale': 0.0},
                'normal',
                True,
            )
        )
    return variants


def _plan_causal_conv1d():
    parameters = product(
        [1, 8],
        [257, 1024, 4096],
        [1, 8],
        ["silu", "none"],
    )
    variants = []
    for batch, dimension, sequence_length, activation in parameters:
        variants.append(
            (
                (batch, dimension, sequence_length),
                'bfloat16',
                {'activation': activation},
                'normal',
                True,
            )
        )
    return variants


def _plan_gather_scatter():
    return _routing_variants((2, 4, 8))


def _plan_binned_gather_scatter():
    return _routing_variants((2, 4))


def _plan_padded_gather_scatter():
    return _routing_variants((2, 4, 8))


PLANS: dict[str, list[tuple]] = {
    "VectorAdd": _plan_vector_add(),
    "VectorCmp": _plan_vector_cmp(),
    "DiscreteMemoryAccess": _plan_discrete_memory_access(),
    "InsertSlice": _plan_insert_slice(),
    "ExtractSlice": _plan_extract_slice(),
    "AssignReqToTokenPool": _plan_assign_req_to_token_pool(),
    "DecodeGroupedAttention": _plan_decode_grouped_attention(),
    "CausalConv1d": _plan_causal_conv1d(),
    # gather 系列：shape = (sl, hs, ne)，与教程 generate_data 一致
    "GatherScatter": _plan_gather_scatter(),
    "BinnedGatherScatter": _plan_binned_gather_scatter(),
    "PaddedGatherScatter": _plan_padded_gather_scatter(),
}


def make_cases(op_name: str, spec: OperatorSpec) -> list[CaseSpec]:
    cases: list[CaseSpec] = []
    for i, (shape, dtype, attrs, vr, contig) in enumerate(PLANS[op_name]):
        attrs = dict(attrs)
        if op_name in ("GatherScatter", "BinnedGatherScatter", "PaddedGatherScatter"):
            tokens, hidden, experts = shape
            shape = (tokens, hidden)
            attrs["num_experts"] = experts
            if op_name == "BinnedGatherScatter":
                attrs["expert_capacity"] = tokens * attrs["top_k"] // experts
        spec.validate_case_params(dtype, list(shape), attrs)
        seed = 1000 + i
        low, high = -1.0, 1.0
        if vr == "extreme":
            low, high = -20.0, 20.0
        case = CaseSpec(
            code=f"{op_name.lower()}-{dtype}-{i:02d}",
            operator=op_name,
            source=CaseSource.VENDOR_SELFTEST,
            dtype=dtype,
            shape=list(shape),
            attrs=dict(attrs),
            value_range={"kind": vr, "low": low, "high": high},
            contiguous=contig,
            seed=seed,
            launch={},
            timeout_s=120,
            feature_atoms=[],
        )
        case.feature_atoms = label_case(case, spec)
        cases.append(case)
    return cases


def select_cases():
    candidates = []
    for name in ASCEND_OP_NAMES:
        candidates.extend(make_cases(name, by_name(name)))
    atoms = sorted({atom for case in candidates for atom in case.feature_atoms})
    return guided_selection(
        candidates, per_category=20, atom_universe=atoms,
        category_of=lambda case: case.operator)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--no-verify-refs", action="store_true")
    args = parser.parse_args()
    from triton_eval.buildlog import LOG_FILENAME, GenerationLog

    log = GenerationLog(CORPUS_DIR / LOG_FILENAME)
    log.step("build_start", version=VERSION, device=args.device,
             verify_refs=not args.no_verify_refs)
    cases, selection = select_cases()
    log.step("guided_selection", **selection)
    from triton_eval.refcheck import check_all
    from triton_eval.specload import load_feature_atoms

    if not args.no_verify_refs:
        if args.device.startswith("npu"):
            import torch_npu
        failures = check_all(cases, args.device, corpus_dir=CORPUS_DIR)
        log.step("reference_gate", device=args.device, failures=failures)
        if failures:
            emit(json.dumps(failures, ensure_ascii=False, indent=2))
            return 1
    operators = []
    for name in ASCEND_OP_NAMES:
        registered = by_name(name)
        reference = f"operators/{registered.source_name}/ref.py"
        if not (CORPUS_DIR / reference).is_file():
            raise FileNotFoundError(reference)
        operators.append(OperatorSpec(
            name=name, source=CaseSource.VENDOR_SELFTEST, weight=1.0,
            ref_path=reference,
            desc=f"{registered.category.value} / {registered.level.value}"))
    manifest = build_manifest(
        CORPUS_DIR, VERSION, load_spec().spec_version, operators, cases,
        feature_atoms_version=str(load_feature_atoms()["version"]))
    target = CORPUS_DIR / "manifest.json"
    if target.exists():
        from triton_eval.corpusmod import load_manifest
        if load_manifest(target).manifest_hash != manifest.manifest_hash:
            raise ValueError("已存在不同内容的修订语料，请使用新版本目录")
    written = write_manifest(manifest, target)
    if not args.no_verify_refs:
        from triton_eval.corpus_assets import freeze_assets

        assets = freeze_assets(CORPUS_DIR, manifest.manifest_hash)
        log.step("evaluation_assets", assets_hash=assets["assets_hash"])
    log.step("build_end", manifest_written=written, manifest_hash=manifest.manifest_hash)
    emit(json.dumps(selection, ensure_ascii=False, indent=2))
    emit(f"{VERSION}: {len(cases)} cases, hash={manifest.manifest_hash}, written={written}")
    emit("参考门禁: " + ("跳过，仅供开发" if args.no_verify_refs else args.device))
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
