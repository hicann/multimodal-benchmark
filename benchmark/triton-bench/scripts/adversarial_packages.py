#!/usr/bin/env python
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""构造一批敌意 zip 提交包并逐个过 ingest，现场打印判定。

用途：
    本地 / 昇腾机验证「zip → 结果」闭环的包级防线。
    构造逻辑与 tests/submissions/test_package.py 同源（那边管回归，这边管演示），
    但用的包是真实大小的（几 MB 的炸弹、真实提交的目录）。

用法：
    python scripts/adversarial_packages.py [--dest /tmp/triton-eval-pkgs] [--run]
    --run 时不只是构造，还对每个包执行 triton_eval.package.ingest_zip
          并打印判定（默认开启；--no-run 只构造）
"""

from __future__ import annotations

import argparse
import random
import sys
import zipfile
from pathlib import Path
import logging
from triton_eval.console import emit

from triton_eval.fixture_sources import fixture_path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

DEMO = fixture_path(PROJECT_ROOT / "examples" / "demo-submission")
ADV = fixture_path(PROJECT_ROOT / "examples" / "adversarial-submissions")

MANIFEST = (
    "submit_id: pkg-adv\n"
    "operators:\n"
    "  - name: Exp\n"
    "    entry: exp\n"
    "    path: operators/exp/kernel.py\n"
)
KERNEL = b"def exp(x, base=-1.0, scale=1.0, shift=0.0):\n    return x\n"


def demo_entries() -> dict[str, bytes]:
    return {
        str(p.relative_to(DEMO)): p.read_bytes()
        for p in sorted(DEMO.rglob("*"))
        if p.is_file() and p.suffix in {".yaml", ".yml", ".py"}
    }


def good(dest: Path) -> Path:
    p = dest / "good.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in demo_entries().items():
            zf.writestr(name, data)
    return p


def nested(dest: Path) -> Path:
    """右键压缩文件夹的形态：全部条目多一层目录。应当剥离后正常摄取。"""
    p = dest / "nested.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in demo_entries().items():
            zf.writestr(f"demo-submission-main/{name}", data)
    return p


def bomb(dest: Path) -> Path:
    """几 KB 声明解压出 4MB：压缩比 >800 倍（单文件上限 8MB 内，
    让它命中的是压缩比规则而不是文件大小规则）。
    """
    p = dest / "bomb.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.yaml", MANIFEST)
        zf.writestr("operators/exp/kernel.py", b"# pad\n" + b"\x00" * (4 * 1024 * 1024))
    return p


def escape(dest: Path) -> Path:
    """路径逃逸：试图把文件写到解包目录之外。"""
    p = dest / "escape.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.yaml", MANIFEST)
        zf.writestr("operators/exp/kernel.py", KERNEL)
        zf.writestr("../escapee.py", b"print('i escaped')\n")
    return p


def absolute(dest: Path) -> Path:
    p = dest / "absolute.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.yaml", MANIFEST)
        zf.writestr("operators/exp/kernel.py", KERNEL)
        zf.writestr("/etc/evil.py", b"x = 1\n")
    return p


def symlink(dest: Path) -> Path:
    p = dest / "symlink.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.yaml", MANIFEST)
        zf.writestr("operators/exp/kernel.py", KERNEL)
        info = zipfile.ZipInfo("operators/exp/link.py")
        info.external_attr = 0o120777 << 16  # S_IFLNK
        zf.writestr(info, "../kernel.py")
    return p


def payload(dest: Path) -> Path:
    """夹带二进制（.so）——扩展名白名单应拒收。"""
    p = dest / "payload.zip"
    rng = random.Random(42)
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.yaml", MANIFEST)
        zf.writestr("operators/exp/kernel.py", KERNEL)
        zf.writestr("operators/exp/kernel.so", bytes(rng.randrange(256) for _ in range(4096)))
    return p


def fat(dest: Path) -> Path:
    """真实大小的总量炸弹：存 9 个 8MB 的不可压 .py 文件，共 72MB > 64MB
    上限。单文件、扩展名、压缩比全部合法，只有总量检查拦得住。
    """
    p = dest / "fat.zip"
    rng = random.Random(7)
    block = bytes(rng.randrange(256) for _ in range(1024))
    with zipfile.ZipFile(p, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("manifest.yaml", MANIFEST)
        for i in range(9):
            zf.writestr(f"operators/exp/big{i}.py", block * (8 * 1024))  # 8 MiB
    return p


def pycache(dest: Path) -> Path:
    """把跑过的目录原样打包（含 __pycache__/*.pyc）。"""
    p = dest / "pycache.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        for name, data in demo_entries().items():
            zf.writestr(name, data)
        zf.writestr(
            "operators/exp/__pycache__/kernel.cpython-311.pyc",
            b"\x33\x0d\x0d\x0a" + b"\x00" * 64,
        )
    return p


def nomani(dest: Path) -> Path:
    p = dest / "nomani.zip"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("operators/exp/kernel.py", KERNEL)
    return p


def poisoned(dest: Path) -> Path:
    """包级全干净、内容是 ADV-D7（import 期投毒 torch.softmax）的整包版：
    zip 层放行；run 时运行时篡改校验（SUB-EXE-005）应当拦下——
    这是「zip → 结果」闭环里内容级防线的现场验证样本。
    """
    p = dest / "d7-oracle-poison.zip"
    manifest = (
        "submit_id: pkg-d7\n"
        "operators:\n"
        "  - name: Softmax\n"
        "    entry: softmax\n"
        "    path: operators/exp/kernel.py\n"
    )
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.yaml", manifest)
        zf.writestr(
            "operators/exp/kernel.py",
            (ADV / "ADV-D7-oracle-poison" / "operators" / "softmax" / "kernel.py").read_bytes(),
        )
    return p


BUILDERS = [
    ("good.zip", good, "应通过（正控制）"),
    ("nested.zip", nested, "应通过（剥离外层目录）"),
    ("bomb.zip", bomb, "应拒：SUB-PKG-005 压缩比异常"),
    ("escape.zip", escape, "应拒：SUB-PKG-006 路径逃逸"),
    ("absolute.zip", absolute, "应拒：SUB-PKG-006 绝对路径"),
    ("symlink.zip", symlink, "应拒：SUB-PKG-007 符号链接"),
    ("payload.zip", payload, "应拒：SUB-PKG-008 扩展名白名单"),
    ("fat.zip", fat, "应拒：SUB-PKG-003 总量超限"),
    ("pycache.zip", pycache, "应拒：SUB-PKG-008 .pyc 不是纯源码"),
    ("nomani.zip", nomani, "应拒：SUB-PKG-009 缺 manifest"),
    ("d7-oracle-poison.zip", poisoned, "包级放行（内容级防线交给 run）"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dest", default="/tmp/triton-eval-pkgs")
    ap.add_argument("--no-run", action="store_true", help="只构造不判定")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    emit(f"构造敌意提交包 → {dest}\n")

    for name, build, expect in BUILDERS:
        path = build(dest)
        size = path.stat().st_size
        emit(f"[构造] {name:<24} {size:>9,} B    预期：{expect}")

    if args.no_run:
        return 0

    emit("\n逐包判定（ingest_zip）\n" + "-" * 70)
    from triton_eval.package import PackageError, ingest_zip

    failures = 0
    for name, _, _expected in BUILDERS:
        try:
            info = ingest_zip(dest / name, dest / "ingest")
            emit(f"[通过] {name:<24} → {info.dest}（{len(info.files)} 个文件）")
        except PackageError as exc:
            emit(f"[拒绝] {name:<24} {exc}")

    # 手工断言（远程验证时肉眼对照上面各行的 rule_id 即可）
    emit("\n判定汇总已打印完毕；预期与实际不一致的行即缺陷。")
    return failures


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(message)s')
    raise SystemExit(main())
