# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""zip 提交包摄取层的测试（L0，不需要 torch / NPU）。

守护什么：
    zip 是「用户上传」的真实入口形态。这一层放过的每一条敌意条目
    （路径逃逸、符号链接、炸弹、可执行文件）都会直接落进磁盘，
    后续所有内容级校验都建立在「磁盘上是干净的」这个前提上。

    这些用例全部现场构造坏包（而不是提交二进制样例进仓库）：
    构造逻辑即攻击逻辑，可读可审，且随限制常量一起演化。
"""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest
from triton_eval.fixture_sources import fixture_path

import triton_eval.package as pkg
from triton_eval.package import PackageError, ingest_zip, sha256_file

DEMO = fixture_path(Path(__file__).resolve().parents[2] / "examples" / "demo-submission")


def _demo_entries() -> dict[str, bytes]:
    """从样例提交读出应当进入包的文件（剔除 __pycache__ 等）。"""
    return {
        str(p.relative_to(DEMO)): p.read_bytes()
        for p in sorted(DEMO.rglob("*"))
        if p.is_file() and p.suffix in {".yaml", ".yml", ".py"}
    }


def _make_zip(path: Path, entries: dict[str, bytes], *, compress=zipfile.ZIP_DEFLATED,
              prefix: str = "") -> None:
    with zipfile.ZipFile(path, "w", compression=compress) as zf:
        for name, data in entries.items():
            zf.writestr(f"{prefix}/{name}" if prefix else name, data)


def _minimal_entries() -> dict[str, bytes]:
    return {
        "manifest.yaml": (
            "submit_id: ztest\n"
            "operators:\n"
            "  - name: Exp\n"
            "    entry: exp\n"
            "    path: operators/exp/kernel.py\n"
        ).encode("utf-8"),
        "operators/exp/kernel.py": b"def exp(x, base=-1.0, scale=1.0, shift=0.0):\n    return x\n",
    }


# ---------------- 好包 ----------------


def test_good_zip_ingests(tmp_path: Path) -> None:
    z = tmp_path / "good.zip"
    _make_zip(z, _demo_entries())
    info = ingest_zip(z, tmp_path / "ingest")

    assert (info.dest / "manifest.yaml").is_file()
    assert info.sha256 == sha256_file(z)
    assert "operators/exp/kernel.py" in info.files
    assert info.total_bytes == sum(len(v) for v in _demo_entries().values())
    assert info.stripped_prefix == ""


def test_ingest_is_idempotent_and_content_addressed(tmp_path: Path) -> None:
    """同一份包解两次落同一处；不同内容各得其所。"""
    z = tmp_path / "a.zip"
    _make_zip(z, _minimal_entries())
    first = ingest_zip(z, tmp_path / "ingest")
    second = ingest_zip(z, tmp_path / "ingest")
    assert first.dest == second.dest

    entries = _minimal_entries()
    entries["operators/exp/kernel.py"] += b"# different\n"
    z2 = tmp_path / "b.zip"
    _make_zip(z2, entries)
    other = ingest_zip(z2, tmp_path / "ingest")
    assert other.dest != first.dest


def test_single_outer_dir_is_stripped(tmp_path: Path) -> None:
    """右键压缩文件夹的包（全部条目多一层目录）应当自动剥离。"""
    z = tmp_path / "nested.zip"
    _make_zip(z, _demo_entries(), prefix="demo-submission-main")
    info = ingest_zip(z, tmp_path / "ingest")

    assert info.stripped_prefix == "demo-submission-main"
    assert (info.dest / "manifest.yaml").is_file()
    assert (info.dest / "operators" / "exp" / "kernel.py").is_file()


def test_mixed_layout_is_not_stripped(tmp_path: Path) -> None:
    """根级文件与顶层目录并存时不得自作主张剥层——宁可拒收。"""
    entries = _demo_entries()
    entries["README.md"] = b"hi"
    z = tmp_path / "mixed.zip"
    # 全部塞进一个顶层目录 + 根级 README：剥离后 manifest 仍在，
    # 但根级还有 README——_strip_root 要求**所有**条目共享前缀，这里不满足
    _make_zip(z, {**{f"pkg/{k}": v for k, v in entries.items()},
                  "README.md": b"hi"})
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    # 根级只有 README、没有 manifest.yaml → 拒收而非错误剥层
    assert ei.value.rule_id == "SUB-PKG-009"


# ---------------- 坏包 ----------------


def test_not_a_zip(tmp_path: Path) -> None:
    z = tmp_path / "fake.zip"
    z.write_bytes(b"this is not a zip file")
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-001"
    assert ei.value.sha256 == sha256_file(z), "坏包也要留指纹"


def test_path_escape_rejected(tmp_path: Path) -> None:
    entries = _minimal_entries()
    entries["../escapee.py"] = b"print('outside')\n"
    z = tmp_path / "escape.zip"
    _make_zip(z, entries)
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-006"
    assert not (tmp_path / "escapee.py").exists(), "逃逸文件不得落盘"


def test_absolute_path_rejected(tmp_path: Path) -> None:
    entries = _minimal_entries()
    entries["/etc/evil.py"] = b"x = 1\n"
    z = tmp_path / "abs.zip"
    _make_zip(z, entries)
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-006"


def test_backslash_name_rejected(tmp_path: Path) -> None:
    entries = _minimal_entries()
    entries["operators\\..\\evil.py"] = b"x = 1\n"
    z = tmp_path / "bs.zip"
    _make_zip(z, entries)
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-006"


def test_symlink_entry_rejected(tmp_path: Path) -> None:
    """敌意打包器可以在 zip 里携带符号链接条目。"""
    entries = _minimal_entries()
    z = tmp_path / "link.zip"
    with zipfile.ZipFile(z, "w") as zf:
        for name, data in entries.items():
            zf.writestr(name, data)
        info = zipfile.ZipInfo("operators/exp/link.py")
        info.external_attr = 0o120777 << 16  # S_IFLNK | 0777
        zf.writestr(info, "../kernel.py")
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-007"
    assert not (tmp_path / "ingest").exists() or \
        not list((tmp_path / "ingest").rglob("link.py")), "链接条目不得落盘"


def test_disallowed_extension_rejected(tmp_path: Path) -> None:
    entries = _minimal_entries()
    entries["operators/exp/kernel.so"] = b"\x7fELF" + b"\x00" * 64
    z = tmp_path / "ext.zip"
    _make_zip(z, entries)
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-008"


def test_pycache_in_package_rejected(tmp_path: Path) -> None:
    """把运行过的目录原样打包（含 __pycache__/*.pyc）必须被拒——
    .pyc 是可执行载荷，不属于纯源码提交。
    """
    entries = _demo_entries()
    entries["operators/exp/__pycache__/kernel.cpython-311.pyc"] = b"\x33\x0d\x0d\x0a" + b"\x00" * 32
    z = tmp_path / "pyc.zip"
    _make_zip(z, entries)
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-008"


def test_missing_manifest_rejected(tmp_path: Path) -> None:
    z = tmp_path / "nomani.zip"
    _make_zip(z, {"operators/exp/kernel.py": b"def exp(x, base=-1.0, scale=1.0, shift=0.0):\n    return x\n"})
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-009"


def test_operators_dir_is_not_treated_as_wrapper(tmp_path: Path) -> None:
    """operators 是契约自身的顶层目录，不得当作外层包装剥掉。"""
    entries = _minimal_entries()
    del entries["manifest.yaml"]  # 只剩 operators/ 下的文件
    z = tmp_path / "opsonly.zip"
    _make_zip(z, entries)
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-009"
    assert "operators" not in ei.value.detail, "报错不得声称剥离了 operators 层"


def test_empty_zip_rejected(tmp_path: Path) -> None:
    z = tmp_path / "empty.zip"
    with zipfile.ZipFile(z, "w"):
        pass
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-009"


# ---------------- 上限类（monkeypatch 缩小阈值，快速触发） ----------------


def test_entry_count_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pkg, "MAX_ENTRIES", 10)
    entries = _minimal_entries()
    for i in range(20):
        entries[f"operators/op{i}/kernel.py"] = b"def f(x):\n    return x\n"
    z = tmp_path / "many.zip"
    _make_zip(z, entries)
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-002"


def test_total_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pkg, "MAX_TOTAL_BYTES", 1000)
    entries = _minimal_entries()
    entries["big.py"] = b"# " + b"x" * 2000  # 存储式：单文件与压缩比都合法
    z = tmp_path / "total.zip"
    _make_zip(z, entries, compress=zipfile.ZIP_STORED)
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-003"


def test_single_file_size_limit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(pkg, "MAX_FILE_BYTES", 100)
    entries = _minimal_entries()
    entries["operators/exp/kernel.py"] += b"# " + b"y" * 500
    z = tmp_path / "big.zip"
    _make_zip(z, entries, compress=zipfile.ZIP_STORED)
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-004"


def test_ratio_bomb_rejected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """压缩比异常：几 KB 的包声明解压出数 MB（zip 炸弹特征）。"""
    monkeypatch.setattr(pkg, "MAX_RATIO", 10.0)
    entries = _minimal_entries()
    entries["operators/exp/kernel.py"] = b"# pad\n" + b"\x00" * 100_000
    z = tmp_path / "bomb.zip"
    _make_zip(z, entries)
    with pytest.raises(PackageError) as ei:
        ingest_zip(z, tmp_path / "ingest")
    assert ei.value.rule_id == "SUB-PKG-005"


# ---------------- 规则登记 ----------------


def test_pkg_rules_registered_in_validator() -> None:
    """包级规则必须在统一规则表里（validate --rules 要能列出来）。"""
    from triton_eval.validator import RULES

    for rid in pkg.RULES:
        assert rid in RULES, f"{rid} 未登记进 validator.RULES"
