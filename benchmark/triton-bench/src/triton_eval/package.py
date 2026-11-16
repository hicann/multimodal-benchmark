# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""zip 提交包的摄取层（L0，无 NPU 依赖）。

层级边界（借鉴 cann-bench 的划分：它的仓库从「已解包的 source_dir」开始，
zip 规则归网站层；本仓库同时扮演「收包方」和「运行器」两个角色，因此把
**包级规则**独立编号为 SUB-PKG-*，与**内容级规则**（SUB-STR/SUB-LOD/SUB-EXE）
分开——两层各自演进、各自测试，不互相掺杂。

为什么解包必须自写而不是 zipfile.extractall：
    extractall 会照单全收压缩包里的一切：`../../etc/x.py` 的路径逃逸、
    符号链接条目、无界解压的 zip 炸弹，全部原样落到磁盘上。
    这里在写盘**之前**逐条把关，并且用「读出字节再写文件」的方式
    落盘，磁盘上只出现检查过的东西。

内容寻址：解包目标是 `<dest>/<sha256[:12]>`，同一个包解两次落在同一处
（幂等），且目录名本身携带身份——与提交快照指纹同一条设计原则。
"""

from __future__ import annotations

import hashlib
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

# ---------------- 包级限制（常量而非配置文件：这些是安全边界，不该被运行时改） ----------------

MAX_ENTRIES = 2000                 # 条目数上限（含目录条目）
MAX_TOTAL_BYTES = 64 * 1024 * 1024  # 解压后总大小上限：64 MiB
MAX_FILE_BYTES = 8 * 1024 * 1024    # 单文件解压后上限：8 MiB
MAX_RATIO = 100.0                   # 单条目压缩比上限：防 zip 炸弹
# 扩展名白名单：提交是纯源码目录，不需要可执行、可链接、可被 import 钩住
# 的任何文件。这里的取向是宁可收窄——缺一种文档后缀的代价远小于放进一个 .so。
ALLOWED_SUFFIXES = frozenset({".py", ".yaml", ".yml", ".md", ".txt", ".json"})

RULES: dict[str, str] = {
    "SUB-PKG-001": "包级：不是有效的 zip 文件（损坏 / 加密 / 头部与内容不一致）",
    "SUB-PKG-002": f"包级：条目数超过上限（{MAX_ENTRIES}）",
    "SUB-PKG-003": f"包级：解压后总大小超过上限（{MAX_TOTAL_BYTES // (1024 * 1024)} MiB）",
    "SUB-PKG-004": f"包级：单个文件解压后超过上限（{MAX_FILE_BYTES // (1024 * 1024)} MiB）",
    "SUB-PKG-005": f"包级：压缩比异常（>{MAX_RATIO:.0f} 倍），疑似压缩炸弹",
    "SUB-PKG-006": "包级：条目路径不安全（绝对路径 / 上跳 / 反斜杠 / 盘符）",
    "SUB-PKG-007": "包级：包含符号链接或其它非常规文件条目",
    "SUB-PKG-008": "包级：扩展名不在白名单（仅允许纯源码与文档）",
    "SUB-PKG-009": "包级：解压后找不到 manifest.yaml（提交清单缺失）",
}


class PackageError(Exception):
    """一条包级规则违规。rule_id + detail，与内容级校验错误同构。"""

    def __init__(self, rule_id: str, detail: str, sha256: str = "") -> None:
        self.rule_id = rule_id
        self.detail = detail
        self.sha256 = sha256
        super().__init__(f"[{rule_id}] {RULES.get(rule_id, rule_id)}：{detail}")

    def render(self) -> str:
        return f"[{self.rule_id}] {RULES.get(self.rule_id, self.rule_id)}\n  详情：{self.detail}"


@dataclass
class IngestResult:
    """一次成功摄取的结果。"""

    dest: Path
    sha256: str
    package_name: str
    files: list[str] = field(default_factory=list)
    total_bytes: int = 0
    stripped_prefix: str = ""  # 自动剥离的外层目录名（GitHub 式下载包）

    @property
    def short_sha(self) -> str:
        return self.sha256[:12]


# ---------------- 工具 ----------------


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _check_name(name: str) -> None:
    """条目名安全性。zip 规范用 `/` 分隔，但敌意打包器什么都写得出来。"""
    # 反斜杠：要么是 Windows 造的包（POSIX 上会变成一个畸形文件名），
    # 要么是冲着「将来同步到 Windows 的仓库」来的。两种都拒收。
    if "\\" in name:
        raise PackageError("SUB-PKG-006", f"条目名含反斜杠：{name!r}")
    p = PurePosixPath(name)
    if p.is_absolute() or name.startswith("/"):
        raise PackageError("SUB-PKG-006", f"绝对路径条目：{name!r}")
    parts = p.parts
    if not parts:
        raise PackageError("SUB-PKG-006", "空路径条目")
    if any(part == ".." for part in parts):
        raise PackageError("SUB-PKG-006", f"路径上跳条目：{name!r}")
    # 盘符（C:evil.py）：zip 规范里没有盘符，出现即敌意
    if ":" in parts[0]:
        raise PackageError("SUB-PKG-006", f"条目名含盘符：{name!r}")


def _strip_root(names: list[str]) -> str:
    """求所有文件条目共享的外层目录（如 GitHub 下载包的顶层文件夹）。

    现实来源：用户在 macOS/Windows 上右键压缩文件夹，包里所有条目都
    多一层目录名。cann-bench 的 runner 同样要做「确定提交根目录」。
    只在「全部条目共享同一个顶层目录」时剥离——混合结构（根级文件
    与顶层目录并存）说明提交方自己组织过结构，不替他做主。
    """
    if not names:
        return ""
    first_parts = PurePosixPath(names[0]).parts
    if len(first_parts) < 2:
        return ""
    candidate = first_parts[0]
    # operators 是契约自身的顶层目录名，不是外层包装——
    # 把它当包装剥掉，缺 manifest 的报错就会变成「剥离后找不到」这种误导话术
    if candidate == "operators":
        return ""
    for name in names:
        parts = PurePosixPath(name).parts
        if len(parts) < 2 or parts[0] != candidate:
            return ""
    return candidate


# ---------------- 主入口 ----------------


def _check_file_info(info, sha):
    _check_name(info.filename)
    mode = (info.external_attr >> 16) & 0o170000
    if mode == 0o120000:
        raise PackageError("SUB-PKG-007", f"符号链接条目：{info.filename!r}", sha)
    if mode not in (0, 0o100000):
        raise PackageError("SUB-PKG-007", f"非常规文件条目（mode={oct(mode)}）：{info.filename!r}", sha)
    suffix = PurePosixPath(info.filename).suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise PackageError("SUB-PKG-008", f"{info.filename!r} 的扩展名 {suffix!r} 不在白名单", sha)
    if info.file_size > MAX_FILE_BYTES:
        raise PackageError("SUB-PKG-004", f"{info.filename!r} 声明 {info.file_size} 字节", sha)
    if info.compress_size > 0:
        ratio = info.file_size / info.compress_size
        if ratio > MAX_RATIO:
            raise PackageError("SUB-PKG-005", f"{info.filename!r} 压缩比 {ratio:.0f} 倍"
                               f"（{info.compress_size} → {info.file_size} 字节）", sha)


def _checked_infos(infos, sha):
    if len(infos) > MAX_ENTRIES:
        raise PackageError("SUB-PKG-002", f"共 {len(infos)} 个条目", sha)
    file_infos = [info for info in infos if not info.is_dir()]
    if not file_infos:
        raise PackageError("SUB-PKG-009", "压缩包没有任何文件条目", sha)
    for info in file_infos:
        _check_file_info(info, sha)
    total = sum(info.file_size for info in file_infos)
    if total > MAX_TOTAL_BYTES:
        raise PackageError("SUB-PKG-003", f"解压后共 {total} 字节", sha)
    return file_infos, total


def _package_names(file_infos, sha):
    prefix = _strip_root([info.filename for info in file_infos])
    has_root_manifest = any(info.filename == "manifest.yaml" for info in file_infos)
    if not has_root_manifest and prefix:
        names = [info.filename[len(prefix) + 1:] if info.filename.startswith(prefix + "/") else info.filename
                 for info in file_infos]
    else:
        prefix = ""
        names = [info.filename for info in file_infos]
    if "manifest.yaml" not in names:
        raise PackageError("SUB-PKG-009", f"未找到 manifest.yaml（剥离外层目录 {prefix!r} 后）" if prefix
                           else "未找到 manifest.yaml", sha)
    return prefix, names


def _write_files(zf, file_infos, names, dest, sha):
    dest.mkdir(parents=True, exist_ok=True)
    written = []
    written_bytes = 0
    for info, rel in zip(file_infos, names):
        with zf.open(info) as src:
            data = src.read(MAX_FILE_BYTES + 1)
        if len(data) > MAX_FILE_BYTES:
            raise PackageError("SUB-PKG-004", f"{info.filename!r} 实际解出 {len(data)} 字节", sha)
        if len(data) != info.file_size:
            raise PackageError("SUB-PKG-001", f"{info.filename!r} 声明 {info.file_size} 字节，实际 {len(data)} 字节"
                               "（头部与内容不一致）", sha)
        target = dest / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        written.append(rel)
        written_bytes += len(data)
    return written, written_bytes


def ingest_zip(zip_path: Path, dest_root: Path) -> IngestResult:
    """解包一个提交 zip 到内容寻址目录。任何规则违规抛 PackageError。

    幂等：目标目录已存在且含 manifest.yaml 时直接复用（同一 sha256
    解两次必然落在同一目录，重跑不产生副本）。
    """
    zip_path = Path(zip_path)
    if not zip_path.is_file():
        raise PackageError("SUB-PKG-001", f"提交包不存在：{zip_path}")
    sha = sha256_file(zip_path)
    dest_root = Path(dest_root)
    dest = dest_root / sha[:12]

    try:
        zf = zipfile.ZipFile(zip_path)
    except (zipfile.BadZipFile, OSError) as exc:
        raise PackageError("SUB-PKG-001", f"无法打开：{exc}", sha) from exc
    with zf:
        file_infos, total = _checked_infos(zf.infolist(), sha)

        # 幂等复用：目标目录已有同内容（目录名即指纹前缀）且结构完整
        if (dest / "manifest.yaml").is_file():
            names = [i.filename for i in file_infos]
            return IngestResult(
                dest=dest, sha256=sha, package_name=zip_path.name,
                files=sorted(names), total_bytes=total,
                stripped_prefix=_strip_root(names),
            )
        if dest.exists():
            # 同名目录却没有 manifest——多半是上次中断的残骸，清掉重来
            import shutil

            shutil.rmtree(dest)

        prefix, names = _package_names(file_infos, sha)
        written, written_bytes = _write_files(zf, file_infos, names, dest, sha)

    return IngestResult(
        dest=dest, sha256=sha, package_name=zip_path.name,
        files=sorted(written), total_bytes=written_bytes,
        stripped_prefix=prefix,
    )
