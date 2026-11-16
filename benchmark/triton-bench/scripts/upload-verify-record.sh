#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# 真机验证记录自动归档 + 上传 gitcode。
#
# 持久环境中的手动归档入口；不内置凭据，也不改写 remote。
#
# 注意：归档内部**不得有名为 runs 的目录**（.gitignore 的 runs/ 是任意
# 深度匹配，会把归档里的证据整个忽略掉）——每轮 run 的产物直接放在
# verify-records/<记录>/run-<编号>/ 下。
#
# 用法（仓库根目录，评测 venv 已激活）：
#     bash scripts/upload-verify-record.sh             # 归档最新一轮 runs/verify-*
#     bash scripts/upload-verify-record.sh <verify目录>
#     （没有 verify 目录时降级为只归档运行产物——单独窗口跑 run 的场景）
#
# 默认由用户手动调用；remote-verify.sh 仅在显式 SKIP_ARCHIVE=0 时调用。
set -uo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/git-auth.sh"

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"
require_safe_git_remote "${TRITON_EVAL_PUSH_URL:-origin}" || exit 1

PYBIN="$(command -v python3 || command -v python)"

# ─────────── 1. 定位要归档的验证目录 ───────────
# 目录参数可省略；连 runs/verify-* 都没有时（单独窗口只跑 run 的场景，
# 如 --operators 分批测速），降级为只归档运行产物与报告。
VDIR="${1:-}"
if [ -z "$VDIR" ]; then
    VDIR="$(ls -td runs/verify-* 2>/dev/null | head -1)"
fi
if [ -n "$VDIR" ] && [ ! -d "$VDIR" ]; then
    echo "[upload] 验证目录不存在：${VDIR}（忽略，只归档运行产物）" >&2
    VDIR=""
fi

STAMP="$(basename "$VDIR" 2>/dev/null | sed 's/^verify-//')"
[ -n "$STAMP" ] || STAMP="$(date +%m%d-%H%M%S)"
HOST="$(hostname -s 2>/dev/null || hostname 2>/dev/null || echo host)"
DEST="verify-records/${HOST}-${STAMP}"

# 幂等：重跑覆盖旧归档（内容相同则 git 无变化、不会产生空提交）
rm -rf "$DEST"
echo "[upload] 归档 ${VDIR:-（仅运行产物）} → $DEST"

# 逐项判定的日志（✓/✗ 判定记录在 checks.log，由 judge() 落盘）
if [ -n "$VDIR" ]; then
    mkdir -p "$DEST/verify"
    cp -r "$VDIR"/. "$DEST/verify/"
fi

# 每轮 run 的关键产物：results.jsonl 是成绩真相源，其余是完整性证据
for run in runs/run-*/; do
    [ -d "$run" ] || continue
    rid="$(basename "$run")"
    mkdir -p "$DEST/$rid"
    for f in results.jsonl events.jsonl env.json integrity.json artifacts.json; do
        [ -f "$run/$f" ] && cp "$run/$f" "$DEST/$rid/"
    done
    [ -d "$run/logs" ] && cp -r "$run/logs" "$DEST/$rid/logs"
done

# 生成的评测报告
if [ -d runs/reports ]; then
    cp -r runs/reports "$DEST/reports"
fi

# 体积护栏：证据不该很大；超限先裁掉逐用例日志（真相源 results.jsonl 已保住）
SIZE_MB="$(du -sm "$DEST" | cut -f1)"
if [ "${SIZE_MB:-0}" -gt 30 ]; then
    echo "[upload] 归档 ${SIZE_MB} MiB 超过 30 MiB 上限，裁掉逐用例日志（原件仍在 runs/run-*/logs）" >&2
    rm -rf "$DEST"/run-*/logs
    echo "逐用例日志因体积未入库（原件在 runs/run-*/logs，需要时可另行导出）" \
        > "$DEST/logs-trimmed.txt"
fi

# ─────────── 2. 生成 SUMMARY.md（给读者的一页报告）───────────
"$PYBIN" - "$DEST" "$HOST" "$STAMP" <<'PY' || { echo "[upload] SUMMARY 生成失败" >&2; exit 1; }
import datetime
import json
import pathlib
import re
import sys

dest = pathlib.Path(sys.argv[1])
host, stamp = sys.argv[2], sys.argv[3]

lines = [f"# 真机验证记录：{host} @ {stamp}", ""]
lines.append(f"- 归档时间：{datetime.datetime.now().isoformat(timespec='seconds')}")
lines.append(f"- 主机：{host}")

# 环境指纹：取编号最大的 run 的 env.json
env = {}
for p in sorted(dest.glob("run-*/env.json")):
    try:
        env = json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
if env:
    lines.append(
        f"- 环境：{env.get('npu_model', '?')}（可见 {env.get('npu_count', '?')} 卡）、"
        f"驱动 {env.get('driver_version', '?')}、CANN {env.get('cann_version', '?')}、"
        f"torch {env.get('torch_version', '?')} / torch_npu {env.get('torch_npu_version', '?')}、"
        f"triton 实测导入 {env.get('triton_import_version', '?')}"
    )

# 全量测试结果行（step-1-pytest.log 的最后一行 passed 统计）
pytest_line = ""
log = dest / "verify" / "step-1-pytest.log"
if log.is_file():
    found = re.findall(
        r"^(\d+ passed[^\n]*)$", log.read_text(encoding="utf-8", errors="replace"), re.M
    )
    if found:
        pytest_line = found[-1].strip()
lines.append(f"- 全量测试：{pytest_line or '（见 verify/step-1-pytest.log）'}")

# 判定汇总：checks.log 由 remote-verify.sh 的 judge() 逐条落盘（[✓]/[✗] 格式）
checks = []
clog = dest / "verify" / "checks.log"
if clog.is_file():
    checks = [ln.rstrip("\n") for ln in clog.read_text(encoding="utf-8",
                                                      errors="replace").splitlines() if ln.strip()]


def _mark(line):
    s = line.strip()
    for good in ("[✓", "✓"):
        if s.startswith(good):
            return "✓"
    for bad in ("[✗", "✗"):
        if s.startswith(bad):
            return "✗"
    return None


ok = sum(1 for ln in checks if _mark(ln) == "✓")
bad = sum(1 for ln in checks if _mark(ln) == "✗")
lines.append(f"- 判定汇总：✓ {ok} 项 / ✗ {bad} 项")

lines += ["", "## 判定明细", ""]
lines += [f"- {ln}" for ln in checks] or ["（checks.log 缺失：判定记录未落盘）"]

lines += [
    "",
    "## 证据目录",
    "",
    "- `verify/`：各步日志（step-*.log）与判定记录（checks.log）",
    "- `run-<编号>/`：每轮的 results.jsonl（成绩真相源）、events.jsonl（事件协议）、"
    "env.json（环境指纹）、integrity.json（完整性档案）、artifacts.json（sha256 产物清单）"
    "与逐用例日志",
    "- `reports/`：生成的评测报告（md + html）",
    "",
    "数据库不入库（*.db）——可从 results.jsonl 用 `import` 子命令完整重建；"
    "重新生成报告无需 NPU（report 命令离线可用）。",
    "",
]
(dest / "SUMMARY.md").write_text("\n".join(lines), encoding="utf-8")
print(f"[upload] SUMMARY.md 生成：判定 ✓ {ok} / ✗ {bad}；测试行：{pytest_line or '未解析到'}")
PY

# ─────────── 3. 提交并推送 gitcode ───────────
# git 身份兜底（仓库级，不动全局）：不能只在 commit 时用 -c 临时传——
# pull --rebase 重放本地归档提交同样需要 committer 身份。2026-10-12
# 实测踩坑：远端一旦有新提交，重放即报 "Committer identity unknown"，
# 整个上传失败（归档退化为本地 tar 包）。
if [ -z "$(git config user.email)" ]; then
    git config user.name "ascend-verify"
    git config user.email "ascend-verify@triton-eval.local"
fi

# remote 推不了归档（拉代码不需要凭证、推需要）。已带认证则不动。

# 判定条数动态取自 checks.log（[✓]/[✗] 各一行）——开关步会增减条数，
# 提交信息如实写；无 checks.log（仅运行产物快照）时同样如实说明
N_CHECKS="$(grep -c '^\[' "${VDIR:-/dev/null}/checks.log" 2>/dev/null || true)"
N_CHECKS="${N_CHECKS:-0}"
# 无判定记录（单独窗口只跑 run 的快照）时，提交信息如实说明
if [ "${N_CHECKS:-0}" -gt 0 ]; then
    EVIDENCE_DESC="${N_CHECKS} 项判定证据、results.jsonl、环境指纹与报告"
else
    EVIDENCE_DESC="results.jsonl、环境指纹与报告（无判定记录的运行快照）"
fi

# git 网络操作加硬超时：git 对断死的连接默认**不超时**，会无限挂起且零
# 输出（2026-10-12 真机实测：先行归档卡在 git 里数小时、无任何提示）。
# 超时走下方已有的失败兜底（本地留档 + tar 包），稍后手动重跑本脚本
# 即可补传（幂等）。每步前打印时间戳——卡住时终端最后一行就是卡点。
gito() {  # gito <超时秒> <git 参数...>
    echo "[upload] $(date +%H:%M:%S) 开始：git ${*:2}（${1}s 超时兜底）"
    if command -v timeout >/dev/null 2>&1; then
        timeout "$1" git "${@:2}"
    else
        git "${@:2}"
    fi
}

git add verify-records
if git diff --cached --quiet; then
    echo "[upload] verify-records 无变化，无需新提交"
else
    echo "[upload] 提交 ${DEST}"
    git commit -m "verify: 真机验证记录 ${HOST}-${STAMP}（自动归档）

${EVIDENCE_DESC}，由
scripts/upload-verify-record.sh 自动归档上传。结果持久保留，可按需导出。" \
        || { echo "[upload] git commit 失败" >&2; exit 1; }
fi

# 本地开发机的 main 可能已有新提交：先 rebase 再推（--autostash 容忍
# 工作树里有未提交的杂项改动，不让归档被它们挡住）
if ! gito 300 pull --rebase --autostash origin main; then
    echo "[upload] git pull --rebase 失败/超时（网络断死，或重放需 git 身份/有冲突）；归档提交留在本地" >&2
    git rebase --abort 2>/dev/null || true
    echo "[upload] 兜底打包：runs/verify-record-${STAMP}.tar.gz" >&2
    tar czf "runs/verify-record-${STAMP}.tar.gz" "$DEST" 2>/dev/null || true
    exit 1
fi

if gito 300 push origin main; then
    echo "[upload] ✓ 已推送 gitcode：${DEST}"
else
    echo "[upload] ✗ 推送失败/超时——gitcode 需要登录凭证或连接断死。一次性配置：" >&2
    echo "  1. gitcode 网页 → 个人设置 → 私人令牌（勾选项目写权限）" >&2
    echo "  2. 通过 SSH 或 Git credential helper 配置认证；不要将令牌写入 URL。" >&2
    echo "  3. 重跑：bash scripts/upload-verify-record.sh" >&2
    echo "[upload] 归档提交已在本地；需要离线转移时，带走兜底包：" >&2
    echo "  runs/verify-record-${STAMP}.tar.gz" >&2
    tar czf "runs/verify-record-${STAMP}.tar.gz" "$DEST" 2>/dev/null || true
    exit 1
fi
