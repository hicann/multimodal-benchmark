#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

# 独立测速窗口（真机环境每两小时自动清空，见手册 §6e）：一条命令完成
# 自举环境 → standard 档测速 → 打分 → 归档上传。验证主流程（21 项判定）
# 在 remote-verify.sh——本脚本只干测速这一件事。
#
# 为什么需要单独的窗口：默认验证流程约 1 小时，正好占满一个清空周期；
# 测速（standard 档每用例现场计时）必须另开窗口单独跑。
#
# 为什么默认跑 demo 而不是 C1：性能锚点资格要求"全对**且都有加速比**"。
# C1 是纯 Python——XL 用例单次执行实测 110-118s（120s 预算擦线过），
# standard 档测速段要在同一预算里连跑 6 次（被测 3 遍 + 参考 3 遍），
# 物理上装不下 → XL 用例永远没有加速比 → C1 永远不满足锚点资格。
# demo 的 triton 内核 XL 档毫秒级，才是锚点轮的料。
#
# 用法（环境刚清空的窗口，从家目录整条粘贴）：
#   cd ~ && { [ -d triton-eval ] || git clone https://gitcode.com/hkhkhkzxk/triton-eval.git; } \
#     && cd triton-eval && git pull --rebase --autostash origin main \
#     && bash scripts/standard-window.sh
#
# 可用环境变量：
#   OPERATORS   只测指定算子（逗号分隔，如 OPERATORS=exp,softmax）
#   SUBMISSION  覆盖被测提交路径（默认 examples/demo-submission）
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

SUBMISSION="${SUBMISSION:-examples/demo-submission}"
OPERATORS="${OPERATORS:-}"

say() { echo; echo "═══ $(date +%H:%M:%S) $* ═══"; }

# ── 1. 开跑前预检（10 秒暴露会白跑的问题）───────────
# 凭证自动配置（与 remote-verify.sh 同款）+ 代码版本核对 + dry-run 探测。
case "$(git remote get-url origin 2>/dev/null || true)" in
    *"@"*) : ;;  # 已带认证，不动
    *)
        TOK="${GITCODE_TOKEN:-2V7WsDGEFQ_Jdx3VRP7Gn8Dj}"
        if [ -n "$TOK" ]; then
            git remote set-url origin \
                "https://hkhkhkzxk:${TOK}@gitcode.com/hkhkhkzxk/triton-eval.git" \
                && echo "  ✓ 已自动配置推送凭证（remote 原为匿名地址）"
        fi
        ;;
esac
if ! git fetch origin main >/dev/null 2>&1; then
    echo "  ⚠ 连不上 gitcode——测速照常可跑，但收尾上传不了；结果请自行带走 runs/"
elif ! git merge-base --is-ancestor origin/main HEAD; then
    echo "  ✗ 本地代码落后/分叉于 gitcode main——先 git pull --rebase origin main 再跑"
    exit 1
elif ! GIT_TERMINAL_PROMPT=0 git push --dry-run origin HEAD:refs/heads/main >/dev/null 2>&1; then
    echo "  ✗ gitcode 推送凭证无效——测完的结果传不上去，现在就停。修复："
    echo "    gitcode 网页 → 个人设置 → 私人令牌（勾选项目写权限），然后"
    echo "    GITCODE_TOKEN=<新令牌> bash scripts/standard-window.sh"
    exit 1
fi

# ── 2. 自举环境（定位/创建 venv，首次约 0.5-2 分钟）───────────
say "自举评测环境"
source scripts/env-bootstrap.sh
pip install -e . -q || { echo "  ✗ pip install -e . 失败"; exit 1; }
python -c "import triton_eval" || { echo "  ✗ triton_eval 不可导入"; exit 1; }

# ── 3. standard 档测速（每用例现场计时 3 遍，demo 全量约 10-30 分钟）───────────
OP_ARGS=()
[ -n "$OPERATORS" ] && OP_ARGS=(--operators "$OPERATORS")
say "standard 档测速：${SUBMISSION}（${OPERATORS:-全部算子}）"
python -m triton_eval run --profile standard "${OP_ARGS[@]}" \
    --submission "$SUBMISSION" \
    || echo "  ✗ run 异常退出（rc=$?）——已产出的部分仍会归档"

say "打分（按算子性能分表：全对算子自锚 100 分，无分算子列原因）"
python -m triton_eval score --run latest \
    || echo "  ✗ score 失败——run 的原始结果仍在 runs/，upload 会带上"

# ── 4. 归档上传（无 verify 目录时自动降级为只归档运行产物）───────────
say "测速记录自动归档上传 gitcode"
bash scripts/upload-verify-record.sh \
    || echo "  （归档未完成——结果留在 runs/，可重试：bash scripts/upload-verify-record.sh）"
