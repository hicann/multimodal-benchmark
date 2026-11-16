#!/usr/bin/env bash
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


# 昇腾机一键闭环验证（zip → 结果 全链路 + 三条关键防线 + 融合冒烟 + 探针
# 收益实证 + 产物/数据库抽查）。
#
# 用法（仓库根目录）：
#     bash scripts/remote-verify.sh
#
# 验证记录持久保存在 runs/verify-<时间戳>/，不依赖定时清空窗口。
# 可选：TRITON_EVAL_VENV 指定环境；SKIP_C1=1 跳过慢正控制；
# WITH_STANDARD=1 追加标准测速。默认不推送；需要发布时手动归档。
# SKIP_ARCHIVE=0 可显式启用旧式归档，凭据必须由外部 Git 工具管理。
set -uo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_ROOT"

STAMP="$(date +%m%d-%H%M%S)"
VDIR="runs/verify-$STAMP"
mkdir -p "$VDIR"
PASS=0; FAIL=0

judge() {  # judge <描述> <0|1>
    if [ "$2" -eq 0 ]; then echo "  ✓ $1"; PASS=$((PASS + 1));
    else echo "  ✗ $1  ——输出见 ${CURLOG}（不是致命错误，继续跑完，证据收尾自动归档）"; FAIL=$((FAIL + 1)); fi
    # 判定逐条落盘：upload-verify-record.sh 依此生成 SUMMARY——
    # 不落盘则验证记录只剩终端回显，环境一清就没了
    echo "[$([ "$2" -eq 0 ] && echo ✓ || echo ✗)] $1" >> "$VDIR/checks.log"
}

section() { echo; echo "═══ $(date +%H:%M:%S) $* ═══"; echo "[记] $(date +%H:%M:%S) $*" >> "$VDIR/checks.log"; }

# git 网络操作统一走这里：git 对断死的连接默认**不超时**，会无限挂起且
# 零输出（2026-10-12 真机实测：先行归档的 git 卡住数小时、脚本"看起来
# 死了"其实活着，终端无法定位卡点）。超时后走各处已有的失败兜底。
git_to() {  # git_to <超时秒> <git 参数...>
    if command -v timeout >/dev/null 2>&1; then
        timeout "$1" git "${@:2}"
    else
        git "${@:2}"
    fi
}

CURLOG="/dev/null"
run_step() {  # run_step <日志名> <命令...>：执行并捕获 rc，输出同步落日志。
    # 长步骤（C1/standard 全量要跑几十分钟）必须有"开始/结束"时间戳——
    # 卡住时终端最后一行就是卡点，不用猜。
    CURLOG="$VDIR/$1"
    shift
    echo "  ── $(date +%H:%M:%S) 开始：$*"
    "$@" 2>&1 | tee -a "$CURLOG"
    local rc="${PIPESTATUS[0]}"
    echo "  ── $(date +%H:%M:%S) 结束（rc=${rc}）"
    return "$rc"
}

latest_run() { ls -td runs/run-* 2>/dev/null | head -1; }

# 收尾兜底：无论脚本以哪种方式退出（正常 / Ctrl-C / 断连 / 被终止），
# 都把已有证据归档上传。checks.log 有内容才归档（开跑前的预检退出
# 没有可归档的东西）。跑完与否由 checks.log 末尾的 [完] 标记自证。
on_exit() {
    rc=$?
    trap - EXIT INT TERM HUP
    if [ "${SKIP_ARCHIVE:-1}" != "1" ] && [ -s "$VDIR/checks.log" ]; then
        echo; echo "═══ 收尾：验证记录自动归档上传（无论成败/是否被中断）═══"
        bash scripts/upload-verify-record.sh "$VDIR" \
            || echo "  （收尾归档未完成——记录已留在本地 ${VDIR}）"
    fi
    exit "$rc"
}
trap on_exit EXIT INT TERM HUP

# ─────────────────────────── 0. 环境 ───────────────────────────
section "0. 定位/建立评测环境"
source scripts/env-bootstrap.sh

pip install -e . 2>&1 | tee "$VDIR/step-0-install.log" >/dev/null
judge "pip install -e ." "${PIPESTATUS[0]}"
command -v pytest >/dev/null 2>&1 || pip install -q pytest

run_step step-0-envcheck.log python -c "import triton_eval, triton, torch_npu; print('环境 OK: triton_eval', triton_eval.__version__, '/ triton', triton.__version__)"
judge "环境自检（triton_eval + triton + torch_npu 可导入）" $?

# ─────────────────────────── 1. 全量测试 ───────────────────────────
section "1. 全量测试（含 NPU 用例；本机无 NPU 基线 509 passed / 33 skipped）"
run_step step-1-pytest.log pytest -q
judge "pytest 全量通过" $?

# ─────────────────────────── 2. 包级 + ingest ───────────────────────────
section "2. 敌意 zip 包级防线 + ingest"
run_step step-2-packages.log python scripts/adversarial_packages.py
rc=$?
grep -q "SUB-PKG-005" "$VDIR/step-2-packages.log"; a=$?
grep -q "SUB-PKG-006" "$VDIR/step-2-packages.log"; b=$?
grep -q "SUB-PKG-007" "$VDIR/step-2-packages.log"; c=$?
grep -q "SUB-PKG-003" "$VDIR/step-2-packages.log"; d=$?
grep -q "SUB-PKG-008" "$VDIR/step-2-packages.log"; e=$?
grep -q "SUB-PKG-009" "$VDIR/step-2-packages.log"; f=$?
[ $rc -eq 0 ] && [ $a -eq 0 ] && [ $b -eq 0 ] && [ $c -eq 0 ] && [ $d -eq 0 ] && [ $e -eq 0 ] && [ $f -eq 0 ]
judge "11 个敌意包全部按预期判定（005/006/007/003/008/009 各就各位）" $?

run_step step-2-ingest.log python -m triton_eval ingest --package /tmp/triton-eval-pkgs/good.zip
judge "ingest good.zip（解包 + 校验 + 落库）" $?

# ─────────────────────────── 3. zip 直进 run → score → report ───────────────────────────
section "3. 核心闭环：zip 直进 run → score → report"
run_step step-3-run.log python -m triton_eval run --profile smoke \
    --submission /tmp/triton-eval-pkgs/good.zip --limit 5
judge "run --submission good.zip（rc=0）" $?

run_step step-3-score.log python -m triton_eval score --run latest
judge "score --run latest" $?
run_step step-3-report.log python -m triton_eval report --run latest
judge "report --run latest" $?

RUN="$(latest_run)"
missing=""
if [ -n "$RUN" ]; then
    for f in results.jsonl events.jsonl env.json integrity.json artifacts.json; do
        [ -s "$RUN/$f" ] || missing="$missing $f"
    done
else
    missing="（没有 runs/run-* 目录）"
fi
[ -z "$missing" ] && [ -n "$RUN" ]
judge "产物齐全（results/events/env/integrity/artifacts）${missing:+缺:$missing}" $?

grep -q '"killed_by_us": false' "$RUN/integrity.json"
judge "integrity.json：子进程自然结束（未被杀）" $?
grep -qh "\[eval\]" "$RUN"/logs/*.log 2>/dev/null
judge "用例日志有真实内容 + [eval] 尾注" $?
python - "$RUN" <<'PY' >> "$VDIR/step-3-artifacts.log" 2>&1
import json, sys, pathlib
run = pathlib.Path(sys.argv[1])
arts = json.loads((run / "artifacts.json").read_text())
assert all(f["sha256"] for f in arts["files"]), "artifacts 清单缺 sha256"
evts = [json.loads(l) for l in (run / "events.jsonl").read_text().splitlines() if l.strip()]
kinds = [e["event"] for e in evts]
assert kinds[0] == "meta" and kinds[-1] == "done", f"事件协议首尾异常: {kinds[:2]}...{kinds[-2:]}"
print("artifacts sha256 完整；事件协议 meta→…→done 正常，事件数", len(evts))
PY
judge "artifacts sha256 清单 + 事件协议 meta→done" $?

# ─────────────────────────── 4. 三条关键防线 ───────────────────────────
section "4a. D7：import 期投毒 torch.softmax（预期 rc=1，INVALID，SUB-EXE-005）"
run_step step-4a-d7.log python -m triton_eval run --profile smoke --limit 3 \
    --submission examples/adversarial-submissions/ADV-D7-oracle-poison
rc=$?; [ $rc -eq 1 ] && grep -q "SUB-EXE-005" "$VDIR/step-4a-d7.log"
judge "D7 被篡改校验拦截（rc=1 + SUB-EXE-005），不再是全零也 PASS" $?

section "4b. E3：import 期 SystemExit（预期 rc=1，结构化报错）"
run_step step-4b-e3.log python -m triton_eval run --profile smoke --limit 3 \
    --submission examples/adversarial-submissions/ADV-E3-systemexit
rc=$?; [ $rc -eq 1 ] && grep -qE "SUB-LOD-005|试图终止进程" "$VDIR/step-4b-e3.log"
judge "E3 结构化报错（rc=1），不再静默消失" $?

section "4c. F4：吞掉超时信号（预期 exp 用例全部 RUNTIME_FAIL，run 整体 OK）"
# 预算压缩到 5s：SIGALRM 被吞后，事后耗时核对兜底。
# 余量给到 240s：语料含 XL 用例（≥2M 元素），输入构造/参考/比对全路径
# 在压缩预算下可能超过旧余量（真机实测：XL exp 用例 65s 窗口内未完成，
# 被监督者杀掉——防线行为正确，但预期写错）。F4 只声明 Exp，
# 其余算子判 SUBMISSION_INVALID（正确行为）。
run_step step-4c-f4.log env TRITON_EVAL_CASE_TIMEOUT_S=5 \
    TRITON_EVAL_IDLE_MARGIN_S=240 TRITON_EVAL_GRACE_S=3 \
    python -m triton_eval run --profile smoke \
    --submission examples/adversarial-submissions/ADV-F4-timeout-swallow
rc=$?
n_fail=$(grep -cE "^  exp-[^:]*: RUNTIME_FAIL" "$VDIR/step-4c-f4.log" || true)
n_pass=$(grep -cE "^  exp-[^:]*: PASS" "$VDIR/step-4c-f4.log" || true)
[ $rc -eq 0 ] && [ "$n_fail" -ge 3 ] && [ "$n_pass" -eq 0 ]
judge "F4 的全部 exp 用例 RUNTIME_FAIL（超时信号被吞也拦得住），run 正常结束" $?

# ─────────────────────────── 5. C1 全量正控制 ───────────────────────────
section "5. C1：14 原语算子全手写正控制（跑 v0.5 全量；11 个融合算子 C1 未提交，其 73 个用例判 SUBMISSION_INVALID 属预期——融合内核的正控制见 5c；纯 Python 较慢，约 15-30 分钟）"
if [ "${SKIP_C1:-0}" = "1" ]; then
    echo "  （SKIP_C1=1，跳过）"
else
    run_step step-5-c1.log python -m triton_eval run --profile smoke \
        --submission examples/adversarial-submissions/ADV-C1-fullhouse
    judge "C1 run 完成（rc=0）" $?
    run_step step-5-score.log python -m triton_eval score --run latest
    RUNC1="$(latest_run)"
    if [ -n "$RUNC1" ] && [ -s "$RUNC1/results.jsonl" ]; then
        python - "$RUNC1/results.jsonl" >> "$VDIR/step-5-score.log" 2>&1 <<'PY'
import collections, json, sys
c = collections.Counter(json.loads(l)["status"] for l in open(sys.argv[1]) if l.strip())
print("C1 终态分布:", dict(c))
print("预期（容差 1.1 修正后）: S/M 档 PASS；XL 档（≥2M 元素）纯 Python")
print("      循环超时按状态机合并为 COMPILE_FAIL——负载分层本来就要淘汰")
print("      纯 Python 实现，不是缺陷；ArgMax 0 维用例 RUNTIME_FAIL 是 C1")
print("      自身缺陷被新考法抓住（战果）；静默错误（WRONG_RESULT）预期 0——")
print("      此前 4 宗已定性为评测方冤案、由 tolerance.yaml 1.1 修复，非 0 即回归")
PY
        tail -8 "$VDIR/step-5-score.log"
    else
        echo "  （C1 没有产出 results.jsonl——看 $VDIR/step-5-c1.log）"
    fi
    # 本轮核心断言升级为正式判定项：静默错误归零（容差 1.1 修正后的复核点）
    WRONG_N5="$(grep -c 'WRONG_RESULT' "$RUNC1/results.jsonl" 2>/dev/null || true)"
    [ -n "$RUNC1" ] && [ -s "$RUNC1/results.jsonl" ] && [ "${WRONG_N5:-0}" = "0" ]
    judge "C1 静默错误为 0（容差 1.1 修正后的核心断言；非 0 即回归，逐条归因）" $?
fi

# ─────────────────────────── 5b. standard 档测速（按需启用）───────────
section "5b. standard 档测速：现场计时 + 按算子性能分（WITH_STANDARD=1 启用）"
if [ "${SKIP_C1:-0}" = "1" ]; then
    echo "  （SKIP_C1=1，跳过）"
elif [ "${WITH_STANDARD:-0}" != "1" ]; then
    echo "  （按需启用：WITH_STANDARD=1；也可单独运行标准测速入口。）"
    echo "  （独立入口：bash scripts/standard-window.sh）"
else
    # 按显式选项归档冒烟证据；运行产物在持久目录保留。
    #
    # 前半程（含"静默错误归零"核心断言）的证据已经安全进了 gitcode。
    if [ "${SKIP_ARCHIVE:-1}" != "1" ]; then
        echo "  —— 冒烟阶段证据先行归档上传（测速跑完收尾还会再归档一次）"
        bash scripts/upload-verify-record.sh "$VDIR" \
            || echo "  （中途归档未完成——收尾还会再试一次）"
    fi
    # 被测提交用 demo（triton 内核）而不是 C1：性能锚点资格要求"全对
    # **且都有加速比**"——C1 是纯 Python，XL 用例单次执行实测 110-118s
    # （120s 预算擦线过），standard 档测速段要在同一预算里连跑 6 次
    # （被测 3 遍 + 参考 3 遍），物理上装不下 → XL 用例无加速比 → C1
    # 永远不满足锚点资格。demo 的 triton 内核 XL 档毫秒级，才是锚点的料。
    run_step step-5b-run.log python -m triton_eval run --profile standard \
        --submission examples/demo-submission
    judge "standard 档 run 完成（rc=0）" $?
    run_step step-5b-score.log python -m triton_eval score --run latest
    judge "standard 档 score（按算子性能分表 + 无分原因）" $?
    run_step step-5b-report.log python -m triton_eval report --run latest
    judge "standard 档 report 生成（含性能分节）" $?
    # 静默错误复核：standard 与冒烟同一判卷口径，容差 1.1 后同样应为 0
    RUNSTD="$(latest_run)"
    WRONG_N="$(grep -c 'WRONG_RESULT' "$RUNSTD/results.jsonl" 2>/dev/null || true)"
    [ -n "$RUNSTD" ] && [ -s "$RUNSTD/results.jsonl" ] && [ "${WRONG_N:-0}" = "0" ]
    judge "standard 档静默错误为 0（demo 全对为正常；非 0 记录并归因）" $?
    echo "  —— 按算子性能分表见上方 score 输出；再来一轮后可用 compare 出两轮对比"
fi

# ─────────────────────────── 5c. 融合算子冒烟（v0.5 新增 11 算子的首次真机执行）───────────
# demo 的 11 个融合内核此前只过静态扫描（开发机无 NPU），本步是它们第一次
# 真机编译执行。预期口径**按用例逐条**（2026-10-14 修正：此前把 MaskedGelu
# 整算子当全显性不支持是错的——demo 一直实现了 tanh 近似路径，erf 精确路径
# 才显性 NotImplementedError；归因见 robustness-findings 第五轮附录）：
#   erf 精确用例（无 approximate 属性，LinearGelu 全部 6 个 + MaskedGelu 3 个）
#     → RUNTIME_FAIL（demo 纪律：erf 依赖 libdevice 未验证，不凑近似）；
#   其余融合用例（含 MaskedGelu 的 5 个 tanh 用例）→ PASS；
# ApplyRotaryPosEmb 的 tl.sin/tl.cos 是否被 triton-ascend 支持正是本步要回答
# 的问题——PASS 则锚点载体成立，COMPILE_FAIL 则如实亮 ✗ 并把归因记入
# checks.log（那是编译器能力发现，不是闭环缺陷）。
section "5c. 融合算子冒烟：demo 11 个融合内核首次真机编译执行（73 用例）"
FUSED_OPS="linear_gelu,rms_norm_linear,grouped_matmul_swiglu,transpose_linear,masked_softmax,masked_gelu,gather_exp,add_rms_norm,apply_rotary_pos_emb,moe_gating_top_k_softmax,unsorted_segment_sum"
run_step step-5c-run.log python -m triton_eval run --profile smoke \
    --operators "$FUSED_OPS" --submission examples/demo-submission
judge "融合冒烟 run 完成（rc=0）" $?
RUNF="$(latest_run)"
python - "$RUNF/results.jsonl" "$VDIR/checks.log" >> "$VDIR/step-5c-analyze.log" 2>&1 <<'PY'
import collections, json, sys
from triton_eval.cli import DEFAULT_CORPUS  # 与 run 相同语料（5c 不传 --corpus）
try:
    recs = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
except OSError:
    recs = []
fusion_ops = {"LinearGelu", "RmsNormLinear", "GroupedMatmulSwiGlu", "TransposeLinear",
              "MaskedSoftmax", "MaskedGelu", "GatherExp", "AddRmsNorm",
              "ApplyRotaryPosEmb", "MoeGatingTopKSoftmax", "UnsortedSegmentSum"}
recs = [r for r in recs if r.get("operator") in fusion_ops]
print(f"融合用例 {len(recs)} 条")
if not recs:
    print("!! 最新 run 的结果里没有融合算子用例——run 目录可能对不上（看 step-5c-run.log）")
    sys.exit(1)
by_op = collections.defaultdict(collections.Counter)
for r in recs:
    by_op[r["operator"]][r["status"]] += 1
for op in sorted(by_op):
    print(f"  {op}: {dict(by_op[op])}")
# demo 纪律按用例模式判（results 记录不带 attrs，模式从冻结清单取）：
# erf 精确用例（无 approximate 属性）显性 NotImplementedError → RUNTIME_FAIL；
# tanh 近似用例已实现 → 与可实现算子同样要求 PASS
_manifest = json.loads((DEFAULT_CORPUS / "manifest.json").read_text(encoding="utf-8"))
erf_codes = {c["code"] for c in _manifest["cases"]
             if c["operator"] in ("LinearGelu", "MaskedGelu")
             and (c.get("attrs") or {}).get("approximate", "none") != "tanh"}
oracle = [r["case_code"] for r in recs if r["status"] == "ORACLE_FAIL"]
bad_expect = [r["case_code"] for r in recs
              if r["case_code"] in erf_codes and r["status"] != "RUNTIME_FAIL"]
unexpected = [f'{r["case_code"]}({r["status"]})' for r in recs
              if r["case_code"] not in erf_codes and r["status"] != "PASS"]
rope = {r["status"] for r in recs if r["operator"] == "ApplyRotaryPosEmb"}
with open(sys.argv[2], "a") as fh:
    if rope == {"PASS"}:
        fh.write("[记] RoPE 编译结论：tl.sin/tl.cos 可用——锚点载体成立\n")
    elif "COMPILE_FAIL" in rope:
        fh.write(f"[记] RoPE 编译结论：triton-ascend 不支持 tl.sin/tl.cos"
                 f"（终态 {sorted(rope)}）——显性失败，如实记录\n")
    elif rope:
        fh.write(f"[记] RoPE 编译结论：非预期终态 {sorted(rope)}——逐用例归因\n")
print("ORACLE_FAIL:", oracle or "无")
print("erf 精确用例的非 RUNTIME_FAIL:", bad_expect or "无（符合 demo 纪律）")
print("已实现用例的非 PASS:", unexpected or "无")
sys.exit(0 if not (oracle or bad_expect or unexpected) else 1)
PY
rc5c=$?
tail -20 "$VDIR/step-5c-analyze.log"
judge "融合冒烟口径：erf 精确用例（LinearGelu 全部+MaskedGelu 3 个）全显性 RUNTIME_FAIL；其余融合用例（含 MaskedGelu 5 个 tanh 用例）全 PASS；无 ORACLE_FAIL" "$rc5c"

# ─────────────────────────── 5d. 手法原子收益实证（探针对真机跑批）───────────
# 9 条性能负载原子的 1.5×/80% 门槛实证（方案 §6 待办①）。逐原子的「实证
# 达标 / 如实降级」都是合法结果——本步只判「跑完且出报告」；结论供覆盖率
# 披露四类分列引用。报告写进验证目录，按需手动归档，原件保存在本地。
section "5d. 探针收益实证：9 条性能负载原子（对拍 → 同口径计时 → 1.5×/80% 门槛）"
run_step step-5d-probes.log python scripts/probe_validation.py --out "$VDIR"
rc5d=$?
if [ $rc5d -eq 0 ] && [ -s "$VDIR/probes-report.json" ]; then
    python - "$VDIR/probes-report.json" >> "$VDIR/step-5d-analyze.log" 2>&1 <<'PY'
import json, sys
rep = json.load(open(sys.argv[1]))
print("逐原子结论：")
for atom, r in sorted(rep["atoms"].items()):
    print(f"  {atom}: 达标占比 {r['pass_ratio']:.0%} → {r['verdict']}")
PY
    tail -12 "$VDIR/step-5d-analyze.log"
    judge "探针收益实证完成（probes-report.{json,md} 已入验证目录）" 0
else
    judge "探针收益实证未完成（rc=${rc5d}，明细见 step-5d-probes.log）" 1
fi

# ─────────────────────────── 5e. 审计项真机复核（2026-10-14 新增）───────────
# 审计三件套（生成日志/演化对照/剔除记录防篡改）是 L0 机制，真机侧补两个证据：
#   ① 冻结门禁（可执行、输出有限、两遍一致）在目标设备上全绿——门禁此前
#      只在开发机 CPU 上跑过。NPU 上若有算子两遍不一致（如散射累加顺序），
#      属真实平台发现，如实亮 ✗ 带清单回报，不是回归；
#   ② 跨机可复现性：本机重跑生成脚本，清单哈希必须与冻结版逐字一致、
#      清单不被重写——「冻结内容与构建机器无关」由这步从机制断言变成实测。
# ② 走纯逻辑路径（--no-verify-refs，① 已单独跑同一份 check_all），不经过
# 门禁剔除分支，不会改写清单。构建脚本会往 generation-log.jsonl 追加本机
# 构建事件——事件行随验证记录归档（step-5e-npubuild-events.jsonl），随后
# 把工作区还原：仓库里的日志只积累开发机构建，机器构建事件在归档记录可查。
section "5e. 审计项真机复核：NPU 参考自检 + 清单跨机可复现"
run_step step-5e-refcheck.log python - "$VDIR/checks.log" <<'PY'
import re
import sys
from pathlib import Path

import torch_npu  # noqa: F401  目标设备执行（与 build_corpus_v3 的 npu 分支同规矩）

from triton_eval.corpusmod import load_manifest
from triton_eval.refcheck import check_all

# 已登记的锚点自噪声（证据：robustness-findings 第六/七轮附录）——
# torch_npu 的 cross_entropy 归约存在执行间 ULP 级噪声：两窗口实测
# 9 例次、max_rel ≤ 9.3e-8，失败用例集合逐轮随机（非固定用例）；
# 判定容差（f32 档 rel 1e-4，rel/abs 析取）余量 3 个数量级以上，
# 历轮 CE 用例判定全部通过。登记界取 1e-6 = 实测最大值的 10 倍、
# 容差的 1%——超界、或出现在其他算子上，仍按新发现报出。
KNOWN_NOISE_OPS = {"CrossEntropyLoss"}
NOISE_REL_BOUND = 1e-6

m = load_manifest(Path("corpus/v0.5-cov20-frozen/manifest.json"))
print(f"冻结清单 {m.version}：{len(m.cases)} 用例，device=npu:0 参考自检"
      "（可执行 + 输出有限 + 两遍一致）")
failures = check_all(m.cases, "npu:0", corpus_dir="corpus/v0.5-cov20-frozen")
op_of = {c.code: c.operator for c in m.cases}
known, unknown = [], []
for code, why in sorted(failures.items()):
    mm = re.search(r"max_rel=([0-9.eE+-]+)", why)
    rel = float(mm.group(1)) if mm else None
    if op_of.get(code) in KNOWN_NOISE_OPS and rel is not None \
            and rel <= NOISE_REL_BOUND:
        known.append((code, rel))
    else:
        unknown.append((code, why))
print(f"自检失败 {len(failures)} 个：已登记锚点自噪声 {len(known)} 个、"
      f"未知 {len(unknown)} 个")
for code, rel in known:
    print(f"  [已登记噪声] {code}: max_rel={rel:.3e} ≤ 界 {NOISE_REL_BOUND:.0e}")
for code, why in unknown:
    print(f"  {code}: {why}")
if known:
    with open(sys.argv[1], "a") as fh:
        fh.write(f"[记] 已登记锚点自噪声 {len(known)} 个（CE 运行间 ULP 噪声，"
                 f"max_rel≤{max(r for _, r in known):.3e}，登记界 1e-6，"
                 f"判定容差余量 3 个数量级）\n")
if unknown:
    with open(sys.argv[1], "a") as fh:
        fh.write(f"[!] NPU 参考自检未知失败 {len(unknown)} 个："
                 f"{', '.join(c for c, _ in unknown)}——新发现，需逐个归因\n")
sys.exit(1 if unknown else 0)
PY
judge "冻结门禁在 npu:0：除已登记的 CE 锚点自噪声（rel≤1e-6）外零失败" $?

run_step step-5e-rebuild.log python scripts/build_corpus_v3.py --no-verify-refs
rc5e=$?
# 本次机器构建的事件行（追加在仓库日志末尾的新增行）随验证记录归档
BASE_LINES=$(git show HEAD:corpus/v0.5-cov20-frozen/generation-log.jsonl 2>/dev/null | wc -l)
tail -n +"$((BASE_LINES + 1))" corpus/v0.5-cov20-frozen/generation-log.jsonl \
    > "$VDIR/step-5e-npubuild-events.jsonl" 2>/dev/null || true
git checkout -- corpus/v0.5-cov20-frozen/generation-log.jsonl 2>/dev/null || true
if [ $rc5e -eq 0 ] \
   && grep -q "内容未变，未重写" "$VDIR/step-5e-rebuild.log" \
   && [ -z "$(git status --porcelain corpus/v0.5-cov20-frozen/manifest.json)" ]; then
    echo "[记] 跨机可复现：aarch64 真机重跑生成脚本，清单哈希逐字一致、未重写；本机构建事件行已随本记录归档" >> "$VDIR/checks.log"
    judge "清单跨机可复现（哈希逐字一致、清单未重写）" 0
else
    judge "清单跨机可复现失败（rc=${rc5e}）——本机生成出不同清单，重大发现；现场已保留：git diff corpus/v0.5-cov20-frozen/manifest.json" 1
fi

# ─────────────────────────── 6. 数据库抽查 ───────────────────────────
section "6. 数据库：提交身份链（包 sha256 → 目录 → 快照）"
python - >> "$VDIR/step-6-db.log" 2>&1 <<'PY'
import sqlite3
conn = sqlite3.connect("runs/triton-eval.db")
conn.row_factory = sqlite3.Row
print("最近 6 条提交记录：")
for r in conn.execute("SELECT submit_id, package_name, substr(package_sha256,1,12) sha, validation_ok FROM submission_record ORDER BY id DESC LIMIT 6"):
    print(" ", dict(r))
print("各 run 状态：")
for r in conn.execute("SELECT id, status, substr(invalid_reason,1,60) reason FROM eval_run ORDER BY id DESC LIMIT 6"):
    print(" ", dict(r))
PY
cat "$VDIR/step-6-db.log"
python - >> "$VDIR/step-6-db.log" 2>&1 <<'PY'
import sqlite3
try:
    conn = sqlite3.connect("runs/triton-eval.db")
    row = conn.execute("SELECT package_name, package_sha256 FROM submission_record WHERE package_sha256 IS NOT NULL ORDER BY id DESC LIMIT 1").fetchone()
except sqlite3.OperationalError as exc:
    print("数据库不可用:", exc)
    raise SystemExit(1)
raise SystemExit(0 if row and row[0] and row[1] else 1)
PY
judge "submission_record 携带包身份（package_name + package_sha256）" $?

# ─────────────────────────── 汇总 ───────────────────────────
# 完成标记：见到此行 = 收尾前的所有节都执行了。归档读它即可分辨
# "脚本跑完"与"中途被断"（2026-10-12 连续两轮在第 5 步后中断，靠它定位）。
echo "[完] 全部步骤执行完毕（此行存在即证明脚本没有中途死掉）" >> "$VDIR/checks.log"
echo
echo "══════════════════════════════════════"
echo "验证完成：✓ $PASS 项通过，✗ $FAIL 项不符"
if [ $FAIL -eq 0 ]; then
    echo "闭环在昇腾机上全部成立"
else
    echo "不符项的完整输出在: $VDIR/step-*.log"
fi

# 归档上传由 on_exit trap 统一兜底（正常结束与被中断都覆盖），
# 此处不再显式调用。
[ $FAIL -eq 0 ] || exit 1
