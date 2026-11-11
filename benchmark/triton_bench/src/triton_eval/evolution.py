# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""测试集版本间演化对照（审计）——两份冻结清单的结构化 diff 与渲染。

任务书要求保留「corpus 演化记录以供审计」。此前两版之间新增/剔除了
什么、为什么，只能人工比对 manifest——v0.4→v0.5 的「按算子用例
再分配」（SwiGlu 20→6 等）就是手工数出来的，不可复用也不可对账。

本模块只做**可从清单推导的事实**：diff_manifests 是纯函数（输入两份
CorpusManifest，输出 dict），render_markdown 把 dict 渲染成文档。
版本内变更的背景与理由（为什么重冻结、为什么再分配）不在此处——
那属于人工叙述，在方案文档的落地实况章节，文档头部互相指引。

编号是弱键的诚实处理：用例 code（如 exp-f32-00）不编码形状/数值，
跨版本同编号完全可能内容不同——diff 把这类「同编号但内容变更」
单列并给出字段级差异，不冒充「未变」。
"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any

from .models import CorpusManifest

# 字段级 diff 的比较范围：CaseSpec 全部内容字段（content_hash 是结果不是内容）
SPEC_FIELDS = (
    "operator", "source", "dtype", "shape", "attrs", "value_range",
    "contiguous", "seed", "launch", "timeout_s", "feature_atoms",
)

# 文档里逐条列出的上限；超过则折叠为按算子计数，避免几百行编号清单
FULL_LIST_CAP = 20


def _numel(shape: list[int]) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _identity(m: CorpusManifest) -> dict[str, Any]:
    return {
        "version": m.version,
        "frozen_at": m.frozen_at,
        "manifest_hash": m.manifest_hash,
        "spec_version": m.spec_version,
        "feature_atoms_version": m.feature_atoms_version,
        "operators": len(m.operators),
        "cases": len(m.cases),
        "excluded": len(m.excluded),
    }


def _field_diff(old, new) -> dict[str, Any]:
    """同编号用例的字段级差异：{字段: (旧值, 新值)}，仅含有差异的字段。"""
    out: dict[str, Any] = {}
    for f in SPEC_FIELDS:
        if getattr(old, f) != getattr(new, f):
            out[f] = (getattr(old, f), getattr(new, f))
    return out


def _bucket_delta(old_cases, new_cases, key_of) -> list[dict[str, Any]]:
    """按 key_of 分桶计数，只留计数有变化的桶（审计看的是再分配）。"""
    old_c: Counter = Counter(key_of(c) for c in old_cases)
    new_c: Counter = Counter(key_of(c) for c in new_cases)
    rows = []
    for k in sorted(set(old_c) | set(new_c)):
        o, n = old_c.get(k, 0), new_c.get(k, 0)
        if o != n:
            rows.append({"key": k, "old": o, "new": n, "delta": n - o})
    return rows


def diff_manifests(old: CorpusManifest, new: CorpusManifest) -> dict[str, Any]:
    """两份冻结清单的结构化 diff。纯函数，可随时重算对账。"""
    old_ops = {op.name: op for op in old.operators}
    new_ops = {op.name: op for op in new.operators}

    old_counts, new_counts, op_rows = _operator_case_deltas(old, new)

    old_by_code = {c.code: c for c in old.cases}
    new_by_code = {c.code: c for c in new.cases}
    added = sorted(set(new_by_code) - set(old_by_code))
    removed = sorted(set(old_by_code) - set(new_by_code))
    changed = []
    for code in sorted(set(old_by_code) & set(new_by_code)):
        fd = _field_diff(old_by_code[code], new_by_code[code])
        if fd:
            changed.append({
                "code": code,
                "operator": new_by_code[code].operator,
                "fields": fd,
            })
    changed_ops: Counter = Counter(c["operator"] for c in changed)
    field_freq: Counter = Counter(f for c in changed for f in c["fields"])

    old_atoms = {a for c in old.cases for a in (c.feature_atoms or [])}
    new_atoms = {a for c in new.cases for a in (c.feature_atoms or [])}

    return {
        "old": _identity(old),
        "new": _identity(new),
        "operators_added": [
            {"name": n, "desc": new_ops[n].desc} for n in sorted(set(new_ops) - set(old_ops))
        ],
        "operators_removed": [
            {"name": n, "desc": old_ops[n].desc} for n in sorted(set(old_ops) - set(new_ops))
        ],
        "op_case_delta": op_rows,
        "op_case_unchanged": len(set(old_counts) | set(new_counts)) - len(op_rows),
        "cases_added": added,
        "cases_removed": removed,
        "cases_changed": changed,
        "cases_changed_by_op": dict(sorted(changed_ops.items())),
        "cases_changed_field_freq": dict(sorted(field_freq.items())),
        "cases_kept_identical": len(old_by_code) - len(removed) - len(changed),
        "atoms_old": sorted(old_atoms),
        "atoms_new": sorted(new_atoms),
        "atoms_gained": sorted(new_atoms - old_atoms),
        "atoms_lost": sorted(old_atoms - new_atoms),
        "size_delta": _bucket_delta(old.cases, new.cases,
                                    lambda c: (c.operator, _size_class(_numel(c.shape)))),
        "dtype_delta": _bucket_delta(
            old.cases, new.cases, lambda c: (c.operator, c.dtype)),
        "excluded_old": list(old.excluded),
        "excluded_new": list(new.excluded),
    }


def _size_class(n: int) -> str:
    # 复用 coverage 的档位口径（S/M/L/XL），不另立标准
    from .coverage import size_class

    return size_class(n)


# ---------- 渲染 ----------


def _short(val: Any) -> str:
    s = json.dumps(val, ensure_ascii=False, sort_keys=True)
    return s if len(s) <= 60 else s[:57] + "…"


def _list_or_fold(codes: list[str], label: str) -> list[str]:
    """≤FULL_LIST_CAP 逐条列出，否则按算子折叠计数。"""
    if not codes:
        return [f"- {label}：无"]
    if len(codes) <= FULL_LIST_CAP:
        return [f"- {label}：{len(codes)} 个"] + [f"  - {c}" for c in codes]
    per_op: Counter = Counter(c.split("-", 1)[0] for c in codes)
    lines = [f"- {label}：{len(codes)} 个（超过 {FULL_LIST_CAP}，按算子折叠）"]
    lines += [f"  - {op}：{n}" for op, n in sorted(per_op.items())]
    return lines


def render_markdown(diff: dict[str, Any], gen_logs: dict[str, Any] | None = None) -> str:
    """渲染单次版本迁移的对照章节（不含文档总标题）。"""
    o, n = diff["old"], diff["new"]
    out: list[str] = []
    out.append(f"## {o['version']} → {n['version']}")
    out.append("")

    _render_version_and_operators(diff, out, o, n)

    _render_case_changes(diff, out)

    _render_distribution_changes(diff, out)

    _render_generation_logs(out, gen_logs, o, n)

    return "\n".join(out)


def _operator_case_deltas(old, new):
    old_counts: Counter = Counter(c.operator for c in old.cases)
    new_counts: Counter = Counter(c.operator for c in new.cases)
    op_rows = []
    for op in sorted(set(old_counts) | set(new_counts)):
        o, n = old_counts.get(op, 0), new_counts.get(op, 0)
        if o != n:
            op_rows.append({"operator": op, "old": o, "new": n, "delta": n - o})
    return old_counts, new_counts, op_rows


def _render_version_and_operators(diff, out, o, n):
    out.append("### 版本身份")
    out.append("")
    out.append("| 项 | 旧 | 新 |")
    out.append("|---|---|---|")
    out.append(f"| 冻结时间 | {o['frozen_at']} | {n['frozen_at']} |")
    out.append(f"| 清单哈希 | `{o['manifest_hash'][:12]}…` | `{n['manifest_hash'][:12]}…` |")
    out.append(f"| 评测规范版本 | {o['spec_version']} | {n['spec_version']} |")
    out.append(f"| 能力点清单版本 | {o['feature_atoms_version']} | {n['feature_atoms_version']} |")
    out.append(f"| 算子数 | {o['operators']} | {n['operators']} |")
    out.append(f"| 用例数 | {o['cases']} | {n['cases']} |")
    out.append(f"| 门禁剔除数 | {o['excluded']} | {n['excluded']} |")
    out.append("")

    out.append("### 算子增删")
    out.append("")
    if diff["operators_added"]:
        names = "、".join(f"{r['name']}（{r['desc']}）" for r in diff["operators_added"])
        out.append(f"- 新增算子 {len(diff['operators_added'])} 个：{names}")
    else:
        out.append("- 新增算子：无")
    if diff["operators_removed"]:
        names = "、".join(f"{r['name']}（{r['desc']}）" for r in diff["operators_removed"])
        out.append(f"- 移除算子 {len(diff['operators_removed'])} 个：{names}")
    else:
        out.append("- 移除算子：无")
    out.append("")

    out.append("### 逐算子用例数变化")
    out.append("")
    if diff["op_case_delta"]:
        out.append("| 算子 | 旧 | 新 | Δ |")
        out.append("|---|---|---|---|")
        for r in diff["op_case_delta"]:
            out.append(f"| {r['operator']} | {r['old']} | {r['new']} | {r['delta']:+d} |")
        out.append(f"（其余 {diff['op_case_unchanged']} 个算子用例数不变）")
    else:
        out.append("所有算子用例数不变。")
    out.append("")


def _render_case_changes(diff, out):
    out.append("### 用例级变化")
    out.append("")
    out.extend(_list_or_fold(diff["cases_added"], "新增用例"))
    out.extend(_list_or_fold(diff["cases_removed"], "移除用例"))
    changed = diff["cases_changed"]
    if changed:
        freq = "、".join(f"{k}×{v}" for k, v in diff["cases_changed_field_freq"].items())
        out.append(f"- **同编号但内容变更** {len(changed)} 个"
                   f"（编号是弱键：不编码形状/数值，跨版本同编号不等于同题；"
                   f"字段变更频次：{freq}）")
        if len(changed) <= FULL_LIST_CAP:
            for c in changed:
                parts = [f"{f}：{_short(v[0])} → {_short(v[1])}"
                         for f, v in c["fields"].items()]
                out.append(f"  - {c['code']}：" + "；".join(parts))
        else:
            per_op = "、".join(f"{k}×{v}" for k, v in diff["cases_changed_by_op"].items())
            out.append(f"  - 超过 {FULL_LIST_CAP} 个，按算子折叠：{per_op}")
    else:
        out.append("- 同编号但内容变更：无")
    out.append(f"- 内容未变的用例：{diff['cases_kept_identical']} 个")
    out.append("")


def _render_distribution_changes(diff, out):
    out.append("### 规模分布变化（Δ≠0 的算子×档位）")
    out.append("")
    if diff["size_delta"]:
        out.append("| 算子/档位 | 旧 | 新 | Δ |")
        out.append("|---|---|---|---|")
        for r in diff["size_delta"]:
            op, sc = r["key"]
            out.append(f"| {op}/{sc} | {r['old']} | {r['new']} | {r['delta']:+d} |")
    else:
        out.append("无变化。")
    out.append("")

    out.append("### dtype 分布变化（Δ≠0 的算子×dtype）")
    out.append("")
    if diff["dtype_delta"]:
        out.append("| 算子/dtype | 旧 | 新 | Δ |")
        out.append("|---|---|---|---|")
        for r in diff["dtype_delta"]:
            op, dt = r["key"]
            out.append(f"| {op}/{dt} | {r['old']} | {r['new']} | {r['delta']:+d} |")
    else:
        out.append("无变化。")
    out.append("")

    out.append("### 覆盖原子得失")
    out.append("")
    out.append(f"- 原子覆盖数：{len(diff['atoms_old'])} → {len(diff['atoms_new'])}")
    gained = "、".join(diff["atoms_gained"]) or "无"
    lost = "、".join(diff["atoms_lost"]) or "无"
    out.append(f"- 新覆盖：{gained}")
    out.append(f"- 不再覆盖：{lost}")
    out.append("")

    out.append("### 门禁剔除记录（随清单冻结，永久可查）")
    out.append("")
    for tag, records in (("旧版", diff["excluded_old"]), ("新版", diff["excluded_new"])):
        if records:
            out.append(f"- {tag}：{len(records)} 条")
            for r in records:
                out.append(f"  - {r['code']}：{r['reason']}")
        else:
            out.append(f"- {tag}：零剔除")
    out.append("")


def _render_generation_logs(out, gen_logs, o, n):
    out.append("### 生成日志")
    out.append("")
    logs = gen_logs or {}
    for tag in ("old", "new"):
        ident = o if tag == "old" else n
        info = logs.get(tag)
        if info is None:
            out.append(f"- {ident['version']}：无 generation-log.yaml"
                       "（该版构建早于日志机制上线）")
        else:
            out.append(f"- {ident['version']}：generation-log.yaml 共 "
                       f"{info['events']} 条记录（日志起点 {info['first_ts']}，"
                       "此前构建无日志）")
    out.append("")
