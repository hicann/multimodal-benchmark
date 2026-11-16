/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import { useEffect, useRef, useState } from "react";
import { Link, useSearchParams } from "react-router-dom";
import useSWR from "swr";
import { ArrowRight } from "@phosphor-icons/react";
import { request, time } from "../api/client";
import type { components } from "../api/schema";
import { Empty, ErrorNotice, Refresh } from "../components";
import { OperatorList } from "../OperatorList";

type Board = components["schemas"]["Leaderboard"];
const ratio = (value: number | null) =>
  value == null
    ? "未测量"
    : `${value < 0.01 ? value.toPrecision(3) : value.toFixed(3)}×`;
export function Leaderboard() {
  const [metric, setMetric] = useState<"functional" | "performance">(
    "performance",
  );
  const [cohort, setCohort] = useState("");
  const [search] = useSearchParams();
  const query = new URLSearchParams({ metric });
  if (cohort) query.set("cohort", cohort);
  else if (search.get("task")) query.set("task_id", search.get("task")!);
  const { data, error, mutate, isValidating } = useSWR<Board>(
    `/api/v1/leaderboard?${query}`,
    request,
    { refreshInterval: 10000 },
  );
  const group = data?.selected_cohort;
  const initialized = useRef(false);
  useEffect(() => {
    // A recent functional-only task should not make the first performance view empty.
    if (!data?.selected_cohort || initialized.current) return;
    initialized.current = true;
    if (data.selected_cohort.profile !== "smoke") return;
    if (!search.get("task")) {
      const performanceGroup = [...data.cohorts]
        .reverse()
        .find((item) => item.profile === "standard");
      if (performanceGroup) {
        setCohort(performanceGroup.id);
        return;
      }
    }
    setMetric("functional");
  }, [data, search]);
  const scoreable =
    data?.anchors.filter((anchor) => anchor.scoreable).length || 0;
  return (
    <>
      <header className="page-heading detail-heading board-heading">
        <div>
          <h1>评测榜单</h1>
          <p>相同算子、用例和运行环境的成绩放在一起比较。</p>
        </div>
        <Link className="button primary" to="/submit">
          提交评测
          <ArrowRight size={18} />
        </Link>
      </header>
      <ErrorNotice error={error} />
      <div className="toolbar board-toolbar">
        <label>
          评测范围
          <select
            aria-label="选择榜单范围"
            value={cohort || group?.id || ""}
            onChange={(event) => {
              setCohort(event.target.value);
              if (
                data?.cohorts.find((item) => item.id === event.target.value)
                  ?.profile === "smoke"
              )
                setMetric("functional");
            }}
            disabled={!data?.cohorts.length}
          >
            {!data?.cohorts.length ? (
              <option value="">暂无完整成绩</option>
            ) : (
              data.cohorts.map((item) => (
                <option key={item.id} value={item.id}>
                  {item.operators.length === 1
                    ? item.operators[0]
                    : `${item.operators.length} 个算子`}{" "}
                  · {item.planned_cases} 用例 ·{" "}
                  {item.profile === "standard" ? "性能评测" : "功能检查"}
                </option>
              ))
            )}
          </select>
        </label>
        <Refresh onClick={() => void mutate()} />
        <span className="muted">
          {isValidating ? "正在更新…" : "每 10 秒自动更新"}
        </span>
      </div>
      {group ? (
        <>
          <section className="board-context">
            <div>
              <span className="muted">当前榜单</span>
              <strong>
                {group.corpus_version.split("-")[0]} · {group.operators.length}{" "}
                个算子 · {group.planned_cases} 个用例
              </strong>
              <p>
                {group.environment.npu_model || "设备型号未记录"} ·{" "}
                {group.environment.compiler_name || "编译器未记录"}{" "}
                {group.environment.triton_import_version} · CANN{" "}
                {group.environment.cann_version || "未记录"}
              </p>
            </div>
            <div>
              <span className="muted">上榜提交</span>
              <strong>{data?.total_entries || 0} 份</strong>
            </div>
          </section>
          <p className="board-scope-mobile">
            {group.corpus_version.split("-")[0]} · {group.operators.length} 算子
            · {group.planned_cases} 用例 · {data?.total_entries || 0} 份成绩
          </p>
          <nav className="tabs" aria-label="榜单类型">
            {(
              [
                ["performance", "性能榜"],
                ["functional", "功能榜"],
              ] as const
            ).map(([value, label]) => (
              <button
                key={value}
                className={metric === value ? "active" : ""}
                aria-pressed={metric === value}
                onClick={() => setMetric(value)}
              >
                {label}
              </button>
            ))}
          </nav>
          {data?.items.length ? (
            <>
              {data.total_entries === 1 ? (
                <p className="muted board-single board-candidate-note">
                  目前只有 1 份成绩。提交其他实现后可比较排名。
                </p>
              ) : null}
              <div className="table-wrap board-desktop">
                <table className="board-table">
                  <thead>
                    <tr>
                      <th>名次</th>
                      <th>提交 / 时间</th>
                      <th>功能得分</th>
                      {metric === "performance" ? (
                        <>
                          <th>相对性能分</th>
                          <th>实测加速比</th>
                        </>
                      ) : null}
                      <th>通过用例</th>
                      {metric === "performance" ? <th>耗时波动</th> : null}
                      <th />
                    </tr>
                  </thead>
                  <tbody>
                    {data.items.map((entry) => (
                      <tr key={entry.task_id}>
                        <td className="board-rank">{entry.rank}</td>
                        <td>
                          <strong className="board-name">
                            {entry.submit_id}
                          </strong>
                          <small>{time(entry.created_at)}</small>
                        </td>
                        <td>{entry.functional_score.toFixed(2)}</td>
                        {metric === "performance" ? (
                          <>
                            <td>
                              {entry.shared_perf_score == null
                                ? "不可计算"
                                : entry.shared_perf_score.toFixed(2)}
                              <small>
                                {entry.scored_operators}/{entry.total_operators}{" "}
                                算子计分
                              </small>
                            </td>
                            <td>{ratio(entry.geometric_speedup)}</td>
                          </>
                        ) : null}
                        <td>
                          {entry.passed_cases}/{entry.planned_cases}
                        </td>
                        {metric === "performance" ? (
                          <td>
                            {entry.unstable_cases} 例
                            <details>
                              <summary>测量详情</summary>
                              <p>
                                {entry.unstable_cases
                                  ? `${entry.unstable_cases} 个用例多次测量的耗时差别较大，建议复测后再比较。`
                                  : "未发现明显的耗时波动。"}
                              </p>
                            </details>
                          </td>
                        ) : null}
                        <td>
                          <Link
                            className="text-link"
                            to={`/workspace/jobs/${entry.task_id}`}
                          >
                            查看结果
                            <ArrowRight size={16} />
                          </Link>
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
              <div className="board-mobile-cards">
                {data.items.map((entry) => (
                  <article className="board-card" key={entry.task_id}>
                    <span className="board-place">第 {entry.rank} 名</span>
                    <h2>{entry.submit_id}</h2>
                    <dl>
                      <div>
                        <dt>功能得分</dt>
                        <dd>{entry.functional_score.toFixed(2)}</dd>
                      </div>
                      {metric === "performance" ? (
                        <>
                          <div>
                            <dt>相对性能分</dt>
                            <dd>
                              {entry.shared_perf_score == null
                                ? "不可计算"
                                : entry.shared_perf_score.toFixed(2)}
                            </dd>
                          </div>
                          <div>
                            <dt>实测加速比</dt>
                            <dd>{ratio(entry.geometric_speedup)}</dd>
                          </div>
                        </>
                      ) : null}
                      <div>
                        <dt>通过用例</dt>
                        <dd>
                          {entry.passed_cases}/{entry.planned_cases}
                        </dd>
                      </div>
                    </dl>
                    <div className="split-line">
                      <span className="muted">
                        {metric === "performance"
                          ? `耗时不稳定：${entry.unstable_cases} 例`
                          : time(entry.created_at)}
                      </span>
                      <Link
                        className="text-link"
                        to={`/workspace/jobs/${entry.task_id}`}
                      >
                        查看结果
                        <ArrowRight size={16} />
                      </Link>
                    </div>
                  </article>
                ))}
              </div>
              {data.total_entries > data.items.length ? (
                <p className="muted board-single">
                  显示前 {data.items.length} 份，共 {data.total_entries} 份成绩
                </p>
              ) : null}
            </>
          ) : (
            <Empty>
              {metric === "performance" && group.profile === "smoke"
                ? "该范围只做功能检查，没有性能数据。请切换功能榜。"
                : "暂无可排名的性能成绩，请先完成一次性能评测。"}
            </Empty>
          )}
          <p className="board-explanation">
            {metric === "performance"
              ? `100 分为本榜基准，当前按 ${scoreable}/${group.operators.length} 个算子计分。加速比以系统参考实现为 1×，大于 1× 更快。`
              : "按功能得分排名。相同代码包只保留最好成绩，同分并列。"}
          </p>
          <details className="technical-details">
            <summary>算子与运行环境</summary>
            <p>
              {group.environment.npu_model} · {group.environment.compiler_name}{" "}
              {group.environment.triton_import_version} · CANN{" "}
              {group.environment.cann_version}
            </p>
            <OperatorList names={group.operators} />
            <p>
              {group.environment.clock_locked
                ? "运行记录显示设备频率已固定。"
                : "本次未固定设备运行频率，耗时可能波动。成绩接近时，建议重复评测后再比较。"}
            </p>
            <p>当前为试运行榜单，尚未完成正式认证。</p>
            <p className="mono">
              评测集：{group.corpus_version}
              <br />
              环境指纹：{group.env_fp}
              <br />
              判定资产：{group.assets_hash}
              <br />
              评分器：{group.scorer_version || "未记录"}
            </p>
          </details>
          <details className="technical-details">
            <summary>计分与排名规则</summary>
            <p>
              每个算子首次全部通过且测速完整的成绩作为固定基准，记为 100
              分；后续成绩都与这份基准比较。 基准加速比接近 1×
              的算子暂不计相对性能分，避免公式放大微小波动。耗时、加速比和功能成绩仍保留。
            </p>
            <details>
              <summary>查看计算方法</summary>
              <p>
                现有公式用基准加速比的对数归一化。基准与 1× 相差不足 0.1
                时分母接近零，因此不计算该项百分制分数。
              </p>
            </details>
            <ul>
              {data?.rules.map((rule) => (
                <li key={rule}>{rule}</li>
              ))}
            </ul>
            <p className="mono">
              基准版本：{data?.baseline_revision || "尚未建立"}
            </p>
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>算子</th>
                    <th>基准加速比</th>
                    <th>是否计分</th>
                    <th>基准来源</th>
                  </tr>
                </thead>
                <tbody>
                  {data?.anchors.map((anchor) => (
                    <tr key={anchor.operator}>
                      <td>{anchor.operator}</td>
                      <td>{ratio(anchor.geometric_speedup)}</td>
                      <td>{anchor.scoreable ? "计分" : "暂不计分"}</td>
                      <td>
                        <Link
                          className="text-link mono"
                          to={`/workspace/jobs/${anchor.task_id}`}
                        >
                          {anchor.task_id.slice(0, 12)}
                        </Link>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </details>
        </>
      ) : !error ? (
        <Empty>
          {data
            ? "还没有完整评测成绩。先提交并完成一次评测，榜单会自动更新。"
            : "正在读取榜单…"}
        </Empty>
      ) : null}
      {data?.excluded.length ? (
        <details className="technical-details">
          <summary>未入榜任务（{data.excluded.length}）</summary>
          {data.excluded.map((item, index) => (
            <p key={`${item.task_id}-${index}`}>
              <Link
                className="text-link mono"
                to={`/workspace/jobs/${item.task_id}`}
              >
                {item.task_id.slice(0, 12)}
              </Link>{" "}
              · {item.reason}
            </p>
          ))}
        </details>
      ) : null}
    </>
  );
}
