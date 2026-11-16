/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import { useDeferredValue, useState } from "react";
import { Link, useParams } from "react-router-dom";
import {
  ArrowLeft,
  DownloadSimple,
  WarningCircle,
} from "@phosphor-icons/react";
import { useJob } from "../api/hooks";
import {
  bytes,
  time,
  post,
  statusText,
  phaseText,
  terminal,
} from "../api/client";
import type { Task } from "../api/client";
import { Badge, Empty, ErrorNotice, Refresh } from "../components";
import { OperatorList } from "../OperatorList";

const actionReason: Record<string, string> = {
  REMOTE_JOB_REQUIRED: "当前不是可恢复的远端任务",
  TASK_NOT_RECOVERABLE: "只有失败或中断的原任务可以补取",
  RELEASE_NOT_CONFIRMED: "实例释放尚未确认",
  REMOTE_RESULT_NOT_LOCATED: "尚无可定位的远端结果",
  INSTANCE_CHANGED: "部署实例已改变",
  RESULTS_ALREADY_INSTALLED: "结果已经安装，无需再次补取",
};
const format = (value: number | null | undefined, digits = 2) =>
  value == null ? "—" : value.toFixed(digits);
export function JobDetail() {
  const { id = "" } = useParams();
  const job = useJob(id);
  const [tab, setTab] = useState("cases"),
    [search, setSearch] = useState(""),
    [filter, setFilter] = useState("");
  const [onlyUnstable, setOnlyUnstable] = useState(false);
  const [busy, setBusy] = useState(false),
    [error, setError] = useState<unknown>(),
    [recoverOpen, setRecoverOpen] = useState(false),
    [ack, setAck] = useState(false);
  const deferred = useDeferredValue(search.toLowerCase());
  const task = job.task,
    results = job.results;
  const action = async (type: "cancel" | "recover") => {
    if (busy) return;
    setBusy(true);
    setError(undefined);
    try {
      await post<Task>(
        `/api/v1/evaluations/${id}/${type}`,
        type === "recover" ? { acknowledge_recovery: true } : {},
      );
      await job.refresh();
      if (type === "recover") {
        job.restart();
        setRecoverOpen(false);
        setAck(false);
      }
    } catch (error) {
      setError(error);
    } finally {
      setBusy(false);
    }
  };
  if (!task)
    return (
      <>
        <header className="page-heading">
          <h1>任务详情</h1>
        </header>
        <ErrorNotice error={job.error} />
        <Empty>
          {job.error
            ? "任务读取失败，请检查任务地址与服务状态。"
            : "正在读取任务…"}
        </Empty>
        <Refresh onClick={() => void job.refresh()} />
      </>
    );
  const cases = (results?.cases || []).filter(
    (item) =>
      (!onlyUnstable ||
        results?.performance?.unstable_cases.includes(item.case_code)) &&
      (!filter || item.status === filter) &&
      (!deferred ||
        `${item.operator} ${item.case_code}`.toLowerCase().includes(deferred)),
  );
  const score = results?.score;
  const prefix = `/api/v1/evaluations/${id}`;
  const names = new Set(job.artifacts.map((item) => item.name));
  const download = (name: string) =>
    prefix + "/artifacts/" + encodeURIComponent(name);
  const passed = results?.status_counts.PASS || 0;
  const performance = results?.performance;
  const geometric = performance?.geometric_speedup;
  const released = task.remote?.release_state === "SHUTDOWN";
  return (
    <>
      <Link className="back-link" to="/workspace/jobs">
        <ArrowLeft size={16} />
        返回任务列表
      </Link>
      <header className="page-heading detail-heading">
        <div>
          <h1>
            评测结果 <Badge status={task.status} />
          </h1>
          <p className="mono task-id">{id}</p>
        </div>
        <div className="actions">
          <Refresh onClick={() => void job.refresh()} />
          {task.available_actions.cancel.enabled ? (
            <button
              className="button"
              disabled={busy}
              onClick={() => void action("cancel")}
            >
              {task.cancel_requested ? "正在取消…" : "取消任务"}
            </button>
          ) : null}
        </div>
      </header>
      <ErrorNotice error={error || job.error} />
      {task.error ? (
        <div className="notice danger" role="alert">
          <WarningCircle size={22} />
          <div>
            <strong>{task.error}</strong>
            <p>
              任务未正常完成，已产生的用例结果与产物仍会保留。参考实现或环境异常需由评测侧排查。
            </p>
          </div>
        </div>
      ) : null}
      <section className="result-verdict">
        <div>
          <p className="eyebrow">
            {task.operators.length === 1
              ? task.operators[0]
              : `${task.operators.length} 个算子`}{" "}
            · {task.profile === "smoke" ? "功能检查" : "性能评测"}
          </p>
          <h2>
            {task.status === "COMPLETED"
              ? passed === task.planned_cases
                ? "全部用例通过"
                : "部分用例未通过"
              : terminal.has(task.status)
                ? "评测未完成"
                : "评测进行中"}
          </h2>
          <p>
            {passed} / {task.planned_cases} 用例通过
            {task.status === "COMPLETED" ? "。" : "，结果会自动更新。"}
          </p>
        </div>
        {names.has("report.html") ? (
          <a className="button primary" href={download("report.html")}>
            <DownloadSimple size={18} />
            下载评测报告
          </a>
        ) : null}
      </section>
      {task.operators.length > 1 ? (
        <details className="operator-roster">
          <summary>查看 {task.operators.length} 个算子</summary>
          <OperatorList names={task.operators} />
        </details>
      ) : null}
      <details className="technical-details" open={!terminal.has(task.status)}>
        <summary>
          {terminal.has(task.status) ? "运行记录与资源状态" : "查看运行进度"}
        </summary>
        <section className="job-progress">
          <div className="split-line">
            <strong>
              {task.remote?.phase
                ? phaseText[task.remote.phase] || task.remote.phase
                : statusText[task.status]}
            </strong>
            <span>
              {task.completed_cases} / {task.planned_cases} 用例
            </span>
          </div>
          <progress
            aria-label="用例完成进度"
            value={task.completed_cases}
            max={task.planned_cases}
          />
          <div className="split-line muted">
            <span>
              {task.completed_cases === task.planned_cases &&
              !terminal.has(task.status)
                ? "用例已跑完，正在计算成绩并整理报告。"
                : "按用例数量统计进度。"}
            </span>
            <span>{job.streamStatus}</span>
          </div>
          {task.remote ? (
            <p
              className={
                !released && terminal.has(task.status) ? "error-text" : "muted"
              }
            >
              资源：
              {released
                ? "已确认释放（SHUTDOWN）"
                : task.remote.release_state === "NOT_ACQUIRED"
                  ? "尚未申请"
                  : terminal.has(task.status)
                    ? "释放尚未确认"
                    : "使用中，最终释放待确认"}{" "}
              · 产物：
              {task.remote.artifacts_verified ? "已核验" : "尚未完成核验"}
            </p>
          ) : null}
        </section>
      </details>
      <nav className="tabs" aria-label="任务详情页签">
        {[
          ["cases", "成绩与用例"],
          ["events", "诊断日志"],
          ["artifacts", "下载文件"],
          ["info", "任务信息"],
        ].map(([value, label]) => (
          <button
            key={value}
            aria-current={tab === value ? "page" : undefined}
            className={tab === value ? "active" : ""}
            onClick={() => setTab(value)}
          >
            {label}
          </button>
        ))}
      </nav>
      {tab === "cases" ? (
        <>
          <section className="score-strip">
            <div>
              <span>功能得分</span>
              <strong>{score ? format(score.coverage_ops_core) : "—"}</strong>
            </div>
            <div>
              <span>严格通过率</span>
              <strong>
                {score ? `${format(score.coverage_strict)}%` : "—"}
              </strong>
            </div>
            <div>
              <span>严重错误数</span>
              <strong>{score ? score.severe_count : "—"}</strong>
            </div>
            <div>
              <span>
                {task.profile === "smoke" ? "性能测量" : "实测加速比"}
              </span>
              <strong>
                {task.profile === "smoke"
                  ? "未测量"
                  : geometric != null
                    ? `${geometric.toFixed(3)}×`
                    : "暂无测量"}
              </strong>
            </div>
          </section>
          <p className="muted score-note">
            {!score
              ? "任务完成后发布最终成绩。"
              : task.profile === "smoke"
                ? "本次为功能检查，未测量速度。"
                : "以系统参考实现的速度为 1×，大于 1× 更快，小于 1× 更慢。"}
          </p>
          {task.profile === "standard" ? (
            <details className="technical-details">
              <summary>性能分与统计口径</summary>
              <p>
                本任务性能分：{format(score?.perf_score)}。100
                分以本任务内首次全对记录为基准，不表示比参考实现更快，不能直接用于跨任务排名。每个用例的测速与原始样本见下表。没有完整、有效的测量时不补零分。
              </p>
              <p>
                实测加速比是 {performance?.measured_cases || 0}{" "}
                个有效用例的几何平均；在下表展开用例可查看原始测速样本。
              </p>
            </details>
          ) : null}
          {task.profile === "standard" && performance?.unstable_cases.length ? (
            <div className="notice neutral">
              <div>
                <strong>
                  {performance.unstable_cases.length} 个用例的测速波动较大
                </strong>
                <p>
                  同一个用例多次运行的耗时差别较大。比较优化效果前，建议再测一次。
                </p>
              </div>
            </div>
          ) : null}
          {task.profile === "standard" && performance?.missing_cases ? (
            <p className="error-text">
              {performance.missing_cases}{" "}
              个通过用例缺少有效测速，加速比仅覆盖已测用例。
            </p>
          ) : null}
          {task.status === "COMPLETED" ? (
            <p className="board-result-link">
              <Link className="text-link" to={`/leaderboard?task=${id}`}>
                查看同类成绩排名
              </Link>
            </p>
          ) : null}
          <div className="toolbar">
            {results?.performance?.unstable_cases.length ? (
              <button
                className="button"
                aria-pressed={onlyUnstable}
                onClick={() => setOnlyUnstable((value) => !value)}
              >
                {onlyUnstable ? "显示全部用例" : "只看测速波动"}
              </button>
            ) : null}
            <input
              aria-label="搜索用例"
              placeholder="搜索算子或用例编号"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
            <select
              aria-label="筛选用例状态"
              value={filter}
              onChange={(event) => setFilter(event.target.value)}
            >
              <option value="">全部用例状态</option>
              {Object.keys(results?.status_counts || {}).map((status) => (
                <option key={status} value={status}>
                  {statusText[status] || status}
                </option>
              ))}
            </select>
            <span className="muted">
              当前结果 {cases.length} 条 · 计入评分{" "}
              {results?.counted_cases || 0} 条
            </span>
          </div>
          {cases.length ? (
            <div className="table-wrap">
              <table>
                <thead>
                  <tr>
                    <th>算子 / 用例</th>
                    <th>结果</th>
                    <th>
                      {task.profile === "standard"
                        ? "提交耗时 / 参考耗时"
                        : "检查耗时"}
                    </th>
                    {task.profile === "standard" ? <th>加速比</th> : null}
                    <th>最大绝对误差</th>
                    <th>详情</th>
                  </tr>
                </thead>
                <tbody>
                  {cases.map((item) => (
                    <tr key={item.case_code}>
                      <td>
                        <strong>{item.operator}</strong>
                        <small className="mono">{item.case_code}</small>
                      </td>
                      <td>
                        <Badge status={item.status} />
                        {performance?.unstable_cases.includes(
                          item.case_code,
                        ) ? (
                          <small className="error-text">测速波动较大</small>
                        ) : null}
                      </td>
                      <td>
                        {task.profile === "standard"
                          ? `${format(item.perf_median_ms, 6)} / ${format(item.baseline_median_ms, 6)} ms`
                          : `${format(item.duration_ms)} ms`}
                      </td>
                      {task.profile === "standard" ? (
                        <td>{format(item.speedup, 3)}×</td>
                      ) : null}
                      <td>
                        {item.max_abs_err == null
                          ? "—"
                          : item.max_abs_err.toExponential(3)}
                      </td>
                      <td>
                        <details className="case-details">
                          <summary>展开</summary>
                          <div>
                            <p>相对误差：{item.max_rel_err ?? "—"}</p>
                            <p>
                              测速变异系数：
                              {item.perf_cv == null
                                ? "未测量"
                                : `${(item.perf_cv * 100).toFixed(2)}%`}
                            </p>
                            <p>
                              性能中位数：{format(item.perf_median_ms, 6)} ms ·
                              加速比：{format(item.speedup, 3)}
                            </p>
                            {item.compile_err ? (
                              <pre>{item.compile_err}</pre>
                            ) : null}
                            {item.samples_ms?.length ? (
                              <pre>
                                每轮测量中位数（ms）：
                                {JSON.stringify(item.samples_ms)}
                              </pre>
                            ) : null}
                            {item.log_path && names.has(item.log_path) ? (
                              <a
                                className="text-link"
                                href={download(item.log_path)}
                              >
                                下载用例日志
                              </a>
                            ) : (
                              <p className="muted">完整日志尚不可用</p>
                            )}
                          </div>
                        </details>
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          ) : (
            <Empty>
              {search || filter
                ? "当前结果中没有匹配的用例。"
                : "尚未收到用例结果；运行后会在这里逐步显示。"}
            </Empty>
          )}
        </>
      ) : null}
      {tab === "events" ? (
        <>
          <div className="split-line">
            <p className="muted">
              显示最近 {job.events.length} 条持久事件；完整日志在产物中下载。
            </p>
            <button className="button quiet" onClick={job.restart}>
              重新同步事件
            </button>
          </div>
          <div className="event-list">
            {job.events.map((event) => (
              <details key={event.sequence}>
                <summary>
                  <span className="mono">#{event.sequence}</span>
                  <time>{time(event.time)}</time>
                  <strong>{event.kind || "state"}</strong>
                  <span>{statusText[event.status] || event.status}</span>
                </summary>
                <pre>{JSON.stringify(event, null, 2)}</pre>
              </details>
            ))}
          </div>
          {!job.events.length ? <Empty>暂无事件</Empty> : null}
        </>
      ) : null}
      {tab === "artifacts" ? (
        <>
          <p className="muted">
            只提供服务端核验并允许下载的产物。HTML
            报告以附件下载，不在页面执行。
          </p>
          {job.artifacts.length ? (
            <div className="artifact-list">
              {job.artifacts.map((item) => (
                <div key={item.name}>
                  <div>
                    <strong>{item.name}</strong>
                    <details>
                      <summary>文件校验值</summary>
                      <small className="mono">SHA-256 {item.sha256}</small>
                    </details>
                  </div>
                  <span>{bytes(item.bytes)}</span>
                  <a className="button" href={download(item.name)}>
                    <DownloadSimple size={18} />
                    下载
                  </a>
                </div>
              ))}
            </div>
          ) : (
            <Empty>暂无可下载产物；结果回收完成后会自动更新。</Empty>
          )}
        </>
      ) : null}
      {tab === "info" ? (
        <>
          <div className="run-summary">
            <div>
              <span>当前范围</span>
              <strong>
                {task.scope === "full"
                  ? `全部 ${task.operators.length} 个算子`
                  : task.operators.length === 1
                    ? task.operators[0]
                    : `${task.operators.length} 个算子`}
              </strong>
            </div>
            <div>
              <span>模式</span>
              <strong>
                {task.profile === "smoke" ? "功能检查" : "性能评测"}
              </strong>
            </div>
            <div>
              <span>评测集</span>
              <strong>{task.corpus_version}</strong>
            </div>
            <div>
              <span>创建时间</span>
              <strong>{time(task.created_at)}</strong>
            </div>
          </div>

          <dl className="metadata">
            {Object.entries({
              任务ID: task.id,
              提交ID: task.submission_id,
              远端Job: task.remote?.id || "无",
              语料指纹: task.manifest_hash,
              提交包指纹: task.package_sha256,
              资产指纹: task.remote?.assets_hash || "暂无",
              同步包指纹: task.remote?.bundle_sha256 || "暂无",
              执行健康: task.remote?.device_health || "NOT_PROBED",
              认证: "NOT_COMPUTED",
              全量验收: "未完成",
            }).map(([key, value]) => (
              <div key={key}>
                <dt>{key}</dt>
                <dd className="mono">{value}</dd>
              </div>
            ))}
          </dl>
          <section className="recovery">
            <h2>补取原任务结果</h2>
            <p>
              只读取原远端任务的事件与结果，不重新执行提交。此操作会启动原实例，补取完成后释放资源。
            </p>
            {!task.available_actions.recover.enabled ? (
              <p className="muted">
                {actionReason[task.available_actions.recover.reason || ""] ||
                  "当前不可补取"}
              </p>
            ) : null}
            <button
              className="button"
              disabled={!task.available_actions.recover.enabled || busy}
              onClick={() => setRecoverOpen(!recoverOpen)}
            >
              补取原任务结果
            </button>
            {recoverOpen ? (
              <div className="recovery-confirm">
                <label>
                  <input
                    type="checkbox"
                    checked={ack}
                    onChange={(event) => setAck(event.target.checked)}
                  />
                  我确认仅补取原任务，并允许启停原实例。
                </label>
                <button
                  className="button primary"
                  disabled={
                    !ack || busy || !task.available_actions.recover.enabled
                  }
                  onClick={() => void action("recover")}
                >
                  确认补取
                </button>
              </div>
            ) : null}
          </section>
        </>
      ) : null}
    </>
  );
}
