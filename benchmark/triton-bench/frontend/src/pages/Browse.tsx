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
import { Link } from "react-router-dom";
import useSWR from "swr";
import useSWRInfinite from "swr/infinite";
import { ArrowRight } from "@phosphor-icons/react";
import {
  request,
  post,
  safeCursor,
  time,
  terminal,
  bytes,
  statusText,
} from "../api/client";
import type { TaskPage, Task, Catalog } from "../api/client";
import { Badge, Empty, ErrorNotice, Refresh } from "../components";
import { ExamplePackages } from "../ExamplePackages";

const taskTitle = (task: Task) =>
  `${task.operators.length === 1 ? task.operators[0] : task.scope === "full" ? "全部算子" : `${task.operators.length} 个算子`} · ${task.profile === "smoke" ? "功能检查" : "性能评测"}`;
const taskAction = (task: Task) =>
  terminal.has(task.status) ? "查看结果" : "查看进度";

export function TaskTable({ items }: { items: Task[] }) {
  return (
    <>
      <div className="table-wrap tasks-desktop">
        <table className="tasks-table">
          <thead>
            <tr>
              <th>任务</th>
              <th>状态</th>
              <th>用例进度</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {items.map((task) => (
              <tr key={task.id}>
                <td>
                  <Link
                    className="task-title text-link"
                    to={`/workspace/jobs/${task.id}`}
                  >
                    {taskTitle(task)}
                  </Link>
                  <small>
                    {time(task.created_at)} ·{" "}
                    <span className="mono">{task.id.slice(0, 12)}</span>
                  </small>
                </td>
                <td>
                  <Badge status={task.status} />
                </td>
                <td className="task-count">
                  {task.completed_cases} / {task.planned_cases}
                </td>
                <td>
                  <Link
                    aria-label={`${taskAction(task)} ${task.id.slice(0, 12)}`}
                    className="text-link task-action"
                    to={`/workspace/jobs/${task.id}`}
                  >
                    {taskAction(task)}
                    <ArrowRight size={16} />
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div className="task-cards">
        {items.map((task) => (
          <article className="task-card" key={task.id}>
            <div className="split-line">
              <h3>
                <Link className="text-link" to={`/workspace/jobs/${task.id}`}>
                  {taskTitle(task)}
                </Link>
              </h3>
              <Badge status={task.status} />
            </div>
            <p className="muted task-card-meta">
              {time(task.created_at)} ·{" "}
              <span className="mono">{task.id.slice(0, 12)}</span>
            </p>
            <div className="split-line">
              <span>
                {task.completed_cases} / {task.planned_cases} 用例
              </span>
              <Link
                aria-label={`${taskAction(task)} ${task.id.slice(0, 12)}`}
                className="text-link task-action"
                to={`/workspace/jobs/${task.id}`}
              >
                {taskAction(task)}
                <ArrowRight size={16} />
              </Link>
            </div>
          </article>
        ))}
      </div>
    </>
  );
}
export function Jobs() {
  const { data, error, size, setSize, mutate, isValidating } =
    useSWRInfinite<TaskPage>(
      (index, previous) => {
        if (previous?.next_cursor === null) return null;
        return `/api/v1/evaluations?limit=30${index ? `&before=${safeCursor(previous!.next_cursor)}` : ""}`;
      },
      request,
      { refreshInterval: 5000, revalidateAll: false },
    );
  const [search, setSearch] = useState(""),
    [filter, setFilter] = useState("");
  const deferred = useDeferredValue(search.toLowerCase());
  const loaded = data?.flatMap((page) => page.items) || [];
  const items = loaded.filter(
    (task) =>
      (!filter || task.status === filter) &&
      `${task.id} ${task.operators.join(" ")}`.toLowerCase().includes(deferred),
  );
  const hasMore =
    !!data &&
    data[data.length - 1]?.items.length === 30 &&
    data[data.length - 1]?.next_cursor !== null;
  return (
    <>
      <header className="page-heading detail-heading">
        <div>
          <h1>评测任务</h1>
          <p>查看评测进度和结果。</p>
        </div>
        <Link className="button primary" to="/submit">
          新建评测
        </Link>
      </header>
      <ErrorNotice error={error} />
      <div className="toolbar">
        <input
          aria-label="搜索任务"
          placeholder="搜索任务 ID 或算子"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <select
          aria-label="筛选任务状态"
          value={filter}
          onChange={(event) => setFilter(event.target.value)}
        >
          <option value="">全部任务状态</option>
          {[...new Set(loaded.map((task) => task.status))].map((status) => (
            <option key={status} value={status}>
              {statusText[status] || status}
            </option>
          ))}
        </select>
        <Refresh onClick={() => void mutate()} />
      </div>
      {data ? (
        <p className="muted task-list-count" aria-live="polite">
          {search || filter
            ? `找到 ${items.length} 个任务`
            : `${loaded.length} 个任务`}
          {hasMore ? ` · 当前显示最近 ${loaded.length} 个` : ""}
          {search || filter ? (
            <button
              className="text-link reset-filter"
              onClick={() => {
                setSearch("");
                setFilter("");
              }}
            >
              重置筛选
            </button>
          ) : null}
        </p>
      ) : null}
      {items.length ? (
        <TaskTable items={items} />
      ) : (
        <Empty>
          {!data && !error ? (
            "正在读取任务…"
          ) : loaded.length ? (
            "没有匹配的任务，请换一个编号或算子名称。"
          ) : (
            <>
              暂无任务，
              <Link className="text-link" to="/submit">
                提交第一个算子实现
              </Link>
              。
            </>
          )}
        </Empty>
      )}
      {hasMore ? (
        <button
          className="button load-more"
          disabled={isValidating}
          onClick={() => void setSize(size + 1)}
        >
          {isValidating ? "加载中…" : "查看更早任务"}
        </button>
      ) : null}
    </>
  );
}
export function Overview({ catalog }: { catalog: Catalog }) {
  const { data, error } = useSWR<TaskPage>(
    "/api/v1/evaluations?limit=6",
    request,
    { refreshInterval: 10000 },
  );
  return (
    <>
      <header className="page-heading">
        <p className="eyebrow">TRITON · ASCEND</p>
        <h1>算子评测工作台</h1>
        <p>上传代码，在昇腾 NPU 上检查正确性、测量执行速度。</p>
        <Link className="button primary hero-action" to="/submit">
          开始一次评测
          <ArrowRight size={18} />
        </Link>
      </header>
      <div className="overview-stats">
        <div>
          <span>支持算子</span>
          <strong>{catalog.operator_count}</strong>
        </div>
        <div>
          <span>测试用例</span>
          <strong>{catalog.case_count.toLocaleString()}</strong>
        </div>
        <div>
          <span>评测集版本</span>
          <strong>{catalog.corpus_version.split("-")[0]}</strong>
        </div>
      </div>
      <section className="overview-section">
        <div className="split-line">
          <h2>最近任务</h2>
          <Link className="text-link" to="/workspace/jobs">
            全部任务
            <ArrowRight size={16} />
          </Link>
        </div>
        <ErrorNotice error={error} />
        {data?.items.length ? (
          <>
            {data.items.some((task) => !terminal.has(task.status)) ? (
              <p className="muted">
                {data.items.filter((task) => !terminal.has(task.status)).length}{" "}
                个任务进行中
              </p>
            ) : null}
            <TaskTable items={data.items} />
          </>
        ) : (
          <Empty>
            {data ? "暂无任务，从提交一个 ZIP 开始。" : "正在读取最近任务…"}
          </Empty>
        )}
      </section>
      <p className="overview-help">
        第一次使用？
        <Link className="text-link" to="/submit">
          下载评测包并开始
        </Link>
      </p>
    </>
  );
}
export function Benchmarks({ catalog }: { catalog: Catalog }) {
  const [search, setSearch] = useState("");
  const deferred = useDeferredValue(search.toLowerCase());
  const operators = catalog.operators.filter((item) =>
    `${item.name} ${item.signature || ""}`.toLowerCase().includes(deferred),
  );
  return (
    <>
      <header className="page-heading">
        <h1>算子目录</h1>
        <p>查找支持的算子，按接口签名编写函数，再将源码打包上传。</p>
      </header>
      <div className="corpus-intro">
        <h2>{catalog.corpus_version}</h2>
        <p>
          {catalog.operator_count} 个算子 ·{" "}
          {catalog.case_count.toLocaleString()} 个用例
        </p>
        <details>
          <summary>版本校验信息</summary>
          <small className="mono">{catalog.manifest_hash}</small>
        </details>
      </div>
      <div className="toolbar">
        <input
          aria-label="搜索算子"
          placeholder="搜索算子名称或接口签名"
          value={search}
          onChange={(event) => setSearch(event.target.value)}
        />
        <span className="muted">{operators.length} 个算子</span>
      </div>
      <div className="table-wrap">
        <table>
          <thead>
            <tr>
              <th>算子</th>
              <th>函数签名</th>
              <th>用例数</th>
              <th>评分权重</th>
            </tr>
          </thead>
          <tbody>
            {operators.map((item) => (
              <tr key={item.name}>
                <td>
                  <strong>{item.name}</strong>
                </td>
                <td>
                  <code>{item.signature || "暂无签名"}</code>
                  <small>参数：{item.params?.join(", ") || "暂无"}</small>
                </td>
                <td>{item.case_count}</td>
                <td>{item.weight}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {!operators.length ? <Empty>没有匹配的算子。</Empty> : null}
    </>
  );
}
export function Docs({ catalog }: { catalog: Catalog }) {
  const policy = catalog.submission_policy;
  const [ack, setAck] = useState(false),
    [busy, setBusy] = useState(false),
    [error, setError] = useState<unknown>();
  const resume = async () => {
    setBusy(true);
    setError(undefined);
    try {
      await post("/api/v1/worker/resume", { acknowledge_health: true });
      location.reload();
    } catch (error) {
      setError(error);
    } finally {
      setBusy(false);
    }
  };
  return (
    <>
      <header className="page-heading">
        <h1>使用指南</h1>
        <p>按以下三步完成第一次评测。技术接入与维护说明在页面底部。</p>
      </header>
      <article className="docs-content">
        <h2>1. 上传评测包</h2>
        <p>第一次使用可直接下载评测包，在“提交评测”页上传。</p>
        <ExamplePackages />
        <details className="technical-details">
          <summary>自己打包：文件结构与函数接口</summary>
          <p>
            根目录包含
            manifest.yaml，清单声明算子名称、入口函数与源码路径。只上传源码和文档；不支持
            wheel 或运行时安装包。浏览器不会执行包内代码。
          </p>
          <pre>
            {
              "manifest.yaml\noperators/\n  kernels.py\n\n# manifest.yaml 示例\nsubmit_id: my-abs\noperators:\n  - name: Abs\n    entry: abs_kernel\n    path: operators/kernels.py"
            }
          </pre>
          <p>
            函数签名须符合
            <Link className="text-link" to="/benchmarks">
              评测集接口契约
            </Link>
            。算子实现与依赖由真实 NPU 环境执行。
          </p>
        </details>
        <details className="technical-details">
          <summary>文件大小与格式限制</summary>
          <p>
            传输上限 {bytes(policy.max_request_bytes)}；解压总量{" "}
            {bytes(policy.max_total_bytes)}；单文件{" "}
            {bytes(policy.max_file_bytes)}
            ；最多 {policy.max_entries} 个条目；压缩比上限{" "}
            {policy.max_compression_ratio} 倍。
          </p>
          <p>
            允许后缀：{policy.allowed_suffixes.join("、")}
            。校验失败会保存拒收记录，但不会创建评测任务。
          </p>
          <details>
            <summary>查看服务端包级规则</summary>
            <dl className="rules">
              {policy.rules.map((rule) => (
                <div key={rule.rule_id}>
                  <dt>{rule.rule_id}</dt>
                  <dd>{rule.detail}</dd>
                </div>
              ))}
            </dl>
          </details>
        </details>
        <h2>2. 选择算子和模式</h2>
        <p>上传后默认选中包内全部算子，展开列表可取消不需要的项目。</p>
        <ul>
          <li>
            <strong>功能检查：</strong>检查计算结果，查看通过数量和错误明细。
          </li>
          <li>
            <strong>性能评测：</strong>同时测量耗时和加速比，适合比较实现速度。
          </li>
        </ul>
        <p>确认范围后，点击“创建评测任务”。</p>
        <h2>3. 查看结果</h2>
        <p>
          在“任务”中打开结果，先看通过数量，再查看未通过的用例。点击“下载评测报告”可保存完整结果。
        </p>
        <p>
          比较不同提交时，进入{" "}
          <Link className="text-link" to="/leaderboard">
            榜单
          </Link>
          。加速比大于 1× 表示比系统参考实现更快，小于 1× 表示更慢。
        </p>
        <details className="technical-details">
          <summary>连接中断或需要恢复时</summary>
          <p>
            关闭浏览器不会停止远端
            Worker。重新打开任务链接可继续查看原任务；事件断开时会重连或退回轮询，不会自动重新提交。取消后需等后端确认终止和资源释放。
          </p>
          <p>
            仅符合条件的远端失败/中断任务可补取原结果。该操作会重新启停原实例，但不重新运行提交；是否可用由服务端判断。
          </p>
        </details>
        <details className="technical-details">
          <summary>开发者 API</summary>
          <p>
            <a
              className="text-link"
              href="/openapi.json"
              target="_blank"
              rel="noreferrer"
            >
              查看当前 OpenAPI 3.1 契约
            </a>
            。上传使用原始 application/zip
            字节；上传与创建任务分别使用独立幂等键。
          </p>
        </details>
        <details className="admin-tools">
          <summary>本机运维：解除 Worker 暂停</summary>
          <p>
            只有人工检查遗留进程、实例租约与设备健康后才可操作，服务端仍会核验是否存在未释放资源。
          </p>
          <ErrorNotice error={error} />
          <label>
            <input
              type="checkbox"
              checked={ack}
              onChange={(event) => setAck(event.target.checked)}
            />
            我已确认进程和设备健康
          </label>
          <button
            className="button"
            disabled={!ack || busy || !catalog.worker_paused}
            onClick={() => void resume()}
          >
            解除暂停
          </button>
        </details>
      </article>
    </>
  );
}
