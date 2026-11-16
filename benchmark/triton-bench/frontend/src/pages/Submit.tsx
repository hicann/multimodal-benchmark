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
import { useNavigate, Link } from "react-router-dom";
import {
  Check,
  CheckCircle,
  FileZip,
  UploadSimple,
} from "@phosphor-icons/react";
import { upload, post, request, bytes, ApiError } from "../api/client";
import type { Catalog, Submission, Task, TaskRequest } from "../api/client";
import { ErrorNotice, SectionTitle } from "../components";
import { ExamplePackages } from "../ExamplePackages";

const SAVED = "triton-eval:submission:v1";
export function Submit({ catalog }: { catalog: Catalog }) {
  const navigate = useNavigate(),
    input = useRef<HTMLInputElement>(null);
  const [submission, setSubmission] = useState<Submission>(),
    [filename, setFilename] = useState("");
  const [operators, setOperators] = useState<string[]>([]),
    [profile, setProfile] = useState<"smoke" | "standard">(
      catalog.default_profile,
    );
  const [scope, setScope] = useState<"operators" | "full">(
    catalog.default_scope,
  );
  const [progress, setProgress] = useState(0),
    [uploading, setUploading] = useState(false),
    [creating, setCreating] = useState(false);
  const [error, setError] = useState<unknown>(),
    [advanced, setAdvanced] = useState(false);
  const fileRef = useRef<File | undefined>(undefined),
    uploadKey = useRef(""),
    controller = useRef<AbortController | undefined>(undefined);
  const revision = useRef(0);
  const taskKey = useRef({ signature: "", key: "" }),
    inFlight = useRef(false);
  useEffect(() => {
    let alive = true;
    const initialRevision = revision.current;
    try {
      const saved = JSON.parse(sessionStorage.getItem(SAVED) || "null");
      if (
        saved &&
        /^[0-9a-f]{32}$/.test(saved.id) &&
        typeof saved.filename === "string"
      )
        void request<Submission>(`/api/v1/submissions/${saved.id}`)
          .then((value) => {
            if (alive && revision.current === initialRevision) {
              setSubmission(value);
              setFilename(saved.filename);
              setOperators(value.operators);
            }
          })
          .catch(() => {
            if (alive && revision.current === initialRevision)
              try {
                sessionStorage.removeItem(SAVED);
              } catch {
                /* storage is optional */
              }
          });
    } catch {
      /* storage is optional */
    }
    return () => {
      alive = false;
      controller.current?.abort();
    };
  }, []);
  const counts = new Map(
    catalog.operators.map((operator) => [operator.name, operator.case_count]),
  );
  const planned =
    scope === "full"
      ? catalog.case_count
      : operators.reduce((sum, name) => sum + (counts.get(name) || 0), 0);
  const ready = submission?.status === "READY";
  const disabled = !catalog.worker_enabled || catalog.worker_paused;
  const runUpload = async (file: File, retry = false) => {
    if (inFlight.current) return;
    setError(undefined);
    if (!file.name.toLowerCase().endsWith(".zip")) {
      setError(new ApiError("FILE-TYPE", "请选择 ZIP 源码包"));
      return;
    }
    if (!file.size || file.size > catalog.submission_policy.max_request_bytes) {
      setError(
        new ApiError(
          "REQUEST-SIZE",
          `ZIP 须非空，且不超过 ${bytes(catalog.submission_policy.max_request_bytes)}`,
        ),
      );
      return;
    }
    revision.current++;
    try {
      sessionStorage.removeItem(SAVED);
    } catch {
      /* storage is optional */
    }
    controller.current?.abort();
    controller.current = new AbortController();
    if (!retry) uploadKey.current = crypto.randomUUID();
    fileRef.current = file;
    setFilename(file.name);
    setSubmission(undefined);
    setOperators([]);
    setProgress(0);
    setUploading(true);
    inFlight.current = true;
    try {
      const value = await upload(
        file,
        uploadKey.current,
        setProgress,
        controller.current.signal,
      );
      setSubmission(value);
      setOperators(value.operators);
      try {
        sessionStorage.setItem(
          SAVED,
          JSON.stringify({ id: value.id, filename: file.name }),
        );
      } catch {
        /* optional */
      }
    } catch (error) {
      if ((error as Error).name !== "AbortError") setError(error);
    } finally {
      inFlight.current = false;
      setUploading(false);
    }
  };
  const create = async () => {
    if (!ready || !planned || disabled || inFlight.current) return;
    inFlight.current = true;
    setCreating(true);
    setError(undefined);
    const body: TaskRequest = {
      submission_id: submission.id,
      scope,
      profile,
      ...(scope === "operators" ? { operators: [...operators].sort() } : {}),
    };
    const signature = JSON.stringify(body);
    if (signature !== taskKey.current.signature)
      taskKey.current = { signature, key: crypto.randomUUID() };
    try {
      const latest = await request<Catalog>("/api/v1/catalog");
      if (!latest.worker_enabled || latest.worker_paused)
        throw new ApiError("WORKER-NOT-READY", "评测服务暂不可用，请稍后再试");
      const task = await post<Task>(
        "/api/v1/evaluations",
        body,
        taskKey.current.key,
      );
      navigate(`/workspace/jobs/${task.id}`);
    } catch (error) {
      setError(error);
    } finally {
      inFlight.current = false;
      setCreating(false);
    }
  };
  return (
    <>
      <header className="page-heading">
        <h1>提交评测</h1>
        <p>上传 ZIP → 选择算子和模式 → 在任务页查看结果。</p>
      </header>
      <div className="submission-grid">
        <ol className="steps" aria-label="提交流程">
          {["上传实现", "确认评测", "查看任务"].map((label, index) => (
            <li
              key={label}
              className={`${ready && index === 0 ? "done" : ""} ${index === (ready ? 1 : 0) ? "current" : ""}`}
            >
              <span className="step-mark">
                {ready && index === 0 ? (
                  <Check size={18} weight="bold" />
                ) : (
                  index + 1
                )}
              </span>
              <div>
                <strong>
                  {index + 1}. {label}
                </strong>
                <small>
                  {ready && index === 0
                    ? "已完成"
                    : index === (ready ? 1 : 0)
                      ? "当前步骤"
                      : "待完成"}
                </small>
              </div>
            </li>
          ))}
        </ol>
        <div className="submission-content">
          <ErrorNotice error={error} />
          <section className="form-section">
            <SectionTitle
              description={`ZIP 根目录需有 manifest.yaml 和算子源码，最大 ${bytes(catalog.submission_policy.max_request_bytes)}。`}
            >
              1. 上传代码
            </SectionTitle>
            <input
              ref={input}
              className="sr-only"
              type="file"
              accept=".zip,application/zip"
              aria-label="选择 ZIP 文件"
              onChange={(event) => {
                const file = event.target.files?.[0];
                if (file) void runUpload(file);
                event.target.value = "";
              }}
            />
            {filename ? (
              <div className="file-row">
                <FileZip size={34} weight="light" />
                <strong className="filename">{filename}</strong>
                {ready ? (
                  <span className="success-text">
                    <CheckCircle weight="fill" size={22} />
                    校验通过
                  </span>
                ) : (
                  <span
                    className={
                      submission?.status === "REJECTED" ? "error-text" : "muted"
                    }
                    role="status"
                  >
                    {uploading
                      ? progress < 1
                        ? `上传中 ${Math.round(progress * 100)}%`
                        : "服务端校验中"
                      : submission
                        ? "未通过校验"
                        : "尚未完成校验"}
                  </span>
                )}
                <button
                  className="button file-change"
                  disabled={uploading || creating}
                  onClick={() => input.current?.click()}
                >
                  重新选择
                </button>
              </div>
            ) : (
              <button
                className="upload-area"
                onClick={() => input.current?.click()}
              >
                <UploadSimple size={32} />
                <strong>选择 ZIP 源码包</strong>
                <span>选择文件后自动检查，不会立即启动评测</span>
              </button>
            )}
            {uploading ? (
              <progress aria-label="文件发送进度" value={progress} max={1} />
            ) : null}
            {!uploading && !submission && fileRef.current && error ? (
              <button
                className="button"
                onClick={() => void runUpload(fileRef.current!, true)}
              >
                重试同一文件
              </button>
            ) : null}
            {submission?.errors.length ? (
              <div className="validation-errors" role="alert">
                {submission.errors.map((item, index) => (
                  <div key={index}>
                    <strong>代码包检查未通过</strong>
                    <p>
                      {item.stage === "package"
                        ? "请检查文件是否为有效 ZIP，根目录是否包含 manifest.yaml，并确认大小和格式符合要求。"
                        : "请检查清单中的算子名称、源码路径和函数签名，修改后重新上传。"}
                    </p>
                    <details>
                      <summary>查看具体错误</summary>
                      <p>{item.detail}</p>
                      <small>
                        {item.rule_id} · {item.stage} · {item.operator}
                        {item.lineno ? ` · 第 ${item.lineno} 行` : ""}
                      </small>
                    </details>
                  </div>
                ))}
              </div>
            ) : null}
          </section>
          {ready ? (
            <>
              <section className="form-section">
                <SectionTitle description="默认选中包内全部算子，可取消不需要评测的项目。">
                  2. 选择算子与模式
                </SectionTitle>
                <details
                  className="operator-selector"
                  key={submission.id}
                  open={submission.operators.length <= 6 ? true : undefined}
                >
                  <summary>
                    已选 {operators.length}/{submission.operators.length} 个算子
                    · {planned} 用例（展开修改）
                  </summary>
                  <div className="operator-actions">
                    <button
                      className="button quiet"
                      disabled={creating || scope === "full"}
                      onClick={() => setOperators(submission.operators)}
                    >
                      全选
                    </button>
                    <button
                      className="button quiet"
                      disabled={creating || scope === "full"}
                      onClick={() => setOperators([])}
                    >
                      清空选择
                    </button>
                  </div>
                  <div className="operator-grid">
                    {submission.operators.map((name) => (
                      <label
                        key={name}
                        className={`operator-choice ${operators.includes(name) ? "selected" : ""}`}
                      >
                        <input
                          type="checkbox"
                          checked={operators.includes(name)}
                          disabled={creating || scope === "full"}
                          onChange={() =>
                            setOperators((current) =>
                              current.includes(name)
                                ? current.filter((value) => value !== name)
                                : [...current, name],
                            )
                          }
                        />
                        <strong>{name}</strong>
                        <span className="choice-divider" />
                        <span>{counts.get(name) || 0} 个测试用例</span>
                      </label>
                    ))}
                  </div>
                </details>
              </section>
              <section className={`form-section ${!ready ? "inactive" : ""}`}>
                <SectionTitle>选择评测模式</SectionTitle>
                {catalog.profile_capabilities.map((item) => (
                  <label
                    className={`profile-choice ${profile === item.id ? "selected" : ""}`}
                    key={item.id}
                  >
                    <input
                      type="radio"
                      name="profile"
                      value={item.id}
                      checked={profile === item.id}
                      disabled={!ready || creating}
                      onChange={() => setProfile(item.id)}
                    />
                    <div>
                      <strong>
                        {item.id === "smoke" ? "功能检查" : "性能评测"}
                      </strong>
                      <p>
                        {item.performance_scoring
                          ? "检查计算结果，同时测量运行耗时和加速比。"
                          : "检查计算结果，查看通过数量和错误明细。"}
                      </p>
                    </div>
                  </label>
                ))}
              </section>
              <section className="confirm-section">
                <h2>确认创建</h2>
                <div className="confirm-strip">
                  <strong>
                    {scope === "full"
                      ? "全部算子"
                      : operators.length > 3
                        ? `${operators.length} 个算子`
                        : operators.join("、") || "尚未选择算子"}{" "}
                    · {scope === "full" ? "全量评测" : "专项评测"} · {planned}{" "}
                    用例 · {profile === "smoke" ? "功能检查" : "性能评测"}
                  </strong>
                  <button
                    className="button primary"
                    disabled={
                      !ready || !planned || disabled || uploading || creating
                    }
                    onClick={() => void create()}
                  >
                    {creating ? "正在创建…" : "创建评测任务"}
                  </button>
                </div>
                {disabled ? (
                  <p className="notice-inline">
                    {catalog.worker_paused
                      ? "执行服务已暂停，请联系管理员检查后恢复。"
                      : "执行服务未开启：代码检查已完成，暂时无法启动真机。请联系管理员开启执行服务。"}
                  </p>
                ) : null}
                <Link className="text-link" to="/docs">
                  使用帮助
                </Link>
                <details
                  open={advanced}
                  onToggle={(event) => setAdvanced(event.currentTarget.open)}
                  className="advanced"
                >
                  <summary>高级范围设置</summary>
                  <label>
                    评测范围{" "}
                    <select
                      aria-label="评测范围"
                      value={scope}
                      disabled={!ready || creating}
                      onChange={(event) =>
                        setScope(event.target.value as "operators" | "full")
                      }
                    >
                      <option value="operators">专项评测</option>
                      <option value="full">全部算子</option>
                    </select>
                  </label>
                  {scope === "full" ? (
                    <p className="error-text">
                      将运行全部 {catalog.operator_count} 个算子的{" "}
                      {catalog.case_count} 个用例。包内未提供的算子会记为缺失。
                    </p>
                  ) : null}
                </details>
              </section>
              <details className="technical-details">
                <summary>评测集版本</summary>
                <p>
                  {catalog.corpus_version} · {catalog.operator_count} 个算子 ·{" "}
                  {catalog.case_count} 个用例
                </p>
              </details>
            </>
          ) : (
            <div className="upload-help">
              <p>
                {submission?.status === "REJECTED"
                  ? "修改代码包后，点击“重新选择”再次上传。也可以下载示例对照文件结构。"
                  : "还没有提交包？下载下方多算子包，可直接评测，也可修改实现后重新打包。"}
              </p>
              <ExamplePackages />
              <Link className="text-link" to="/docs">
                查看打包方法
              </Link>
            </div>
          )}
        </div>
      </div>
    </>
  );
}
