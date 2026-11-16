/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import type { components } from "./schema";

export type Catalog = Required<components["schemas"]["Catalog"]>;
export type Submission = Required<components["schemas"]["Submission"]>;
export type Task = Required<components["schemas"]["Task"]>;
export type Results = Required<components["schemas"]["Results"]>;
export type Artifact = NonNullable<
  components["schemas"]["Artifacts"]["items"]
>[number];
export type PublicEvent = components["schemas"]["PublicEvent"];
export type CaseResult = components["schemas"]["CaseResult"];
export type TaskRequest = components["schemas"]["TaskRequest"];
export type TaskPage = { items: Task[]; next_cursor: number | null };
export type EventBatch = { items: PublicEvent[]; next_cursor: number };
export const terminal = new Set([
  "COMPLETED",
  "REJECTED",
  "FAILED",
  "CANCELLED",
  "INTERRUPTED",
]);

export class ApiError extends Error {
  code: string;
  status: number;
  constructor(code: string, detail: string, status = 0) {
    super(detail);
    this.code = code;
    this.status = status;
  }
}
function failure(value: unknown, status: number): ApiError {
  const error = (value as { error?: { code?: string; detail?: string } })
    ?.error;
  return new ApiError(
    error?.code || "HTTP-ERROR",
    error?.detail || `请求失败（${status}）`,
    status,
  );
}
export async function request<T>(
  path: string,
  options: RequestInit = {},
): Promise<T> {
  let response: Response;
  try {
    response = await fetch(path, options);
  } catch (error) {
    if ((error as Error).name === "AbortError") throw error;
    throw new ApiError("NETWORK", "无法连接评测服务，请检查本机入口与后端状态");
  }
  const value = await response.json().catch(() => {
    throw new ApiError("RESPONSE", "服务返回了无法解析的响应");
  });
  if (!response.ok) throw failure(value, response.status);
  return value as T;
}
export function post<T>(path: string, body: unknown, key?: string) {
  return request<T>(path, {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      ...(key ? { "Idempotency-Key": key } : {}),
    },
    body: JSON.stringify(body),
  });
}
export function upload(
  file: File,
  key: string,
  progress: (value: number) => void,
  signal: AbortSignal,
): Promise<Submission> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    const abort = () => xhr.abort();
    xhr.open("POST", "/api/v1/submissions");
    xhr.setRequestHeader("Content-Type", "application/zip");
    xhr.setRequestHeader("Idempotency-Key", key);
    xhr.timeout = 120000;
    xhr.upload.onprogress = (event) => {
      if (event.lengthComputable) progress(event.loaded / event.total);
    };
    xhr.onload = () => {
      try {
        const value = JSON.parse(xhr.responseText);
        if (xhr.status >= 200 && xhr.status < 300) resolve(value);
        else reject(failure(value, xhr.status));
      } catch {
        reject(new ApiError("RESPONSE", "上传响应无法解析，请用相同文件重试"));
      }
    };
    xhr.onerror = () =>
      reject(new ApiError("NETWORK", "上传连接中断，可以安全重试同一文件"));
    xhr.ontimeout = () =>
      reject(new ApiError("TIMEOUT", "上传或校验超时，可以重试同一文件"));
    xhr.onabort = () => reject(new DOMException("Aborted", "AbortError"));
    xhr.onloadend = () => signal.removeEventListener("abort", abort);
    signal.addEventListener("abort", abort, { once: true });
    if (signal.aborted) {
      reject(new DOMException("Aborted", "AbortError"));
      return;
    }
    xhr.send(file);
  });
}
export function safeCursor(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0)
    throw new ApiError(
      "EVENT-CURSOR",
      "事件游标超出浏览器安全范围，请升级接口契约",
    );
  return value;
}
export const bytes = (n: number) =>
  n >= 1048576
    ? `${(n / 1048576).toFixed(1)} MiB`
    : `${(n / 1024).toFixed(1)} KiB`;
export const time = (value: string) =>
  new Date(value).toLocaleString("zh-CN", { hour12: false });
export const statusText: Record<string, string> = {
  QUEUED: "排队中",
  RUNNING: "运行中",
  SCORING: "评分中",
  REPORTING: "生成报告",
  COMPLETED: "已结束",
  FAILED: "评测失败",
  REJECTED: "已拒收",
  CANCELLED: "已取消",
  INTERRUPTED: "已中断",
  PASS: "通过",
  COMPILE_FAIL: "编译失败",
  RUNTIME_FAIL: "运行失败",
  WRONG_RESULT: "结果错误",
  SHAPE_MISMATCH: "输出不匹配",
  SPEC_ERROR: "用例定义异常",
  ORACLE_FAIL: "参考实现异常",
  ENV_ABORT: "环境异常",
  SUBMISSION_INVALID: "提交不合规",
};
export const phaseText: Record<string, string> = {
  queued: "等待调度",
  preparing: "准备资产",
  starting: "启动环境",
  connecting: "建立连接",
  syncing: "传输资产",
  running: "执行用例",
  scoring: "计算分数",
  reporting: "生成报告",
  collecting: "回收核验产物",
  releasing: "释放资源",
  recovering: "补取原任务结果",
  completed: "已完成",
  failed: "已失败",
  interrupted: "已中断",
  cancelled: "已取消",
};
