/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import type { ReactNode } from "react";
import { WarningCircle, ArrowClockwise } from "@phosphor-icons/react";
import { ApiError, statusText } from "./api/client";

export function ErrorNotice({ error }: { error: unknown }) {
  if (!error) return null;
  return (
    <div className="notice danger" role="alert">
      <WarningCircle size={20} />
      <div>
        <strong>{error instanceof ApiError ? error.code : "请求未完成"}</strong>
        <p>{error instanceof Error ? error.message : String(error)}</p>
      </div>
    </div>
  );
}
export function Badge({ status }: { status: string }) {
  const tone = ["PASS", "READY"].includes(status)
    ? "good"
    : [
          "FAILED",
          "REJECTED",
          "WRONG_RESULT",
          "ORACLE_FAIL",
          "ENV_ABORT",
          "COMPILE_FAIL",
          "RUNTIME_FAIL",
          "SHAPE_MISMATCH",
        ].includes(status)
      ? "bad"
      : "neutral";
  return (
    <span className={`badge ${tone}`}>{statusText[status] || status}</span>
  );
}
export function Empty({ children }: { children: ReactNode }) {
  return <div className="empty">{children}</div>;
}
export function Refresh({ onClick }: { onClick: () => void }) {
  return (
    <button className="button quiet" onClick={onClick}>
      <ArrowClockwise size={17} />
      刷新
    </button>
  );
}
export function SectionTitle({
  children,
  description,
}: {
  children: ReactNode;
  description?: string;
}) {
  return (
    <div className="section-heading">
      <h2>{children}</h2>
      {description ? <p>{description}</p> : null}
    </div>
  );
}
