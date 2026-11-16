/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import { request, safeCursor, terminal, ApiError } from "./client.ts";
import type { EventBatch, PublicEvent, Task } from "./client.ts";

type Stream = Pick<
  EventSource,
  "addEventListener" | "close" | "onerror" | "onopen"
>;
type Options = {
  onEvent: (event: PublicEvent) => void;
  onSignal: () => void;
  onEnd: () => void;
  onStatus: (status: string) => void;
  onError: (error: Error) => void;
  createStream?: (url: string) => Stream;
};

// One owner per detail page. Nothing here creates or reruns an evaluation.
export function subscribeJob(id: string, options: Options): () => void {
  const prefix = `/api/v1/evaluations/${id}`;
  let cursor = 0,
    closed = false,
    source: Stream | null = null,
    failures = 0,
    polling = false;
  let timer: ReturnType<typeof setTimeout> | undefined;
  let retryAt = 0;
  const controller = new AbortController();
  const close = () => {
    closed = true;
    source?.close();
    clearTimeout(timer);
    controller.abort();
  };
  const accept = (value: unknown) => {
    if (closed) return;
    const event = value as PublicEvent;
    const next = safeCursor(event?.sequence);
    if (!next || typeof event?.status !== "string")
      throw new ApiError("EVENT-PROTOCOL", "事件格式不符合契约");
    if (next <= cursor) return;
    options.onEvent(event);
    cursor = next;
    options.onSignal();
  };
  const read = async () => {
    while (!closed) {
      const batch = await request<EventBatch>(
        `${prefix}/events?after=${cursor}`,
        { signal: controller.signal },
      );
      if (closed) return;
      const next = safeCursor(batch.next_cursor);
      if (!Array.isArray(batch.items))
        throw new ApiError("EVENT-PROTOCOL", "事件列表格式不符合契约");
      if (!batch.items.length) return;
      const previous = cursor;
      batch.items.forEach(accept);
      if (next !== cursor || cursor <= previous)
        throw new ApiError("EVENT-PROTOCOL", "事件游标未正常推进");
    }
  };
  const fatal = (error: unknown) => {
    if (closed) return;
    if (
      error instanceof ApiError &&
      ["EVENT-CURSOR", "EVENT-PROTOCOL"].includes(error.code)
    ) {
      close();
      options.onStatus("协议异常");
      options.onError(error);
    }
  };
  const poll = async () => {
    if (closed) return;
    try {
      await read();
      if (closed) return;
      options.onSignal();
      const task = await request<Task>(prefix, { signal: controller.signal });
      if (closed) return;
      if (terminal.has(task.status)) {
        close();
        options.onStatus("已同步");
        options.onEnd();
        return;
      }
      options.onStatus("轮询同步");
      if (Date.now() >= retryAt) {
        polling = false;
        connect();
        return;
      }
    } catch (error) {
      fatal(error);
      if (!closed) options.onStatus("等待重连");
    }
    if (!closed) timer = setTimeout(poll, 3000);
  };
  const fallback = () => {
    source?.close();
    source = null;
    if (closed || polling) return;
    polling = true;
    retryAt = Date.now() + 30000;
    void poll();
  };
  const connect = () => {
    if (closed) return;
    clearTimeout(timer);
    failures = 0;
    options.onStatus("连接中");
    source = (options.createStream || ((url) => new EventSource(url)))(
      `${prefix}/stream?after=${cursor}`,
    );
    source.onopen = () => {
      failures = 0;
      options.onStatus("实时同步");
    };
    source.addEventListener("update", (raw: Event) => {
      if (closed) return;
      try {
        accept(JSON.parse((raw as MessageEvent).data));
      } catch (error) {
        fatal(
          error instanceof SyntaxError
            ? new ApiError("EVENT-PROTOCOL", "事件 JSON 无法解析")
            : error,
        );
      }
    });
    source.addEventListener("end", () => {
      if (!closed) {
        close();
        options.onStatus("已同步");
        options.onEnd();
      }
    });
    source.onerror = () => {
      if (!closed) {
        options.onStatus("等待重连");
        if (++failures >= 3) fallback();
      }
    };
  };
  void read()
    .then(() => {
      if (!closed) connect();
    })
    .catch((error) => {
      fatal(error);
      if (!closed) fallback();
    });
  return close;
}
