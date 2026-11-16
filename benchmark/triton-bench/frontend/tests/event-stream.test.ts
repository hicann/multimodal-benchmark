/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import { test } from "node:test";
import assert from "node:assert/strict";
import { subscribeJob } from "../src/api/event-stream.ts";
import { ApiError, request, safeCursor } from "../src/api/client.ts";

const tick = () => new Promise((resolve) => setImmediate(resolve));
const event = (sequence: number) => ({
  sequence,
  time: "2026-11-13T00:00:00Z",
  status: "RUNNING",
  kind: "state",
});
class FakeStream extends EventTarget {
  onopen: (() => void) | null = null;
  onerror: (() => void) | null = null;
  closed = false;
  close() {
    this.closed = true;
  }
  emit(value: unknown) {
    this.dispatchEvent(
      new MessageEvent("update", { data: JSON.stringify(value) }),
    );
  }
}

test("event backfill uses global cursor, deduplicates replays and closes on end", async (t) => {
  const calls: string[] = [],
    received: number[] = [];
  const batches = [
    { items: [event(3), event(7)], next_cursor: 7 },
    { items: [], next_cursor: 7 },
  ];
  t.mock.method(
    globalThis,
    "fetch",
    async (url: string, options: { method?: string }) => {
      calls.push(String(url));
      assert.equal(options?.method, undefined);
      return Response.json(batches.shift());
    },
  );
  const stream = new FakeStream();
  let sourceUrl = "",
    ended = 0;
  const stop = subscribeJob("a".repeat(32), {
    onEvent: (e) => received.push(e.sequence),
    onSignal: () => {},
    onStatus: () => {},
    onError: (e) => assert.fail(e.message),
    onEnd: () => ended++,
    createStream: (url) => {
      sourceUrl = url;
      return stream as any;
    },
  });
  await tick();
  await tick();
  assert.match(sourceUrl, /after=7$/);
  assert.equal(calls.length, 2);
  stream.emit(event(7));
  stream.emit(event(8));
  assert.deepEqual(received, [3, 7, 8]);
  stream.dispatchEvent(new Event("end"));
  assert.equal(ended, 1);
  assert.ok(stream.closed);
  stop();
  t.mock.restoreAll();
});

test("unsafe cursor aborts subscription instead of rounding and skipping events", async (t) => {
  t.mock.method(globalThis, "fetch", async () =>
    Response.json({
      items: [event(Number.MAX_SAFE_INTEGER + 1)],
      next_cursor: Number.MAX_SAFE_INTEGER + 1,
    }),
  );
  let error: Error | undefined,
    created = false;
  const stop = subscribeJob("a".repeat(32), {
    onEvent: () => assert.fail("unsafe event accepted"),
    onSignal: () => {},
    onStatus: () => {},
    onError: (e) => {
      error = e;
    },
    onEnd: () => {},
    createStream: () => {
      created = true;
      return new FakeStream() as any;
    },
  });
  await tick();
  await tick();
  assert.ok(error instanceof ApiError);
  assert.equal((error as ApiError).code, "EVENT-CURSOR");
  assert.equal(created, false);
  stop();
  t.mock.restoreAll();
});

test("repeated SSE errors fall back to GET polling and drain a terminal task", async (t) => {
  const calls: string[] = [];
  t.mock.method(
    globalThis,
    "fetch",
    async (url: string, options: { method?: string }) => {
      calls.push(String(url));
      assert.equal(options?.method, undefined);
      return Response.json(
        String(url).includes("/events?")
          ? { items: [], next_cursor: 0 }
          : { status: "CANCELLED" },
      );
    },
  );
  const stream = new FakeStream();
  let ended = 0;
  const stop = subscribeJob("b".repeat(32), {
    onEvent: () => {},
    onSignal: () => {},
    onStatus: () => {},
    onError: (e) => assert.fail(e.message),
    onEnd: () => ended++,
    createStream: () => stream as any,
  });
  await tick();
  await tick();
  stream.onerror?.();
  stream.onerror?.();
  stream.onerror?.();
  await tick();
  await tick();
  assert.ok(stream.closed);
  assert.equal(ended, 1);
  assert.ok(calls.some((url) => url.endsWith("b".repeat(32))));
  stop();
  t.mock.restoreAll();
});

test("cleanup aborts pending backfill and never creates a late subscription", async (t) => {
  let signal: AbortSignal | undefined,
    created = false;
  t.mock.method(
    globalThis,
    "fetch",
    (_url: string, options: { signal: AbortSignal }) => {
      signal = options.signal;
      return new Promise((_resolve, reject) =>
        signal?.addEventListener("abort", () =>
          reject(new DOMException("Aborted", "AbortError")),
        ),
      );
    },
  );
  const stop = subscribeJob("c".repeat(32), {
    onEvent: () => {},
    onSignal: () => {},
    onStatus: () => {},
    onError: (e) => assert.fail(e.message),
    onEnd: () => {},
    createStream: () => {
      created = true;
      return new FakeStream() as any;
    },
  });
  stop();
  await tick();
  assert.ok(signal?.aborted);
  assert.equal(created, false);
  t.mock.restoreAll();
});

test("HTTP errors retain structured code and zero cursor remains valid", async (t) => {
  t.mock.method(globalThis, "fetch", async () =>
    Response.json(
      { error: { code: "IDEMPOTENCY-CONFLICT", detail: "内容不同" } },
      { status: 409 },
    ),
  );
  await assert.rejects(
    request("/api/v1/evaluations"),
    (e) =>
      e instanceof ApiError &&
      e.code === "IDEMPOTENCY-CONFLICT" &&
      e.status === 409,
  );
  assert.equal(safeCursor(0), 0);
  assert.throws(() => safeCursor("1"), ApiError);
  t.mock.restoreAll();
});

test("late backfill ignored after cleanup even when the transport ignores abort", async (t) => {
  let resolveFetch: (response: Response) => void = () => {};
  t.mock.method(
    globalThis,
    "fetch",
    () =>
      new Promise((resolve) => {
        resolveFetch = resolve;
      }),
  );
  const stop = subscribeJob("d".repeat(32), {
    onEvent: () => assert.fail("late event accepted"),
    onSignal: () => assert.fail("late refresh"),
    onStatus: () => {},
    onError: (e) => assert.fail(e.message),
    onEnd: () => {},
    createStream: () => {
      assert.fail("late subscription");
      return new FakeStream() as any;
    },
  });
  stop();
  resolveFetch(Response.json({ items: [event(1)], next_cursor: 1 }));
  await tick();
  await tick();
  t.mock.restoreAll();
});
