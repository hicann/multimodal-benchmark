/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import { useCallback, useEffect, useState } from "react";
import useSWR from "swr";
import { request } from "./client";
import type { Catalog, Task, Results, Artifact, PublicEvent } from "./client";
import { subscribeJob } from "./event-stream";

export const useCatalog = () =>
  useSWR<Catalog>("/api/v1/catalog", request, { refreshInterval: 15000 });
export function useJob(id: string) {
  const prefix = `/api/v1/evaluations/${id}`;
  const task = useSWR<Task>(prefix, request);
  const results = useSWR<Results>(prefix + "/results", request);
  const artifacts = useSWR<{ items: Artifact[] }>(
    prefix + "/artifacts",
    request,
  );
  const [events, setEvents] = useState<PublicEvent[]>([]);
  const [streamStatus, setStreamStatus] = useState("连接中");
  const [streamError, setStreamError] = useState<Error>();
  const [epoch, setEpoch] = useState(0);
  const { mutate: taskMutate } = task,
    { mutate: resultsMutate } = results,
    { mutate: artifactsMutate } = artifacts;
  const refresh = useCallback(
    () =>
      Promise.allSettled([taskMutate(), resultsMutate(), artifactsMutate()]),
    [taskMutate, resultsMutate, artifactsMutate],
  );
  useEffect(() => {
    setEvents([]);
    setStreamError(undefined);
    let timer: ReturnType<typeof setTimeout> | undefined;
    const stop = subscribeJob(id, {
      onEvent: (event) =>
        setEvents((current) => [...current, event].slice(-500)),
      onStatus: setStreamStatus,
      onError: setStreamError,
      onSignal: () => {
        if (!timer)
          timer = setTimeout(() => {
            timer = undefined;
            void refresh();
          }, 1000);
      },
      onEnd: () => {
        clearTimeout(timer);
        timer = undefined;
        void refresh();
      },
    });
    return () => {
      stop();
      clearTimeout(timer);
    };
  }, [id, epoch, refresh]);
  return {
    task: task.data,
    results: results.data,
    artifacts: artifacts.data?.items || [],
    events,
    streamStatus,
    error: task.error || results.error || artifacts.error || streamError,
    refresh,
    restart: () => setEpoch((current) => current + 1),
  };
}
