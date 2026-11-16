/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import { BrowserRouter, NavLink, Route, Routes, Link } from "react-router-dom";
import { SWRConfig } from "swr";
import {
  House,
  Trophy,
  Stack,
  UploadSimple,
  ListBullets,
  BookOpen,
  ArrowUpRight,
} from "@phosphor-icons/react";
import { useCatalog } from "./api/hooks";
import { Submit } from "./pages/Submit";
import { Leaderboard } from "./pages/Leaderboard";
import { JobDetail } from "./pages/JobDetail";
import { Benchmarks, Docs, Jobs, Overview } from "./pages/Browse";
import { Empty, ErrorNotice, Refresh } from "./components";

const navigation = [
  { to: "/", label: "概览", icon: House },
  { to: "/benchmarks", label: "算子目录", icon: Stack },
  { to: "/submit", label: "提交评测", icon: UploadSimple },
  { to: "/workspace/jobs", label: "任务", icon: ListBullets },
  { to: "/leaderboard", label: "榜单", icon: Trophy },
  { to: "/docs", label: "使用指南", icon: BookOpen },
];
function Workspace() {
  const { data: catalog, error, mutate } = useCatalog();
  return (
    <div className="workspace">
      <aside className="sidebar">
        <Link className="brand" to="/">
          <img src="/assets/mark.svg" alt="" />
          Triton Eval
        </Link>
        <nav aria-label="主导航">
          {navigation.map(({ to, label, icon: Icon }) => (
            <NavLink
              to={to}
              end={to === "/"}
              key={to}
              className={({ isActive }) =>
                isActive ? "nav-item active" : "nav-item"
              }
            >
              <Icon size={24} weight="regular" />
              {label}
            </NavLink>
          ))}
        </nav>
        <div className="sidebar-footer">
          <span>TRITON · ASCEND</span>
          <p>本机评测工作台</p>
          <Link to="/docs">
            接入与使用说明
            <ArrowUpRight size={15} />
          </Link>
        </div>
      </aside>
      <main className="main-content">
        <div className="service-status" role="status">
          <span
            className={`service-dot ${error || !catalog ? "offline" : catalog.worker_paused || !catalog.worker_enabled ? "idle" : "online"}`}
          />
          {error
            ? "评测服务不可达"
            : !catalog
              ? "正在连接评测服务"
              : catalog.worker_paused
                ? "执行服务已暂停"
                : catalog.worker_enabled
                  ? "可启动评测"
                  : "仅检查代码 · 执行服务未开启"}
        </div>
        {catalog ? (
          <Routes>
            <Route path="/" element={<Overview catalog={catalog} />} />
            <Route path="/submit" element={<Submit catalog={catalog} />} />
            <Route
              path="/benchmarks"
              element={<Benchmarks catalog={catalog} />}
            />
            <Route path="/workspace/jobs" element={<Jobs />} />
            <Route path="/workspace/jobs/:id" element={<JobDetail />} />
            <Route path="/leaderboard" element={<Leaderboard />} />
            <Route path="/docs" element={<Docs catalog={catalog} />} />
            <Route
              path="*"
              element={
                <Empty>
                  页面不存在，<Link to="/">返回概览</Link>。
                </Empty>
              }
            />
          </Routes>
        ) : (
          <>
            <header className="page-heading">
              <h1>评测工作台</h1>
              <p>正在读取服务绑定的评测集与运行能力。</p>
            </header>
            <ErrorNotice error={error} />
            {error ? (
              <Refresh onClick={() => void mutate()} />
            ) : (
              <Empty>连接中…</Empty>
            )}
          </>
        )}
        {catalog && error ? <ErrorNotice error={error} /> : null}
      </main>
    </div>
  );
}
export function App() {
  return (
    <SWRConfig
      value={{
        revalidateOnFocus: true,
        errorRetryCount: 2,
        dedupingInterval: 1500,
      }}
    >
      <BrowserRouter>
        <Workspace />
      </BrowserRouter>
    </SWRConfig>
  );
}
