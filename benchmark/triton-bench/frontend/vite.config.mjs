/*
 * Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.

 */
import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendPort = Number(process.env.TRITON_EVAL_BACKEND_PORT || 8765);
if (!Number.isInteger(backendPort) || backendPort < 1 || backendPort > 65535)
  throw new Error("TRITON_EVAL_BACKEND_PORT must be a TCP port");
const target = `http://127.0.0.1:${backendPort}`;
const proxy = () => ({
  target,
  changeOrigin: true,
  configure(instance) {
    instance.on("proxyReq", (outgoing) => outgoing.setHeader("Origin", target));
  },
});

const sameOriginApi = {
  name: "triton-loopback-api-boundary",
  configureServer(server) {
    server.middlewares.use((req, res, next) => {
      const path = (req.url || "").split("?", 1)[0];
      if (
        !(
          path === "/api" ||
          path.startsWith("/api/") ||
          path === "/openapi.json"
        )
      )
        return next();
      const address = server.httpServer?.address();
      const port =
        address && typeof address === "object"
          ? address.port
          : server.config.server.port;
      const host = req.headers.host;
      const duplicate = (name) =>
        req.rawHeaders.filter(
          (value, index) => index % 2 === 0 && value.toLowerCase() === name,
        ).length > 1;
      if (
        ![`127.0.0.1:${port}`, `localhost:${port}`].includes(host) ||
        duplicate("host") ||
        duplicate("origin") ||
        (req.headers.origin && req.headers.origin !== `http://${host}`) ||
        req.headers["sec-fetch-site"] === "cross-site"
      ) {
        res.writeHead(403, {
          "Content-Type": "application/json",
          "Cache-Control": "no-store",
        });
        res.end(
          JSON.stringify({
            error: {
              code: "GATEWAY-ORIGIN",
              detail: "只接受当前本机前端同源请求",
            },
          }),
        );
        return;
      }
      if (
        (path === "/api" || path.startsWith("/api/")) &&
        !path.startsWith("/api/v1/")
      ) {
        res.writeHead(404, { "Content-Type": "application/json" });
        res.end(
          JSON.stringify({
            error: { code: "NOT-FOUND", detail: "接口不存在" },
          }),
        );
        return;
      }
      next();
    });
  },
};

export default defineConfig({
  build: {
    outDir: "dist/client",
  },
  optimizeDeps: {
    include: ["react", "react-dom/client"],
  },
  server: {
    host: "127.0.0.1",
    allowedHosts: ["localhost"],
    cors: false,
    proxy: { "^/api/v1/": proxy(), "^/openapi\\.json(?:\\?|$)": proxy() },
    warmup: {
      clientFiles: ["./src/main.jsx"],
    },
  },
  plugins: [sameOriginApi, react()],
});
