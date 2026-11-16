# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Loopback-only static SPA entry and fixed-target streaming API gateway.

Run after building the frontend: python -m triton_eval.frontend_gateway
--static-dir frontend/dist --backend-port 8765 --port 5173
"""

from __future__ import annotations

import argparse
from functools import partialmethod
from http.client import HTTPConnection, HTTPException
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path, PurePosixPath
import re
from urllib.parse import unquote, urlsplit

from triton_eval.console import emit
from triton_eval.submission_policy import MAX_REQUEST_BYTES

SPA_ROUTES = re.compile(
    r"/(?:submit|benchmarks|docs|leaderboard|workspace/jobs(?:/[0-9a-f]{32})?)?/?$"
)
REQUEST_HEADERS = ("Accept", "Content-Type", "Content-Length", "Idempotency-Key", "Last-Event-ID")
RESPONSE_HEADERS = (
    "Content-Type",
    "Content-Length",
    "Content-Disposition",
    "Cache-Control",
    "X-Content-Type-Options",
    "X-Accel-Buffering",
)


class GatewayRequestError(Exception):
    def __init__(self, status, code, detail):
        super().__init__(code)
        self.status, self.code, self.detail = status, code, detail


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, format_value, *args):
        pass

    def _error(self, status, code, detail):
        data = json.dumps(
            dict(error=dict(code=code, detail=detail)), ensure_ascii=False
        ).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.send_header("Connection", "close")
        self.end_headers()
        self.close_connection = True
        if self.command != "HEAD":
            self.wfile.write(data)

    def _handle(self):
        self.connection.settimeout(30)
        try:
            length = self._request_length()
            parsed, path = self._request_path()
        except GatewayRequestError as exc:
            return self._error(exc.status, exc.code, exc.detail)
        return self._route(parsed, path, length)

    def _request_length(self):
        host = self.headers.get("Host", "")
        hosts = {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}
        if len(self.headers.get_all("Host", [])) != 1 or host not in hosts:
            raise GatewayRequestError(403, "GATEWAY-HOST", "只接受当前本机入口")
        if (
            len(self.headers.get_all("Origin", [])) > 1
            or self.headers.get("Origin") not in (None, f"http://{host}")
            or self.headers.get("Sec-Fetch-Site") == "cross-site"
        ):
            raise GatewayRequestError(403, "GATEWAY-ORIGIN", "只接受同源请求")
        if (
            self.headers.get("Transfer-Encoding")
            or len(self.headers.get_all("Content-Length", [])) > 1
        ):
            raise GatewayRequestError(400, "GATEWAY-FRAMING", "不支持该请求分帧")
        raw_length = self.headers.get("Content-Length", "0")
        if not re.fullmatch(r"[0-9]{1,10}", raw_length):
            raise GatewayRequestError(400, "CONTENT-LENGTH", "请求长度非法")
        length = int(raw_length)
        if length > MAX_REQUEST_BYTES:
            raise GatewayRequestError(413, "REQUEST-SIZE", "请求超过传输上限")
        return length

    def _request_path(self):
        try:
            parsed = urlsplit(self.path)
            path = unquote(parsed.path, errors="strict")
        except ValueError as exc:
            raise GatewayRequestError(400, "GATEWAY-PATH", "请求路径非法") from exc
        if (
            any((parsed.scheme, parsed.netloc, not path.startswith("/"), "\\" in path))
            or any(
                part in {".", ".."} or part.startswith(".") for part in path.split("/") if part
            )
        ):
            raise GatewayRequestError(400, "GATEWAY-PATH", "请求路径非法")
        return parsed, path

    def _route(self, parsed, path, length):
        if parsed.path.startswith("/api/v1/") or parsed.path == "/openapi.json":
            if self.command not in {"GET", "POST"}:
                return self._error(405, "METHOD", "接口只支持 GET/POST")
            return self._proxy(length)
        if self.command not in {"GET", "HEAD"}:
            return self._error(405, "METHOD", "静态页面只支持 GET/HEAD")
        if length:
            return self._error(400, "GATEWAY-FRAMING", "静态请求不能携带请求体")
        return self._static(path)

    def _static(self, path):
        root = self.server.static_root
        target = root.joinpath(*PurePosixPath(path).parts[1:])
        if not target.resolve().is_relative_to(root):
            return self._error(404, "NOT-FOUND", "文件不存在")
        if not target.is_file():
            if not SPA_ROUTES.fullmatch(path):
                return self._error(404, "NOT-FOUND", "页面不存在")
            target = root / "index.html"
        # Do not publish symlink assets, including an internal symlink directory.
        if any(
            item.is_symlink()
            for item in [target, *target.parents]
            if item != root and item.is_relative_to(root)
        ):
            return self._error(404, "NOT-FOUND", "文件不存在")
        data = target.read_bytes()
        self.send_response(200)
        self.send_header(
            "Content-Type", mimetypes.guess_type(target.name)[0] or "application/octet-stream"
        )
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)
        return None

    def _proxy(self, length):
        data = self.rfile.read(length) if length else None
        if data is not None and len(data) != length:
            return self._error(400, "REQUEST-INCOMPLETE", "请求体未完整接收")
        authority = f"127.0.0.1:{self.server.backend_port}"
        headers = {
            key: self.headers[key]
            for key in REQUEST_HEADERS
            if self.headers.get(key) is not None
        }
        # Only normalize AFTER validating the browser-facing Host and Origin.
        headers.update(Host=authority, Origin=f"http://{authority}", Connection="close")
        upstream = HTTPConnection("127.0.0.1", self.server.backend_port, timeout=40)
        sent = False
        response = None
        try:
            upstream.request(self.command, self.path, body=data, headers=headers)
            response = upstream.getresponse()
            self.send_response(response.status)
            for key in RESPONSE_HEADERS:
                if response.getheader(key) is not None:
                    self.send_header(key, response.getheader(key))
            self.send_header("Connection", "close")
            self.end_headers()
            sent = True
            self.close_connection = True
            # read() would wait to fill a large buffer and stall SSE updates.
            while chunk := response.read1(64 * 1024):
                self.wfile.write(chunk)
                self.wfile.flush()
        except (OSError, HTTPException):
            if not sent:
                self._error(502, "GATEWAY-UPSTREAM", "评测后端不可达")
        finally:
            if response is not None:
                response.close()
            upstream.close()
            self.close_connection = True
        return None

    do_GET = partialmethod(_handle)
    do_POST = do_GET
    do_HEAD = do_GET
    do_PUT = do_GET
    do_PATCH = do_GET
    do_DELETE = do_GET
    do_OPTIONS = do_GET


def make_gateway(static_dir: Path, backend_port: int, port: int = 5173) -> ThreadingHTTPServer:
    root = Path(static_dir).resolve(strict=True)
    if (
        not root.is_dir()
        or not (root / "index.html").is_file()
        or (root / "index.html").is_symlink()
    ):
        raise ValueError("需要含 index.html 的前端构建目录")
    if not 1 <= backend_port <= 65535:
        raise ValueError("后端端口非法")


    server = ThreadingHTTPServer(("127.0.0.1", port), GatewayHandler)
    if server.server_port == backend_port:
        server.server_close()
        raise ValueError("前端入口端口不能与后端相同")
    server.static_root = root
    server.backend_port = backend_port
    return server


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--static-dir", type=Path, required=True)
    parser.add_argument("--backend-port", type=int, default=8765)
    parser.add_argument("--port", type=int, default=5173)
    args = parser.parse_args()
    server = make_gateway(args.static_dir, args.backend_port, args.port)
    emit(f"本机前端入口：http://127.0.0.1:{server.server_port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
