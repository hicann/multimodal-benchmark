# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

"""Dependency-free loopback HTTP adapter for controlled frontend integration."""

from __future__ import annotations

import json
import re
from functools import partialmethod
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
import logging

import argparse


from triton_eval.submission_service import EvaluationService, ServiceError
from triton_eval.submission_simulation import DEFAULT_CORPUS
from triton_eval.models import CaseResultRecord, RunScoreRecord
from triton_eval.console import emit

MAX_REQUEST_BYTES = 64 * 1024 * 1024


def openapi() -> dict:
    def reference(name):
        return {"$ref": f"#/components/schemas/{name}"}

    error = {
        "description": "结构化错误；不会以 HTTP 成功伪装请求失败",
        "content": {"application/json": {"schema": reference("Error")}},
    }
    identifier = dict(name="id", **{"in": "path"}, required=True, schema=dict(type="string", pattern="^[0-9a-f]{32}$"))
    idem = dict(name="Idempotency-Key", **{"in": "header"}, required=False, schema=dict(type="string", maxLength=200))

    def operation(summary, response, *, body=None, parameters=None, status="200"):
        return _openapi_operation(summary, response, body, parameters, status, reference, error)

    paths = {
        '/api/v1/catalog': _openapi_paths_1(operation, reference, identifier, idem, error),
        '/api/v1/submissions': _openapi_paths_2(operation, reference, identifier, idem, error),
        '/api/v1/submissions/{id}': _openapi_paths_3(operation, reference, identifier, idem, error),
        '/api/v1/evaluations': _openapi_paths_4(operation, reference, identifier, idem, error),
        '/api/v1/evaluations/{id}': _openapi_paths_5(operation, reference, identifier, idem, error),
        '/api/v1/evaluations/{id}/events': _openapi_paths_6(operation, reference, identifier, idem, error),
        '/api/v1/evaluations/{id}/results': _openapi_paths_7(operation, reference, identifier, idem, error),
        '/api/v1/evaluations/{id}/cancel': _openapi_paths_8(operation, reference, identifier, idem, error),
        '/api/v1/evaluations/{id}/artifacts': _openapi_paths_9(operation, reference, identifier, idem, error),
        '/api/v1/evaluations/{id}/artifacts/{name}': _openapi_paths_10(operation, reference, identifier, idem, error),
    }
    string = dict(type="string")
    array = dict(type="array", items=string)
    schemas = {
        'Catalog': _openapi_schemas_1(reference, string, array),
        'Error': _openapi_schemas_2(reference, string, array),
        'Submission': _openapi_schemas_3(reference, string, array),
        'TaskRequest': _openapi_schemas_4(reference, string, array),
        'Task': _openapi_schemas_5(reference, string, array),
        'Events': _openapi_schemas_6(reference, string, array),
        'Results': _openapi_schemas_7(reference, string, array),
        'ValidationError': _openapi_schemas_8(reference, string, array),
        'Certification': _openapi_schemas_9(reference, string, array),
        'Artifacts': _openapi_schemas_10(reference, string, array),
    }
    schemas.update(_model_schemas())
    return dict(
        openapi="3.1.0",
        info=dict(title="Triton 提交联调接口", version="0.1.0"),
        paths=paths,
        components=dict(schemas=schemas),
    )


def _model_schemas():
    schemas = {}
    for name, model in [("CaseResult", CaseResultRecord), ("Score", RunScoreRecord)]:
        schema = model.model_json_schema(ref_template="#/components/schemas/{model}")
        schemas.update(schema.pop("$defs", {}))
        schemas[name] = schema
    try:
        schemas["CaseResult"]["properties"]["log_path"]["description"] = "任务产物文件名，不返回服务器路径"
    except KeyError as exc:
        raise ValueError("case result schema is missing its log_path property") from exc
    return schemas


def _event_cursor(parsed):
    try:
        after = int(parse_qs(parsed.query).get("after", ["0"])[0])
        if after < 0:
            raise ValueError
    except ValueError:
        raise ServiceError("EVENT-CURSOR", "事件游标须为非负整数") from None
    return after


def handler_class(service: EvaluationService):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, format_value, *args):
            pass

        def _write(self, status, value, *, artifact=False, name=""):
            if not artifact:
                text = json.dumps(value, ensure_ascii=False)
                for path in (str(service.root), str(service.corpus.parent.parent), str(Path.home())):
                    text = text.replace(path, "<local>")
                data = text.encode()
            else:
                data = value
            self.send_response(status)
            self.send_header(
                "Content-Type", "application/octet-stream" if artifact else "application/json; charset=utf-8"
            )
            self.send_header("Content-Length", str(len(data)))
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Cache-Control", "no-store")
            if artifact:
                self.send_header("Content-Disposition", f'attachment; filename="{name}"')
            self.end_headers()
            self.wfile.write(data)

        def _body(self, content_type):
            if self.headers.get_content_type() != content_type:
                raise ServiceError("CONTENT-TYPE", f"需要 {content_type}", 415)
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                raise ServiceError("CONTENT-LENGTH", "请求长度非法") from None
            if not 0 < length <= MAX_REQUEST_BYTES:
                raise ServiceError("REQUEST-SIZE", "请求为空或超过传输上限", 413)
            data = self.rfile.read(length)
            if len(data) != length:
                raise ServiceError("REQUEST-INCOMPLETE", "请求体未完整接收")
            return data

        def _evaluation_get(self, identifier, suffix, parsed):
            if suffix is None:
                return self._write(200, service.task(identifier))
            if suffix == "results":
                return self._write(200, service.results(identifier))
            if suffix == "events":
                return self._write(200, service.events(identifier, _event_cursor(parsed)))
            if suffix == "artifacts":
                return self._write(200, dict(items=service.artifacts(identifier)))
            if suffix and suffix.startswith("artifacts/"):
                name = suffix.removeprefix("artifacts/")
                return self._write(200, service.artifact(identifier, name), artifact=True, name=name)
            raise ServiceError("NOT-FOUND", "接口不存在", 404)

        def _route(self, method):
            host = self.headers.get("Host", "")
            if host not in {f"127.0.0.1:{self.server.server_port}", f"localhost:{self.server.server_port}"}:
                raise ServiceError("HOST", "仅支持 loopback 联调", 403)
            if self.headers.get("Origin") not in (None, f"http://{host}"):
                raise ServiceError("ORIGIN", "不接受跨源请求；前端开发使用同源代理", 403)
            parsed = urlsplit(self.path)
            key = self.headers.get("Idempotency-Key")
            if method == "GET" and parsed.path == "/openapi.json":
                return self._write(200, openapi())
            if method == "GET" and parsed.path == "/api/v1/catalog":
                return self._write(200, service.catalog())
            if method == "POST" and parsed.path == "/api/v1/submissions":
                return self._write(201, service.upload(self._body("application/zip"), key=key))
            if method == "POST" and parsed.path == "/api/v1/evaluations":
                try:
                    request = json.loads(self._body("application/json"))
                except (UnicodeDecodeError, json.JSONDecodeError):
                    raise ServiceError("REQUEST-JSON", "JSON 请求不可解析") from None
                return self._write(202, service.create_task(request, key=key))
            match = re.fullmatch(r"/api/v1/(submissions|evaluations)/([0-9a-f]{32})(?:/(.*))?", parsed.path)
            if not match:
                raise ServiceError("NOT-FOUND", "接口不存在", 404)
            kind, identifier, suffix = match.groups()
            if kind == "submissions" and method == "GET" and suffix is None:
                return self._write(200, service.submission(identifier))
            if kind == "evaluations":
                if method == "POST" and suffix == "cancel":
                    return self._write(200, service.cancel(identifier))
                if method == "GET":
                    return self._evaluation_get(identifier, suffix, parsed)
            raise ServiceError("NOT-FOUND", "接口不存在", 404)

        def _handle(self, method):
            self.connection.settimeout(30)
            try:
                self._route(method)
            except ServiceError as exc:
                self._write(exc.http_status, dict(error=dict(code=exc.code, detail=exc.detail)))
            except Exception:
                self._write(500, dict(error=dict(code="INTERNAL", detail="服务内部异常；检查服务端状态")))

        def _unsupported(self):
            self._write(405, dict(error=dict(code="METHOD", detail="该方法不受支持")))

        do_GET = partialmethod(_handle, "GET")
        do_POST = partialmethod(_handle, "POST")
        do_PUT = _unsupported
        do_PATCH = _unsupported
        do_DELETE = _unsupported
        do_OPTIONS = _unsupported

    return Handler


def make_server(service: EvaluationService, port: int = 8765) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(("127.0.0.1", port), handler_class(service))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/submission-service"))
    parser.add_argument("--corpus", type=Path, default=DEFAULT_CORPUS)
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--execute", action="store_true", help="在本机启动真实评测 Worker")
    parser.add_argument("--acknowledge-isolation", action="store_true", help="确认这是受控隔离执行环境")
    args = parser.parse_args()
    if args.execute and not args.acknowledge_isolation:
        parser.error("执行不可信提交前必须提供 --acknowledge-isolation")
    service = EvaluationService(args.root, args.corpus, execute=args.execute)
    try:
        server = make_server(service, args.port)
        service.start_worker()
        emit(f"联调接口：http://127.0.0.1:{server.server_port}，真实 Worker：{args.execute}", flush=True)
        try:
            server.serve_forever()
        except KeyboardInterrupt:
            pass
        finally:
            server.server_close()
    finally:
        service.close()
    return 0




def _openapi_paths_1(operation, reference, identifier, idem, error):
    return {"get": operation("查询服务绑定的冻结测试集、算子契约及 Worker 配置", "Catalog")}


def _openapi_paths_2(operation, reference, identifier, idem, error):
    return {
        "post": operation(
            "上传 ZIP；拒收结论也持久化",
            "Submission",
            status="201",
            parameters=[idem],
            body=dict(required=True, content={"application/zip": {"schema": dict(type="string", format="binary")}}),
        )
    }


def _openapi_paths_3(operation, reference, identifier, idem, error):
    return {"get": operation("查询提交内容检查结果", "Submission", parameters=[identifier])}


def _openapi_paths_4(operation, reference, identifier, idem, error):
    return {
        "post": operation(
            "创建持久任务",
            "Task",
            status="202",
            parameters=[idem],
            body=dict(required=True, content={"application/json": {"schema": reference("TaskRequest")}}),
        )
    }


def _openapi_paths_5(operation, reference, identifier, idem, error):
    return {"get": operation("查询任务和真实用例进度", "Task", parameters=[identifier])}


def _openapi_paths_6(operation, reference, identifier, idem, error):
    return {
        "get": operation(
            "游标增量事件",
            "Events",
            parameters=[identifier, dict(name="after", **{"in": "query"}, schema=dict(type="integer", minimum=0))],
        )
    }


def _openapi_paths_7(operation, reference, identifier, idem, error):
    return {
        "get": operation("查询用例及评分，认证单独标注", "Results", parameters=[identifier])
    }


def _openapi_paths_8(operation, reference, identifier, idem, error):
    return {
        "post": operation("请求取消；运行中须等待 Worker 终止确认", "Task", parameters=[identifier])
    }


def _openapi_paths_9(operation, reference, identifier, idem, error):
    return {
        "get": operation("列出允许下载的产物及哈希", "Artifacts", parameters=[identifier])
    }


def _openapi_paths_10(operation, reference, identifier, idem, error):
    return {
        "get": dict(
            summary="下载固定白名单产物",
            parameters=[identifier, dict(name="name", **{"in": "path"}, required=True, schema=dict(type="string"))],
            responses={
                "200": dict(
                    description="产物；强制附件下载",
                    content={"application/octet-stream": {"schema": dict(type="string", format="binary")}},
                ),
                "404": error,
            },
        )
    }


def _openapi_schemas_1(reference, string, array):
    return dict(
        type="object",
        required=["manifest_hash", "operators", "profiles", "scopes"],
        properties=dict(
            corpus_version=string,
            manifest_hash=string,
            case_count=dict(type="integer"),
            operator_count=dict(type="integer"),
            profiles=array,
            scopes=array,
            worker_enabled=dict(type="boolean"),
            worker_paused=dict(type="boolean"),
            device_health=dict(enum=["NOT_PROBED"]),
            full_suite_accepted=dict(type="boolean", const=False),
            operators=dict(
                type="array",
                items=dict(
                    type="object",
                    required=["name", "weight", "case_count"],
                    properties=dict(
                        name=string,
                        weight=dict(type="number"),
                        case_count=dict(type="integer"),
                        entry=dict(type=["string", "null"]),
                        params=array,
                        signature=dict(type=["string", "null"]),
                    ),
                ),
            ),
        ),
    )


def _openapi_schemas_2(reference, string, array):
    return dict(
        type="object",
        required=["error"],
        properties=dict(
            error=dict(type="object", required=["code", "detail"], properties=dict(code=string, detail=string))
        ),
    )


def _openapi_schemas_3(reference, string, array):
    return dict(
        type="object",
        required=["id", "status", "package_sha256", "errors"],
        properties=dict(
            id=string,
            status=dict(enum=["READY", "REJECTED"]),
            package_sha256=string,
            declared_submit_id=dict(type=["string", "null"]),
            operators=array,
            stage=string,
            created_at=dict(type="string", format="date-time"),
            errors=dict(type="array", items=reference("ValidationError")),
        ),
    )


def _openapi_schemas_4(reference, string, array):
    return dict(
        type="object",
        additionalProperties=False,
        required=["submission_id"],
        properties=dict(
            submission_id=string,
            scope=dict(enum=["operators", "full"], default="operators"),
            operators=dict(type="array", items=string, minItems=1, uniqueItems=True),
            profile=dict(enum=["smoke", "standard"], default="smoke"),
        ),
    )


def _openapi_schemas_5(reference, string, array):
    return dict(
        type="object",
        required=["id", "status", "planned_cases", "completed_cases"],
        properties=dict(
            id=string,
            submission_id=string,
            scope=dict(enum=["operators", "full"]),
            operators=array,
            status=dict(
                enum=[
                    "QUEUED",
                    "RUNNING",
                    "SCORING",
                    "REPORTING",
                    "COMPLETED",
                    "REJECTED",
                    "FAILED",
                    "CANCELLED",
                    "INTERRUPTED",
                ]
            ),
            planned_cases=dict(type="integer"),
            completed_cases=dict(type="integer"),
            progress=dict(type="number", minimum=0, maximum=1),
            manifest_hash=string,
            corpus_version=string,
            package_sha256=string,
            case_codes=array,
            profile=dict(enum=["smoke", "standard"]),
            created_at=dict(type="string", format="date-time"),
            updated_at=dict(type="string", format="date-time"),
            worker_enabled=dict(type="boolean"),
            worker_paused=dict(type="boolean"),
            cancel_requested=dict(type="boolean"),
            error=dict(type=["string", "null"]),
        ),
    )


def _openapi_schemas_6(reference, string, array):
    return dict(
        type="object",
        properties=dict(
            next_cursor=dict(type="integer"),
            items=dict(
                type="array",
                items=dict(
                    type="object", properties=dict(sequence=dict(type="integer"), time=string, status=string)
                ),
            ),
        ),
    )


def _openapi_schemas_7(reference, string, array):
    return dict(
        type="object",
        required=["task_id", "status", "cases", "score", "certification"],
        properties=dict(
            task_id=string,
            status=string,
            scope=string,
            manifest_hash=string,
            counted_cases=dict(type="integer"),
            status_counts=dict(type="object"),
            corpus_version=string,
            package_sha256=string,
            planned_cases=dict(type="integer"),
            completed_cases=dict(type="integer"),
            declared_operators=array,
            counted_operators=array,
            scope_operators=array,
            error=dict(type=["string", "null"]),
            engine_error=dict(type=["string", "null"]),
            score=dict(anyOf=[reference("Score"), dict(type="null")]),
            score_reason=dict(type=["string", "null"]),
            performance_available=dict(type="boolean"),
            certification=reference("Certification"),
            validation=dict(type="array", items=reference("ValidationError")),
            cases=dict(type="array", items=reference("CaseResult")),
        ),
    )


def _openapi_schemas_8(reference, string, array):
    return dict(
        type="object",
        required=["rule_id", "stage", "operator", "detail"],
        properties=dict(
            rule_id=string, stage=string, operator=string, detail=string, lineno=dict(type=["integer", "null"])
        ),
    )


def _openapi_schemas_9(reference, string, array):
    return dict(
        type="object",
        required=["static", "compilation", "execution", "full_suite_accepted"],
        properties=dict(
            static=dict(enum=["NOT_COMPUTED"]),
            compilation=dict(enum=["NOT_COMPUTED"]),
            execution=dict(enum=["NOT_COMPUTED"]),
            full_suite_accepted=dict(type="boolean", const=False),
        ),
    )


def _openapi_schemas_10(reference, string, array):
    return dict(
        type="object",
        properties=dict(
            items=dict(
                type="array",
                items=dict(
                    type="object",
                    required=["name", "bytes", "sha256"],
                    properties=dict(name=string, bytes=dict(type="integer"), sha256=string),
                ),
            )
        ),
    )




def _openapi_operation(summary, response, body, parameters, status, reference, error):
    value = dict(
        summary=summary,
        responses={
            status: dict(description="成功", content={"application/json": {"schema": reference(response)}}),
            "400": error,
            "403": error,
            "404": error,
            "405": error,
            "409": error,
            "413": error,
            "415": error,
            "500": error,
        },
    )
    if body:
        value["requestBody"] = body
    if parameters:
        value["parameters"] = parameters
    return value


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    raise SystemExit(main())
