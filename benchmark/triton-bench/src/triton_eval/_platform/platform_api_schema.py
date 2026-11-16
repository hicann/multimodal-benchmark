# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Platform API additions, preserving the reviewed base document builder."""


def extend_document(document):

    def reference(name):
        return {"$ref": "#/components/schemas/" + name}

    error = {
        "description": "结构化错误",
        "content": {"application/json": {"schema": reference("Error")}},
    }

    def operation(summary, response, *, body=None, parameters=None, status="200"):
        result = {
            "summary": summary,
            "responses": {
                status: {
                    "description": "成功",
                    "content": {"application/json": {"schema": reference(response)}},
                },
                **{str(code): error for code in (400, 403, 404, 405, 409, 413, 415, 500)},
            },
        }
        if body:
            result["requestBody"] = body
        if parameters:
            result["parameters"] = parameters
        return result

    string = {"type": "string"}
    array = {"type": "array", "items": string}
    identifier = {
        "name": "id",
        "in": "path",
        "required": True,
        "schema": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
    }
    idem = {
        "name": "Idempotency-Key",
        "in": "header",
        "required": False,
        "schema": {"type": "string", "maxLength": 200},
    }
    context = (operation, reference, identifier, idem, error, string, array)
    _platform_paths(document, context)
    _platform_models(document, context)
    from triton_eval.submission_api_schema import extend_schemas
    from triton_eval.leaderboard_schema import extend_schemas as extend_leaderboard

    extend_schemas(document["components"]["schemas"])
    extend_leaderboard(document["components"]["schemas"])
    document["info"]["version"] = "0.3.0"
    return document


def _paths_1(operation, reference, identifier, idem, error, string, array):
    return {
        "post": operation(
            "人工确认健康后解除 Worker 暂停；存在遗留租约时拒绝",
            "Catalog",
            body=dict(
                required=True,
                content={
                    "application/json": {
                        "schema": dict(
                            type="object",
                            additionalProperties=False,
                            required=["acknowledge_health"],
                            properties=dict(acknowledge_health=dict(type="boolean", const=True)),
                        )
                    }
                },
            ),
        )
    }


def _paths_4(routes, operation):
    routes["get"] = operation(
        "按创建顺序倒序分页查询任务",
        "Tasks",
        parameters=[
            dict(name="before", **{"in": "query"}, schema=dict(type="integer", minimum=0)),
            dict(
                name="limit",
                **{"in": "query"},
                schema=dict(type="integer", minimum=1, maximum=100, default=30)
            ),
        ],
    )
    return routes


def _paths_7(operation, reference, identifier, idem, error, string, array):
    return {
        "get": dict(
            summary="SSE：状态和逐用例事件；全局 sequence 续读",
            parameters=[
                identifier,
                dict(
                    name="Last-Event-ID",
                    **{"in": "header"},
                    schema=dict(type="string", pattern="^[0-9]+$")
                ),
                dict(name="after", **{"in": "query"}, schema=dict(type="integer", minimum=0)),
            ],
            responses={
                "200": dict(
                    description="retry=2000；15 秒心跳；终态排空后 end；最多 8 条连接",
                    content={"text/event-stream": {"schema": dict(type="string")}},
                ),
                "400": error,
                "403": error,
                "404": error,
                "429": error,
            },
        )
    }


def _paths_10(operation, reference, identifier, idem, error, string, array):
    return {
        "post": operation(
            "仅补取原远端任务结果，不同步、不重新执行提交",
            "Task",
            parameters=[identifier],
            body=dict(
                required=True,
                content={
                    "application/json": {
                        "schema": dict(
                            type="object",
                            additionalProperties=False,
                            required=["acknowledge_recovery"],
                            properties=dict(acknowledge_recovery=dict(type="boolean", const=True)),
                        )
                    }
                },
            ),
        )
    }


def _schemas_5(operation, reference, identifier, idem, error, string, array):
    return dict(
        type="object",
        properties=dict(
            items=dict(type="array", items=reference("Task")),
            next_cursor=dict(type=["integer", "null"]),
        ),
    )


def _platform_paths(document, context):
    from triton_eval.submission_http import base_openapi_paths

    operation = context[0]
    document["paths"].update(base_openapi_paths(*context[:5]))
    document["paths"]["/api/v1/leaderboard"] = {
        "get": operation(
            "查看榜单",
            "Leaderboard",
            parameters=[
                {
                    "name": "metric",
                    "in": "query",
                    "schema": {"enum": ["functional", "performance"]},
                },
                {
                    "name": "cohort",
                    "in": "query",
                    "schema": {"type": "string", "pattern": "^[0-9a-f]{64}$"},
                },
                {
                    "name": "task_id",
                    "in": "query",
                    "schema": {"type": "string", "pattern": "^[0-9a-f]{32}$"},
                },
            ],
        )
    }
    document["paths"]["/api/v1/worker/resume"] = _paths_1(*context)
    document["paths"]["/api/v1/evaluations"] = _paths_4(document["paths"]["/api/v1/evaluations"], operation)
    document["paths"]["/api/v1/evaluations/{id}/stream"] = _paths_7(*context)
    document["paths"]["/api/v1/evaluations/{id}/recover"] = _paths_10(*context)


def _platform_models(document, context):
    schemas = document["components"]["schemas"]
    schemas["Catalog"]["properties"]["execution_backend"] = dict(enum=["disabled", "local", "remote"])
    schemas["Tasks"] = _schemas_5(*context)
