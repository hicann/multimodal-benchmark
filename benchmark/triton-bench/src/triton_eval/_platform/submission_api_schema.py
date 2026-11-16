# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""Additive frontend capability and event definitions for the v1 HTTP contract."""

from __future__ import annotations


from types import SimpleNamespace


def _reference(name):
    return {"$ref": f"#/components/schemas/{name}"}


def extend_schemas(schemas):
    types = SimpleNamespace(
        string=dict(type="string"),
        nullable=dict(type=["string", "null"]),
        integer=dict(type="integer", minimum=0),
        boolean=dict(type="boolean"),
    )
    _remote_task(schemas, types)
    _actions(schemas, types)
    _submission_policy(schemas, types)
    _profiles(schemas, types)
    _catalog(schemas, types)
    _performance(schemas, types)
    _engine_events(schemas, types)
    kinds = _payloads(types)
    _envelopes(schemas, types, kinds)
    _public_events(schemas, types, kinds)


def _remote_task(schemas, types):
    schemas["Task"]["properties"]["remote"] = dict(
        type=["object", "null"],
        properties=dict(
            id=types.string,
            phase=types.string,
            release_state=dict(enum=["NOT_ACQUIRED", "UNCONFIRMED", "SHUTDOWN"]),
            artifacts_verified=types.boolean,
            device_health=dict(enum=["NOT_PROBED", "HEALTHY", "UNHEALTHY"]),
            event_seq=types.integer,
            bundle_sha256=types.nullable,
            manifest_hash=types.nullable,
            assets_hash=types.nullable,
        ),
    )


def _actions(schemas, types):
    action = dict(
        type="object",
        required=["enabled", "reason"],
        properties=dict(
            enabled=types.boolean,
            reason=types.nullable,
            requires_acknowledgement=types.boolean,
            starts_instance=types.boolean,
        ),
    )
    schemas["ActionAvailability"] = action
    schemas["AvailableActions"] = dict(
        type="object",
        required=["cancel", "recover"],
        properties=dict(
            cancel=_reference("ActionAvailability"), recover=_reference("ActionAvailability")
        ),
    )
    schemas["Task"]["properties"]["available_actions"] = _reference("AvailableActions")
    schemas["Task"]["required"].append("available_actions")


def _submission_policy(schemas, types):
    policy_properties = dict(
        content_type=dict(type="string", const="application/zip"),
        max_request_bytes=types.integer,
        max_entries=types.integer,
        max_total_bytes=types.integer,
        max_file_bytes=types.integer,
        max_compression_ratio=dict(type="number", minimum=1),
        allowed_suffixes=dict(type="array", items=types.string),
        required_manifest=types.string,
        rules=dict(
            type="array",
            items=dict(
                type="object",
                required=["rule_id", "detail"],
                properties=dict(rule_id=types.string, detail=types.string),
            ),
        ),
    )
    schemas["SubmissionPolicy"] = dict(
        type="object", required=list(policy_properties), properties=policy_properties
    )


def _profiles(schemas, types):
    profile_properties = dict(
        id=dict(enum=["smoke", "standard"]),
        label=types.string,
        default=types.boolean,
        experimental=types.boolean,
        performance_scoring=types.boolean,
        performance_accepted=types.boolean,
        full_suite_accepted=types.boolean,
    )
    schemas["ProfileCapability"] = dict(
        type="object", required=list(profile_properties), properties=profile_properties
    )


def _catalog(schemas, types):
    schemas["Catalog"]["properties"].update(
        submission_policy=_reference("SubmissionPolicy"),
        profile_capabilities=dict(type="array", items=_reference("ProfileCapability")),
        default_profile=dict(enum=["smoke", "standard"]),
        default_scope=dict(enum=["operators", "full"]),
    )
    schemas["Catalog"]["required"] += [
        "submission_policy",
        "profile_capabilities",
        "default_profile",
        "default_scope",
    ]


def _performance(schemas, types):
    perf_properties = dict(
        measured_cases=types.integer,
        passed_cases=types.integer,
        missing_cases=types.integer,
        unstable_cases=dict(type="array", items=types.string),
        cv_gate=dict(type="number"),
        geometric_speedup=dict(type=["number", "null"]),
        score_basis=dict(const="TASK_LOCAL_FIRST_ALL_PASS"),
        cross_task_comparable=dict(const=False),
    )
    schemas["PerformanceView"] = dict(
        type="object", required=list(perf_properties), properties=perf_properties
    )
    schemas["Results"]["properties"]["performance"] = _reference("PerformanceView")
    schemas["Results"]["required"].append("performance")


def _engine_events(schemas, types):
    schemas["EngineCaseEvent"] = dict(
        type="object",
        required=["event", "record"],
        properties=dict(event=dict(const="case"), record=_reference("CaseResult")),
    )
    schemas["EngineEvent"] = dict(
        oneOf=[
            _reference("EngineCaseEvent"),
            dict(
                type="object",
                properties=dict(event=dict(type="string", **{"not": dict(const="case")})),
                additionalProperties=True,
            ),
        ],
        description="case 有完整结果模型；其余 meta/env_check/probe/fatal/done 保留扩展字段",
    )


def _payloads(types):
    kinds = {
        "phase": dict(
            type="object",
            required=["phase"],
            properties=dict(phase=dict(enum=["running", "scoring", "reporting"])),
        ),
        "health": dict(
            type="object",
            properties=dict(
                ok=types.boolean,
                torch=types.string,
                torch_npu=types.string,
                triton=types.string,
                devices=types.integer,
            ),
        ),
        "engine": _reference("EngineEvent"),
        "heartbeat": dict(type="object", properties=dict(elapsed_s=types.integer)),
        "final": dict(
            type="object",
            properties=dict(
                status=types.string,
                error=types.nullable,
                archive_bytes=types.integer,
                archive_sha256=types.string,
                files=dict(
                    type="object",
                    additionalProperties=dict(
                        type="object",
                        required=["bytes", "sha256"],
                        properties=dict(bytes=types.integer, sha256=types.string),
                    ),
                ),
            ),
        ),
    }
    return kinds


def _envelopes(schemas, types, kinds):
    envelopes = []
    for kind, payload in kinds.items():
        name = "Remote" + kind.title() + "Envelope"
        schemas[name] = dict(
            type="object",
            additionalProperties=False,
            required=["protocol_version", "job_id", "event_seq", "kind", "payload"],
            properties=dict(
                protocol_version=dict(type="integer", const=1),
                job_id=types.string,
                event_seq=dict(type="integer", minimum=1),
                kind=dict(const=kind),
                payload=payload,
            ),
        )
        envelopes.append(_reference(name))
    schemas["RemoteEnvelope"] = dict(oneOf=envelopes)


def _public_events(schemas, types, kinds):
    status = schemas["Task"]["properties"]["status"]
    common = dict(
        sequence=dict(type="integer", minimum=1, format="int64"), time=types.string, status=status
    )
    schemas["StateEvent"] = dict(
        type="object",
        required=["sequence", "time", "status"],
        properties={
            **common,
            "kind": dict(const="state"),
            "error": types.nullable,
            "cancel_requested": types.boolean,
            "phase": types.string,
            "release_state": dict(enum=["NOT_ACQUIRED", "UNCONFIRMED", "SHUTDOWN"]),
            "remote": False,
        },
    )
    schemas["RemotePublicEvent"] = dict(
        type="object",
        required=["sequence", "time", "status", "kind", "remote"],
        properties={
            **common,
            "kind": dict(enum=list(kinds)),
            "remote": _reference("RemoteEnvelope"),
        },
    )
    schemas["PublicEvent"] = dict(
        oneOf=[_reference("StateEvent"), _reference("RemotePublicEvent")],
        description="sequence 是全服务游标；remote.event_seq 是 job 内序号，不可互换",
    )
    schemas["Events"]["properties"]["items"]["items"] = _reference("PublicEvent")
