# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.


"""OpenAPI models for trial leaderboard responses."""

from types import SimpleNamespace


def _types():
    string = dict(type="string")
    number = dict(type="number")
    integer = dict(type="integer", minimum=0)
    strings = dict(type="array", items=string)
    nullable = dict(type=["number", "null"])

    def obj(properties):
        return dict(type="object", required=list(properties), properties=properties)

    def ref(name):
        return {"$ref": "#/components/schemas/" + name}

    def array(name):
        return dict(type="array", items=ref(name))

    return SimpleNamespace(
        string=string,
        number=number,
        integer=integer,
        strings=strings,
        nullable=nullable,
        obj=obj,
        ref=ref,
        array=array,
    )


def extend_schemas(schemas):
    types = _types()
    schemas["LeaderboardEnvironment"] = _leaderboardenvironment(types)
    schemas["LeaderboardCohort"] = _leaderboardcohort(types)
    schemas["LeaderboardAnchor"] = _leaderboardanchor(types)
    schemas["LeaderboardEntry"] = _leaderboardentry(types)
    schemas["LeaderboardExcluded"] = _leaderboardexcluded(types)
    schemas["Leaderboard"] = _leaderboard(types)


def _leaderboardenvironment(types):
    field_names = ['compiler_name', 'triton_import_version', 'cann_version', 'npu_model', 'driver_version']
    return types.obj(
        {
            **{key: dict(type=['string', 'null']) for key in field_names},
            "clock_locked": dict(type=["boolean", "null"]),
        }
    )


def _leaderboardcohort(types):
    return types.obj(
        dict(
            id=types.string,
            manifest_hash=types.string,
            assets_hash=dict(type=["string", "null"]),
            corpus_version=types.string,
            profile=dict(enum=["smoke", "standard"]),
            scope=types.string,
            operators=types.strings,
            env_fp=types.string,
            execution=types.string,
            scorer_version=dict(type=["string", "null"]),
            protocol_version=dict(const=1),
            planned_cases=types.integer,
            task_count=types.integer,
            environment=types.ref("LeaderboardEnvironment"),
        )
    )


def _leaderboardanchor(types):
    return types.obj(
        dict(
            operator=types.string,
            task_id=types.string,
            geometric_speedup=types.number,
            scoreable=dict(type="boolean"),
        )
    )


def _leaderboardentry(types):
    return types.obj(
        dict(
            rank=types.integer,
            task_id=types.string,
            submit_id=types.string,
            package_sha256=types.string,
            created_at=types.string,
            functional_score=types.number,
            strict_score=types.number,
            passed_cases=types.integer,
            planned_cases=types.integer,
            geometric_speedup=types.nullable,
            shared_perf_score=types.nullable,
            scored_operators=types.integer,
            total_operators=types.integer,
            unstable_cases=types.integer,
            warnings=types.strings,
        )
    )


def _leaderboardexcluded(types):
    return types.obj(dict(task_id=types.string, reason=types.string))


def _leaderboard(types):
    return types.obj(
        dict(
            protocol_version=dict(const=1),
            mode=dict(const="trial"),
            metric=dict(enum=["functional", "performance"]),
            cohorts=types.array("LeaderboardCohort"),
            selected_cohort=dict(anyOf=[types.ref("LeaderboardCohort"), dict(type="null")]),
            baseline_revision=dict(type=["string", "null"]),
            anchors=types.array("LeaderboardAnchor"),
            items=types.array("LeaderboardEntry"),
            total_entries=types.integer,
            excluded=types.array("LeaderboardExcluded"),
            rules=types.strings,
        )
    )
