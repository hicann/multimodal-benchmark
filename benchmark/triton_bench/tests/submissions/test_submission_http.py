# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
import zipfile
from io import BytesIO
from types import SimpleNamespace

import pytest

from triton_eval.submission_http import handler_class, make_server, openapi
from triton_eval.submission_service import EvaluationService
from triton_eval.submission_simulation import DEFAULT_CORPUS


@pytest.fixture
def endpoint(tmp_path):
    service = EvaluationService(tmp_path / 'http', DEFAULT_CORPUS)
    try:
        try:
            server = make_server(service, 0)
        except PermissionError:
            pytest.skip('当前沙箱禁止绑定 loopback；不是接口测试通过')
        thread = threading.Thread(target=server.serve_forever)
        thread.start()
        try:
            yield service, f'http://127.0.0.1:{server.server_port}'
        finally:
            server.shutdown()
            thread.join()
            server.server_close()
    finally:
        service.close()


def call(base, path, data=None, content_type='application/json', headers=None):
    if isinstance(data, dict):
        data = json.dumps(data).encode()
    request = urllib.request.Request(base + path, data=data,
                                     headers={'Content-Type': content_type, **(headers or {})})
    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.load(response)
    except urllib.error.HTTPError as exc:
        return exc.code, json.load(exc)


def upload_bytes():
    stream = BytesIO()
    with zipfile.ZipFile(stream, 'w') as archive:
        archive.writestr(zipfile.ZipInfo('manifest.yaml', (2026, 1, 1, 0, 0, 0)),
                         'submit_id: http\noperators:\n- name: Exp\n  entry: exp\n  path: kernel.py\n')
        archive.writestr(zipfile.ZipInfo('kernel.py', (2026, 1, 1, 0, 0, 0)),
                         'def exp(x, base=-1.0, scale=1.0, shift=0.0):\n    return x\n')
    return stream.getvalue()


def test_real_http_upload_queue_query_cancel(endpoint):
    service, base = endpoint
    code, submission = call(base, '/api/v1/submissions', upload_bytes(), 'application/zip',
                            {'Idempotency-Key': 'one'})
    assert code == 201 and submission['status'] == 'READY'
    assert call(base, '/api/v1/submissions', upload_bytes(), 'application/zip',
                {'Idempotency-Key': 'one'})[1]['id'] == submission['id']
    code, task = call(base, '/api/v1/evaluations', dict(submission_id=submission['id']))
    assert code == 202 and task['status'] == 'QUEUED'
    prefix = f'/api/v1/evaluations/{task["id"]}'
    assert call(base, prefix)[1]['planned_cases'] == task['planned_cases']
    assert call(base, prefix + '/results')[1]['score'] is None
    assert call(base, prefix + '/events')[1]['items'][0]['status'] == 'QUEUED'
    assert call(base, prefix + '/cancel', {})[1]['status'] == 'CANCELLED'
    assert call(base, prefix + '/artifacts')[1] == dict(items=[])
    assert call(base, '/openapi.json')[1] == openapi()
    assert service.task(task['id'])['status'] == 'CANCELLED'


def test_http_errors_are_structured_and_no_cross_origin(endpoint):
    service, base = endpoint
    code, error = call(base, '/api/v1/evaluations', b'{')
    assert code == 400 and error['error']['code'] == 'REQUEST-JSON'
    assert call(base, '/api/v1/submissions', upload_bytes())[0] == 415
    assert call(base, '/openapi.json', headers={'Origin': 'https://other.invalid'})[0] == 403
    assert call(base, '/openapi.json', headers={'Host': 'other.invalid'})[0] == 403
    assert call(base, '/api/v1/submissions/' + '0' * 32)[0] == 404


def test_openapi_enumerates_complete_orchestration_surface():
    contract = openapi()
    assert contract['openapi'] == '3.1.0'
    assert len(contract['paths']) == 10
    assert contract['components']['schemas']['TaskRequest']['additionalProperties'] is False
    for path, methods in contract['paths'].items():
        for operation in methods.values():
            assert operation['responses']
            assert operation['summary']
            _assert_path_parameters(path, operation)


def _assert_path_parameters(path, operation):
    for parameter in operation.get('parameters', []):
        if parameter['in'] == 'path':
            assert '{' + parameter['name'] + '}' in path


def raw_http(service, method, path, data=b'', content_type='application/json', extra_headers=None):
    headers = {'Host': '127.0.0.1:8765', 'Connection': 'close',
               'Content-Length': str(len(data)), 'Content-Type': content_type, **(extra_headers or {})}
    serialized = f'{method} {path} HTTP/1.1\r\n' + ''.join(f'{name}: {value}\r\n' for name, value in headers.items())

    class MemoryConnection:
        def __init__(self):
            self.response = BytesIO()

        @staticmethod
        def makefile(*args):
            return BytesIO(serialized.encode() + b'\r\n' + data)

        @staticmethod
        def settimeout(value):
            pass

        def sendall(self, content):
            self.response.write(content)

    connection = MemoryConnection()
    handler_class(service)(connection, ('127.0.0.1', 12345), SimpleNamespace(server_port=8765))
    header, body = connection.response.getvalue().split(b'\r\n\r\n', 1)
    return int(header.split()[1]), body


def test_http_dispatch_without_socket_privilege(tmp_path):
    service = EvaluationService(tmp_path / 'memory-http', DEFAULT_CORPUS)
    try:
        code, body = raw_http(service, 'POST', '/api/v1/submissions', upload_bytes(), 'application/zip')
        submission = json.loads(body)
        assert code == 201 and submission['status'] == 'READY'
        code, body = raw_http(service, 'POST', '/api/v1/evaluations',
                              json.dumps(dict(submission_id=submission['id'])).encode())
        task = json.loads(body)
        assert code == 202 and task['status'] == 'QUEUED'
        prefix = f'/api/v1/evaluations/{task["id"]}'
        for suffix in ('', '/results', '/events', '/artifacts'):
            assert raw_http(service, 'GET', prefix + suffix)[0] == 200
        assert json.loads(raw_http(service, 'POST', prefix + '/cancel')[1])['status'] == 'CANCELLED'
        assert raw_http(service, 'GET', prefix + '/events?after=-1')[0] == 400
        assert raw_http(service, 'GET', prefix + '/artifacts/../../service.sqlite')[0] == 404
        assert raw_http(service, 'POST', '/api/v1/evaluations', b'{')[0] == 400
        assert raw_http(service, 'POST', '/api/v1/evaluations', b'[]')[0] == 400
        assert raw_http(service, 'PUT', '/api/v1/evaluations')[0] == 405
        assert raw_http(service, 'GET', '/openapi.json', extra_headers={'Origin': 'https://evil.invalid'})[0] == 403
        assert raw_http(service, 'GET', '/openapi.json', extra_headers={'Host': 'evil.invalid'})[0] == 403
        assert json.loads(raw_http(service, 'GET', '/openapi.json')[1]) == openapi()
        catalog = json.loads(raw_http(service, 'GET', '/api/v1/catalog')[1])
        assert catalog['case_count'] == 1385 and catalog['operator_count'] == 74
        assert sum(operator['case_count'] for operator in catalog['operators']) == 1385
        assert catalog['device_health'] == 'NOT_PROBED'
    finally:
        service.close()


def test_openapi_schema_references_are_resolvable():
    contract = openapi()

    def visit(value):
        if isinstance(value, dict):
            if '$ref' in value:
                assert value['$ref'].startswith('#/components/schemas/')
                assert value['$ref'].rsplit('/', 1)[-1] in contract['components']['schemas']
            for child in value.values():
                visit(child)
        elif isinstance(value, list):
            for child in value:
                visit(child)

    visit(contract)
    assert contract['components']['schemas']['CaseResult']['additionalProperties'] is False
    assert contract['components']['schemas']['Score']['additionalProperties'] is False


def test_protocol_bindings_do_not_share_method_or_service_state(tmp_path):
    first = EvaluationService(tmp_path / 'first', DEFAULT_CORPUS)
    second = EvaluationService(tmp_path / 'second', DEFAULT_CORPUS)
    try:
        for service in (first, second, first):
            assert raw_http(service, 'GET', '/api/v1/catalog')[0] == 200
            assert raw_http(service, 'POST', '/api/v1/catalog')[0] == 404
            assert raw_http(service, 'GET', '/api/v1/submissions')[0] == 404
            assert raw_http(service, 'PUT', '/api/v1/catalog')[0] == 405
        status, body = raw_http(first, 'POST', '/api/v1/submissions', upload_bytes(), 'application/zip')
        assert status == 201
        identifier = json.loads(body)['id']
        assert raw_http(first, 'GET', f'/api/v1/submissions/{identifier}')[0] == 200
        assert raw_http(second, 'GET', f'/api/v1/submissions/{identifier}')[0] == 404
    finally:
        first.close()
        second.close()


def test_content_scenarios_cross_serialized_http_entry(tmp_path):
    from triton_eval.submission_simulation import build_suite, check_outcome

    directory = tmp_path / 'suite'
    suite = build_suite(directory)
    service = EvaluationService(tmp_path / 'http-content', DEFAULT_CORPUS)
    try:
        for scenario in suite['scenarios']:
            if scenario['stage'] != 'content':
                continue
            code, body = raw_http(service, 'POST', '/api/v1/submissions',
                                  (directory / scenario['package']).read_bytes(), 'application/zip')
            assert code == 201
            submission = json.loads(body)
            assert check_outcome(scenario, submission, None, None) == []
            assert submission['stage'] == 'content'
    finally:
        service.close()
