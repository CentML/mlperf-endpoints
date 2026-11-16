# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import asyncio
import threading

import pytest
from aiohttp.test_utils import TestClient, TestServer
from inference_endpoint.evaluation.osworld_service.osworld_service.config import (
    ServiceConfig,
)
from inference_endpoint.evaluation.osworld_service.osworld_service.schemas import (
    RunRequest,
)
from inference_endpoint.evaluation.osworld_service.osworld_service.server import (
    create_app,
)
from pydantic import ValidationError

from tests.unit.evaluation.osworld_service.fakes import FakeRunner

pytestmark = pytest.mark.unit

TOKEN = "service-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}


def _body(**overrides):
    body = {
        "model_name": "qwen-test",
        "endpoint_urls": ["http://gpu-node:30000"],
        "endpoint_api_key": "endpoint-secret",
        "generation_params": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768},
        "task_ids": ["calc-1", "os-2"],
        "n_attempts": 2,
        "max_turns": 30,
        "concurrency": 4,
    }
    body.update(overrides)
    return body


async def _client(tmp_path, runner):
    app = create_app(ServiceConfig(artifact_root=tmp_path, auth_token=TOKEN), runner)
    client = TestClient(TestServer(app))
    await client.start_server()
    return client


async def _wait_terminal(client, run_id):
    for _ in range(500):
        resp = await client.get(f"/v1/runs/{run_id}", headers=AUTH)
        status = await resp.json()
        if status["status"] in {"succeeded", "failed", "cancelled"}:
            return status
        await asyncio.sleep(0.01)
    raise AssertionError("run did not finish")


class TestRunRequest:
    def test_ignore_eos_and_unknown_sampling_are_refused(self):
        with pytest.raises(ValidationError, match="unsupported generation params"):
            RunRequest.model_validate(_body(generation_params={"ignore_eos": True}))
        with pytest.raises(ValidationError, match="unsupported generation params"):
            RunRequest.model_validate(_body(generation_params={"max_new_tokens": 9}))

    def test_endpoint_url_is_normalized_to_its_base(self):
        req = RunRequest.model_validate(
            _body(endpoint_urls=["http://gpu-node:30000/v1/"])
        )
        assert req.endpoint_url == "http://gpu-node:30000"

    @pytest.mark.parametrize(
        "task_ids", [["../etc"], ["a", "a"], [], [".hidden"]], ids=str
    )
    def test_bad_task_ids_are_refused(self, task_ids):
        with pytest.raises(ValidationError):
            RunRequest.model_validate(_body(task_ids=task_ids))

    def test_one_endpoint_only(self):
        with pytest.raises(ValidationError):
            RunRequest.model_validate(_body(endpoint_urls=["http://a:1", "http://b:2"]))


class TestService:
    def test_needs_exactly_one_auth_mode(self, tmp_path):
        with pytest.raises(ValueError, match="exactly one"):
            create_app(ServiceConfig(artifact_root=tmp_path), FakeRunner())
        with pytest.raises(ValueError, match="exactly one"):
            create_app(
                ServiceConfig(
                    artifact_root=tmp_path, auth_token="t", allow_unauthenticated=True
                ),
                FakeRunner(),
            )

    @pytest.mark.asyncio
    async def test_health_reports_endpoint_without_auth(self, tmp_path):
        runner = FakeRunner()
        client = await _client(tmp_path, runner)
        try:
            resp = await client.get("/health", params={"endpoint": "http://gpu:30000"})
            health = await resp.json()
        finally:
            await client.close()
        assert health["api_version"] == "v1"
        assert "osworld.run" in health["capabilities"]
        assert health["endpoint_reachable"] is True
        assert runner.health_queries == ["http://gpu:30000"]

    @pytest.mark.asyncio
    async def test_runs_require_the_token(self, tmp_path):
        client = await _client(tmp_path, FakeRunner())
        try:
            resp = await client.post("/v1/runs", json=_body())
        finally:
            await client.close()
        assert resp.status == 401

    @pytest.mark.asyncio
    async def test_run_succeeds_with_artifacts_and_redacted_request(self, tmp_path):
        runner = FakeRunner({"calc-1": 1.0})
        client = await _client(tmp_path, runner)
        try:
            resp = await client.post("/v1/runs", json=_body(), headers=AUTH)
            assert resp.status == 202
            submitted = await resp.json()
            assert submitted["trials_total"] == 4
            status = await _wait_terminal(client, submitted["run_id"])
            names = {a["name"] for a in status["artifacts"]}
            art = await client.get(
                f"/v1/runs/{status['run_id']}/artifacts/osworld_results.json",
                headers=AUTH,
            )
            bad = await client.get(
                f"/v1/runs/{status['run_id']}/artifacts/request.json", headers=AUTH
            )
        finally:
            await client.close()
        assert status["status"] == "succeeded"
        assert len(status["result"]["trials"]) == 4
        assert {"osworld_results.json", "preflight.json", "status.json"} <= names
        assert art.status == 200
        assert bad.status == 404
        request_json = (tmp_path / status["run_id"] / "request.json").read_text()
        assert "endpoint-secret" not in request_json
        assert runner.requests[0].generation_params == {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 32768,
        }

    @pytest.mark.asyncio
    async def test_runner_failure_becomes_a_failed_run(self, tmp_path):
        client = await _client(tmp_path, FakeRunner(fail="preflight refused: x"))
        try:
            resp = await client.post("/v1/runs", json=_body(), headers=AUTH)
            status = await _wait_terminal(client, (await resp.json())["run_id"])
        finally:
            await client.close()
        assert status["status"] == "failed"
        assert "preflight refused" in status["error"]

    @pytest.mark.asyncio
    async def test_one_active_run_and_cancel(self, tmp_path):
        block = threading.Event()
        client = await _client(tmp_path, FakeRunner(block=block))
        try:
            first = await (
                await client.post("/v1/runs", json=_body(), headers=AUTH)
            ).json()
            second = await client.post("/v1/runs", json=_body(), headers=AUTH)
            progress = await (
                await client.get(f"/v1/runs/{first['run_id']}", headers=AUTH)
            ).json()
            cancelled = await (
                await client.post(f"/v1/runs/{first['run_id']}/cancel", headers=AUTH)
            ).json()
        finally:
            block.set()
            await client.close()
        assert second.status == 429
        assert progress["trials_completed"] == 1
        assert cancelled["status"] == "cancelled"

    @pytest.mark.asyncio
    async def test_unknown_run_is_404(self, tmp_path):
        client = await _client(tmp_path, FakeRunner())
        try:
            resp = await client.get("/v1/runs/nope", headers=AUTH)
        finally:
            await client.close()
        assert resp.status == 404
