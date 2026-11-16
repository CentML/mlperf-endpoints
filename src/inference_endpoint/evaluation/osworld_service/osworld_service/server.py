# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HTTP API of the OSWorld service (one run at a time per node)."""

from __future__ import annotations

import asyncio
import hmac
import logging
import shutil
import time
import uuid
from pathlib import Path
from typing import Any

import msgspec.json
from aiohttp import web
from pydantic import ValidationError

from . import API_VERSION, CAPABILITIES
from .artifacts import (
    SAFE_ARTIFACT_NAMES,
    atomic_write_bytes,
    redact_secrets,
    redact_text,
    resolve_artifact,
)
from .config import ServiceConfig
from .process import CancellationToken, RunCancelled
from .runner import RunnerProtocol
from .schemas import ArtifactInfo, RunRequest, RunStatus

logger = logging.getLogger(__name__)
_TERMINAL = frozenset({"succeeded", "failed", "cancelled"})


class RunManager:
    def __init__(self, config: ServiceConfig, runner: RunnerProtocol):
        self.config = config
        self.runner = runner
        self.runs: dict[str, RunStatus] = {}
        self._tokens: dict[str, CancellationToken] = {}
        self._secrets: dict[str, set[str]] = {}
        self._tasks: dict[str, asyncio.Task] = {}
        self._lock = asyncio.Lock()

    def run_dir(self, run_id: str) -> Path:
        return self.config.artifact_root / run_id

    def public(self, status: RunStatus) -> dict[str, Any]:
        return redact_secrets(
            status.model_dump(mode="json"), self._secrets.get(status.run_id, set())
        )

    async def submit(self, request: RunRequest) -> RunStatus:
        async with self._lock:
            if any(r.status in {"queued", "running"} for r in self.runs.values()):
                raise web.HTTPTooManyRequests(text="an OSWorld run is already active")
            run_id = uuid.uuid4().hex
            now = time.time()
            status = RunStatus(
                run_id=run_id,
                status="queued",
                phase="queued",
                created_at=now,
                updated_at=now,
                trials_total=len(request.task_ids) * request.n_attempts,
            )
            self.runs[run_id] = status
            self._tokens[run_id] = CancellationToken()
            self._secrets[run_id] = {
                s for s in (request.endpoint_api_key, self.config.auth_token) if s
            }
        run_dir = self.run_dir(run_id)
        run_dir.mkdir(parents=True)
        atomic_write_bytes(
            run_dir / "request.json",
            msgspec.json.encode(
                redact_secrets(request.model_dump(mode="json"), self._secrets[run_id])
            ),
        )
        self._persist(status)
        self._tasks[run_id] = asyncio.create_task(self._execute(run_id, request))
        return status

    def _update(self, run_id: str, **updates: Any) -> RunStatus:
        data = self.runs[run_id].model_dump()
        data.update(updates)
        data["updated_at"] = time.time()
        if data["status"] in _TERMINAL and data["finished_at"] is None:
            data["finished_at"] = data["updated_at"]
        status = RunStatus.model_validate(data)
        self.runs[run_id] = status
        self._persist(status)
        return status

    def _persist(self, status: RunStatus) -> None:
        try:
            atomic_write_bytes(
                self.run_dir(status.run_id) / "status.json",
                msgspec.json.encode(self.public(status)),
            )
        except OSError:
            logger.exception("Could not persist status of run %s", status.run_id)

    def _artifacts(self, run_id: str) -> list[ArtifactInfo]:
        run_dir = self.run_dir(run_id)
        return [
            ArtifactInfo(name=name, url=f"/v1/runs/{run_id}/artifacts/{name}")
            for name in SAFE_ARTIFACT_NAMES
            if (run_dir / name).is_file()
        ]

    async def _execute(self, run_id: str, request: RunRequest) -> None:
        token = self._tokens[run_id]
        loop = asyncio.get_running_loop()

        def on_phase(phase: str) -> None:
            loop.call_soon_threadsafe(
                lambda: self._update(
                    run_id, status="running", phase=phase, message=phase
                )
            )

        try:
            self._update(run_id, status="running", phase="preflight")
            result = await asyncio.to_thread(
                self.runner.run, request, self.run_dir(run_id), token, on_phase
            )
        except RunCancelled:
            self._update(run_id, status="cancelled", phase="cancelled")
        except Exception as exc:  # noqa: BLE001 - every failure becomes the run's error
            if token.is_cancelled():
                self._update(run_id, status="cancelled", phase="cancelled")
            else:
                logger.exception("OSWorld run %s failed", run_id)
                self._update(
                    run_id,
                    status="failed",
                    phase="failed",
                    error=redact_text(str(exc), self._secrets[run_id]),
                )
        else:
            self._update(
                run_id,
                status="succeeded",
                phase="succeeded",
                result=result,
                trials_completed=result.get("trials_reported", 0),
            )
        finally:
            self.runs[run_id] = self.runs[run_id].model_copy(
                update={"artifacts": self._artifacts(run_id)}
            )
            self._persist(self.runs[run_id])
            self._tasks.pop(run_id, None)
            await asyncio.to_thread(self._prune)

    def _prune(self) -> None:
        finished = sorted(
            (r for r in self.runs.values() if r.status in _TERMINAL),
            key=lambda r: r.finished_at or 0,
        )
        for run in finished[: max(0, len(finished) - self.config.max_stored_runs)]:
            self.runs.pop(run.run_id, None)
            self._tokens.pop(run.run_id, None)
            self._secrets.pop(run.run_id, None)
            shutil.rmtree(self.run_dir(run.run_id), ignore_errors=True)

    async def get(self, run_id: str) -> RunStatus:
        status = self.runs.get(run_id)
        if status is None:
            raise web.HTTPNotFound(text="unknown run_id")
        if status.status == "running":
            done = await asyncio.to_thread(
                self.runner.trials_completed, self.run_dir(run_id)
            )
            if done != status.trials_completed:
                status = self._update(run_id, trials_completed=done)
        return status

    async def cancel(self, run_id: str) -> RunStatus:
        status = await self.get(run_id)
        if status.status in _TERMINAL:
            return status
        await asyncio.to_thread(self._tokens[run_id].cancel)
        task = self._tasks.get(run_id)
        if task is not None:
            await asyncio.wait({task}, timeout=120)
        return self.runs[run_id]

    async def shutdown(self) -> None:
        for run_id, status in list(self.runs.items()):
            if status.status not in _TERMINAL:
                await self.cancel(run_id)


MANAGER_KEY = web.AppKey("manager", RunManager)


def create_app(config: ServiceConfig, runner: RunnerProtocol) -> web.Application:
    if bool(config.auth_token) == config.allow_unauthenticated:
        raise ValueError(
            "OSWorld service needs exactly one of --auth-token or "
            "--allow-unauthenticated (trusted development only)"
        )
    config.artifact_root.mkdir(parents=True, exist_ok=True)
    manager = RunManager(config, runner)

    @web.middleware
    async def auth(request: web.Request, handler: Any) -> web.StreamResponse:
        if (
            request.path != "/health"
            and config.auth_token
            and not hmac.compare_digest(
                request.headers.get("Authorization", ""), f"Bearer {config.auth_token}"
            )
        ):
            raise web.HTTPUnauthorized(text="unauthorized")
        return await handler(request)

    app = web.Application(middlewares=[auth])
    app[MANAGER_KEY] = manager

    async def on_shutdown(app: web.Application) -> None:
        await app[MANAGER_KEY].shutdown()

    app.on_shutdown.append(on_shutdown)

    async def health(request: web.Request) -> web.Response:
        endpoint = request.query.get("endpoint")
        info = await asyncio.to_thread(runner.health, endpoint)
        return web.json_response(
            {
                "api_version": API_VERSION,
                "capabilities": CAPABILITIES,
                "status": "ok",
                "active_runs": sum(
                    1 for r in manager.runs.values() if r.status not in _TERMINAL
                ),
                **info,
            }
        )

    async def post_run(request: web.Request) -> web.Response:
        try:
            run_request = RunRequest.model_validate(await request.json())
        except (ValidationError, ValueError) as exc:
            raise web.HTTPBadRequest(text=f"invalid run request: {exc}") from exc
        status = await manager.submit(run_request)
        return web.json_response(manager.public(status), status=202)

    async def get_run(request: web.Request) -> web.Response:
        status = await manager.get(request.match_info["run_id"])
        return web.json_response(manager.public(status))

    async def cancel_run(request: web.Request) -> web.Response:
        status = await manager.cancel(request.match_info["run_id"])
        return web.json_response(manager.public(status))

    async def get_artifact(request: web.Request) -> web.StreamResponse:
        run_id = request.match_info["run_id"]
        await manager.get(run_id)
        try:
            path = resolve_artifact(manager.run_dir(run_id), request.match_info["name"])
        except FileNotFoundError as exc:
            raise web.HTTPNotFound(text="artifact not found") from exc
        return web.FileResponse(path)

    app.router.add_get("/health", health)
    app.router.add_post("/v1/runs", post_run)
    app.router.add_get("/v1/runs/{run_id}", get_run)
    app.router.add_post("/v1/runs/{run_id}/cancel", cancel_run)
    app.router.add_get("/v1/runs/{run_id}/artifacts/{name}", get_artifact)
    return app
