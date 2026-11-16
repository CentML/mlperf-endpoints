# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Real HTTP fakes for the OSWorld service and scorer tests.

``FakeVLLMServer`` answers the vLLM routes the service reads; ``ServiceThread``
serves the real OSWorld service app (with a fake runner) on a local port.
"""

import asyncio
import json
import threading
import time
from pathlib import Path
from typing import Any

from aiohttp import web
from inference_endpoint.evaluation.osworld_service.osworld_service.config import (
    ServiceConfig,
)
from inference_endpoint.evaluation.osworld_service.osworld_service.process import (
    RunCancelled,
)
from inference_endpoint.evaluation.osworld_service.osworld_service.server import (
    create_app,
)
from inference_endpoint.testing.echo_server import EchoServer

MODEL = "qwen-test"


class FakeVLLMServer(EchoServer):
    """vLLM's model, version and chat routes; records any chat request."""

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self.max_model_len = 131072
        self.chat_requests: list[dict[str, Any]] = []

    def _register_routes(self, app: web.Application) -> None:
        app.router.add_get("/v1/models", self._models)
        app.router.add_get("/version", self._version)
        app.router.add_post("/v1/chat/completions", self._chat)

    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response(
            {"data": [{"id": MODEL, "max_model_len": self.max_model_len}]}
        )

    async def _version(self, request: web.Request) -> web.Response:
        return web.json_response({"version": "0.32.0-test"})

    async def _chat(self, request: web.Request) -> web.Response:
        self.chat_requests.append(await request.json())
        return web.json_response(
            {"choices": [{"message": {"role": "assistant", "content": "1"}}]}
        )


class FakeRunner:
    """Stands in for ``OSWorldRunner``: rewards from ``rewards[task]``."""

    def __init__(
        self,
        rewards: dict[str, float] | None = None,
        *,
        errored: set[str] | None = None,
        fail: str | None = None,
        block: threading.Event | None = None,
        platform: str = "x86_64",
    ):
        self.rewards = rewards or {}
        self.errored = errored or set()
        self.fail = fail
        self.block = block
        self.platform = platform
        self.requests: list[Any] = []
        self.health_queries: list[str | None] = []

    def health(self, endpoint_url):
        self.health_queries.append(endpoint_url)
        info = {"platform": self.platform, "cua_harbor_commit": "abc"}
        if endpoint_url:
            reachable = "unreachable" not in endpoint_url
            info["endpoint_reachable"] = reachable
            info["endpoint_error"] = None if reachable else "connection refused"
        return info

    def trials_completed(self, run_dir: Path) -> int:
        return 1 if self.block is not None else 0

    def run(self, request, run_dir: Path, cancel_token, on_phase):
        self.requests.append(request)
        on_phase("preflight")
        (run_dir / "preflight.json").write_text('{"ok": true, "checks": []}')
        on_phase("trials")
        if self.block is not None:
            while not self.block.is_set():
                if cancel_token.is_cancelled():
                    raise RunCancelled("cancelled")
                time.sleep(0.01)
        if self.fail:
            raise RuntimeError(self.fail)
        trials = [
            {
                "task_id": task,
                "trial": f"{task}__{attempt}",
                "reward": None if task in self.errored else self.rewards.get(task, 0.0),
                "exception": "AgentTimeoutError" if task in self.errored else None,
            }
            for task in request.task_ids
            for attempt in range(request.n_attempts)
        ]
        result = {"trials": trials, "job_dir": str(run_dir / "jobs" / "osworld")}
        (run_dir / "osworld_results.json").write_text(json.dumps(result))
        return result


class ServiceThread:
    """The OSWorld service app served from a background thread."""

    def __init__(self, runner: FakeRunner, root: Path, auth_token: str | None = None):
        self.app = create_app(
            ServiceConfig(
                artifact_root=root,
                auth_token=auth_token,
                allow_unauthenticated=auth_token is None,
            ),
            runner,
        )
        self._ready = threading.Event()
        self._loop: asyncio.AbstractEventLoop | None = None
        self._runner: web.AppRunner | None = None
        self.port = 0
        self._thread = threading.Thread(target=self._serve, daemon=True)

    def _serve(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._loop.run_until_complete(self._start())
        self._loop.run_forever()

    async def _start(self) -> None:
        self._runner = web.AppRunner(self.app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = site._server.sockets[0].getsockname()[1]  # type: ignore[union-attr]
        self._ready.set()

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/"

    def start(self) -> "ServiceThread":
        self._thread.start()
        assert self._ready.wait(5)
        return self

    def stop(self) -> None:
        assert self._loop is not None and self._runner is not None
        asyncio.run_coroutine_threadsafe(self._runner.cleanup(), self._loop).result(10)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(5)
