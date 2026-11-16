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

"""One OSWorld run: preflight of the external server, then the trials."""

from __future__ import annotations

import time
from collections.abc import Callable
from pathlib import Path
from typing import Any, Protocol

import msgspec.json

from . import cua_harbor, pins
from .artifacts import atomic_write_bytes, redact_text
from .config import HarborConfig
from .preflight import PreflightError, endpoint_reachable, served_model
from .process import CancellationToken, RunCancelled, RunnerError, run_logged
from .schemas import RunRequest

PhaseCallback = Callable[[str], None]


class RunnerProtocol(Protocol):
    def run(
        self,
        request: RunRequest,
        run_dir: Path,
        cancel_token: CancellationToken,
        on_phase: PhaseCallback,
    ) -> dict[str, Any]: ...

    def trials_completed(self, run_dir: Path) -> int: ...

    def health(self, endpoint_url: str | None) -> dict[str, Any]: ...


def _write_json(path: Path, value: Any) -> None:
    atomic_write_bytes(path, msgspec.json.encode(value))


class OSWorldRunner:
    """Runs cua-harbor on this node; verified against the pins at construction."""

    def __init__(
        self,
        cfg: HarborConfig,
        *,
        verify_image: bool = True,
        pinned_commit: str = pins.CUA_HARBOR_COMMIT,
    ):
        self.cfg = cfg
        self.platform = cua_harbor.host_platform()
        self.commit = cua_harbor.verify_checkout(cfg, pinned_commit)
        self.image_sha256 = (
            cua_harbor.verify_image(cfg, self.platform) if verify_image else None
        )
        self.model_files_sha256 = cua_harbor.model_files_digest(cfg)

    def health(self, endpoint_url: str | None) -> dict[str, Any]:
        info: dict[str, Any] = {
            "platform": self.platform,
            "cua_harbor_commit": self.commit,
            "image_sqsh_sha256": self.image_sha256,
            "model_files_sha256": self.model_files_sha256,
        }
        if endpoint_url:
            error = endpoint_reachable(endpoint_url.rstrip("/").removesuffix("/v1"))
            info["endpoint_reachable"] = error is None
            info["endpoint_error"] = error
        return info

    def trials_completed(self, run_dir: Path) -> int:
        return cua_harbor.finished_trials(run_dir / "jobs")

    def _check_platform(self, request: RunRequest) -> str:
        if request.platform and request.platform != self.platform:
            raise PreflightError(
                f"service runs on {self.platform}; the request asks for "
                f"{request.platform}"
            )
        return self.platform

    def _check_tasks(self, request: RunRequest) -> dict[str, int]:
        missing = cua_harbor.missing_tasks(self.cfg, request.task_ids)
        if missing:
            raise PreflightError(f"unknown tasks: {missing[:10]}")
        return {"tasks": len(request.task_ids)}

    def _preflight(self, request: RunRequest, run_dir: Path) -> Path:
        """Checks before any desktop starts; returns the attached server dir.

        ``serve.sh --use`` checks template parity and the image limit; the
        served name and context window are checked here first.
        """
        cfg = self.cfg
        url, key, model = (
            request.endpoint_url,
            request.endpoint_api_key,
            request.model_name,
        )
        checks: list[dict[str, Any]] = []

        def check(name: str, fn: Callable[[], Any]) -> Any:
            try:
                detail = fn()
            except (PreflightError, RunnerError, OSError, ValueError) as exc:
                checks.append({"name": name, "ok": False, "detail": str(exc)})
                raise PreflightError(f"{name}: {exc}") from exc
            checks.append({"name": name, "ok": True, "detail": detail})
            return detail

        try:
            check("platform", lambda: self._check_platform(request))
            check("tasks", lambda: self._check_tasks(request))
            check(
                "served_model",
                lambda: served_model(url, key, model, cfg.min_max_model_len),
            )
            server_dir = Path(
                check(
                    "attach",
                    lambda: str(cua_harbor.attach(cfg, run_dir, url, key, model)),
                )
            )
        finally:
            _write_json(
                run_dir / "preflight.json",
                {"ok": all(c["ok"] for c in checks), "checks": checks},
            )
        record: dict[str, Any] = {
            "server_dir": str(server_dir),
            "serve_args": msgspec.json.decode(
                (server_dir / "serve.args.json").read_bytes()
            ),
            "cua_harbor_commit": self.commit,
            "image_sqsh_sha256": self.image_sha256,
            "model_files_sha256": self.model_files_sha256,
            "platform": self.platform,
        }
        observed = server_dir / "serve.observed.json"
        if observed.is_file():
            record["observed"] = msgspec.json.decode(observed.read_bytes())
        _write_json(run_dir / "server_record.json", record)
        return server_dir

    def run(
        self,
        request: RunRequest,
        run_dir: Path,
        cancel_token: CancellationToken,
        on_phase: PhaseCallback,
    ) -> dict[str, Any]:
        cfg = self.cfg
        secrets = {request.endpoint_api_key or ""}
        on_phase("preflight")
        try:
            server_dir = self._preflight(request, run_dir)
        except PreflightError as exc:
            raise RunnerError(f"preflight refused the run: {exc}") from exc
        if cancel_token.is_cancelled():
            raise RunCancelled("cancelled")

        on_phase("trials")
        jobs_dir = run_dir / "jobs"
        jobs_dir.mkdir(exist_ok=True)
        log = run_dir / "harbor.raw.log"
        cmd = cua_harbor.harbor_command(
            cfg,
            model=request.model_name,
            generation_params=request.generation_params,
            task_ids=request.task_ids,
            n_attempts=request.n_attempts,
            max_turns=request.max_turns,
            concurrency=request.concurrency,
            server_dir=server_dir,
            jobs_dir=jobs_dir,
        )
        env = cua_harbor.harbor_env(
            cfg,
            request.endpoint_url,
            request.endpoint_api_key,
            server_dir,
            request.model_name,
        )
        started = time.time()
        try:
            exit_code = run_logged(
                cmd,
                log,
                cwd=cfg.cua_harbor_dir,
                env=env,
                timeout_s=cfg.run_timeout_s,
                cancel_token=cancel_token,
            )
        finally:
            if log.exists():
                (run_dir / "harbor.log").write_text(
                    redact_text(log.read_text(errors="replace"), secrets)
                )
                log.unlink()
        return self._summarize(
            request, run_dir, jobs_dir, exit_code, time.time() - started
        )

    def _summarize(
        self,
        request: RunRequest,
        run_dir: Path,
        jobs_dir: Path,
        exit_code: int,
        seconds: float,
    ) -> dict[str, Any]:
        trials = cua_harbor.read_trials(jobs_dir)
        if not trials:
            raise RunnerError(
                f"harbor exited {exit_code} without finishing a trial; see harbor.log"
            )
        expected = len(request.task_ids) * request.n_attempts
        errored = sum(1 for t in trials if t["exception"] or t["reward"] is None)
        total = sum(t["reward"] or 0.0 for t in trials if not t["exception"])
        result = {
            "trials": trials,
            "trials_expected": expected,
            "trials_reported": len(trials),
            "trials_errored": errored,
            "mean_reward": total / expected,
            "harbor_exit_code": exit_code,
            "seconds": round(seconds, 1),
            "job_dir": str(jobs_dir / cua_harbor.JOB_NAME),
            "generation_params": request.generation_params,
            "n_attempts": request.n_attempts,
            "max_turns": request.max_turns,
            "concurrency": request.concurrency,
            "cua_harbor_commit": self.commit,
            "image_sqsh_sha256": self.image_sha256,
            "model_files_sha256": self.model_files_sha256,
            "platform": self.platform,
        }
        _write_json(run_dir / "osworld_results.json", result)
        rows = ["task_id\ttrial\treward\texception"] + [
            f"{t['task_id']}\t{t['trial']}\t{'' if t['reward'] is None else t['reward']}"
            f"\t{t['exception'] or ''}"
            for t in trials
        ]
        (run_dir / "trials.tsv").write_text("\n".join(rows) + "\n")
        return result
