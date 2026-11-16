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

"""Service-backed OSWorld computer-use accuracy scorer."""

from __future__ import annotations

import logging
import os
import re
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import TYPE_CHECKING, Any, ClassVar
from urllib import error as urllib_error
from urllib import request as urllib_request
from urllib.parse import quote, urljoin, urlparse

import msgspec
from tqdm import tqdm

from ..dataset_manager.dataset import Dataset
from ..dataset_manager.predefined.osworld_cua import OSWorldCUA
from ..exceptions import SetupError
from ..utils.atomic_write import atomic_write_bytes
from .extractor import Extractor
from .scoring import Scorer

if TYPE_CHECKING:
    from ..config.schema import EndpointConfig, ModelParams

logger = logging.getLogger(__name__)

_SAFE_RUN_ID_RE = re.compile(r"^[A-Za-z0-9_-]+$")


class _RejectRedirectHandler(urllib_request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        return None


_NO_REDIRECT_OPENER = urllib_request.build_opener(_RejectRedirectHandler())

# model_params field -> request field of the OSWorld agent. ignore_eos and the
# perf replay's per-row output lengths are never sent: with a forced output
# length the model keeps writing past its turn, and the agent executes the tool
# calls it invents there.
GENERATION_PARAM_MAP: dict[str, str] = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_new_tokens": "max_tokens",
}


class OSWorldScorer(Scorer, scorer_id="osworld_cua"):
    """Run OSWorld tasks through the OSWorld service against the benchmark endpoint.

    The service runs cua-harbor's computer-1 agent in one desktop container per
    trial on its own CPU allocation, querying the same endpoint and model as the
    performance phase. It is called from benchmark finalization, after the
    performance phase has finished. The score is the mean reward over all
    ``tasks x n_attempts`` trials; a trial that errored or never reported counts
    as 0 and marks the score incomplete.
    """

    REQUIRES_EXTRACTOR: ClassVar[bool] = False
    SKIP_ENDPOINT_PHASE: ClassVar[bool] = True
    NEEDS_ENDPOINT: ClassVar[bool] = True
    SERVICE_API_VERSION: ClassVar[str] = "v1"
    REQUIRED_SERVICE_CAPABILITIES: ClassVar[set[str]] = {
        "osworld.run",
        "osworld.cancel",
        "artifacts.download",
    }
    SAFE_ARTIFACT_NAMES: ClassVar[set[str]] = {
        "osworld_results.json",
        "preflight.json",
        "server_record.json",
        "trials.tsv",
        "harbor.log",
        "status.json",
    }
    PLATFORMS: ClassVar[set[str]] = {"x86_64", "aarch64"}
    DEFAULTS: ClassVar[dict[str, Any]] = {
        "task_set": "osworld_cua_84",
        "n_attempts": 2,
        "max_turns": 30,
        "concurrency": 16,
        "service_timeout_s": 12 * 60 * 60,
        "poll_interval_s": 10.0,
    }

    def __init__(
        self,
        dataset_name: str,
        dataset: Dataset,
        report_dir: os.PathLike,
        extractor: type[Extractor] | None = None,
        ground_truth_column: str | None = "task_id",
        osworld_service_url: str | None = None,
        osworld_service_auth_token: str | None = None,
        task_set: str | None = None,
        num_tasks: int | None = None,
        n_attempts: int | None = None,
        max_turns: int | None = None,
        concurrency: int | None = None,
        platform: str | None = None,
        service_timeout_s: int | None = None,
        poll_interval_s: float | None = None,
        model_params: ModelParams | None = None,
        endpoint_config: EndpointConfig | None = None,
    ):
        super().__init__(
            dataset_name=dataset_name,
            dataset=dataset,
            report_dir=report_dir,
            extractor=extractor,
            ground_truth_column=ground_truth_column or "task_id",
        )
        self.report_dir: Path = Path(report_dir).resolve()
        options = self._resolve_options(
            {
                "osworld_service_url": osworld_service_url,
                "osworld_service_auth_token": osworld_service_auth_token,
                "task_set": task_set,
                "num_tasks": num_tasks,
                "n_attempts": n_attempts,
                "max_turns": max_turns,
                "concurrency": concurrency,
                "platform": platform,
                "service_timeout_s": service_timeout_s,
                "poll_interval_s": poll_interval_s,
            }
        )
        self.service_url: str = options["osworld_service_url"]
        self.auth_token: str | None = options["osworld_service_auth_token"]
        self.task_set: str = options["task_set"]
        self.num_tasks: int | None = options["num_tasks"]
        self.n_attempts: int = options["n_attempts"]
        self.max_turns: int = options["max_turns"]
        self.concurrency: int = options["concurrency"]
        self.platform: str | None = options["platform"]
        self.service_timeout_s: int = options["service_timeout_s"]
        self.poll_interval_s: float = options["poll_interval_s"]
        self.model_params = model_params
        self.endpoint_config = endpoint_config
        self._breakdown: dict[str, Any] | None = None

    # ------------------------------------------------------------------ options

    @staticmethod
    def _normalize_service_url(value: Any) -> str:
        if value is None or str(value).strip() == "":
            raise SetupError(
                "accuracy_config.extras.osworld_service_url is required for "
                "osworld_cua. Start the OSWorld service and pass its URL."
            )
        raw = str(value).strip()
        message = (
            "accuracy_config.extras.osworld_service_url must be an HTTP(S) "
            "service root URL with no path, query, or fragment"
        )
        try:
            parsed = urlparse(raw)
            _ = parsed.port
        except ValueError as exc:
            raise SetupError(message) from exc
        if (
            parsed.scheme not in {"http", "https"}
            or parsed.hostname is None
            or parsed.username is not None
            or parsed.password is not None
            or parsed.path not in {"", "/"}
            or parsed.params
            or parsed.query
            or parsed.fragment
        ):
            raise SetupError(message)
        return parsed._replace(path="/").geturl()

    @staticmethod
    def _positive_int(extras: dict[str, Any], key: str, default: Any) -> int | None:
        value = extras.get(key)
        if value is None or value == "":
            value = default
        if value is None:
            return None
        if isinstance(value, bool):
            raise SetupError(f"accuracy_config.extras.{key} must be an integer")
        try:
            parsed = int(value)
        except (TypeError, ValueError) as exc:
            raise SetupError(
                f"accuracy_config.extras.{key} must be an integer; got {value!r}"
            ) from exc
        if parsed < 1:
            raise SetupError(f"accuracy_config.extras.{key} must be >= 1; got {parsed}")
        return parsed

    @classmethod
    def _resolve_options(cls, extras: dict[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {
            "osworld_service_url": cls._normalize_service_url(
                extras.get("osworld_service_url")
            )
        }
        token = extras.get("osworld_service_auth_token")
        options["osworld_service_auth_token"] = (
            str(token) if token not in (None, "") else None
        )
        task_set = str(extras.get("task_set") or cls.DEFAULTS["task_set"])
        try:
            OSWorldCUA.task_set_path(task_set)
        except ValueError as exc:
            raise SetupError(str(exc)) from exc
        options["task_set"] = task_set
        options["num_tasks"] = cls._positive_int(extras, "num_tasks", None)
        for key in ("n_attempts", "max_turns", "concurrency", "service_timeout_s"):
            options[key] = cls._positive_int(extras, key, cls.DEFAULTS[key])
        poll = extras.get("poll_interval_s")
        poll = cls.DEFAULTS["poll_interval_s"] if poll is None else poll
        try:
            options["poll_interval_s"] = float(poll)
        except (TypeError, ValueError) as exc:
            raise SetupError(
                f"accuracy_config.extras.poll_interval_s must be numeric; got {poll!r}"
            ) from exc
        if options["poll_interval_s"] <= 0:
            raise SetupError("accuracy_config.extras.poll_interval_s must be > 0")
        platform = extras.get("platform")
        if platform not in (None, "") and platform not in cls.PLATFORMS:
            raise SetupError(
                f"accuracy_config.extras.platform must be one of "
                f"{sorted(cls.PLATFORMS)}; got {platform!r}"
            )
        options["platform"] = platform or None
        return options

    @classmethod
    def dataset_loader_kwargs(cls, extras: dict[str, Any]) -> dict[str, Any]:
        """Select the configured task set for loading."""
        return {"task_set": cls._resolve_options(extras)["task_set"]}

    @classmethod
    def external_sample_count(cls, extras: dict[str, Any]) -> int | None:
        """Tasks run by the service (before clamping to the task-set size)."""
        try:
            num_tasks = cls._positive_int(extras, "num_tasks", None)
        except SetupError:
            return None
        return num_tasks

    @staticmethod
    def generation_params(model_params: ModelParams) -> dict[str, Any]:
        """The sampling fields of ``model_params`` the agent sends each turn."""
        dumped = model_params.model_dump(exclude_none=True)
        return {
            wire: dumped[field]
            for field, wire in GENERATION_PARAM_MAP.items()
            if field in dumped
        }

    # --------------------------------------------------------------------- HTTP

    @staticmethod
    def _http_json(
        url: str,
        *,
        method: str = "GET",
        payload: dict[str, Any] | None = None,
        timeout_s: float = 30.0,
        auth_token: str | None = None,
    ) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        if auth_token:
            headers["Authorization"] = f"Bearer {auth_token}"
        data = None
        if payload is not None:
            data = msgspec.json.encode(payload)
            headers["Content-Type"] = "application/json"
        req = urllib_request.Request(url, data=data, headers=headers, method=method)
        try:
            with _NO_REDIRECT_OPENER.open(req, timeout=timeout_s) as resp:
                body = resp.read()
        except urllib_error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise SetupError(
                f"OSWorld service request failed: {url} returned HTTP "
                f"{exc.code}: {detail}"
            ) from exc
        except (urllib_error.URLError, OSError) as exc:
            reason = getattr(exc, "reason", exc)
            raise SetupError(
                f"OSWorld service is unreachable at {url}: {reason}"
            ) from exc
        try:
            return msgspec.json.decode(body, type=dict)
        except msgspec.DecodeError as exc:
            raise SetupError(
                f"OSWorld service returned invalid JSON from {url}"
            ) from exc

    @classmethod
    def _check_health(
        cls,
        service_url: str,
        auth_token: str | None,
        endpoint_url: str | None,
        platform: str | None,
    ) -> dict[str, Any]:
        url = urljoin(service_url, "health")
        if endpoint_url:
            url += "?endpoint=" + quote(endpoint_url, safe="")
        health = cls._http_json(url, timeout_s=30.0, auth_token=auth_token)
        if health.get("api_version") != cls.SERVICE_API_VERSION:
            raise SetupError(
                "OSWorld service API version mismatch: expected "
                f"{cls.SERVICE_API_VERSION!r}, got {health.get('api_version')!r}"
            )
        missing = cls.REQUIRED_SERVICE_CAPABILITIES - set(
            health.get("capabilities") or []
        )
        if missing:
            raise SetupError(
                "OSWorld service is missing required capabilities: "
                + ", ".join(sorted(missing))
            )
        if platform and health.get("platform") != platform:
            raise SetupError(
                f"OSWorld service runs on {health.get('platform')!r}; the config "
                f"asks for {platform!r}"
            )
        if endpoint_url and health.get("endpoint_reachable") is not True:
            raise SetupError(
                f"OSWorld service cannot reach the endpoint {endpoint_url}: "
                f"{health.get('endpoint_error') or 'no answer'}"
            )
        return health

    @classmethod
    def preflight(
        cls, extras: dict[str, Any], endpoint_config: EndpointConfig | None = None
    ) -> None:
        """Before the benchmark starts: the service is up and reaches the endpoint.

        Only ``GET /health`` is sent; the service checks the endpoint with
        ``GET /v1/models`` and starts no desktop.
        """
        options = cls._resolve_options(extras)
        endpoint_url = None
        if endpoint_config is not None and len(endpoint_config.endpoints) == 1:
            endpoint_url = endpoint_config.endpoints[0]
        cls._check_health(
            options["osworld_service_url"],
            options["osworld_service_auth_token"],
            endpoint_url,
            options["platform"],
        )

    @staticmethod
    def _validate_run_id(value: Any) -> str:
        if not isinstance(value, str) or not _SAFE_RUN_ID_RE.fullmatch(value):
            raise SetupError("OSWorld service returned an invalid run_id")
        return value

    def _cancel(self, run_id: str) -> None:
        try:
            self._http_json(
                urljoin(self.service_url, f"v1/runs/{run_id}/cancel"),
                method="POST",
                timeout_s=30.0,
                auth_token=self.auth_token,
            )
        except SetupError:
            logger.warning("Could not cancel OSWorld service run %s", run_id)

    def _download_artifacts(self, status: dict[str, Any], run_dir: Path) -> None:
        run_id = self._validate_run_id(status.get("run_id"))
        for artifact in status.get("artifacts") or []:
            if not isinstance(artifact, dict):
                continue
            name = str(artifact.get("name") or "")
            href = str(artifact.get("url") or "")
            if (
                name not in self.SAFE_ARTIFACT_NAMES
                or href != f"/v1/runs/{run_id}/artifacts/{name}"
            ):
                logger.warning("Ignoring unexpected OSWorld artifact %r", name)
                continue
            req = urllib_request.Request(
                urljoin(self.service_url, href.lstrip("/")),
                headers={"Accept": "application/octet-stream"},
            )
            if self.auth_token:
                req.add_header("Authorization", f"Bearer {self.auth_token}")
            target = run_dir / name
            tmp = target.with_suffix(target.suffix + ".tmp")
            try:
                with (
                    _NO_REDIRECT_OPENER.open(req, timeout=120.0) as resp,
                    tmp.open("wb") as out,
                ):
                    shutil.copyfileobj(resp, out, length=1024 * 1024)
                os.replace(tmp, target)
            except (urllib_error.URLError, OSError):
                logger.warning("Could not download OSWorld artifact %s", name)
            finally:
                tmp.unlink(missing_ok=True)

    # ------------------------------------------------------------------ scoring

    def score_single_sample(self, value: str, ground_truth: str) -> float:
        raise RuntimeError("OSWorldScorer runs in the OSWorld service; call score().")

    def score_breakdown(self) -> dict[str, Any] | None:
        """Per-domain mean reward, trial and error counts, pass@k, job location."""
        return self._breakdown

    def _selected_tasks(self) -> tuple[list[str], dict[str, str]]:
        df = self.dataset.dataframe
        if df is None:
            raise RuntimeError("OSWorld dataset must be loaded before scoring")
        if self.num_tasks is not None and self.num_tasks < len(df):
            df = df.iloc[: self.num_tasks]
        task_ids = [str(t) for t in df[self.ground_truth_column].tolist()]
        domains = dict(zip(task_ids, (str(d) for d in df["domain"]), strict=True))
        return task_ids, domains

    def _payload(self, task_ids: list[str]) -> dict[str, Any]:
        if self.model_params is None or not self.model_params.name:
            raise ValueError("model_params with a name is required for OSWorldScorer")
        if self.endpoint_config is None:
            raise ValueError("endpoint_config is required for OSWorldScorer")
        if len(self.endpoint_config.endpoints) != 1:
            raise SetupError(
                "OSWorld service mode supports exactly one endpoint URL; "
                f"got {len(self.endpoint_config.endpoints)}."
            )
        return {
            "model_name": self.model_params.name,
            "endpoint_urls": list(self.endpoint_config.endpoints),
            "endpoint_api_key": self.endpoint_config.api_key,
            "generation_params": self.generation_params(self.model_params),
            "task_ids": task_ids,
            "n_attempts": self.n_attempts,
            "max_turns": self.max_turns,
            "concurrency": self.concurrency,
            "platform": self.platform,
        }

    def _run_service(self, payload: dict[str, Any]) -> dict[str, Any] | None:
        run_id = ""
        bar = tqdm(
            total=len(payload["task_ids"]) * payload["n_attempts"],
            desc="OSWorld trials",
            unit="trial",
        )
        try:
            status = self._http_json(
                urljoin(self.service_url, "v1/runs"),
                method="POST",
                payload=payload,
                timeout_s=60.0,
                auth_token=self.auth_token,
            )
            run_id = self._validate_run_id(status.get("run_id"))
            deadline = time.monotonic() + self.service_timeout_s
            while status.get("status") not in {"succeeded", "failed", "cancelled"}:
                if time.monotonic() >= deadline:
                    raise SetupError(f"Timed out waiting for OSWorld run {run_id}")
                time.sleep(self.poll_interval_s)
                status = self._http_json(
                    urljoin(self.service_url, f"v1/runs/{run_id}"),
                    timeout_s=60.0,
                    auth_token=self.auth_token,
                )
                if status.get("run_id") != run_id:
                    raise SetupError("OSWorld service returned a mismatched run_id")
                done = status.get("trials_completed")
                if isinstance(done, int) and done > bar.n:
                    bar.update(done - bar.n)
        except (KeyboardInterrupt, SystemExit):
            if run_id:
                self._cancel(run_id)
            raise
        except SetupError:
            if run_id:
                self._cancel(run_id)
            logger.error("OSWorld service run failed", exc_info=True)
            return None
        finally:
            bar.close()

        run_dir = self.report_dir / "osworld_runs" / run_id
        run_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_bytes(run_dir / "service_status.json", msgspec.json.encode(status))
        self._download_artifacts(status, run_dir)
        if status.get("status") != "succeeded":
            logger.error(
                "OSWorld service run %s ended %s: %s",
                run_id,
                status.get("status"),
                status.get("error"),
            )
            return None
        result = status.get("result")
        return result if isinstance(result, dict) else None

    def score(self) -> tuple[float | None, int]:
        """Run the task set in the service. Returns (mean reward, 1)."""
        self.complete = True
        task_ids, domains = self._selected_tasks()
        if not task_ids:
            self.complete = False
            return None, 1
        payload = self._payload(task_ids)
        result = self._run_service(payload)
        if result is None:
            self.complete = False
            return None, 1
        atomic_write_bytes(
            self.report_dir / "osworld_results.json", msgspec.json.encode(result)
        )
        return self._score_result(result, task_ids, domains), 1

    def _score_result(
        self, result: dict[str, Any], task_ids: list[str], domains: dict[str, str]
    ) -> float | None:
        expected = len(task_ids) * self.n_attempts
        rewards: dict[str, list[float]] = defaultdict(list)
        errors = 0
        trials = result.get("trials")
        if not isinstance(trials, list):
            logger.error("OSWorld service result has no trial list")
            self.complete = False
            return None
        for trial in trials:
            task = trial.get("task_id") if isinstance(trial, dict) else None
            if task not in domains:
                logger.warning("Ignoring OSWorld trial of unrequested task %r", task)
                continue
            reward = trial.get("reward")
            if trial.get("exception") or not isinstance(reward, int | float):
                errors += 1
                reward = 0.0
            rewards[task].append(float(reward))
        reported = sum(len(v) for v in rewards.values())
        if reported > expected or any(
            len(v) > self.n_attempts for v in rewards.values()
        ):
            logger.error("OSWorld service reported more trials than requested")
            self.complete = False
            return None
        if errors or reported != expected:
            logger.warning(
                "OSWorld: %d of %d trials reported, %d errored; errored and "
                "missing trials score 0 and the score is marked incomplete",
                reported,
                expected,
                errors,
            )
            self.complete = False

        by_domain: dict[str, dict[str, Any]] = {}
        for domain in sorted(set(domains.values())):
            tasks = [t for t in task_ids if domains[t] == domain]
            total = sum(sum(rewards.get(t, [])) for t in tasks)
            by_domain[domain] = {
                "mean_reward": total / (len(tasks) * self.n_attempts),
                "tasks": len(tasks),
            }
        solved = sum(1 for t in task_ids if any(r >= 1.0 for r in rewards.get(t, [])))
        mean = sum(sum(v) for v in rewards.values()) / expected
        self._breakdown = {
            "mean_reward": mean,
            f"pass_at_{self.n_attempts}": solved / len(task_ids),
            "tasks": len(task_ids),
            "n_attempts": self.n_attempts,
            "trials_expected": expected,
            "trials_reported": reported,
            "trials_errored": errors,
            "by_domain": by_domain,
            "job_dir": result.get("job_dir"),
        }
        logger.info(
            "OSWorld: mean reward %.4f over %d trials (%d errored)",
            mean,
            expected,
            errors,
        )
        return mean
