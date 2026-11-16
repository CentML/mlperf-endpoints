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

"""Everything the service knows about cua-harbor.

The service drives cua-harbor from a local checkout pinned in ``pins.py``:

- ``scripts/serve.sh --use URL --model DIR`` attaches to the external model
  server, given the model's non-weight files. It writes a server directory and
  refuses a server whose ``/tokenize`` of a probe chat differs from the model
  entry's served template, or that does not take exactly the entry's images per
  prompt.
- ``scripts/run.sh`` runs the trials with the computer-1 agent and the
  osworld-novm Enroot environment, one desktop per trial on this node.
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

from . import pins
from .config import HarborConfig
from .preflight import PreflightError, sha256_file
from .process import RunnerError

ENVIRONMENT_IMPORT = "osworld_novm.enroot.environment:EnrootEnvironment"
JOB_NAME = "osworld"
_ATTACH_TIMEOUT_S = 900
# OpenAI-style request field -> computer-1 format field (``--ak``).
_FORMAT_FIELDS = {
    "temperature": "temperature",
    "top_p": "top_p",
    "top_k": "top_k",
    "max_tokens": "max_new_tokens",
}


def host_platform() -> str:
    return platform.machine()


def harbor_python(cfg: HarborConfig) -> Path:
    return cfg.cua_harbor_dir / ".venv" / "bin" / "python"


def verify_checkout(cfg: HarborConfig, pinned: str = pins.CUA_HARBOR_COMMIT) -> str:
    """The checkout's commit; refuses another commit or modified tracked files."""
    try:
        head = subprocess.run(
            ["git", "-C", str(cfg.cua_harbor_dir), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        dirty = subprocess.run(
            ["git", "-C", str(cfg.cua_harbor_dir), "status", "--porcelain"]
            + ["--untracked-files=no"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise RunnerError(f"{cfg.cua_harbor_dir} is not a git checkout: {exc}") from exc
    if head != pinned:
        raise RunnerError(f"cua-harbor checkout is at {head}; the pin is {pinned}")
    if dirty:
        raise RunnerError(f"cua-harbor checkout has modified files:\n{dirty}")
    if not harbor_python(cfg).is_file():
        raise RunnerError(
            f"{harbor_python(cfg)} is missing; run `uv sync` in the checkout"
        )
    return head


def verify_image(cfg: HarborConfig, arch: str) -> str:
    """The desktop image's sha256; refuses an image other than the pinned one."""
    pin = pins.IMAGES.get(arch)
    if pin is None:
        raise RunnerError(f"no desktop image is pinned for {arch}")
    digest = sha256_file(cfg.image_sqsh)
    if digest != pin.sqsh_sha256:
        raise RunnerError(
            f"{cfg.image_sqsh} has sha256 {digest}; the pin for {arch} is "
            f"{pin.sqsh_sha256}"
        )
    return digest


def model_files_digest(cfg: HarborConfig) -> str:
    """One sha256 over the model's non-weight files, recorded with each run."""
    digest = hashlib.sha256()
    for path in sorted(cfg.model_files_dir.iterdir()):
        if path.is_file():
            digest.update(path.name.encode())
            digest.update(path.read_bytes())
    return digest.hexdigest()


def missing_tasks(cfg: HarborConfig, task_ids: list[str]) -> list[str]:
    tasks_dir = cfg.cua_harbor_dir / cfg.tasks_subdir
    return [t for t in task_ids if not (tasks_dir / t).is_dir()]


def _base_env(base_url: str, api_key: str | None) -> dict[str, str]:
    """The service's environment minus its own virtualenv, for cua-harbor's tools."""
    env = dict(os.environ)
    venv = env.pop("VIRTUAL_ENV", None)
    if venv:
        env["PATH"] = os.pathsep.join(
            p for p in env.get("PATH", "").split(os.pathsep) if not p.startswith(venv)
        )
    env.pop("PYTHONHOME", None)
    env.pop("CUA_MODEL_SERVER_API_KEY", None)
    if api_key:
        env["CUA_MODEL_SERVER_API_KEY"] = api_key
    host = urlparse(base_url).hostname or ""
    no_proxy = {h for h in env.get("NO_PROXY", "").split(",") if h}
    no_proxy |= {host, "127.0.0.1", "localhost"}
    env["NO_PROXY"] = env["no_proxy"] = ",".join(sorted(no_proxy))
    return env


def attach(
    cfg: HarborConfig, run_dir: Path, base_url: str, api_key: str | None, model: str
) -> Path:
    """``serve.sh --use``: the run's server directory for the external server."""
    servers = run_dir / "servers"
    env = _base_env(base_url, api_key)
    env["SERVERS_DIR"] = str(servers)
    log = run_dir / "attach.log"
    with log.open("w") as out:
        try:
            done = subprocess.run(
                [
                    "bash",
                    "scripts/serve.sh",
                    "--use",
                    base_url,
                    "--model",
                    str(cfg.model_files_dir),
                ],
                cwd=str(cfg.cua_harbor_dir),
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                timeout=_ATTACH_TIMEOUT_S,
            )
        except subprocess.TimeoutExpired as exc:
            raise PreflightError(f"serve.sh --use timed out; see {log}") from exc
    if done.returncode != 0:
        tail = log.read_text(errors="replace").strip().splitlines()[-5:]
        raise PreflightError(f"serve.sh --use refused the server: {tail}")
    server_dir = servers / model
    if not (server_dir / "serve.args.json").is_file():
        raise PreflightError(
            f"serve.sh --use attached no server named {model!r} under {servers}"
        )
    return server_dir


def harbor_command(
    cfg: HarborConfig,
    *,
    model: str,
    generation_params: dict[str, Any],
    task_ids: list[str],
    n_attempts: int,
    max_turns: int,
    concurrency: int,
    server_dir: Path,
    jobs_dir: Path,
) -> list[str]:
    width, height = cfg.model_image_size
    cmd = [
        "scripts/run.sh",
        "-c",
        cfg.agent_config,
        "-m",
        f"openai/{model}",
        "--ak",
        f"serve_args={server_dir / 'serve.args.json'}",
        "--ak",
        f"model_image_size=[{width},{height}]",
        "--ak",
        f"max_turns={max_turns}",
    ]
    for key, value in generation_params.items():
        cmd += ["--ak", f"{_FORMAT_FIELDS[key]}={value}"]
    cmd += ["-p", cfg.tasks_subdir]
    for task in task_ids:
        cmd += ["-i", task]
    cmd += [
        "--env",
        ENVIRONMENT_IMPORT,
        "--n-attempts",
        str(n_attempts),
        "-n",
        str(concurrency),
        "--jobs-dir",
        str(jobs_dir),
        "--job-name",
        JOB_NAME,
    ]
    return cmd


def harbor_env(
    cfg: HarborConfig, base_url: str, api_key: str | None, server_dir: Path, model: str
) -> dict[str, str]:
    env = _base_env(base_url, api_key)
    env["OSW_ENROOT_SQUASHFS"] = str(cfg.image_sqsh)
    env["OSW_ENROOT_WORK_ROOT"] = str(cfg.work_root)
    env["CUA_MODEL_SERVER_URL"] = base_url
    env["SERVE_DIR"] = str(server_dir)
    env["M"] = model
    env.update(cfg.extra_env)
    return env


def finished_trials(jobs_dir: Path) -> int:
    return sum(1 for _ in (jobs_dir / JOB_NAME).glob("*/result.json"))


def read_trials(jobs_dir: Path) -> list[dict[str, Any]]:
    """One record per finished trial of the job: task, reward, exception."""
    trials: list[dict[str, Any]] = []
    for path in sorted((jobs_dir / JOB_NAME).glob("*/result.json")):
        try:
            record = json.loads(path.read_text())
        except (OSError, ValueError):
            trials.append(
                {
                    "task_id": path.parent.name.split("__")[0],
                    "trial": path.parent.name,
                    "reward": None,
                    "exception": "unreadable result.json",
                }
            )
            continue
        verifier = record.get("verifier_result") or {}
        reward = (verifier.get("rewards") or {}).get("reward")
        exception = (record.get("exception_info") or {}).get("exception_type")
        trials.append(
            {
                "task_id": record.get("task_name") or path.parent.name.split("__")[0],
                "trial": path.parent.name,
                "reward": reward if isinstance(reward, int | float) else None,
                "exception": exception,
            }
        )
    return trials
