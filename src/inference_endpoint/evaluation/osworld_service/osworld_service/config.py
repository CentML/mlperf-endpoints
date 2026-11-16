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

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True)
class ServiceConfig:
    host: str = "127.0.0.1"
    port: int = 18090
    artifact_root: Path = Path("osworld_service_artifacts")
    auth_token: str | None = None
    allow_unauthenticated: bool = False
    max_stored_runs: int = 20


@dataclass(frozen=True)
class HarborConfig:
    """Where cua-harbor, the desktop image and the served model's files live.

    ``model_files_dir`` holds the model's non-weight files (``config.json``,
    tokenizer, ``generation_config.json``, processor config, chat template):
    what ``serve.sh --use`` needs to check a server it did not start.
    """

    cua_harbor_dir: Path
    model_files_dir: Path
    image_sqsh: Path
    work_root: Path
    tasks_subdir: str = "adapters/osworld-novm/tasks"
    agent_config: str = "configs/agents/local/computer-1.yaml"
    model_image_size: tuple[int, int] = (1280, 720)
    min_max_model_len: int = 131072
    run_timeout_s: int = 12 * 60 * 60
    extra_env: dict[str, str] = field(default_factory=dict)
