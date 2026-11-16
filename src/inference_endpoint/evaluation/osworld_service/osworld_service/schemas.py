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

from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RunState = Literal["queued", "running", "succeeded", "failed", "cancelled"]
RunPhase = Literal["queued", "preflight", "trials", "succeeded", "failed", "cancelled"]
Platform = Literal["x86_64", "aarch64"]

# Sampling fields the agent sends each turn; anything else is refused.
GENERATION_PARAM_KEYS = frozenset({"temperature", "top_p", "top_k", "max_tokens"})


class RunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    model_name: str = Field(min_length=1)
    endpoint_urls: list[str] = Field(min_length=1, max_length=1)
    endpoint_api_key: str | None = None
    generation_params: dict[str, Any] = Field(default_factory=dict)
    task_ids: list[str] = Field(min_length=1)
    n_attempts: int = Field(ge=1, le=16)
    max_turns: int = Field(ge=1, le=500)
    concurrency: int = Field(ge=1, le=256)
    platform: Platform | None = None

    @field_validator("endpoint_urls")
    @classmethod
    def _validate_endpoint_urls(cls, urls: list[str]) -> list[str]:
        for url in urls:
            try:
                parsed = urlparse(url)
                _ = parsed.port
            except ValueError as exc:
                raise ValueError(f"endpoint URL must be HTTP(S): {url!r}") from exc
            if parsed.scheme not in {"http", "https"} or parsed.hostname is None:
                raise ValueError(f"endpoint URL must be HTTP(S): {url!r}")
        return urls

    @field_validator("generation_params")
    @classmethod
    def _validate_generation_params(cls, params: dict[str, Any]) -> dict[str, Any]:
        unknown = set(params) - GENERATION_PARAM_KEYS
        if unknown:
            raise ValueError(
                f"unsupported generation params {sorted(unknown)}; the agent sends "
                f"only {sorted(GENERATION_PARAM_KEYS)}"
            )
        return params

    @field_validator("task_ids")
    @classmethod
    def _validate_task_ids(cls, task_ids: list[str]) -> list[str]:
        for task in task_ids:
            if not task or "/" in task or task.startswith("."):
                raise ValueError(f"invalid task id {task!r}")
        if len(set(task_ids)) != len(task_ids):
            raise ValueError("task_ids must be unique")
        return task_ids

    @model_validator(mode="after")
    def _endpoint_url(self) -> RunRequest:
        base = self.endpoint_urls[0].rstrip("/")
        self.endpoint_urls = [base[: -len("/v1")] if base.endswith("/v1") else base]
        return self

    @property
    def endpoint_url(self) -> str:
        return self.endpoint_urls[0]


class ArtifactInfo(BaseModel):
    name: str
    url: str


class RunStatus(BaseModel):
    run_id: str
    status: RunState
    phase: RunPhase
    created_at: float
    updated_at: float
    finished_at: float | None = None
    trials_total: int = 0
    trials_completed: int = 0
    message: str | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    artifacts: list[ArtifactInfo] = Field(default_factory=list)
