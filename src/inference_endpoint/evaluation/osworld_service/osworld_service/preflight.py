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

"""Checks of the external model server that cua-harbor's attach leaves out.

Nothing here starts the server or sends a generation request: every fact is
read from its HTTP API.
"""

from __future__ import annotations

import hashlib
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request

import msgspec.json


class PreflightError(RuntimeError):
    pass


def http_json(
    url: str,
    *,
    api_key: str | None = None,
    payload: dict[str, Any] | None = None,
    timeout_s: float = 30.0,
) -> Any:
    headers = {"Accept": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    data = None
    if payload is not None:
        data = msgspec.json.encode(payload)
        headers["Content-Type"] = "application/json"
    req = urllib_request.Request(url, data=data, headers=headers)
    try:
        with urllib_request.urlopen(req, timeout=timeout_s) as resp:
            body = resp.read()
    except urllib_error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise PreflightError(f"{url}: HTTP {exc.code}: {detail}") from exc
    except (urllib_error.URLError, OSError) as exc:
        raise PreflightError(f"{url}: {getattr(exc, 'reason', exc)}") from exc
    try:
        return msgspec.json.decode(body)
    except msgspec.DecodeError as exc:
        raise PreflightError(f"{url}: invalid JSON") from exc


def endpoint_reachable(base_url: str, api_key: str | None = None) -> str | None:
    """``None`` when ``GET /v1/models`` answers, else the error (no generation)."""
    try:
        http_json(f"{base_url}/v1/models", api_key=api_key, timeout_s=10.0)
    except PreflightError as exc:
        return str(exc)
    return None


def served_model(
    base_url: str, api_key: str | None, model_name: str, min_max_model_len: int
) -> dict[str, Any]:
    """The ``/v1/models`` entry of ``model_name``; refuses a short context window."""
    models = http_json(f"{base_url}/v1/models", api_key=api_key)
    entries = models.get("data") if isinstance(models, dict) else None
    entry = next(
        (m for m in entries or [] if isinstance(m, dict) and m.get("id") == model_name),
        None,
    )
    if entry is None:
        served = [m.get("id") for m in entries or [] if isinstance(m, dict)]
        raise PreflightError(f"model {model_name!r} is not served; served: {served}")
    max_len = entry.get("max_model_len")
    if not isinstance(max_len, int) or max_len < min_max_model_len:
        raise PreflightError(
            f"max_model_len {max_len!r} is below the agent's {min_max_model_len}"
        )
    return entry


def sha256_file(path: Any) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()
