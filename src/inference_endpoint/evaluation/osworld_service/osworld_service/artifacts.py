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

"""Run artifacts: atomic writes, secret redaction and the downloadable names."""

from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

SAFE_ARTIFACT_NAMES = (
    "osworld_results.json",
    "preflight.json",
    "server_record.json",
    "trials.tsv",
    "harbor.log",
    "attach.log",
    "status.json",
)
_SECRET_TEXT_PATTERNS = (
    re.compile(r"(?i)(authorization:\s*(?:bearer|basic)\s+)[^\s,;]+"),
    re.compile(r"(?i)((?:api[_-]?key|access[_-]?token|token|password)=)[^&\s]+"),
)
_SECRET_KEY_RE = re.compile(r"(?i)(api[_-]?key|token|secret|password|authorization)")


def atomic_write_bytes(path: Path, data: bytes) -> None:
    """Publish ``data`` at ``path`` without exposing a partial file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("wb") as file:
        file.write(data)
        file.flush()
        os.fsync(file.fileno())
    os.replace(tmp, path)


def redact_text(text: str, secret_values: set[str]) -> str:
    for secret in secret_values:
        if secret:
            text = text.replace(secret, "<redacted>")
    for pattern in _SECRET_TEXT_PATTERNS:
        text = pattern.sub(r"\1<redacted>", text)
    return text


def redact_secrets(value: Any, secret_values: set[str]) -> Any:
    if isinstance(value, dict):
        return {
            k: (
                "<redacted>"
                if _SECRET_KEY_RE.search(str(k)) and v is not None
                else redact_secrets(v, secret_values)
            )
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [redact_secrets(v, secret_values) for v in value]
    if isinstance(value, str):
        return redact_text(value, secret_values)
    return value


def resolve_artifact(run_dir: Path, name: str) -> Path:
    if name not in SAFE_ARTIFACT_NAMES:
        raise FileNotFoundError(name)
    path = (run_dir / name).resolve()
    if path.parent != run_dir.resolve() or not path.is_file():
        raise FileNotFoundError(name)
    return path
