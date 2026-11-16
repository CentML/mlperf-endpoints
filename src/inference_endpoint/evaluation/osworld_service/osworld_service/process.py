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

"""Cancellable subprocesses: each runs in its own session, killed as a group."""

from __future__ import annotations

import os
import signal
import subprocess
import threading
import time
from pathlib import Path

_TERMINATE_GRACE_S = 30.0


class RunnerError(RuntimeError):
    pass


class RunCancelled(RunnerError):
    pass


def terminate_process(process: subprocess.Popen) -> None:
    """SIGTERM the process group, then SIGKILL it after a grace period."""
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=_TERMINATE_GRACE_S)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            # The group exited between the timeout and the kill.
            return
        process.wait()


class CancellationToken:
    """Cancels a run: every process attached to it is terminated, now or later."""

    def __init__(self) -> None:
        self._event = threading.Event()
        self._lock = threading.Lock()
        self._processes: set[subprocess.Popen] = set()

    def is_cancelled(self) -> bool:
        return self._event.is_set()

    def cancel(self) -> None:
        self._event.set()
        with self._lock:
            processes = list(self._processes)
        # Signal every group first, so the grace periods run concurrently.
        for process in processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    # The group exited since poll(); nothing left to signal.
                    pass
        for process in processes:
            terminate_process(process)

    def attach(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.add(process)
            cancelled = self._event.is_set()
        if cancelled:
            terminate_process(process)

    def detach(self, process: subprocess.Popen) -> None:
        with self._lock:
            self._processes.discard(process)


def run_logged(
    cmd: list[str],
    log_path: Path,
    *,
    cwd: Path,
    env: dict[str, str],
    timeout_s: float,
    cancel_token: CancellationToken,
) -> int:
    """Run ``cmd`` with output appended to ``log_path``; return its exit code."""
    if cancel_token.is_cancelled():
        raise RunCancelled("cancelled before start")
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"$ {' '.join(cmd)}\n")
        log.flush()
        process = subprocess.Popen(
            cmd,
            stdin=subprocess.DEVNULL,
            stdout=log,
            stderr=subprocess.STDOUT,
            cwd=str(cwd),
            env=env,
            start_new_session=True,
        )
        cancel_token.attach(process)
        try:
            deadline = time.monotonic() + timeout_s
            while process.poll() is None:
                if cancel_token.is_cancelled():
                    terminate_process(process)
                    raise RunCancelled("cancelled")
                if time.monotonic() > deadline:
                    terminate_process(process)
                    raise RunnerError(f"timed out after {timeout_s:.0f}s: {cmd[0]}")
                time.sleep(0.5)
        finally:
            cancel_token.detach(process)
    if cancel_token.is_cancelled():
        raise RunCancelled("cancelled")
    return process.returncode
