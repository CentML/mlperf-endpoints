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

"""A run as several ``harbor run`` processes of at most N desktops each.

One harbor process drives every agent of its trials from a single Python
process. At 48 desktops that process saturates one core and the agents queue
behind it (pdx, 84 tasks x 2: 21.2 min in one process, 9.4 min as three of
16). The service therefore splits a run into shards.

The task x attempt list is partitioned by task: harbor takes one
``--n-attempts`` per process, so every attempt of a task stays in one shard.
Tasks go round-robin in request order, which spreads each app's tasks over the
shards. The run's concurrency is split as evenly as possible.
"""

from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass
from math import ceil
from pathlib import Path


@dataclass(frozen=True)
class Shard:
    index: int
    task_ids: tuple[str, ...]
    concurrency: int

    @property
    def name(self) -> str:
        return f"shard-{self.index}"


def plan_shards(
    task_ids: list[str], concurrency: int, max_per_process: int
) -> list[Shard]:
    """Deterministic shards: ceil(concurrency / max_per_process), at most one per task."""
    if max_per_process < 1:
        raise ValueError("max_per_process must be at least 1")
    count = max(1, min(len(task_ids), ceil(concurrency / max_per_process)))
    base, extra = divmod(concurrency, count)
    return [
        Shard(
            index=i,
            task_ids=tuple(task_ids[i::count]),
            concurrency=max(1, base + (1 if i < extra else 0)),
        )
        for i in range(count)
    ]


@dataclass
class ShardOutcome:
    shard: Shard
    jobs_dir: Path
    exit_code: int | None = None
    error: BaseException | None = None


def run_all(
    shards: list[Shard], jobs_root: Path, run_one: Callable[[Shard, Path], int]
) -> list[ShardOutcome]:
    """Run ``run_one(shard, jobs_dir) -> exit code`` for every shard at once.

    Each shard runs in its own thread; a shard that raises is recorded, not
    propagated, so the others finish (or are cancelled) and the run returns.
    """
    outcomes = [ShardOutcome(s, jobs_root / s.name) for s in shards]

    def work(outcome: ShardOutcome) -> None:
        outcome.jobs_dir.mkdir(parents=True, exist_ok=True)
        try:
            outcome.exit_code = run_one(outcome.shard, outcome.jobs_dir)
        except BaseException as exc:  # recorded and re-raised by the caller
            outcome.error = exc

    threads = [
        threading.Thread(target=work, args=(o,), name=o.shard.name, daemon=True)
        for o in outcomes
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    return outcomes
