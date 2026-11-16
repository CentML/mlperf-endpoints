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

"""Deterministic selection of the ``osworld_cua_84`` task set.

Input is ``tasks_scan.csv``: one row per task of cua-harbor's
``adapters/osworld-novm`` (OSWorld-Verified without a VM), with its domain,
apps, oracle step count and evaluator facts.

Rules, applied per domain with ``random.Random(SEED)``:

- only feasible tasks that have an oracle solution and contact no live website
  (live sites drift between runs);
- ``QUOTA`` tasks per domain, spread round-robin over strata of (oracle-step
  tercile, single vs multi app);
- ``UID_KEEP`` tasks per domain whose result depends on the desktop user's uid
  are kept on purpose, as a known difference class between container runtimes.

That gives 88 tasks. ``SUDO_EXCLUDED`` then removes the 4 tasks that need
``sudo`` inside the desktop, which an unprivileged (rootless) desktop cannot
grant, leaving 84.

Regenerate the committed task list with::

    python -m inference_endpoint.dataset_manager.predefined.osworld_cua.select_tasks
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

SEED = 398
QUOTA = {
    "calc": 11,
    "chrome": 8,
    "gimp": 6,
    "impress": 11,
    "multi_apps": 20,
    "os": 7,
    "thunderbird": 6,
    "vlc": 6,
    "vs_code": 6,
    "writer": 7,
}
UID_KEEP = {"os": 3, "multi_apps": 1, "chrome": 1}
SUDO_EXCLUDED = (
    "multi_apps-e1fc0df3",
    "os-5812b315",
    "os-94d95f96",
    "os-b6781586",
)

PACKAGE_DIR = Path(__file__).resolve().parent
SCAN_PATH = PACKAGE_DIR / "tasks_scan.csv"
TASK_SET_NAME = "osworld_cua_84"
TASK_SET_PATH = PACKAGE_DIR / f"{TASK_SET_NAME}.json"


def _stratum(row: dict[str, str], cut: tuple[int, int]) -> tuple[int, bool]:
    steps = int(row["oracle_steps"])
    tercile = 0 if steps <= cut[0] else 1 if steps <= cut[1] else 2
    return tercile, int(row["n_apps"]) > 1


def select_candidates(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """The 88-task stratified sample, sorted by task id."""
    rows = list(rows)
    rng = random.Random(SEED)
    picked: list[dict[str, str]] = []
    for domain, quota in QUOTA.items():
        pool = [
            r
            for r in rows
            if r["domain"] == domain
            and r["has_oracle"] == "1"
            and r["infeasible"] == "0"
            and not r["live_hosts"]
        ]
        uid_sensitive = [r for r in pool if r["uid_sensitive"] == "1"]
        rest = [r for r in pool if r["uid_sensitive"] == "0"]
        take = rng.sample(
            uid_sensitive, min(UID_KEEP.get(domain, 0), len(uid_sensitive))
        )
        steps = sorted(int(r["oracle_steps"]) for r in rest)
        cut = (steps[len(steps) // 3], steps[2 * len(steps) // 3])
        strata: dict[tuple[int, bool], list[dict[str, str]]] = defaultdict(list)
        for r in rest:
            strata[_stratum(r, cut)].append(r)
        for members in strata.values():
            rng.shuffle(members)
        keys = sorted(strata)
        while len(take) < quota and any(strata[k] for k in keys):
            for k in keys:
                if strata[k] and len(take) < quota:
                    take.append(strata[k].pop())
        picked += take
    return sorted(picked, key=lambda r: r["task"])


def select_task_set(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """``osworld_cua_84``: the candidates minus the sudo tasks."""
    return [
        {"task_id": r["task"], "domain": r["domain"]}
        for r in select_candidates(rows)
        if r["task"] not in SUDO_EXCLUDED
    ]


def read_scan(path: Path = SCAN_PATH) -> list[dict[str, str]]:
    with path.open(newline="") as f:
        return list(csv.DictReader(f))


def task_set_document(rows: Iterable[dict[str, str]]) -> dict:
    return {
        "name": TASK_SET_NAME,
        "seed": SEED,
        "quota": QUOTA,
        "uid_keep": UID_KEEP,
        "sudo_excluded": list(SUDO_EXCLUDED),
        "tasks": select_task_set(rows),
    }


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else TASK_SET_PATH
    out.write_text(json.dumps(task_set_document(read_scan()), indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
