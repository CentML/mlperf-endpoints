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

"""Deterministic selection of the ``osworld_cua_stable200`` task set.

A subset of ``osworld_cua_full`` meant to give a steadier score and to compare
x86_64 with aarch64 desktops. It starts from the 344 full-set tasks and drops:

- **unstable** tasks: ``stability_classes.csv`` holds each task's class from the
  Qwen3.8-27B runs at the perf sampling settings (temperature 1.0, top_p 0.95,
  max_tokens 32768): 4 attempts per task from two full-set runs, 16 for the
  ``osworld_cua_84`` tasks. A task is unstable when neither success nor failure
  reached 75% of its attempts;
- **infeasible** tasks (``infeasible`` in ``tasks_scan.csv``): every attempt
  scored 0 for a reason not yet understood, so they only hold the score down;
- ``ARCH_EXCLUDED``: tasks known to behave differently on aarch64 desktops,
  each with its reason.

From the rest, ``STABLE_TARGET`` tasks are drawn with ``random.Random(SEED)``,
split over domains in proportion to the full set (largest remainder).

Regenerate the committed task list with::

    python -m inference_endpoint.dataset_manager.predefined.osworld_cua.select_stable
"""

from __future__ import annotations

import csv
import json
import random
import sys
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

from . import select_tasks

SEED = 719
STABLE_TARGET = 200
UNSTABLE_CLASS = "unstable"

_APT = (
    "its setup runs apt at task time; the aarch64 image's runtime apt sources"
    " are live ports.ubuntu.com, the x86_64 image's a fixed snapshot"
)
ARCH_EXCLUDED = {
    "chrome-3299584d": _APT,
    "chrome-6766f2b8": _APT,
    "chrome-9656a811": _APT,
    "multi_apps-2373b66a": _APT,
    "multi_apps-716a6079": _APT,
    "multi_apps-a74b607e": _APT,
    "chrome-44ee5668": (
        "the no-op agent passes on aarch64 (0 on x86_64): the YouTube entries its"
        " setup seeds into Chrome's history are not there when the verifier reads it"
    ),
    "multi_apps-69acbb55": (
        "its oracle pip-installs torch==2.2.2+cpu, which has no aarch64 wheel; the"
        " oracle failed in 2 of 2 aarch64 attempts and passes on x86_64"
    ),
    "multi_apps-02ce9a50": (
        "its verifier OCRs a screenshot with easyocr, whose quantized recognizer"
        " runs on a different torch engine on aarch64 (QNNPACK) than on x86_64"
        " (fbgemm)"
    ),
}

PACKAGE_DIR = Path(__file__).resolve().parent
STABILITY_PATH = PACKAGE_DIR / "stability_classes.csv"
STABLE_TASK_SET_NAME = "osworld_cua_stable200"
STABLE_TASK_SET_PATH = PACKAGE_DIR / f"{STABLE_TASK_SET_NAME}.json"


def read_stability(path: Path = STABILITY_PATH) -> dict[str, str]:
    """Task id -> stability class."""
    with path.open(newline="") as f:
        return {r["task_id"]: r["class"] for r in csv.DictReader(f)}


def stable_pool(
    rows: Iterable[dict[str, str]], classes: dict[str, str]
) -> list[dict[str, str]]:
    """Full-set tasks that are not unstable, infeasible or arch-dependent."""
    full = select_tasks.full_task_set_document(rows)["tasks"]
    infeasible = {r["task"] for r in rows if r["infeasible"] == "1"}
    return [
        t
        for t in full
        if classes[t["task_id"]] != UNSTABLE_CLASS
        and t["task_id"] not in infeasible
        and t["task_id"] not in ARCH_EXCLUDED
    ]


def domain_quota(full_counts: Counter, target: int) -> dict[str, int]:
    """``target`` split over domains in proportion to ``full_counts``."""
    total = sum(full_counts.values())
    exact = {d: target * n / total for d, n in full_counts.items()}
    quota = {d: int(v) for d, v in exact.items()}
    by_remainder = sorted(exact, key=lambda d: (quota[d] - exact[d], d))
    for d in by_remainder[: target - sum(quota.values())]:
        quota[d] += 1
    return quota


def select_stable(
    rows: Iterable[dict[str, str]], classes: dict[str, str]
) -> list[dict[str, str]]:
    rows = list(rows)
    full = select_tasks.full_task_set_document(rows)["tasks"]
    quota = domain_quota(Counter(t["domain"] for t in full), STABLE_TARGET)
    pool = stable_pool(rows, classes)
    rng = random.Random(SEED)
    picked: list[dict[str, str]] = []
    for domain in sorted(quota):
        members = sorted(
            (t for t in pool if t["domain"] == domain), key=lambda t: t["task_id"]
        )
        if len(members) < quota[domain]:
            raise ValueError(
                f"{domain}: {len(members)} stable tasks, quota {quota[domain]}"
            )
        picked += rng.sample(members, quota[domain])
    return sorted(picked, key=lambda t: t["task_id"])


def stable_task_set_document(
    rows: Iterable[dict[str, str]], classes: dict[str, str]
) -> dict:
    rows = list(rows)
    tasks = select_stable(rows, classes)
    return {
        "name": STABLE_TASK_SET_NAME,
        "seed": SEED,
        "target": STABLE_TARGET,
        "excluded_unstable": sorted(
            t for t, c in classes.items() if c == UNSTABLE_CLASS
        ),
        "excluded_infeasible": sorted(
            r["task"] for r in rows if r["infeasible"] == "1" and r["task"] in classes
        ),
        "excluded_arch": ARCH_EXCLUDED,
        "class_mix": dict(
            sorted(Counter(classes[t["task_id"]] for t in tasks).items())
        ),
        "tasks": tasks,
    }


def main() -> None:
    out_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else PACKAGE_DIR
    document = stable_task_set_document(select_tasks.read_scan(), read_stability())
    path = out_dir / STABLE_TASK_SET_PATH.name
    path.write_text(json.dumps(document, indent=2) + "\n")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
