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

"""OSWorld computer-use task sets for the service-backed ``osworld_cua`` scorer."""

from pathlib import Path

import msgspec.json
import pandas as pd

from ...dataset import Dataset

_TASK_SET_DIR = Path(__file__).resolve().parent
TASK_SETS = ("osworld_cua_84",)


class OSWorldCUA(Dataset, dataset_id="osworld_cua"):
    """Accuracy-only OSWorld task set; the OSWorld service runs the tasks.

    Rows are task ids of cua-harbor's ``adapters/osworld-novm``. ``prompt`` holds
    the task id only: the agent's real prompts are built by the service.
    """

    ACCURACY_ONLY = True
    COLUMN_NAMES = ["task_id", "domain", "prompt"]

    @classmethod
    def task_set_path(cls, task_set: str) -> Path:
        if task_set not in TASK_SETS:
            raise ValueError(
                f"Unknown OSWorld task set {task_set!r}; choose from: {list(TASK_SETS)}"
            )
        return _TASK_SET_DIR / f"{task_set}.json"

    @classmethod
    def generate(
        cls,
        datasets_dir: Path,
        task_set: str = "osworld_cua_84",
        force: bool = False,
    ) -> pd.DataFrame:
        """Load a committed task set (no download; ``datasets_dir`` is unused)."""
        document = msgspec.json.decode(cls.task_set_path(task_set).read_bytes())
        tasks = document["tasks"]
        return pd.DataFrame(
            {
                "task_id": [t["task_id"] for t in tasks],
                "domain": [t["domain"] for t in tasks],
                "prompt": [t["task_id"] for t in tasks],
            }
        )
