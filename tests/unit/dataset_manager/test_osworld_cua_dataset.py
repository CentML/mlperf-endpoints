# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter

import pytest
from inference_endpoint.dataset_manager import Dataset, OSWorldCUA
from inference_endpoint.dataset_manager.predefined.osworld_cua import select_tasks

pytestmark = pytest.mark.unit


def test_committed_task_set_is_the_selection_output():
    committed = json.loads(select_tasks.TASK_SET_PATH.read_text())
    assert committed == select_tasks.task_set_document(select_tasks.read_scan())


def test_selection_is_the_stratified_88_minus_sudo_tasks():
    rows = select_tasks.read_scan()
    candidates = [r["task"] for r in select_tasks.select_candidates(rows)]
    tasks = select_tasks.select_task_set(rows)
    assert len(candidates) == 88
    assert set(select_tasks.SUDO_EXCLUDED) <= set(candidates)
    assert len(tasks) == 84
    assert not set(select_tasks.SUDO_EXCLUDED) & {t["task_id"] for t in tasks}
    expected = dict(select_tasks.QUOTA, os=4, multi_apps=19)
    assert Counter(t["domain"] for t in tasks) == expected


def test_selection_rules_hold():
    rows = {r["task"]: r for r in select_tasks.read_scan()}
    for task in select_tasks.select_task_set(rows.values()):
        row = rows[task["task_id"]]
        assert row["has_oracle"] == "1"
        assert row["infeasible"] == "0"
        assert not row["live_hosts"]


def test_dataset_loads_the_task_set():
    assert Dataset.PREDEFINED["osworld_cua"] is OSWorldCUA
    df = OSWorldCUA.generate(datasets_dir=None, task_set="osworld_cua_84")
    assert list(df.columns) == ["task_id", "domain", "prompt"]
    assert len(df) == 84
    assert df["task_id"].is_unique
    assert (df["prompt"] == df["task_id"]).all()


def test_unknown_task_set():
    with pytest.raises(ValueError, match="Unknown OSWorld task set"):
        OSWorldCUA.generate(datasets_dir=None, task_set="osworld_all")


def test_full_task_set_matches_selection():
    committed = json.loads(select_tasks.FULL_TASK_SET_PATH.read_text())
    assert committed == select_tasks.full_task_set_document(select_tasks.read_scan())


def test_full_task_set_is_every_task_minus_the_oracle_failures():
    rows = select_tasks.read_scan()
    tasks = {t["task_id"] for t in select_tasks.full_task_set_document(rows)["tasks"]}
    scanned = {r["task"] for r in rows}
    assert set(select_tasks.FULL_EXCLUDED) <= scanned
    assert tasks == scanned - set(select_tasks.FULL_EXCLUDED)
    assert len(tasks) == 344
    assert set(select_tasks.SUDO_EXCLUDED) <= set(select_tasks.FULL_EXCLUDED)
    assert all(select_tasks.FULL_EXCLUDED.values())


def test_dataset_loads_the_full_task_set():
    df = OSWorldCUA.generate(datasets_dir=None, task_set="osworld_cua_full")
    assert len(df) == 344
    assert df["task_id"].is_unique
