# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
from collections import Counter

import pytest
from inference_endpoint.dataset_manager import Dataset, OSWorldCUA
from inference_endpoint.dataset_manager.predefined.osworld_cua import (
    select_stable,
    select_tasks,
)

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


def _stable_document():
    return select_stable.stable_task_set_document(
        select_tasks.read_scan(), select_stable.read_stability()
    )


def test_stable_task_set_matches_selection():
    committed = json.loads(select_stable.STABLE_TASK_SET_PATH.read_text())
    assert committed == _stable_document()


def test_stable_task_set_drops_unstable_infeasible_and_arch_tasks():
    rows = {r["task"]: r for r in select_tasks.read_scan()}
    classes = select_stable.read_stability()
    full = {
        t["task_id"]
        for t in select_tasks.full_task_set_document(rows.values())["tasks"]
    }
    document = _stable_document()
    tasks = [t["task_id"] for t in document["tasks"]]
    assert len(tasks) == select_stable.STABLE_TARGET
    assert set(classes) == full
    assert set(tasks) <= full
    assert all(classes[t] != select_stable.UNSTABLE_CLASS for t in tasks)
    assert all(rows[t]["infeasible"] == "0" for t in tasks)
    assert not set(tasks) & set(select_stable.ARCH_EXCLUDED)
    assert set(select_stable.ARCH_EXCLUDED) <= full
    assert all(select_stable.ARCH_EXCLUDED.values())
    assert sum(document["class_mix"].values()) == len(tasks)


def test_stable_task_set_follows_the_full_set_domain_mix():
    rows = select_tasks.read_scan()
    full = Counter(
        t["domain"] for t in select_tasks.full_task_set_document(rows)["tasks"]
    )
    quota = select_stable.domain_quota(full, select_stable.STABLE_TARGET)
    assert sum(quota.values()) == select_stable.STABLE_TARGET
    assert Counter(t["domain"] for t in _stable_document()["tasks"]) == quota
    for domain, n in full.items():
        assert (
            abs(quota[domain] - select_stable.STABLE_TARGET * n / sum(full.values()))
            < 1
        )


def test_dataset_loads_the_stable_task_set():
    df = OSWorldCUA.generate(datasets_dir=None, task_set="osworld_cua_stable200")
    assert len(df) == select_stable.STABLE_TARGET
    assert df["task_id"].is_unique
