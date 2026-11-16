# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""How a run is split into harbor processes of at most N desktops."""

import pytest
from inference_endpoint.evaluation.osworld_service.osworld_service.shards import (
    plan_shards,
)

pytestmark = pytest.mark.unit

FULL = [f"app{i % 10}-{i:04d}" for i in range(344)]


class TestPlanShards:
    def test_48_desktops_become_three_processes_of_16(self):
        shards = plan_shards(FULL, concurrency=48, max_per_process=16)
        assert [s.concurrency for s in shards] == [16, 16, 16]
        assert [len(s.task_ids) for s in shards] == [115, 115, 114]
        assert [s.name for s in shards] == ["shard-0", "shard-1", "shard-2"]

    def test_every_task_lands_in_exactly_one_shard(self):
        shards = plan_shards(FULL, concurrency=48, max_per_process=16)
        placed = [t for s in shards for t in s.task_ids]
        assert sorted(placed) == sorted(FULL)
        assert len(placed) == len(set(placed))

    def test_round_robin_in_request_order_is_deterministic(self):
        first = plan_shards(FULL, concurrency=48, max_per_process=16)
        assert first == plan_shards(list(FULL), concurrency=48, max_per_process=16)
        assert first[1].task_ids[:3] == (FULL[1], FULL[4], FULL[7])

    @pytest.mark.parametrize(
        "concurrency, cap, expected",
        [
            (16, 16, [16]),
            (4, 16, [4]),
            (40, 16, [14, 13, 13]),
            (17, 16, [9, 8]),
            (64, 16, [16, 16, 16, 16]),
        ],
    )
    def test_concurrency_is_split_evenly(self, concurrency, cap, expected):
        shards = plan_shards(FULL, concurrency=concurrency, max_per_process=cap)
        assert [s.concurrency for s in shards] == expected
        assert sum(s.concurrency for s in shards) == concurrency

    def test_never_more_shards_than_tasks(self):
        shards = plan_shards(["a-1", "b-2"], concurrency=48, max_per_process=16)
        assert [s.task_ids for s in shards] == [("a-1",), ("b-2",)]
        assert [s.concurrency for s in shards] == [24, 24]

    def test_cap_must_be_positive(self):
        with pytest.raises(ValueError, match="at least 1"):
            plan_shards(FULL, concurrency=16, max_per_process=0)
