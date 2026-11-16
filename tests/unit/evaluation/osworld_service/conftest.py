# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from tests.unit.evaluation.osworld_service.fakes import (
    FakeRunner,
    FakeVLLMServer,
    ServiceThread,
)


@pytest.fixture
def fake_vllm():
    server = FakeVLLMServer(port=0)
    server.start()
    yield server
    server.stop()


@pytest.fixture
def service_factory(tmp_path):
    started: list[ServiceThread] = []

    def factory(runner: FakeRunner, auth_token: str | None = None) -> ServiceThread:
        service = ServiceThread(runner, tmp_path / f"svc{len(started)}", auth_token)
        started.append(service.start())
        return service

    yield factory
    for service in started:
        service.stop()
