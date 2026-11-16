# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import json
import threading

import msgspec.json
import pytest
from inference_endpoint.config.schema import APIType, EndpointConfig, ModelParams
from inference_endpoint.dataset_manager import OSWorldCUA
from inference_endpoint.evaluation.osworld_scorer import OSWorldScorer
from inference_endpoint.evaluation.osworld_service.osworld_service.server import (
    MANAGER_KEY,
)
from inference_endpoint.evaluation.scoring import Scorer
from inference_endpoint.exceptions import SetupError

from tests.unit.evaluation.osworld_service.fakes import FakeRunner

pytestmark = pytest.mark.unit

PERF_PARAMS = ModelParams(
    name="qwen-test", temperature=1.0, top_p=0.95, max_new_tokens=32768
)


@pytest.fixture
def dataset():
    ds = OSWorldCUA.get_dataloader(task_set="osworld_cua_84")
    ds.load(api_type=APIType.OPENAI, model_params=PERF_PARAMS)
    return ds


def _scorer(tmp_path, dataset, service, **extras):
    (tmp_path / "sample_idx_map.json").write_bytes(
        msgspec.json.encode({"osworld_cua": {}})
    )
    return OSWorldScorer(
        "osworld_cua",
        dataset,
        tmp_path,
        osworld_service_url=service.url,
        poll_interval_s=0.01,
        model_params=PERF_PARAMS,
        endpoint_config=EndpointConfig(endpoints=["http://gpu-node:30000"]),
        **extras,
    )


class TestOptions:
    def test_registered_as_external_endpoint_scorer(self):
        cls = Scorer.get("osworld_cua")
        assert cls is OSWorldScorer
        assert cls.SKIP_ENDPOINT_PHASE and cls.NEEDS_ENDPOINT
        assert not cls.REQUIRES_EXTRACTOR

    def test_sends_only_the_agents_sampling_fields(self):
        params = ModelParams(
            name="m",
            temperature=1.0,
            top_p=0.95,
            max_new_tokens=32768,
            seed=3,
            repetition_penalty=1.1,
            chat_template_kwargs={"enable_thinking": False},
        )
        assert OSWorldScorer.generation_params(params) == {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 32768,
        }

    @pytest.mark.parametrize(
        "extras, message",
        [
            ({}, "osworld_service_url is required"),
            ({"osworld_service_url": "http://svc:1/path"}, "service root URL"),
            ({"osworld_service_url": "http://svc:1", "n_attempts": 0}, ">= 1"),
            ({"osworld_service_url": "http://svc:1", "platform": "ppc"}, "platform"),
            ({"osworld_service_url": "http://svc:1", "task_set": "x"}, "task set"),
        ],
    )
    def test_invalid_extras(self, extras, message):
        with pytest.raises(SetupError, match=message):
            OSWorldScorer.preflight(extras)

    def test_loader_kwargs_and_sample_count(self):
        extras = {"osworld_service_url": "http://svc:1", "num_tasks": 2}
        assert OSWorldScorer.dataset_loader_kwargs(extras) == {
            "task_set": "osworld_cua_84"
        }
        assert OSWorldScorer.external_sample_count(extras) == 2
        assert OSWorldScorer.external_sample_count({"num_tasks": "x"}) is None


class TestPreflight:
    def test_checks_health_and_endpoint(self, service_factory):
        runner = FakeRunner()
        service = service_factory(runner)
        OSWorldScorer.preflight(
            {"osworld_service_url": service.url, "platform": "x86_64"},
            endpoint_config=EndpointConfig(endpoints=["http://gpu-node:30000"]),
        )
        assert runner.health_queries == ["http://gpu-node:30000"]
        assert runner.requests == []

    def test_unreachable_endpoint_fails_before_the_benchmark(self, service_factory):
        service = service_factory(FakeRunner())
        with pytest.raises(SetupError, match="cannot reach the endpoint"):
            OSWorldScorer.preflight(
                {"osworld_service_url": service.url},
                endpoint_config=EndpointConfig(endpoints=["http://unreachable:1"]),
            )

    def test_platform_mismatch(self, service_factory):
        service = service_factory(FakeRunner(platform="aarch64"))
        with pytest.raises(SetupError, match="aarch64"):
            OSWorldScorer.preflight(
                {"osworld_service_url": service.url, "platform": "x86_64"}
            )

    def test_service_down(self):
        with pytest.raises(SetupError, match="unreachable"):
            OSWorldScorer.preflight({"osworld_service_url": "http://127.0.0.1:9"})


class TestScore:
    def test_mean_reward_over_all_trials(self, tmp_path, dataset, service_factory):
        rewards = {"calc-035f41ba": 1.0, "os-37887e8c": 1.0, "vlc-8ba5ae7a": 0.5}
        runner = FakeRunner(rewards)
        scorer = _scorer(tmp_path, dataset, service_factory(runner), num_tasks=84)

        score, repeats = scorer.score()

        assert repeats == 1
        assert score == pytest.approx(2 * 2.5 / 168)
        assert scorer.complete
        request = runner.requests[0]
        assert len(request.task_ids) == 84 and request.n_attempts == 2
        assert request.max_turns == 30 and request.concurrency == 16
        assert request.endpoint_url == "http://gpu-node:30000"
        assert request.generation_params == {
            "temperature": 1.0,
            "top_p": 0.95,
            "max_tokens": 32768,
        }
        breakdown = scorer.score_breakdown()
        assert breakdown["by_domain"]["calc"]["tasks"] == 11
        assert breakdown["by_domain"]["os"]["mean_reward"] == pytest.approx(1 / 4)
        assert breakdown["pass_at_2"] == pytest.approx(2 / 84)
        assert breakdown["trials_errored"] == 0
        saved = json.loads((tmp_path / "osworld_results.json").read_text())
        assert len(saved["trials"]) == 168
        run_dirs = list((tmp_path / "osworld_runs").iterdir())
        assert (run_dirs[0] / "osworld_results.json").is_file()

    def test_errored_trials_count_zero_and_mark_incomplete(
        self, tmp_path, dataset, service_factory
    ):
        runner = FakeRunner({"calc-035f41ba": 1.0}, errored={"calc-035f41ba"})
        scorer = _scorer(tmp_path, dataset, service_factory(runner), num_tasks=2)

        score, _ = scorer.score()

        assert score == 0.0
        assert not scorer.complete
        assert scorer.score_breakdown()["trials_errored"] == 2

    def test_failed_run_returns_none(self, tmp_path, dataset, service_factory):
        service = service_factory(FakeRunner(fail="preflight refused the run"))
        scorer = _scorer(tmp_path, dataset, service, num_tasks=2)

        assert scorer.score() == (None, 1)
        assert not scorer.complete

    def test_missing_trials_are_incomplete(self, tmp_path, dataset, service_factory):
        class HalfRunner(FakeRunner):
            def run(self, request, run_dir, cancel_token, on_phase):
                result = super().run(request, run_dir, cancel_token, on_phase)
                result["trials"] = result["trials"][::2]
                return result

        scorer = _scorer(
            tmp_path,
            dataset,
            service_factory(HalfRunner({"calc-035f41ba": 1.0})),
            num_tasks=2,
        )
        score, _ = scorer.score()
        assert score == pytest.approx(1 / 4)
        assert not scorer.complete

    def test_extra_trials_are_rejected(self, tmp_path, dataset, service_factory):
        class DoubleRunner(FakeRunner):
            def run(self, request, run_dir, cancel_token, on_phase):
                result = super().run(request, run_dir, cancel_token, on_phase)
                result["trials"] = result["trials"] * 2
                return result

        scorer = _scorer(
            tmp_path, dataset, service_factory(DoubleRunner()), num_tasks=2
        )
        assert scorer.score() == (None, 1)

    def test_keyboard_interrupt_cancels_the_run(
        self, tmp_path, dataset, service_factory, monkeypatch
    ):
        block = threading.Event()
        runner = FakeRunner(block=block)
        service = service_factory(runner)
        scorer = _scorer(tmp_path, dataset, service, num_tasks=2)
        real_http_json = OSWorldScorer._http_json

        def interrupt_on_poll(url, **kwargs):
            if kwargs.get("method", "GET") == "GET" and "/v1/runs/" in url:
                raise KeyboardInterrupt
            return real_http_json(url, **kwargs)

        monkeypatch.setattr(
            OSWorldScorer, "_http_json", staticmethod(interrupt_on_poll)
        )
        try:
            with pytest.raises(KeyboardInterrupt):
                scorer.score()
        finally:
            block.set()
        manager = service.app[MANAGER_KEY]
        assert [r.status for r in manager.runs.values()] == ["cancelled"]

    def test_requires_one_endpoint(self, tmp_path, dataset, service_factory):
        scorer = _scorer(tmp_path, dataset, service_factory(FakeRunner()))
        scorer.endpoint_config = EndpointConfig(endpoints=["http://a:1", "http://b:2"])
        with pytest.raises(SetupError, match="exactly one endpoint"):
            scorer.score()
