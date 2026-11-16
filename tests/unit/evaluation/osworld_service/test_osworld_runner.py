# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""OSWorldRunner end to end on a 2-task set: a fake cua-harbor checkout whose
``scripts/serve.sh --use`` writes a server dir and whose ``scripts/run.sh``
writes Harbor trial results, against a fake vLLM server."""

import hashlib
import json
import os
import subprocess
import sys

import pytest
from inference_endpoint.evaluation.osworld_service.osworld_service import (
    cua_harbor,
    pins,
)
from inference_endpoint.evaluation.osworld_service.osworld_service.config import (
    HarborConfig,
)
from inference_endpoint.evaluation.osworld_service.osworld_service.process import (
    CancellationToken,
    RunnerError,
)
from inference_endpoint.evaluation.osworld_service.osworld_service.runner import (
    OSWorldRunner,
)
from inference_endpoint.evaluation.osworld_service.osworld_service.schemas import (
    RunRequest,
)

from tests.unit.evaluation.osworld_service.fakes import MODEL

pytestmark = pytest.mark.unit

TASKS = ["calc-035f41ba", "os-37887e8c"]

RUN_SH = """\
#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
Path(os.environ["FAKE_ARGS_OUT"]).write_text(json.dumps({
    "args": args,
    "url": os.environ.get("CUA_MODEL_SERVER_URL"),
    "squashfs": os.environ.get("OSW_ENROOT_SQUASHFS"),
    "work_root": os.environ.get("OSW_ENROOT_WORK_ROOT"),
    "serve_dir": os.environ.get("SERVE_DIR"),
}))
tasks = [args[i + 1] for i, a in enumerate(args) if a == "-i"]
k = int(args[args.index("--n-attempts") + 1])
job = Path(args[args.index("--jobs-dir") + 1]) / args[args.index("--job-name") + 1]
for t in tasks:
    for a in range(k):
        d = job / f"{t}__{a}"
        d.mkdir(parents=True)
        failed = t.startswith("os") and a == 1
        d.joinpath("result.json").write_text(json.dumps({
            "task_name": t,
            "verifier_result": None if failed else {"rewards": {"reward": 1.0 if t.startswith("calc") else 0.0}},
            "exception_info": {"exception_type": "AgentTimeoutError"} if failed else None,
        }))
"""

SERVE_SH = """\
#!/usr/bin/env bash
# serve.sh --use URL --model DIR; refuses when FAKE_ATTACH_FAIL is set.
set -e
[ "$1" = --use ] && [ "$3" = --model ] || exit 2
if [ -n "${FAKE_ATTACH_FAIL:-}" ]; then
  echo "serve.sh: error: $FAKE_ATTACH_FAIL" >&2
  exit 1
fi
d=$SERVERS_DIR/${FAKE_SERVED_NAME:-qwen-test}
mkdir -p "$d"
echo '{"max_model_len": 131072, "max_images": 100, "engine": "vllm"}' > "$d/serve.args.json"
echo "{\\"url\\": \\"$2\\", \\"images\\": {\\"max_images\\": 100}}" > "$d/serve.observed.json"
echo "$4" > "$d/serve.model"
echo "attached: at $2"
"""


def _git(repo, *args: str) -> str:
    return subprocess.run(
        ["git", "-C", str(repo), *args],
        capture_output=True,
        text=True,
        check=True,
        env={
            **os.environ,
            "GIT_AUTHOR_NAME": "t",
            "GIT_AUTHOR_EMAIL": "t@t",
            "GIT_COMMITTER_NAME": "t",
            "GIT_COMMITTER_EMAIL": "t@t",
        },
    ).stdout.strip()


@pytest.fixture
def harbor_cfg(tmp_path, monkeypatch):
    repo = tmp_path / "cua-harbor"
    for task in TASKS:
        (repo / "adapters/osworld-novm/tasks" / task).mkdir(parents=True)
    scripts = repo / "scripts"
    scripts.mkdir()
    (scripts / "run.sh").write_text(RUN_SH)
    (scripts / "run.sh").chmod(0o755)
    (scripts / "serve.sh").write_text(SERVE_SH)
    venv_bin = repo / ".venv" / "bin"
    venv_bin.mkdir(parents=True)
    (venv_bin / "python").symlink_to(sys.executable)
    (repo / ".gitignore").write_text(".venv/\n")
    _git(repo, "init", "-q")
    _git(repo, "add", ".")
    _git(repo, "commit", "-qm", "fake")
    monkeypatch.setenv("FAKE_ARGS_OUT", str(tmp_path / "run_args.json"))
    monkeypatch.delenv("FAKE_ATTACH_FAIL", raising=False)

    image = tmp_path / "osworld.sqsh"
    image.write_bytes(b"squashfs")
    monkeypatch.setitem(
        pins.IMAGES,
        cua_harbor.host_platform(),
        pins.ImagePin(
            docker_image_id="", sqsh_sha256=hashlib.sha256(b"squashfs").hexdigest()
        ),
    )
    model_files = tmp_path / "model-files"
    model_files.mkdir()
    (model_files / "config.json").write_text("{}")
    cfg = HarborConfig(
        cua_harbor_dir=repo,
        model_files_dir=model_files,
        image_sqsh=image,
        work_root=tmp_path / "work",
    )
    return cfg, _git(repo, "rev-parse", "HEAD")


def _request(url: str, **overrides) -> RunRequest:
    body = {
        "model_name": MODEL,
        "endpoint_urls": [url + "/v1"],
        "generation_params": {"temperature": 1.0, "top_p": 0.95, "max_tokens": 32768},
        "task_ids": TASKS,
        "n_attempts": 2,
        "max_turns": 30,
        "concurrency": 2,
    }
    body.update(overrides)
    return RunRequest.model_validate(body)


class TestRunnerChecks:
    def test_pin_is_the_upstream_enroot_commit(self):
        assert pins.CUA_HARBOR_COMMIT == "09b42370f4abad7a91fd7c01e41e0a3a86ff9e8c"
        assert cua_harbor.ENVIRONMENT_IMPORT == (
            "osworld_novm.enroot.environment:EnrootEnvironment"
        )

    def test_refuses_a_checkout_off_the_pin(self, harbor_cfg):
        cfg, _ = harbor_cfg
        with pytest.raises(RunnerError, match="the pin is"):
            OSWorldRunner(cfg, pinned_commit="0" * 40)

    def test_refuses_modified_tracked_files(self, harbor_cfg):
        cfg, commit = harbor_cfg
        (cfg.cua_harbor_dir / "scripts" / "run.sh").write_text("changed")
        with pytest.raises(RunnerError, match="modified files"):
            OSWorldRunner(cfg, pinned_commit=commit)

    def test_refuses_another_image(self, harbor_cfg):
        cfg, commit = harbor_cfg
        cfg.image_sqsh.write_bytes(b"other")
        with pytest.raises(RunnerError, match="sha256"):
            OSWorldRunner(cfg, pinned_commit=commit)

    def test_health_checks_the_endpoint_without_generating(self, harbor_cfg, fake_vllm):
        cfg, commit = harbor_cfg
        runner = OSWorldRunner(cfg, pinned_commit=commit)
        assert runner.health(fake_vllm.url + "/v1")["endpoint_reachable"] is True
        down = runner.health("http://127.0.0.1:9")
        assert down["endpoint_reachable"] is False and down["endpoint_error"]
        assert fake_vllm.chat_requests == []


class TestRun:
    def test_two_task_run(self, harbor_cfg, fake_vllm, tmp_path):
        cfg, commit = harbor_cfg
        runner = OSWorldRunner(cfg, pinned_commit=commit)
        run_dir = tmp_path / "run"
        run_dir.mkdir()
        phases: list[str] = []

        result = runner.run(
            _request(fake_vllm.url), run_dir, CancellationToken(), phases.append
        )

        assert phases == ["preflight", "trials"]
        assert result["trials_expected"] == 4
        assert result["trials_errored"] == 1
        assert result["mean_reward"] == pytest.approx(2 / 4)
        assert result["cua_harbor_commit"] == commit
        preflight = json.loads((run_dir / "preflight.json").read_text())
        assert preflight["ok"]
        assert [c["name"] for c in preflight["checks"]] == [
            "platform",
            "tasks",
            "served_model",
            "attach",
        ]
        server_dir = run_dir / "servers" / MODEL
        record = json.loads((run_dir / "server_record.json").read_text())
        assert record["observed"]["url"] == fake_vllm.url
        assert record["serve_args"]["max_model_len"] == 131072
        assert (server_dir / "serve.model").read_text().strip() == str(
            cfg.model_files_dir
        )
        # The service itself sends no generation request; attach and the
        # agent own every model call.
        assert fake_vllm.chat_requests == []
        sent = json.loads((tmp_path / "run_args.json").read_text())
        args = sent["args"]
        assert sent["url"] == fake_vllm.url
        assert sent["squashfs"] == str(cfg.image_sqsh)
        assert sent["work_root"] == str(cfg.work_root)
        assert sent["serve_dir"] == str(server_dir)
        assert f"serve_args={server_dir / 'serve.args.json'}" in args
        for ak in (
            "temperature=1.0",
            "top_p=0.95",
            "max_new_tokens=32768",
            "max_turns=30",
            "model_image_size=[1280,720]",
        ):
            assert ak in args
        assert not any("ignore_eos" in a for a in args)
        assert args[args.index("--env") + 1] == cua_harbor.ENVIRONMENT_IMPORT
        assert args[args.index("-n") + 1] == "2"
        tsv = (run_dir / "trials.tsv").read_text().splitlines()
        assert len(tsv) == 5 and "AgentTimeoutError" in tsv[-1]
        assert (run_dir / "harbor.log").is_file()

    @pytest.mark.parametrize(
        "change, message",
        [
            ("short_context", "max_model_len"),
            ("unknown_model", "is not served"),
            ("unknown_task", "unknown tasks"),
            ("attach_refuses", "template differs"),
            ("attach_other_name", "attached no server named"),
        ],
    )
    def test_preflight_refuses_before_any_desktop(
        self, harbor_cfg, fake_vllm, tmp_path, monkeypatch, change, message
    ):
        cfg, commit = harbor_cfg
        runner = OSWorldRunner(cfg, pinned_commit=commit)
        overrides: dict = {}
        if change == "short_context":
            fake_vllm.max_model_len = 32768
        elif change == "unknown_model":
            overrides["model_name"] = "other"
        elif change == "unknown_task":
            overrides["task_ids"] = ["writer-nope"]
        elif change == "attach_refuses":
            monkeypatch.setenv("FAKE_ATTACH_FAIL", "the served template differs")
        elif change == "attach_other_name":
            monkeypatch.setenv("FAKE_SERVED_NAME", "someone-else")
        run_dir = tmp_path / "run"
        run_dir.mkdir()

        with pytest.raises(RunnerError, match=message):
            runner.run(
                _request(fake_vllm.url, **overrides),
                run_dir,
                CancellationToken(),
                lambda phase: None,
            )

        assert not (tmp_path / "run_args.json").exists()
        assert not json.loads((run_dir / "preflight.json").read_text())["ok"]

    def test_platform_mismatch(self, harbor_cfg, fake_vllm, tmp_path):
        cfg, commit = harbor_cfg
        runner = OSWorldRunner(cfg, pinned_commit=commit)
        other = "aarch64" if runner.platform == "x86_64" else "x86_64"
        with pytest.raises(RunnerError, match="service runs on"):
            runner.run(
                _request(fake_vllm.url, platform=other),
                tmp_path,
                CancellationToken(),
                lambda phase: None,
            )

    def test_harbor_without_trials_fails(self, harbor_cfg, fake_vllm, tmp_path):
        cfg, _ = harbor_cfg
        (cfg.cua_harbor_dir / "scripts" / "run.sh").write_text("#!/bin/sh\nexit 3\n")
        _git(cfg.cua_harbor_dir, "commit", "-qam", "broken")
        runner = OSWorldRunner(
            cfg, pinned_commit=_git(cfg.cua_harbor_dir, "rev-parse", "HEAD")
        )
        with pytest.raises(RunnerError, match="without finishing a trial"):
            runner.run(
                _request(fake_vllm.url), tmp_path, CancellationToken(), lambda p: None
            )
