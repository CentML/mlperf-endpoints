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

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from aiohttp import web

from .config import HarborConfig, ServiceConfig
from .runner import OSWorldRunner
from .server import create_app


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the OSWorld service on a CPU node of the model server's cluster"
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=18090)
    parser.add_argument("--artifact-root", default="osworld_service_artifacts")
    auth = parser.add_mutually_exclusive_group()
    auth.add_argument("--auth-token")
    auth.add_argument("--allow-unauthenticated", action="store_true")
    parser.add_argument("--max-stored-runs", type=int, default=20)
    parser.add_argument(
        "--cua-harbor-dir",
        required=True,
        help="local cua-harbor checkout at the pinned commit, with its .venv",
    )
    parser.add_argument(
        "--model-files-dir",
        required=True,
        help="the served model's non-weight files (config, tokenizer, processor, template)",
    )
    parser.add_argument(
        "--image-sqsh",
        required=True,
        help="the cua-harbor/osworld desktop image as an Enroot squashfs, on local disk",
    )
    parser.add_argument(
        "--work-root", required=True, help="node-local directory for desktops"
    )
    parser.add_argument(
        "--max-desktops-per-process",
        type=int,
        default=16,
        help="desktops per harbor process; a run is split into several processes",
    )
    parser.add_argument(
        "--skip-image-hash",
        action="store_true",
        help="do not hash the image at startup (development only)",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s"
    )

    artifact_root = Path(args.artifact_root).resolve()
    harbor = HarborConfig(
        cua_harbor_dir=Path(args.cua_harbor_dir).resolve(),
        model_files_dir=Path(args.model_files_dir).resolve(),
        image_sqsh=Path(args.image_sqsh).resolve(),
        work_root=Path(args.work_root).resolve(),
        max_desktops_per_process=args.max_desktops_per_process,
    )
    runner = OSWorldRunner(harbor, verify_image=not args.skip_image_hash)
    config = ServiceConfig(
        host=args.host,
        port=args.port,
        artifact_root=artifact_root,
        auth_token=args.auth_token,
        allow_unauthenticated=args.allow_unauthenticated,
        max_stored_runs=args.max_stored_runs,
    )
    web.run_app(create_app(config, runner), host=config.host, port=config.port)


if __name__ == "__main__":
    main()
