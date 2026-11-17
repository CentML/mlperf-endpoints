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

"""What the service runs, pinned: the cua-harbor commit and the desktop images.

cua-harbor is used from a local checkout (``--cua-harbor-dir``), not fetched:
the service refuses to start unless its ``HEAD`` is ``CUA_HARBOR_COMMIT`` and
its tracked files are unmodified. Changing the commit or an image is a reviewed
edit of this file.
"""

from __future__ import annotations

from dataclasses import dataclass

CUA_HARBOR_COMMIT = "09b42370f4abad7a91fd7c01e41e0a3a86ff9e8c"


@dataclass(frozen=True)
class ImagePin:
    """The ``cua-harbor/osworld:0.2`` desktop image as an Enroot squashfs."""

    docker_image_id: str
    sqsh_sha256: str


IMAGES: dict[str, ImagePin] = {
    "x86_64": ImagePin(
        docker_image_id=(
            "sha256:24435afff80fbdd5f04cd1cf3a3d061d2bfaca258657250fdc859a9b2784ce48"
        ),
        sqsh_sha256="ccf67a9e235147003cd0eab3e8cc2ce2ba779ac6bbb81b6df740f92c623a4f3f",
    ),
    # The arm64 build imported under the tag cua-harbor/osworld:0.2-arm64, with
    # only its Enroot provenance relabelled to cua-harbor/osworld:0.2 (the tag
    # every task.toml names), which cua-harbor's Enroot environment requires.
    "aarch64": ImagePin(
        docker_image_id="",
        sqsh_sha256="17e7b01c742b5b9df540f9d9579fa077bf7b00fc44a510ebe7daa2a3f5ff099d",
    ),
}
