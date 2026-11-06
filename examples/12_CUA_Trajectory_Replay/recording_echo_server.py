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

"""OpenAI-compatible stub server that records every chat-completions request.

Used as the payload-correctness gate for trajectory replay: run the benchmark
against this server, then compare the recorded bodies with the source trace
using ``verify_payloads.py``. Each output line holds the session header, the
receive and response-complete timestamps, and the request body exactly as sent.
Replies are a fixed short text, so no GPU or model is needed.
"""

import argparse
import json
import time
import uuid
from pathlib import Path

from aiohttp import web

SESSION_HEADER = "X-Session-ID"
# Recorded CUA requests carry base64 screenshots and reach a few MB; aiohttp's
# default 1 MiB body limit would reject them with HTTP 413.
MAX_BODY_BYTES = 64 * 1024 * 1024


class RequestRecorder:
    def __init__(self, out_path: Path, reply_text: str):
        self._out = out_path.open("w")
        self._reply_text = reply_text

    def close(self) -> None:
        self._out.close()

    async def handle(self, request: web.Request) -> web.StreamResponse:
        received_at = time.time()
        raw = await request.read()
        body = json.loads(raw)
        completion_id = f"chatcmpl-{uuid.uuid4().hex}"
        model = body.get("model", "")
        response: web.StreamResponse
        if body.get("stream"):
            response = web.StreamResponse(
                status=200, headers={"Content-Type": "text/event-stream"}
            )
            await response.prepare(request)
            for delta, finish in (
                ({"role": "assistant", "content": self._reply_text}, None),
                ({}, "length"),
            ):
                chunk = {
                    "id": completion_id,
                    "object": "chat.completion.chunk",
                    "created": int(received_at),
                    "model": model,
                    "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
                }
                await response.write(f"data: {json.dumps(chunk)}\n\n".encode())
            await response.write(b"data: [DONE]\n\n")
            await response.write_eof()
        else:
            response = web.json_response(
                {
                    "id": completion_id,
                    "object": "chat.completion",
                    "created": int(received_at),
                    "model": model,
                    "choices": [
                        {
                            "index": 0,
                            "message": {
                                "role": "assistant",
                                "content": self._reply_text,
                            },
                            "finish_reason": "length",
                        }
                    ],
                }
            )
        # Single-threaded event loop: writes from concurrent handlers never interleave.
        self._out.write(
            json.dumps(
                {
                    "session_id": request.headers.get(SESSION_HEADER),
                    "received_at": received_at,
                    "completed_at": time.time(),
                    "body_bytes": len(raw),
                    "body": body,
                }
            )
            + "\n"
        )
        self._out.flush()
        return response


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=12345)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--reply-text", default="ok")
    args = parser.parse_args()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    recorder = RequestRecorder(args.out, args.reply_text)
    app = web.Application(client_max_size=MAX_BODY_BYTES)
    app.router.add_post("/v1/chat/completions", recorder.handle)
    try:
        print(f"recording requests to {args.out}", flush=True)
        web.run_app(app, host=args.host, port=args.port, print=None)
    finally:
        recorder.close()


if __name__ == "__main__":
    main()
