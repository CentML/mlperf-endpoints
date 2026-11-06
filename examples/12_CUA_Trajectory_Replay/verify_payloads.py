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

"""Compare replayed HTTP requests with the source h_cua_perf trace, 1:1.

Inputs are the original trace records and the request log written by
``recording_echo_server.py``. For every session and turn it checks that the
request the server received matches the recorded request: ``messages``
(images, placeholders and history exactly), ``tools``, ``tool_choice``,
``max_completion_tokens == output_length`` and ``ignore_eos``. It also checks
that turns of a session arrive in order without overlap and that each turn
waited at least its recorded delay after the previous response completed.
Exits non-zero on any mismatch.
"""

import argparse
import json
import subprocess
import sys
from collections import Counter, defaultdict
from collections.abc import Iterator
from pathlib import Path

# Request keys derived from the trace; everything else comes from YAML
# model_params (sampling, model name, streaming) and is only reported.
TRACE_KEYS = {"messages", "tools", "tool_choice", "max_completion_tokens", "ignore_eos"}


def iter_jsonl(path: Path) -> Iterator[dict]:
    if path.suffix == ".zst":
        proc = subprocess.Popen(
            ["zstd", "-dc", str(path)], stdout=subprocess.PIPE, text=True
        )
        assert proc.stdout is not None
        try:
            for line in proc.stdout:
                if line.strip():
                    yield json.loads(line)
        finally:
            proc.stdout.close()
            if proc.wait() != 0:
                raise RuntimeError(f"zstd failed to decompress {path}")
        return
    with path.open() as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def drop_null_message_keys(messages: list[dict]) -> tuple[list[dict], int]:
    """Treat ``"key": null`` and a missing key as equal (the client omits nulls)."""
    dropped = 0
    out = []
    for m in messages:
        clean = {k: v for k, v in m.items() if v is not None}
        dropped += len(m) - len(clean)
        out.append(clean)
    return out, dropped


def first_diff(a, b, path: str = "") -> str:
    """Return a short description of the first difference between two JSON values."""
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                return f"{path}.{k}: present only in {'trace' if k in a else 'sent'}"
            if a[k] != b[k]:
                return first_diff(a[k], b[k], f"{path}.{k}")
    elif isinstance(a, list):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            if x != y:
                return first_diff(x, y, f"{path}[{i}]")
    return f"{path}: {str(a)[:80]!r} != {str(b)[:80]!r}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--trace", type=Path, required=True, help="Source h_cua_perf JSONL(.zst)"
    )
    parser.add_argument(
        "--captured", type=Path, required=True, help="recording_echo_server.py output"
    )
    parser.add_argument(
        "--no-ignore-eos", action="store_true", help="Expect ignore_eos to be absent"
    )
    parser.add_argument(
        "--skip-delay-check",
        action="store_true",
        help="Run used inject_tool_delay: false",
    )
    parser.add_argument("--delay-tolerance-s", type=float, default=0.05)
    parser.add_argument("--max-errors", type=int, default=20)
    args = parser.parse_args()

    captured: dict[str, list[dict]] = defaultdict(list)
    for rec in iter_jsonl(args.captured):
        captured[rec["session_id"]].append(rec)
    for reqs in captured.values():
        reqs.sort(key=lambda r: r["received_at"])

    errors: list[str] = []
    stats: Counter = Counter()
    other_keys: Counter = Counter()
    delay_lag_s: list[float] = []

    def fail(msg: str) -> None:
        stats["errors"] += 1
        if len(errors) < args.max_errors:
            errors.append(msg)

    trace_by_session: dict[str, list[dict]] = defaultdict(list)
    for rec in iter_jsonl(args.trace):
        if rec["session_id"] in captured:
            trace_by_session[rec["session_id"]].append(rec)

    for sid in captured:
        if sid not in trace_by_session:
            fail(f"session {sid}: captured but not in trace")

    for sid, records in trace_by_session.items():
        sent = captured[sid]
        stats["sessions"] += 1
        if len(sent) != len(records):
            fail(
                f"session {sid}: trace has {len(records)} requests, server got {len(sent)}"
            )
        for turn, (rec, req) in enumerate(zip(records, sent, strict=False), start=1):
            stats["requests"] += 1
            body = req["body"]
            where = f"session {sid} turn {turn}"
            other_keys.update(set(body) - TRACE_KEYS)

            want_msgs, n_null = drop_null_message_keys(rec["messages"])
            got_msgs, _ = drop_null_message_keys(body.get("messages", []))
            stats["null_keys_normalized"] += n_null
            if want_msgs != got_msgs:
                fail(
                    f"{where}: messages differ at {first_diff(want_msgs, got_msgs, 'messages')}"
                )
            if body.get("tools") != rec.get("tools"):
                fail(f"{where}: tools differ")
            want_choice = (rec.get("extra") or {}).get("tool_choice")
            if body.get("tool_choice") != want_choice:
                fail(
                    f"{where}: tool_choice {body.get('tool_choice')!r} != {want_choice!r}"
                )
            if body.get("max_completion_tokens") != rec["output_length"]:
                fail(
                    f"{where}: max_completion_tokens {body.get('max_completion_tokens')} "
                    f"!= output_length {rec['output_length']}"
                )
            want_ignore_eos = None if args.no_ignore_eos else True
            if body.get("ignore_eos") != want_ignore_eos:
                fail(
                    f"{where}: ignore_eos {body.get('ignore_eos')!r} != {want_ignore_eos!r}"
                )

            stats["images_sent"] += sum(
                1
                for m in body.get("messages", [])
                if isinstance(m.get("content"), list)
                for p in m["content"]
                if isinstance(p, dict) and p.get("type") == "image_url"
            )
            if turn > 1:
                prev = sent[turn - 2]
                gap = req["received_at"] - prev["completed_at"]
                if gap < 0:
                    fail(
                        f"{where}: issued {-gap:.3f}s before turn {turn - 1} completed"
                    )
                if not args.skip_delay_check and rec.get("delay") is not None:
                    lag = gap - rec["delay"] / 1000.0
                    delay_lag_s.append(lag)
                    if lag < -args.delay_tolerance_s:
                        fail(
                            f"{where}: waited {gap:.3f}s, recorded delay {rec['delay'] / 1000:.3f}s"
                        )

    summary = {
        "result": "PASS" if stats["errors"] == 0 and stats["requests"] else "FAIL",
        **dict(stats),
        "non_trace_request_keys": dict(other_keys),
    }
    if delay_lag_s:
        lags = sorted(delay_lag_s)
        summary["delay_lag_s"] = {
            "min": round(lags[0], 4),
            "p50": round(lags[len(lags) // 2], 4),
            "max": round(lags[-1], 4),
        }
    print(json.dumps(summary, indent=1))
    for e in errors:
        print("MISMATCH:", e, file=sys.stderr)
    sys.exit(0 if summary["result"] == "PASS" else 1)


if __name__ == "__main__":
    main()
