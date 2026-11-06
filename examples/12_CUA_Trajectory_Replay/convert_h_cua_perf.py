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

"""Convert H Company CUA perf traces into verbatim agentic-inference rows.

Each h_cua_perf record is the complete chat-completions request the agent sent
at one step. Every record becomes one verbatim client turn (see
``AgenticInferenceDataset``): its ``messages`` are replayed unchanged, so
screenshot placeholders and context-compaction resets are reproduced exactly.

Field mapping (one output row per input record)::

    session_id            -> conversation_id
    position in session   -> turn (1..N)
    messages              -> messages (unchanged)
    tools                 -> tools (omitted when the record has none)
    extra.tool_choice     -> tool_choice
    output_length         -> max_new_tokens (+ ignore_eos)
    delay (milliseconds)  -> delay_seconds (absent on the first request)

Input is an h_cua_perf JSONL file (``.jsonl``, or ``.jsonl.zst`` decompressed
with the ``zstd`` CLI), e.g. the dataset itself or a subset produced by the
dataset's ``trace_processor.py``.
"""

import argparse
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path

# Request fields of h_cua_perf ``extra`` that map onto the chat-completions
# request; any other key would be dropped by the client, so it is rejected.
SUPPORTED_EXTRA_KEYS = {"tool_choice"}


def iter_records(path: Path) -> Iterator[dict]:
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


def convert_record(record: dict, turn: int, ignore_eos: bool) -> dict:
    extra = record.get("extra") or {}
    unsupported = set(extra) - SUPPORTED_EXTRA_KEYS
    if unsupported:
        raise ValueError(
            f"session {record['session_id']} turn {turn}: unsupported extra "
            f"keys {sorted(unsupported)}"
        )
    output_length = int(record["output_length"])
    if output_length <= 0:
        raise ValueError(
            f"session {record['session_id']} turn {turn}: output_length must be > 0"
        )

    row: dict = {
        "conversation_id": record["session_id"],
        "turn": turn,
        "role": "user",
        "content": "",
        "messages": record["messages"],
        "max_new_tokens": output_length,
    }
    if ignore_eos:
        row["ignore_eos"] = True
    if record.get("tools") is not None:
        row["tools"] = record["tools"]
    if "tool_choice" in extra:
        row["tool_choice"] = extra["tool_choice"]
    if record.get("delay") is not None:
        row["delay_seconds"] = record["delay"] / 1000.0
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--session-ids",
        nargs="+",
        default=None,
        help="Keep only these sessions (in input order).",
    )
    parser.add_argument(
        "--max-sessions",
        type=int,
        default=None,
        help="Keep only the first N sessions (after --session-ids filtering).",
    )
    parser.add_argument(
        "--no-ignore-eos",
        action="store_true",
        help="Let the model stop at EOS instead of generating output_length tokens.",
    )
    args = parser.parse_args()

    wanted = set(args.session_ids) if args.session_ids else None
    seen_sessions: list[str] = []
    finished: set[str] = set()
    turn = 0
    n_rows = n_no_tools = total_output = 0

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as out:
        for record in iter_records(args.input):
            sid = record["session_id"]
            if wanted is not None and sid not in wanted:
                continue
            if not seen_sessions or seen_sessions[-1] != sid:
                if sid in finished or sid in seen_sessions:
                    raise ValueError(f"records of session {sid} are not contiguous")
                if seen_sessions:
                    finished.add(seen_sessions[-1])
                if args.max_sessions is not None and len(seen_sessions) >= (
                    args.max_sessions
                ):
                    break
                seen_sessions.append(sid)
                turn = 0
            turn += 1
            row = convert_record(record, turn, ignore_eos=not args.no_ignore_eos)
            out.write(json.dumps(row, ensure_ascii=False) + "\n")
            n_rows += 1
            n_no_tools += "tools" not in row
            total_output += row["max_new_tokens"]

    if wanted is not None and wanted - set(seen_sessions):
        missing = sorted(wanted - set(seen_sessions))
        print(f"warning: sessions not found: {missing}", file=sys.stderr)
    print(
        json.dumps(
            {
                "output": str(args.output),
                "sessions": len(seen_sessions),
                "requests": n_rows,
                "requests_without_tools": n_no_tools,
                "total_output_tokens": total_output,
            }
        )
    )


if __name__ == "__main__":
    main()
