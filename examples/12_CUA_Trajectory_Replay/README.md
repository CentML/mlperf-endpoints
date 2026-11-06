# CUA Trajectory Replay (H Company `h_cua_perf`)

This example replays recorded Computer-Use Agent (CUA) inference requests
against an OpenAI-compatible server to measure serving performance on a
multimodal, multi-turn agentic workload. It reuses the
[agentic inference](../10_Agentic_Inference/README.md) load pattern: one
in-flight turn per session, `X-Session-ID` routing headers, and recorded think
time between turns.

It measures inference performance only. Model responses are not fed back into
later turns, so task success is not evaluated.

## Dataset

[`Hcompany/h_cua_perf`](https://huggingface.co/datasets/Hcompany/h_cua_perf)
holds 18,224 recorded requests from 477 agent sessions (2.5 GB compressed, 9.3
GB uncompressed). Each record is the complete chat-completions request the agent
sent at one step:

| Field           | Meaning                                                              |
| --------------- | -------------------------------------------------------------------- |
| `session_id`    | Agent session; its records are contiguous and in order               |
| `messages`      | Full request history, with one PNG screenshot as an `image_url` part |
| `tools`         | Tool schemas; absent on the 25 context-compaction / OCR calls        |
| `extra`         | `{"tool_choice": "auto"}`; absent when `tools` is absent             |
| `output_length` | Recorded completion tokens                                           |
| `delay`         | Milliseconds between the previous response and this request          |

History is not append-only: each step replaces older screenshots with an
`[Image omitted by context cleaning]` placeholder, and compaction calls restart
the history from a summary. Rebuilding requests from per-message rows cannot
reproduce this, so the converter emits **verbatim rows**: each record becomes
one client turn whose `messages` are sent exactly as recorded (see
`AgenticInferenceDataset`).

```bash
hf download Hcompany/h_cua_perf --repo-type dataset --local-dir h_cua_perf
sha256sum h_cua_perf/h_cua.jsonl.zst   # compare with h_cua.meta.json
```

The dataset ships `trace_processor.py` for sampling sessions, capping session
length and widening the screenshot window; its output can be converted the same
way.

## Convert

```bash
python convert_h_cua_perf.py \
  --input h_cua_perf/h_cua.jsonl.zst \
  --output datasets/h_cua_perf_endpoints.jsonl
```

| Trace field         | Converted row                              |
| ------------------- | ------------------------------------------ |
| `session_id`        | `conversation_id`                          |
| position in session | `turn` (1..N), `role: "user"`              |
| `messages`          | `messages` (unchanged)                     |
| `tools`             | `tools` (omitted when absent)              |
| `extra.tool_choice` | `tool_choice`                              |
| `output_length`     | `max_new_tokens`, plus `ignore_eos: true`  |
| `delay` (ms)        | `delay_seconds` (absent on the first turn) |

`ignore_eos` makes the server generate exactly the recorded number of tokens, so
the decode workload matches the trace for any model. Use `--no-ignore-eos` to
let the model stop naturally. `--session-ids` and `--max-sessions` select a
subset.

## Payload correctness gate

Before measuring performance, check that the client sends what was recorded.
`recording_echo_server.py` is a stub server that logs every request body;
`verify_payloads.py` compares them with the trace and fails on any difference in
`messages`, `tools`, `tool_choice`, `max_completion_tokens`, `ignore_eos`,
per-session order, or recorded delays.

```bash
python recording_echo_server.py --port 12345 --out gate/captured.jsonl &
CUA_DATASET=datasets/subset_endpoints.jsonl ENDPOINT_URL=http://127.0.0.1:12345 \
TOKENIZER=../../tests/assets/tokenizers/char_chat REPORT_DIR=gate/report CONCURRENCY=4 \
  inference-endpoint benchmark from-config -c minimax_cua_replay.yaml
python verify_payloads.py --trace subset.jsonl --captured gate/captured.jsonl
```

## Server

`serve_minimax_vllm.sh` starts vLLM with MiniMax M3.1 Flash on one 4-GPU node
using the single-node command from the official
[vLLM recipe](https://recipes.vllm.ai/MiniMaxAI/MiniMax-M3.1-Flash) (TP4,
MiniMax tool-call and reasoning parsers, built-in DSpark speculative decoding;
`SPEC_DECODE=0` turns it off). Keep image input enabled: check that any recipe
preset or feature you add (e.g. `text_only`) does not pass
`--language-model-only`. Size `--max-model-len` for the longest recorded prompts
(up to about 800K characters of text plus one screenshot).

## Run

```bash
CUA_DATASET=datasets/h_cua_perf_endpoints.jsonl ENDPOINT_URL=http://<server>:8000 \
TOKENIZER=<model dir> CONCURRENCY=8 REPORT_DIR=logs/cua_replay \
  inference-endpoint benchmark from-config -c minimax_cua_replay.yaml
```

`minimax_cua_replay.yaml` disables salting (it would rewrite the recorded system
prompt) and client-side ISL (it would tokenize base64 screenshots as text). Read
prompt-token and prefix-cache totals from the server's `/metrics` before and
after the run instead.
