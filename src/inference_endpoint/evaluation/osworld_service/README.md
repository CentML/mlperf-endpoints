# OSWorld Service

Runs OSWorld computer-use tasks for the `osworld_cua` accuracy scorer: the
computer-1 agent of [cua-harbor](https://github.com/cua-lab/cua-harbor) drives
one Linux desktop container per trial and queries the benchmark's own model
endpoint. The benchmark client sends the endpoint URL, model name, sampling
parameters and task ids; the service returns per-trial rewards and keeps the
Harbor job directory.

The service is trusted infrastructure: it receives the endpoint URL and optional
endpoint credentials, starts containers on its node and serves run artifacts.
This subproject commits its own `uv.lock`.

## Where it runs

- **One CPU node of the cluster that serves the model.** The desktops need the
  CPUs; the model server's node keeps its CPUs for the server. The CPU node must
  reach the server's URL: start the service on the same cluster as the server
  (for example x86 desktops on pdx CPU nodes for a pdx B300 server, arm64
  desktops on aga CPU nodes for an aga GB300 server).
- **After the performance phase.** The scorer calls the service from benchmark
  finalization, once the performance phase has finished, so desktops never run
  while performance is measured. Before the benchmark starts, the scorer only
  calls `GET /health`, which checks the endpoint with `GET /v1/models` and
  starts nothing.
- **Desktops per node.** 48 at once per exclusive CPU node ran without errors;
  16 is the scorer's default. One 84-task × 2-attempt pass takes about 21
  minutes at 16 desktops.

## What it needs on the node

| Flag                | What                                                                                                                                                                       |
| ------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `--cua-harbor-dir`  | A local cua-harbor checkout with its `.venv` (`uv sync` in it). Its `HEAD` must be `pins.CUA_HARBOR_COMMIT` and its tracked files clean.                                   |
| `--model-files-dir` | The served model's non-weight files: `config.json`, tokenizer files, `generation_config.json`, processor configs and `chat_template.jinja`. No weights are needed.         |
| `--image-sqsh`      | The `cua-harbor/osworld:0.2` desktop image as an Enroot squashfs (cua-harbor's osworld-novm README, "Enroot"), on local disk; its sha256 must equal the pin for this node. |
| `--work-root`       | A node-local directory for the desktops' root filesystems.                                                                                                                 |

The node needs `enroot`, `nsenter` and `slirp4netns` (the desktops), and `jq`,
`curl` and `uv` (cua-harbor's scripts).

cua-harbor is not fetched by the service and cannot be given as a URL: it is a
checkout on the cluster, pinned by commit hash in
[`osworld_service/pins.py`](osworld_service/pins.py), together with the sha256
of each architecture's desktop image. The service refuses to start on another
commit, a modified checkout or another image. Moving to a new cua-harbor commit
or image is a reviewed edit of `pins.py`. Everything the service knows about
cua-harbor is in
[`osworld_service/cua_harbor.py`](osworld_service/cua_harbor.py).

## Start

Inside a one-node CPU allocation on the server's cluster:

```bash
uv run --project src/inference_endpoint/evaluation/osworld_service \
  python -m osworld_service --host 0.0.0.0 --port 18090 \
  --auth-token "$OSWORLD_SERVICE_TOKEN" \
  --artifact-root /shared/fs/osworld_runs \
  --cua-harbor-dir /shared/fs/cua-harbor \
  --model-files-dir /shared/fs/qwen3.8-27b-files \
  --image-sqsh /local/osworld-0.2.sqsh \
  --work-root /local/osworld-desktops
```

Startup hashes the desktop image (a few seconds to half a minute). Then point
the benchmark config's `osworld_service_url` at `http://<cpu node>:18090` and
run `inference-endpoint benchmark from-config --mode both`.

## What one run does

1. **Preflight, before any desktop starts.** Each check is recorded in
   `preflight.json`; the first failure refuses the run.
   - `platform`: the request's platform (if any) is the node's.
   - `tasks`: every task id exists in the checkout.
   - `served_model`: `/v1/models` serves the model name with `max_model_len` of
     at least 131072, the agent's context window.
   - `attach`: `scripts/serve.sh --use URL --model DIR` writes the run's server
     directory. It refuses a server whose `/tokenize` of a probe chat differs
     from the model entry's served template, or that does not take exactly the
     entry's 100 images per prompt. A server started with another chat template
     would otherwise give plausible but wrong scores and no error. Its output is
     `attach.log`.
2. **Server record.** `server_record.json` holds the attached `serve.args.json`
   and `serve.observed.json` (what the server answered; the launch facts it does
   not show stay null), the pinned commit and image hash, and a hash of the
   model files.
3. **Trials.** `scripts/run.sh` with the computer-1 agent and the osworld-novm
   Enroot environment, one desktop per trial on this node, with the requested
   attempts, turn budget and desktops at once. The sampling parameters are the
   request's `temperature`, `top_p`, `top_k` and `max_tokens` (computer-1's
   `max_new_tokens`); anything else is refused.
4. **Result.** `osworld_results.json` and `trials.tsv` list every finished trial
   with its reward and exception, plus the job directory on the service host.

## API

| Route                                | Purpose                                                                                                                  |
| ------------------------------------ | ------------------------------------------------------------------------------------------------------------------------ |
| `GET /health[?endpoint=URL]`         | Version, capabilities, platform, pins; with `endpoint`, whether it answers.                                              |
| `POST /v1/runs`                      | Start a run (one active run per service).                                                                                |
| `GET /v1/runs/{id}`                  | Status, phase (`preflight`, `trials`), trials finished so far, result.                                                   |
| `POST /v1/runs/{id}/cancel`          | Stop the run and its desktops.                                                                                           |
| `GET /v1/runs/{id}/artifacts/{name}` | `osworld_results.json`, `preflight.json`, `server_record.json`, `trials.tsv`, `harbor.log`, `attach.log`, `status.json`. |

All routes except `/health` need `Authorization: Bearer <token>` unless the
service was started with `--allow-unauthenticated`.
