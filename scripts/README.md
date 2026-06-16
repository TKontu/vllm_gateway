# Scripts

## `smoke_test.sh` — end-to-end validation on the inference host

A version-agnostic smoke test that drives a **running** gateway over its HTTP API
(OpenAI-compatible endpoints + `/gateway/status`). It verifies the things unit tests can't:
real model load, routing, streaming, and concurrency against live Docker + GPUs.

It checks:
1. Gateway reachable (`/v1/models`) and lists models.
2. `/gateway/status` is well-formed (VRAM probe, pools, GPUs).
3. **Cold start** — first completion downloads + loads the model (can take minutes).
4. Status reflects the loaded container (and its `gpu_uuids`/pool on multi-GPU builds).
5. **Warm** completion (fast path).
6. **Streaming** (SSE chunks + `[DONE]`).
7. **Concurrency** — N parallel requests all return 200 (exercises the lock model).
8. **Placement/eviction** — a second model loads; model 1 still serves afterward.

### Prerequisites
- The gateway stack is up (see repo README / `docker-compose.yml`):
  ```bash
  docker build -t my-vllm-gateway:latest ./gateway
  docker compose up -d            # exposes the gateway on :9003
  docker logs -f vllm_gateway     # watch model loads / per-model vLLM command
  ```
- `bash`, `curl`, `python3` on the host. NVIDIA container runtime + `docker.sock` mounted
  (the gateway needs them to spawn vLLM containers and probe VRAM).

### Run
```bash
# Defaults: GATEWAY_URL=http://localhost:9003, models auto-picked from /v1/models
./scripts/smoke_test.sh

# Pick specific models (primary + a second for the placement test):
./scripts/smoke_test.sh gemma3-4B qwen2.5

# Tuning via env:
GATEWAY_URL=http://10.0.0.5:9003 COLD_TIMEOUT=1800 CONCURRENCY=8 \
  ./scripts/smoke_test.sh
```

| Env | Default | Meaning |
|-----|---------|---------|
| `GATEWAY_URL` | `http://localhost:9003` | Gateway base URL |
| `SMOKE_MODEL` / arg 1 | first from `/v1/models` | Primary model to exercise |
| `SMOKE_MODEL2` / arg 2 | second from `/v1/models` | Second model (placement test) |
| `COLD_TIMEOUT` | `1200` | Seconds allowed for a cold start (weights download) |
| `WARM_TIMEOUT` | `120` | Seconds for a warm request |
| `CONCURRENCY` | `5` | Parallel warm requests |
| `MAX_TOKENS` | `16` | Tokens per completion (keep small) |
| `SKIP_MODEL2` | `0` | Set `1` to skip the placement test |

Exit code is `0` only if all **required** checks pass; some checks are WARN-only (e.g.
`gpu_uuids` absent on older single-pool builds). On failure, inspect `docker logs vllm_gateway`.

> Note: this issues **real inference** — the first run for a model downloads its weights and
> can take several minutes. Use small/quantized models for a quick pass.
