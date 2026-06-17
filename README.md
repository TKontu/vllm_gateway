# vLLM Smart Gateway

This project provides a smart, dynamic gateway for the [vLLM inference engine](https://github.com/vllm-project/vllm). It enables the serving of multiple large language models on a single GPU, with intelligent, on-demand model loading, unloading, and VRAM management.

The gateway can run multiple models concurrently if they fit in VRAM, or swap them based on a least-recently-used (LRU) policy if they don't. It automatically learns the VRAM footprint of each model to make efficient packing decisions.

**✨ Now supports GGUF quantized models** for memory-efficient inference with automatic tokenizer detection.

> [!WARNING] > **Security Notice: Requires Root-Equivalent Host Access**
> This gateway requires direct access to the host's Docker socket (`/var/run/docker.sock`). This is equivalent to granting root access to the host system. Please be fully aware of the security implications before deploying this project. It is strongly recommended to run this in a trusted and isolated environment.

## How It Works

The architecture consists of a single, persistent `gateway` container that has access to the host's Docker socket.

1.  An API request is sent to the gateway's `/v1/chat/completions` endpoint, specifying a model (e.g., `"model": "gemma3-4B"`).
2.  The gateway checks if the requested model is already running in a `vllm_server` container.
3.  **If the model is running**, the request is forwarded.
4.  **If the model is not running**, the gateway checks its VRAM footprint.
    - **Unknown Footprint (First Run):** The gateway stops all other models, starts the requested one in isolation, measures its VRAM usage, and saves the footprint to `memory_footprints.json`.
    - **Known Footprint:** The gateway checks if there is enough free VRAM.
      - If it fits, the model is started alongside other models.
      - If it doesn't fit, the gateway evicts the least recently used model(s) until there is enough space.
5.  Once the model is ready, the request is forwarded.

This process is transparent to the end-user, who simply experiences a longer response time for the first request to a new or swapped model.

## Key Features

- **Dynamic VRAM Management:** Intelligently loads and unloads models to maximize GPU utilization.
- **Concurrent Model Serving:** Runs multiple models simultaneously if they fit in VRAM.
- **Footprint Learning:** Automatically measures and records the VRAM footprint of new models.
- **LRU Eviction Policy:** Gracefully swaps models when space is needed.
- **Inactivity Timeout:** Automatically shuts down idle model containers to free up resources.
- **Standard OpenAI API:** Uses the familiar `/v1/chat/completions` and `/v1/models` endpoints.
- **Concurrent Request Safety:** Uses an `asyncio.Lock` to handle simultaneous requests safely.
- **Highly Configurable:** Control models, networking, and resource limits via environment variables.

## Deployment (Docker Compose)

This gateway is designed to be deployed as a standalone service that dynamically manages other containers.

```yaml
version: "3.9"

networks:
  vllm_network:
    driver: bridge

services:
  gateway:
    # Build the image locally using the gateway/Dockerfile
    image: my-vllm-gateway:latest
    container_name: vllm_gateway
    ports:
      - "9003:9000" # Expose the gateway on port 9003
    environment:
      # --- REQUIRED ---
      # Your Hugging Face token for accessing gated models.
      HUGGING_FACE_HUB_TOKEN: ${HUGGING_FACE_HUB_TOKEN}
      # The name of this gateway container, used to resolve the Docker network.
      GATEWAY_CONTAINER_NAME: "vllm_gateway"

      # --- NETWORKING ---
      # The network to attach the vLLM container to. Must match the network defined above.
      DOCKER_NETWORK_NAME: "vllm_network"

      # --- MODEL & RESOURCE CONFIGURATION ---
      # A JSON string defining user-friendly names and their model paths.
      # Supports regular HF models and GGUF files (local, URLs, or repo/file.gguf format).
      ALLOWED_MODELS_JSON: '{"gemma3-4B":"google/gemma-3-4b-it", "qwen2.5":"Qwen/Qwen2.5-Coder-7B-Instruct", "tinyllama-gguf":"TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"}'
      # The percentage of GPU memory vLLM should be allowed to use (e.g., "0.90" for 90%).
      VLLM_GPU_MEMORY_UTILIZATION: ${VLLM_GPU_MEMORY_UTILIZATION:-0.90}
      # Timeout in seconds to shut down inactive model containers. Set to 0 to disable.
      VLLM_INACTIVITY_TIMEOUT: "1800" # 30 minutes
      # Global cap on model context length. 0 means use model's native max length.
      VLLM_MAX_MODEL_LEN_GLOBAL: "0"
      # --- LOGGING ---
      LOG_LEVEL: "INFO" # DEBUG, INFO, WARNING, ERROR

    volumes:
      # [SECURITY WARNING] Mount the Docker socket to allow the gateway to manage containers.
      - /var/run/docker.sock:/var/run/docker.sock
      # Mount the Hugging Face cache to avoid re-downloading models.
      - /root/.cache/huggingface:/root/.cache/huggingface
      # Mount a data directory for persistent files (app creates files inside automatically).
      - ./data:/app/data
    networks:
      - vllm_network
    restart: unless-stopped
```

### Building the Gateway Image

To build the `my-vllm-gateway:latest` image, navigate to the `gateway` directory and run:

```bash
docker build -t my-vllm-gateway:latest .
```

## Model Configuration File

Models and their per-model vLLM tuning live in a mounted YAML file. Its path comes from
`MODELS_CONFIG_FILE` (default `/app/config/models.yaml`); the compose file mounts
`./config` read-only at `/app/config`. Copy the example and edit it:

```bash
cp config/models.yaml.example config/models.yaml
```

```yaml
defaults:                      # optional; applies to all models
  gpu_memory_utilization: 0.90
  max_model_len: 0             # 0 = use the model's native max (omit the flag)
  tensor_parallel_size: 1
  quantization: null           # awq | gptq | fp8 | null
  dtype: auto
  inactivity_timeout: 1800     # seconds; 0 = never unload
  always_on: false
  extra_args: []               # raw vLLM CLI args, appended verbatim
models:                        # required, non-empty
  gemma3-4B:
    repo: google/gemma-3-4b-it
  qwen3-30b-awq:
    repo: cyankiwi/Qwen3-30B-A3B-Instruct-2507-AWQ-4bit
    quantization: awq          # per-model override
  bge-m3:
    repo: BAAI/bge-m3
    always_on: true            # never auto-unloaded
    inactivity_timeout: 0
```

**Precedence (per setting):** `per-model entry` → `defaults` block → **built-in default**
(the corresponding `VLLM_*` env var). So an unset field falls back to the `defaults` block,
and if that's unset too, to the legacy global env var.

**Per-model command construction.** Each container's vLLM command is built from its resolved
config: always `--model`, `--gpu-memory-utilization`, `--tensor-parallel-size`; plus
`--max-model-len` (only when `max_model_len > 0`, capped to the model's native max),
`--quantization` (only when not null), `--dtype` (only when not `auto`); then each `extra_args`
string is appended verbatim. `extra_args` override any flag the gateway generated (no
duplicates). The final command is logged per model.

**Lifecycle.** Containers whose model sets `always_on: true` (or `inactivity_timeout: 0`) are
never auto-unloaded by the inactivity monitor; all others use their own resolved
`inactivity_timeout`.

**Validation / fail-fast.** The file is validated at startup — `models` must be present and
non-empty, every model needs a non-empty `repo`, unknown keys are rejected, and types/ranges
are checked (`0 < gpu_memory_utilization <= 1`, integer fields must be integers, `extra_args`
must be a list). Any error is logged and the gateway refuses to start.

**Backward compatibility.** If `MODELS_CONFIG_FILE` is missing, the gateway falls back to the
legacy `ALLOWED_MODELS_JSON` env var with the global `VLLM_*` settings — behaving exactly as
before this feature existed.

## Multi-GPU pools

One gateway can manage several GPUs by declaring a `pools:` section and assigning each model to a
pool with `pool:`. Get the full UUIDs from `nvidia-smi -L`.

```yaml
pools:
  llm:  [GPU-aaaa-…]   # RTX 3090 — LLMs
  util: [GPU-bbbb-…]   # RTX A2000 — small/utility models
defaults: { pool: llm, gpu_memory_utilization: 0.90 }
models:
  qwen3-30b-awq: { repo: …, quantization: awq, pool: llm }
  small-util:    { repo: …, pool: util, gpu_memory_utilization: 0.45 }
```

- **Placement.** On each request the gateway picks a GPU *within the model's pool*: a card with
  enough free VRAM (tie-break: most free); if none fits, it evicts idle models (LRU, never
  `always_on` or in-flight) on a pool GPU; if it still can't fit, it returns **503** (it never
  over-commits and risks an OOM). Per-GPU accounting uses *actual* free VRAM from `nvidia-smi`, so
  it tolerates VRAM used by processes the gateway didn't start.
- **Tensor parallel.** Set `tensor_parallel_size: k` to spread one model across `k` GPUs of its
  pool (launched with `--tensor-parallel-size k`, pinned to those cards). TP requires a
  **homogeneous** pool (equal-VRAM GPUs — never a 3090+A2000 mix; the util pool's single A2000
  never TPs) and `k` must not exceed the GPUs declared in the pool (validated at startup). A model
  with `tensor_parallel_size: 1` whose **weights exceed one card** auto-falls-back to the minimal TP
  that fits. TP over PCIe (no NVLink) has real comms cost, so prefer single-GPU when a model fits.
  Each card is still filled to `gpu_memory_utilization` (TP fits bigger *weights*, not more KV).
- **Co-location.** Set `colocate: true` to let small models **share a card**. A co-locatable
  model's `gpu_memory_utilization` is its intended per-card *share* (e.g. `0.45`, so two fit a 24 GB
  card); the gateway caps the launch util to the chosen card's *actual free* VRAM (minus a margin,
  `COLOCATE_MARGIN_MIB`, default 1024) so a second model never OOMs the first. Co-locatable models
  only share with each other (a whole-card model never shares), are evicted only as needed (idle,
  LRU), and co-location is **single-GPU only** (not combinable with `tensor_parallel_size > 1`).
  Default (`colocate: false`) keeps whole-card placement. Expect per-model throughput to drop when a
  card is shared (contended SMs/KV) — opt in for lightly-used helpers, not your hot path.
- **GPU source precedence.** `pools:` in config → else `GATEWAY_GPU_UUID` (a single-GPU pool) →
  else all visible GPUs (the legacy single-pool behavior). When no `pools:` are declared, behavior
  is unchanged.
- **Embeddings/rerankers** are expected to run **out of band** — a separate always-on
  [TEI](https://github.com/huggingface/text-embeddings-inference) or
  [infinity](https://github.com/michaelfeil/infinity) container pinned to the A2000 — **not**
  through this swapping gateway. The gateway's `util` pool can still place small models on the same
  A2000 because it reads the card's real free VRAM. See `/gateway/status` for per-GPU pool/usage/residents.

## Placement modes

`PLACEMENT_MODE` controls how the gateway fits models onto a card.

### `budget` (default)

`GPU_BUDGET_FRACTION` becomes a **per-GPU total cap** (fraction of each card the gateway may fill
across *all* its models). Each model is launched sized to **only what it needs** — its weights plus
the KV cache for its `max_model_len` × `max_num_seqs`, plus a safety cushion — and the gateway packs
**multiple models onto one card** until the budget is full. When the budget can't fit a new model it
evicts idle models LRU, or returns **503** rather than over-commit.

Why this is safe: `--gpu-memory-utilization` is a hard fraction of total card memory, so launching a
model at `util = need / card_total` makes it **physically unable** to exceed its budgeted share. An
under-estimate can therefore only fail *that* model's own startup — it can never OOM a co-resident.
Estimates are biased generous via `BUDGET_OVERHEAD_FACTOR` / `BUDGET_OVERHEAD_MIB`.

**Ground-truth accounting.** The sizing estimate (weights from HF metadata + sliding-window-aware KV)
only picks the launch util. After a model is READY the gateway measures its **actual** per-process
VRAM (`nvidia-smi` compute-apps attributed to the container) and uses *that* as the footprint for
placement math — so the gateway's view always matches reality, including the effects of quantization,
sliding-window attention, and KV dtype. Each footprint is stamped with a **signature**
(mode + `max_model_len` + `max_num_seqs` + tensor-parallel + util basis) and is reused only when that
signature matches — so a mode switch or a config change **re-measures automatically** instead of
trusting a stale value.

**Restrictions in budget mode.** Memory-affecting flags must be set via the structured keys, not
`extra_args`: `--max-model-len`, `--max-num-seqs`, `--kv-cache-dtype`, `--gpu-memory-utilization` in
`extra_args` are rejected at startup (they would diverge from the gateway's sizing). Models the gateway
can't auto-size (GGUF, or no readable HF config) load **alone once** at their per-model
`gpu_memory_utilization`, are measured, then pack at that measured size on later requests — so set that
value to the share you want such a model to take.

**Requirement:** every model must set `max_model_len > 0` (and ideally a modest `max_num_seqs`),
because vLLM's KV-cache need is otherwise unbounded. The gateway **fails fast at startup** if a model
omits it. The `colocate` flag is ignored in this mode (packing is universal).

**Tradeoff:** a lone model is sized to its need and leaves the rest of the card idle (so others can
join). That caps single-model peak throughput (smaller KV pool) in exchange for density. If you run
one model at a time and want maximum throughput, use `whole_card`.

```yaml
# config/models.yaml — two models share one 24 GB card under budget mode
defaults: { pool: llm, max_num_seqs: 8 }
models:
  qwen9b:  { repo: cyankiwi/Qwen3.5-9B-AWQ-4bit,    quantization: awq, max_model_len: 32768 }
  gemma:   { repo: cyankiwi/gemma-4-E4B-it-AWQ-INT4, quantization: awq, max_model_len: 32768 }
```

#### Concurrency vs. packing: `kv_reservation_seqs`

`max_num_seqs` is vLLM's scheduler width *and*, by default, the amount of KV the gateway reserves
(`max_model_len × max_num_seqs`). For a model that takes **many short requests**, reserving for the full
worst-case concurrency wastes VRAM. Set **`kv_reservation_seqs`** below `max_num_seqs` to reserve KV for
fewer sequences while still admitting the full width — extra requests queue/page within the reserved
pool instead of demanding proportional VRAM:

```yaml
judge-fast:
  repo: cyankiwi/Qwen3.5-4B-AWQ-4bit
  max_model_len: 8192
  max_num_seqs: 32          # admit up to 32 concurrent
  kv_reservation_seqs: 4    # but only reserve KV for 4 -> packs like a 4-way model
```

This works because the requests are short: 4 sequences' worth of KV blocks holds far more than 4 short
requests. Caveats: it does **not** guarantee 32-way at long context (you still only fit ~4 long ones,
the rest defer); the pool is static; and it's a tuning dial (over-reserve → packs poorly, under-reserve →
queuing) — but never an OOM (the util cap holds). Leave it unset for the default (= `max_num_seqs`).

#### Multiple profiles of one model

Model identity is the **config name**, not the repo, so you can register one repo under several names
with different launch profiles and have your application route to the right one — e.g. a
short-context/high-concurrency profile for burst work and a long-context profile for big jobs (you can't
change `max_model_len` per request, so this genuinely needs two launches):

```yaml
coder-short: { repo: Qwen/Qwen2.5-Coder-7B-Instruct, max_model_len: 8192,  max_num_seqs: 16, kv_reservation_seqs: 4 }
coder-long:  { repo: Qwen/Qwen2.5-Coder-7B-Instruct, max_model_len: 32768, max_num_seqs: 1 }
```

Each name is an independent model (own container, footprint, queue). **Caveat:** if both are resident at
once the weights occupy VRAM **twice**; otherwise the gateway swaps between them (a reload on switch). Best
when used in batches rather than interleaved request-by-request.

### `whole_card`

The legacy behavior: each model fills its card at its own `gpu_memory_utilization`, and the gateway
swaps models via LRU eviction when a different model is requested. `colocate: true` lets small models
share a card here. No `max_model_len` requirement. Set `PLACEMENT_MODE=whole_card` to keep this.

> **Upgrade note:** the default changed to `budget`. A deployment using the legacy
> `ALLOWED_MODELS_JSON` (no per-model `max_model_len`) will now refuse to start until you either set
> `max_model_len` per model (via `models.yaml`) or set `PLACEMENT_MODE=whole_card`.

## Configuration

All configuration is done via environment variables, making it easy to deploy with Portainer, Kubernetes, or directly from GitHub.

### Using .env File

For easier management, create a `.env` file in your project root:

```bash
# Copy the example file
cp .env.example .env

# Edit with your values
nano .env
```

The `.env` file will be automatically loaded by Docker Compose.

### Environment Variables Reference

#### Required Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `HUGGING_FACE_HUB_TOKEN` | - | **Required** for gated models. Get from https://huggingface.co/settings/tokens |

#### vLLM Engine Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `VLLM_IMAGE` | `vllm/vllm-openai:v0.10.2` | Docker image for vLLM model containers |
| `VLLM_GPU_MEMORY_UTILIZATION` | `0.90` | GPU memory utilization (0.0 to 1.0) |
| `VLLM_MAX_MODEL_LEN_GLOBAL` | `0` | Global max context length (0 = use model default) |
| `VLLM_MAX_NUM_SEQS` | `16` | Maximum concurrent sequences |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs to split model across |
| `VLLM_PORT` | `8000` | Internal port used by vLLM containers |

#### Placement Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `PLACEMENT_MODE` | `budget` | `budget` (pack many models per card under a cap) or `whole_card` (legacy LRU swap). See [Placement modes](#placement-modes). |
| `GPU_BUDGET_FRACTION` | `VLLM_GPU_MEMORY_UTILIZATION` | Per-GPU total VRAM cap (fraction) the gateway may fill across all its models in `budget` mode |
| `BUDGET_OVERHEAD_FACTOR` | `1.1` | Multiplier on (weights + KV) when estimating a model's need |
| `BUDGET_OVERHEAD_MIB` | `1024` | Fixed per-card margin (MiB) added to each model's need estimate |

#### Networking Settings

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCKER_NETWORK_NAME` | `vllm_network` | Docker network name for vLLM containers |
| `GATEWAY_CONTAINER_NAME` | `vllm_gateway` | Gateway container name (must match compose) |
| `VLLM_CONTAINER_PREFIX` | `vllm_server` | Prefix for vLLM model server containers |
| `VLLM_INACTIVITY_TIMEOUT` | `1800` | Seconds before idle containers shutdown (0 = disabled) |

#### Path Settings (Host Machine)

| Variable | Default | Description |
|----------|---------|-------------|
| `HOST_CACHE_DIR` | `/root/.cache/huggingface` | HuggingFace cache directory on host |
| `HOST_DATA_DIR` | `./data` | Data directory on host (app creates files inside) |
| `HOST_TEMP_DIR` | `/tmp` | Temporary directory on host |

#### Path Settings (Container)

| Variable | Default | Description |
|----------|---------|-------------|
| `MEMORY_FOOTPRINT_FILE` | `/app/data/memory_footprints.json` | Memory footprints file inside gateway container |
| `VLLM_TEMP_DIR` | `/tmp` | Temporary directory inside vLLM containers |

#### Docker Images

| Variable | Default | Description |
|----------|---------|-------------|
| `NVIDIA_UTILITY_IMAGE` | `nvidia/cuda:12.1.0-base-ubuntu22.04` | NVIDIA utility container for GPU queries |

#### Logging

| Variable | Default | Description |
|----------|---------|-------------|
| `LOG_LEVEL` | `INFO` | Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL) |

#### Model Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `ALLOWED_MODELS_JSON` | See docker-compose.yml | JSON mapping of model names to HuggingFace IDs |

**Example ALLOWED_MODELS_JSON:**

For `.env` file (single line):
```bash
ALLOWED_MODELS_JSON={"gemma3-4B":"google/gemma-3-4b-it","llama3-8B":"meta-llama/Meta-Llama-3-8B-Instruct","qwen2.5":"Qwen/Qwen2.5-Coder-7B-Instruct"}
```

For Portainer or docker-compose.yml (multi-line):
```json
{
  "gemma3-4B": "google/gemma-3-4b-it",
  "llama3-8B": "meta-llama/Meta-Llama-3-8B-Instruct",
  "qwen2.5": "Qwen/Qwen2.5-Coder-7B-Instruct",
  "tinyllama-gguf": "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
}
```

The docker-compose.yml includes a default list of models that will be used if you don't override this variable.

### Deployment with Portainer

When deploying from GitHub using Portainer:

1. **Stack Configuration**: In Portainer, create a new stack from the repository URL

2. **Environment Variables**: Use the Portainer UI to set environment variables

3. **Data Directory**: The app uses a `./data` directory for persistent files
   - Portainer will create this automatically when the stack starts
   - No need to pre-create files - the app handles it

4. **Optional - Custom Paths**: Override default paths if needed:
   ```bash
   HOST_DATA_DIR=/mnt/data/vllm-gateway
   HOST_CACHE_DIR=/mnt/data/huggingface
   ```

5. **Permissions**: Portainer must have access to `/var/run/docker.sock`

**Example Portainer Environment Variables:**

```bash
HUGGING_FACE_HUB_TOKEN=hf_xxxxxxxxxxxxx
HOST_CACHE_DIR=/mnt/data/huggingface
HOST_DATA_DIR=/mnt/data/vllm-gateway
LOG_LEVEL=INFO
VLLM_INACTIVITY_TIMEOUT=3600

# Override models (uses default from docker-compose.yml if not set)
ALLOWED_MODELS_JSON={"my-model":"org/model-repo","another-model":"org/another-repo"}
```

**Note:** In Portainer's web UI, you can enter `ALLOWED_MODELS_JSON` in multi-line format for better readability.

## Troubleshooting

### memory_footprints.json created as directory (legacy issue - now fixed)

**Note:** As of the latest version, this issue is **resolved** by using directory mounts instead of file mounts.

If you're running an older deployment where `memory_footprints.json` was created as a directory:

**Fix:**

```bash
# SSH into your server
ssh user@your-server

# Stop the stack in Portainer
# Then fix the directory issue
cd /path/to/your/deployment
rm -rf memory_footprints.json

# Update your docker-compose.yml to use the new directory mount:
# - ${HOST_DATA_DIR:-./data}:/app/data

# Set environment variable:
# MEMORY_FOOTPRINT_FILE=/app/data/memory_footprints.json

# Start the stack again in Portainer
```

The application will now create the file automatically inside the mounted data directory.

## API Usage

### Making an API Call

Send requests to the gateway as you would to the OpenAI API. The `model` parameter must be one of the keys defined in `ALLOWED_MODELS_JSON`.

> **Note on Model Loading:** The first time you request a new model, there will be a delay as the gateway downloads the model and starts the container. If it's a "discovery run," other models will be temporarily stopped.

**Example using `curl`:**

```bash
curl http://<your-server-ip>:9003/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma3-4B",
    "messages": [
      {
        "role": "user",
        "content": "What are the top 3 benefits of using Docker?"
      }
    ]
  }'
```

### Gateway Endpoints

- **`GET /v1/models`**: Lists the models allowed by the gateway (as defined in `ALLOWED_MODELS_JSON`).
- **`GET /gateway/status`**: Returns a JSON object with the current status of the gateway, including total VRAM, known footprints, and a list of active model containers.
