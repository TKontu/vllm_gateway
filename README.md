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
      # vLLM CPU swap space in GB.
      VLLM_SWAP_SPACE: "16"

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
| `VLLM_SWAP_SPACE` | `16` | CPU swap space in GB for offloading |
| `VLLM_MAX_MODEL_LEN_GLOBAL` | `0` | Global max context length (0 = use model default) |
| `VLLM_MAX_NUM_SEQS` | `16` | Maximum concurrent sequences |
| `VLLM_TENSOR_PARALLEL_SIZE` | `1` | Number of GPUs to split model across |
| `VLLM_PORT` | `8000` | Internal port used by vLLM containers |

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
