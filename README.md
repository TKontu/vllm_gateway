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
    image: my-vlmm-gateway:latest
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
      # Mount a file to persist learned VRAM footprints. Must be a file, not a directory.
      - ./memory_footprints.json:/app/memory_footprints.json
    networks:
      - vllm_network
    restart: unless-stopped
```

### Building the Gateway Image

To build the `my-vlmm-gateway:latest` image, navigate to the `gateway` directory and run:

```bash
docker build -t my-vlmm-gateway:latest .
```

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
