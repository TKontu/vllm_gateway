# vLLM Dynamic Gateway

This project provides a smart gateway that acts as a dynamic proxy for the [vLLM inference engine](https://github.com/vllm-project/vllm). Its primary goal is to enable the seamless, local serving of multiple large language models on a single GPU with **zero-effort model switching**.

When an API request for a specific model is received, the gateway automatically handles the backend logistics: if the requested model isn't already loaded, it will gracefully stop the current vLLM container and launch a new one with the correct model, all without any manual intervention.

## How It Works

The architecture consists of a single, persistent `gateway` container that has access to the host's Docker socket.

1.  An API request is sent to the gateway's `/v1/chat/completions` endpoint, specifying a model (e.g., `"model": "gemma3-4B"`).
2.  The gateway checks if a `vllm_server` container is running and if it's loaded with the requested model.
3.  **If the correct model is already running**, the request is simply forwarded.
4.  **If a different model is running (or no model is running)**, the gateway stops and removes the old `vllm_server` container. It then launches a new `vllm_server` container, instructing it to load the newly requested model.
5.  Once the new model is loaded and ready, the original request is forwarded.

This entire process is transparent to the end-user, who simply experiences a slightly longer response time for the first request to a new model.

## Key Features

-   **Dynamic Model Switching:** Automatically manages and swaps vLLM containers based on incoming API requests.
-   **Efficient Resource Management:** Only one resource-intensive vLLM model occupies GPU VRAM at any given time.
-   **Standard OpenAI API:** Uses the familiar `/v1/chat/completions` endpoint, making it a drop-in replacement for many applications.
-   **Concurrent Request Safety:** Uses an `asyncio.Lock` to ensure that simultaneous requests for different models are handled safely and queud, preventing race conditions.
-   **Highly Configurable:** Control allowed models, GPU memory utilization, and networking via environment variables.

## Deployment (Portainer / Docker Compose)

This gateway is designed to be deployed as a standalone service that dynamically manages other containers. Below is a sample `docker-compose.yml` for deploying it in Portainer.

```yaml
version: "3.9"

# Define the shared network
networks:
  vllm_network:
    driver: bridge

services:
  gateway:
    # Build the image locally or pull from a registry
    image: my-vlmm-gateway:latest
    container_name: vllm_gateway
    ports:
      - "9000:9000" # Expose the gateway on port 9000
    environment:
      # --- REQUIRED ---
      # Your Hugging Face token for accessing gated models
      HUGGING_FACE_HUB_TOKEN: "hf_..."

      # --- NETWORKING ---
      # The network to attach the vLLM container to. Must match the network defined above.
      DOCKER_NETWORK_NAME: "vllm_network"
      # The hostname for the vLLM container.
      VLLM_HOST: "vllm_server"

      # --- CONFIGURATION ---
      # A JSON string defining the user-friendly names and their corresponding Hugging Face model IDs.
      ALLOWED_MODELS_JSON: '{"gemma3-4B":"google/gemma-3-4b-it", "qwen2.5":"Qwen/Qwen2.5-Coder-7B-Instruct"}'
      
      # The percentage of GPU memory vLLM should be allowed to use (e.g., "0.85" for 85%).
      VLLM_GPU_MEMORY_UTILIZATION: "0.85"

    volumes:
      # Mount the Docker socket to allow the gateway to manage containers
      - /var/run/docker.sock:/var/run/docker.sock
      # Mount the Hugging Face cache to avoid re-downloading models
      - /root/.cache/huggingface:/root/.cache/huggingface
    
    # Attach the gateway to the shared network
    networks:
      - vllm_network
    restart: unless-stopped
```

### Building the Gateway Image

To build the `my-vlmm-gateway:latest` image, navigate to the `gateway` directory and run:

```bash
docker build -t my-vlmm-gateway:latest .
```

## Making an API Call

Once the gateway is running, you can send requests to it as you would to the OpenAI API. The `model` parameter should be one of the keys you defined in the `ALLOWED_MODELS_JSON` environment variable.

**Example using PowerShell:**

```powershell
$headers = @{
    "Content-Type" = "application/json"
}

$body = @{
    model    = "gemma3-4B" # This must match a key in ALLOWED_MODELS_JSON
    messages = @(
        @{ 
            role    = "user"
            content = "What are the top 3 benefits of using Docker?"
        }
    )
} | ConvertTo-Json

Invoke-WebRequest -Uri http://<your-server-ip>:9000/v1/chat/completions -Method POST -Headers $headers -Body $body
```

**Example using `curl`:**

```bash
curl http://<your-server-ip>:9000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d 
  {
    "model": "gemma3-4B",
    "messages": [
      {
        "role": "user",
        "content": "What are the top 3 benefits of using Docker?"
      }
    ]
  }
```
