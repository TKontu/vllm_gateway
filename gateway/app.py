import os
import asyncio
import httpx
import docker
import json
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from docker.types import DeviceRequest
from docker.errors import NotFound, APIError

# --- Configuration ---
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
# The vLLM container will be discoverable by its name on the shared Docker network
VLLM_HOST = os.getenv("VLLM_HOST", "vllm_server") 
VLLM_PORT = 8001
VLLM_CONTAINER_NAME = "vllm_server"
VLLM_IMAGE = "vllm/vllm-openai:latest"
VLLM_GPU_MEMORY_UTILIZATION = os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")
# The name of the shared network, configured in docker-compose.yml
DOCKER_NETWORK_NAME = os.getenv("DOCKER_NETWORK_NAME", "vllm_network")

def load_allowed_models():
    """Load allowed models from environment variable or use a default."""
    models_json = os.getenv("ALLOWED_MODELS_JSON")
    if models_json:
        try:
            return json.loads(models_json)
        except json.JSONDecodeError:
            print("Warning: Invalid JSON in ALLOWED_MODELS_JSON. Using default models.")
    
    # Default models if environment variable is not set or invalid
    return {
        "gemma3": "google/gemma-3-8b-it",
        "openai/gpt-oss-20b": "openai/gpt-oss-20b",
    }

ALLOWED_MODELS = load_allowed_models()

# --- Docker and HTTP Clients ---
docker_client = docker.from_env()
http_client = httpx.AsyncClient()
model_management_lock = asyncio.Lock()

app = FastAPI()

# --- Helper Functions ---

def get_current_model_id():
    """Check the running container to see which model is loaded."""
    try:
        container = docker_client.containers.get(VLLM_CONTAINER_NAME)
        args = container.attrs['Args']
        try:
            # The model id is the argument that follows the --model flag.
            model_flag_index = args.index('--model')
            return args[model_flag_index + 1]
        except (ValueError, IndexError):
            # This handles cases where --model is not in the args or is the last arg.
            print("Warning: Could not determine model from running container args.")
            return None
    except NotFound:
        return None

async def start_model(model_id: str):
    """
    Stops the current vLLM container (if any) and starts a new one on the shared network.
    """
    print(f"Attempting to switch model to: {model_id}")

    try:
        container = docker_client.containers.get(VLLM_CONTAINER_NAME)
        print(f"Stopping container {VLLM_CONTAINER_NAME}...")
        container.stop()
        container.remove()
        print("Container stopped and removed.")
    except NotFound:
        pass
    except APIError as e:
        print(f"Error stopping or removing container: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop existing model container: {e}")

    try:
        print(f"Starting new container with model: {model_id} on network {DOCKER_NETWORK_NAME}")
        docker_client.containers.run(
            VLLM_IMAGE,
            command=[
            "--model", model_id,
            "--gpu-memory-utilization", VLLM_GPU_MEMORY_UTILIZATION
        ],
            name=VLLM_CONTAINER_NAME,
            hostname=VLLM_CONTAINER_NAME,
            detach=True,
            # No need to publish ports to the host, as communication is internal
            # ports={f"{VLLM_PORT}/tcp": VLLM_PORT}, 
            network=DOCKER_NETWORK_NAME, # Attach to the shared network
            environment={"HUGGING_FACE_HUB_TOKEN": HF_TOKEN},
            ipc_mode="host",
            device_requests=[DeviceRequest(count=-1, capabilities=[['gpu']])],
            volumes={
                os.path.expanduser("~/.cache/huggingface"): {'bind': '/root/.cache/huggingface', 'mode': 'rw'}
            }
        )
    except APIError as e:
        print(f"Error starting container: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start model container: {e}")

    vllm_base_url = f"http://{VLLM_HOST}:{VLLM_PORT}"
    for i in range(60):
        try:
            await asyncio.sleep(2)
            response = await http_client.get(f"{vllm_base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                print(f"Model {model_id} started successfully.")
                return
        except httpx.RequestError:
            if i > 5:
                 print(f"Waiting for model to be ready... (attempt {i+1})")
    
    raise HTTPException(status_code=500, detail="Model failed to start in the allocated time.")

# --- API Endpoints ---

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    body = await request.json()
    model_name = body.get("model")

    if not model_name or model_name not in ALLOWED_MODELS:
        return JSONResponse({"error": f"Model not allowed. Please choose from: {list(ALLOWED_MODELS.keys())}"}, status_code=400)

    target_model_id = ALLOWED_MODELS[model_name]

    # Acquire a lock to ensure that only one request can manage the model container at a time.
    # This prevents race conditions if multiple requests for different models arrive simultaneously.
    async with model_management_lock:
        current_model_id = get_current_model_id()
        if target_model_id != current_model_id:
            # If the requested model is not the one currently running, start the correct one.
            # The lock is held until the new model is started and ready.
            await start_model(target_model_id)

    # Once the correct model is running and the lock is released, forward the request.

    try:
        vllm_url = f"http://{VLLM_HOST}:{VLLM_PORT}/v1/chat/completions"
        response = await http_client.post(vllm_url, json=body, timeout=300)
        response.raise_for_status()
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.HTTPStatusError as e:
        return JSONResponse({"error": "Error from vLLM service", "details": str(e)}, status_code=e.response.status_code)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to vLLM service: {e}")

@app.get("/v1/models")
def list_models():
    return {"allowed_models": list(ALLOWED_MODELS.keys())}
