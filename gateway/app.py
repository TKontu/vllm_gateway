import os
import asyncio
import httpx
import docker
import json
import time
import logging
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from docker.types import DeviceRequest
from docker.errors import NotFound, APIError
from starlette.middleware.base import BaseHTTPMiddleware

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
VLLM_HOST = os.getenv("VLLM_HOST", "vllm_server") 
VLLM_PORT = 8000
VLLM_CONTAINER_NAME = "vllm_server"
VLLM_IMAGE = "vllm/vllm-openai:latest"
VLLM_GPU_MEMORY_UTILIZATION = os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.85")
VLLM_SWAP_SPACE = os.getenv("VLLM_SWAP_SPACE", "16") # Default to 16 GiB
VLLM_MAX_MODEL_LEN_GLOBAL = int(os.getenv("VLLM_MAX_MODEL_LEN_GLOBAL", "0")) # Optional: A global cap.
# The base name of the shared network, as defined in the compose file.
DOCKER_NETWORK_NAME = os.getenv("DOCKER_NETWORK_NAME", "vllm_network")
# The gateway's own container name, used for self-inspection to find the real network name.
GATEWAY_CONTAINER_NAME = os.getenv("GATEWAY_CONTAINER_NAME", "vllm_gateway")
VLLM_INACTIVITY_TIMEOUT = int(os.getenv("VLLM_INACTIVITY_TIMEOUT", 1800))

# This will be populated at startup with the real, stack-prefixed network name.
RESOLVED_DOCKER_NETWORK = None

def load_allowed_models():
    """Load allowed models from environment variable or use a default."""
    models_json = os.getenv("ALLOWED_MODELS_JSON")
    if models_json:
        try:
            return json.loads(models_json)
        except json.JSONDecodeError:
            logging.warning("Invalid JSON in ALLOWED_MODELS_JSON. Using default models.")
    
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

# --- Logging Middleware ---
class RequestLoggingMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next):
        logging.info(f"Request received: {request.method} {request.url.path} from {request.client.host}")
        response = await call_next(request)
        logging.info(f"Response sent: {response.status_code} for {request.method} {request.url.path}")
        return response

app.add_middleware(RequestLoggingMiddleware)

# --- Graceful Shutdown Logic ---
last_request_time = None

async def shutdown_if_inactive():
    """A background task that checks for inactivity and shuts down the vLLM container."""
    global last_request_time
    logging.info(f"Starting inactivity monitor with a {VLLM_INACTIVITY_TIMEOUT}s timeout.")
    while True:
        await asyncio.sleep(60)
        
        if last_request_time is not None and VLLM_INACTIVITY_TIMEOUT > 0:
            idle_time = time.time() - last_request_time
            if idle_time > VLLM_INACTIVITY_TIMEOUT:
                logging.info(f"Container has been idle for {idle_time:.0f}s. Shutting down.")
                async with model_management_lock:
                    try:
                        container = docker_client.containers.get(VLLM_CONTAINER_NAME)
                        logging.info(f"Stopping container {VLLM_CONTAINER_NAME} due to inactivity...")
                        container.stop()
                        container.remove()
                        logging.info("Container stopped and removed.")
                    except NotFound:
                        logging.info("Inactivity shutdown triggered, but container was not found.")
                    except APIError as e:
                        logging.error(f"Error stopping inactive container: {e}")
                    finally:
                        last_request_time = None

@app.on_event("startup")
async def startup_event():
    """
    On application startup:
    1. Resolve the real Docker network name.
    2. Launch the background inactivity monitor.
    """
    global RESOLVED_DOCKER_NETWORK
    try:
        logging.info(f"Attempting to find gateway container with name: {GATEWAY_CONTAINER_NAME}")
        gateway_container = docker_client.containers.get(GATEWAY_CONTAINER_NAME)
        networks = gateway_container.attrs['NetworkSettings']['Networks']
        logging.info(f"Gateway is in networks: {list(networks.keys())}. Searching for network ending with '{DOCKER_NETWORK_NAME}'.")
        
        for network_name in networks:
            if network_name.endswith(DOCKER_NETWORK_NAME):
                RESOLVED_DOCKER_NETWORK = network_name
                logging.info(f"Successfully resolved Docker network to: {RESOLVED_DOCKER_NETWORK}")
                break
        
        if not RESOLVED_DOCKER_NETWORK:
            logging.error(f"Could not find a network ending with '{DOCKER_NETWORK_NAME}' for the gateway container.")
            # In a real-world scenario, you might want to exit here.
            # For now, we will fall back to the base name.
            RESOLVED_DOCKER_NETWORK = DOCKER_NETWORK_NAME

    except NotFound:
        logging.error(f"Could not find gateway container named '{GATEWAY_CONTAINER_NAME}'. Network resolution failed. Falling back to '{DOCKER_NETWORK_NAME}'.")
        RESOLVED_DOCKER_NETWORK = DOCKER_NETWORK_NAME
    except Exception as e:
        logging.error(f"An unexpected error occurred during network resolution: {e}")
        RESOLVED_DOCKER_NETWORK = DOCKER_NETWORK_NAME

    # Launch the background inactivity monitor
    asyncio.create_task(shutdown_if_inactive())

# --- Helper Functions ---

def get_current_model_id():
    """Check the running container to see which model is loaded."""
    try:
        container = docker_client.containers.get(VLLM_CONTAINER_NAME)
        args = container.attrs['Args']
        try:
            model_flag_index = args.index('--model')
            return args[model_flag_index + 1]
        except (ValueError, IndexError):
            logging.warning("Could not determine model from running container args.")
            return None
    except NotFound:
        return None

async def get_model_max_len(model_id: str) -> int:
    """Fetches the model's config.json from Hugging Face to find its max length."""
    config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(config_url, follow_redirects=True)
            response.raise_for_status()
            config = response.json()
            
            # Common keys for max model length
            keys_to_check = ['max_position_embeddings', 'n_positions', 'model_max_length']
            for key in keys_to_check:
                if key in config and isinstance(config[key], int):
                    model_max_len = config[key]
                    logging.info(f"Found '{key}' in config for {model_id}: {model_max_len}")
                    return model_max_len
            
            logging.warning(f"Could not find a max length key in config.json for {model_id}.")
            return 0 # Return 0 if no key is found
    except httpx.HTTPStatusError as e:
        logging.error(f"Failed to fetch config for {model_id}. Status code: {e.response.status_code}")
        return 0
    except Exception as e:
        logging.error(f"An error occurred while getting max length for {model_id}: {e}")
        return 0

async def start_model(model_id: str):
    """
    Stops the current vLLM container (if any) and starts a new one with a
    dynamically determined max_model_len.
    """
    logging.info(f"Attempting to switch model to: {model_id}")

    try:
        container = docker_client.containers.get(VLLM_CONTAINER_NAME)
        logging.info(f"Stopping container {VLLM_CONTAINER_NAME}...")
        container.stop()
        container.remove()
        logging.info("Container stopped and removed.")
    except NotFound:
        pass
    except APIError as e:
        logging.error(f"Error stopping or removing container: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to stop existing model container: {e}")

    try:
        # Dynamically determine the max_model_len for the new model
        model_max_len = await get_model_max_len(model_id)
        
        # Use the smaller of the model's max length and the global cap (if set)
        final_max_len = model_max_len
        if VLLM_MAX_MODEL_LEN_GLOBAL > 0:
            if model_max_len > 0:
                final_max_len = min(model_max_len, VLLM_MAX_MODEL_LEN_GLOBAL)
                logging.info(f"Using smaller of model max ({model_max_len}) and global max ({VLLM_MAX_MODEL_LEN_GLOBAL}): {final_max_len}")
            else:
                final_max_len = VLLM_MAX_MODEL_LEN_GLOBAL
                logging.info(f"Model max length not found, using global max: {final_max_len}")

        # Dynamically build the command for the vLLM container.
        command = [
            "--model", model_id,
            "--gpu-memory-utilization", VLLM_GPU_MEMORY_UTILIZATION
        ]
        if int(VLLM_SWAP_SPACE) > 0:
            command.extend(["--swap-space", VLLM_SWAP_SPACE])
        if final_max_len > 0:
            command.extend(["--max-model-len", str(final_max_len)])

        logging.info(f"Starting new container with command: {' '.join(command)}")
        
        new_container = docker_client.containers.run(
            VLLM_IMAGE,
            command=command,
            name=VLLM_CONTAINER_NAME,
            hostname=VLLM_CONTAINER_NAME,
            detach=True,
            network=RESOLVED_DOCKER_NETWORK,
            environment={"HUGGING_FACE_HUB_TOKEN": HF_TOKEN},
            ipc_mode="host",
            device_requests=[DeviceRequest(count=-1, capabilities=[['gpu']])],
            volumes={
                os.path.expanduser("~/.cache/huggingface"): {'bind': '/root/.cache/huggingface', 'mode': 'rw'}
            }
        )
        
        new_container.reload()
        vllm_ip_address = new_container.attrs['NetworkSettings']['Networks'][RESOLVED_DOCKER_NETWORK]['IPAddress']
        logging.info(f"Container started with IP: {vllm_ip_address}")

    except APIError as e:
        logging.error(f"Error starting container: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start model container: {e}")

    vllm_base_url = f"http://{vllm_ip_address}:{VLLM_PORT}"
    for i in range(60):
        try:
            await asyncio.sleep(2)
            response = await http_client.get(f"{vllm_base_url}/v1/models", timeout=2)
            if response.status_code == 200:
                logging.info(f"Model {model_id} started successfully at {vllm_base_url}.")
                return
        except httpx.RequestError:
            if i > 5:
                logging.info(f"Waiting for model to be ready... (attempt {i+1})")
    
    raise HTTPException(status_code=500, detail="Model failed to start in the allocated time.")

# --- API Endpoints ---

async def proxy_request(request: Request, target_path: str):
    """
    A generic proxy handler that manages model switching and forwards requests.
    """
    global last_request_time
    body = await request.json()
    model_name = body.get("model")

    if not model_name or model_name not in ALLOWED_MODELS:
        return JSONResponse({"error": f"Model not allowed. Please choose from: {list(ALLOWED_MODELS.keys())}"}, status_code=400)

    target_model_id = ALLOWED_MODELS[model_name]

    async with model_management_lock:
        last_request_time = time.time()
        current_model_id = get_current_model_id()
        if target_model_id != current_model_id:
            await start_model(target_model_id)

    try:
        # Before forwarding, replace the user-friendly model name
        # with the actual model ID that the vLLM server was started with.
        body['model'] = target_model_id
        
        vllm_url = f"http://{VLLM_HOST}:{VLLM_PORT}{target_path}"
        response = await http_client.post(vllm_url, json=body, timeout=300)
        response.raise_for_status()
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.HTTPStatusError as e:
        return JSONResponse({"error": "Error from vLLM service", "details": str(e)}, status_code=e.response.status_code)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to vLLM service: {e}")

@app.post("/v1/chat/completions")
async def chat_completions(request: Request):
    return await proxy_request(request, "/v1/chat/completions")

@app.post("/v1/embeddings")
async def embeddings(request: Request):
    return await proxy_request(request, "/v1/embeddings")

@app.get("/v1/models")
def list_models():
    return {"allowed_models": list(ALLOWED_MODELS.keys())}
