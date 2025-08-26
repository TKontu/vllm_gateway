import os
import asyncio
import httpx
import docker
import json
import time
import logging
import subprocess
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from docker.types import DeviceRequest
from docker.errors import NotFound, APIError
from starlette.middleware.base import BaseHTTPMiddleware
from dataclasses import dataclass, field

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
VLLM_PORT = 8000
VLLM_IMAGE = "vllm/vllm-openai:latest"
VLLM_GPU_MEMORY_UTILIZATION = os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
VLLM_SWAP_SPACE = os.getenv("VLLM_SWAP_SPACE", "16")
VLLM_MAX_MODEL_LEN_GLOBAL = int(os.getenv("VLLM_MAX_MODEL_LEN_GLOBAL", "0"))
DOCKER_NETWORK_NAME = os.getenv("DOCKER_NETWORK_NAME", "vllm_network")
GATEWAY_CONTAINER_NAME = os.getenv("GATEWAY_CONTAINER_NAME", "vllm_gateway")
VLLM_INACTIVITY_TIMEOUT = int(os.getenv("VLLM_INACTIVITY_TIMEOUT", 1800))
VLLM_CONTAINER_PREFIX = "vllm_server"
NVIDIA_UTILITY_IMAGE = "nvidia/cuda:12.1.0-base-ubuntu22.04"
MEMORY_FOOTPRINT_FILE = "/app/memory_footprints.json"

# --- Global State ---
RESOLVED_DOCKER_NETWORK = None
TOTAL_GPU_VRAM = 0  # in MiB
known_footprints = {}
active_containers = {}
model_management_lock = asyncio.Lock()

@dataclass
class ContainerState:
    model_id: str
    container_name: str
    ip_address: str
    port: int
    last_request_time: float
    vram_footprint: float # in MiB

# --- Docker and HTTP Clients ---
docker_client = docker.from_env()
http_client = httpx.AsyncClient()
app = FastAPI()

# --- Helper Functions ---

def run_nvidia_smi_in_container(command: list[str]) -> str:
    """Runs an nvidia-smi command in a temporary container and returns the output."""
    try:
        smi_output = docker_client.containers.run(
            NVIDIA_UTILITY_IMAGE,
            command=command,
            remove=True,
            device_requests=[DeviceRequest(count=-1, capabilities=[['gpu']])]
        )
        return smi_output.decode('utf-8').strip()
    except APIError as e:
        logging.error(f"Error running nvidia-smi container: {e}")
        return ""

def get_total_vram():
    """Gets total GPU VRAM in MiB."""
    global TOTAL_GPU_VRAM
    logging.info("Getting total GPU VRAM...")
    output = run_nvidia_smi_in_container(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    if output and output.isdigit():
        TOTAL_GPU_VRAM = int(output)
        logging.info(f"Total GPU VRAM: {TOTAL_GPU_VRAM} MiB")
    else:
        logging.error("Could not determine total GPU VRAM. Disabling dynamic memory management.")
        TOTAL_GPU_VRAM = 0

def load_known_footprints():
    """Loads the known model memory footprints from the JSON file."""
    global known_footprints
    try:
        if os.path.exists(MEMORY_FOOTPRINT_FILE):
            with open(MEMORY_FOOTPRINT_FILE, 'r') as f:
                known_footprints = json.load(f)
            logging.info(f"Loaded known model footprints: {known_footprints}")
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Could not load memory footprints file: {e}")
        known_footprints = {}

def save_known_footprints():
    """Saves the known footprints back to the JSON file."""
    try:
        with open(MEMORY_FOOTPRINT_FILE, 'w') as f:
            json.dump(known_footprints, f, indent=4)
    except IOError as e:
        logging.error(f"Could not save memory footprints file: {e}")

def get_used_vram() -> float:
    """Gets currently used GPU VRAM in MiB."""
    output = run_nvidia_smi_in_container(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"])
    if output and output.isdigit():
        return float(output)
    else:
        logging.error("Could not determine used GPU VRAM.")
        return 0.0

def load_allowed_models():
    """Load allowed models from environment variable."""
    models_json = os.getenv("ALLOWED_MODELS_JSON")
    if models_json:
        try:
            return json.loads(models_json)
        except json.JSONDecodeError:
            logging.warning("Invalid JSON in ALLOWED_MODELS_JSON. Using default empty set.")
    return {}

ALLOWED_MODELS = load_allowed_models()

# --- Startup and Shutdown Events ---

@app.on_event("startup")
async def startup_event():
    global RESOLVED_DOCKER_NETWORK
    try:
        gateway_container = docker_client.containers.get(GATEWAY_CONTAINER_NAME)
        networks = gateway_container.attrs['NetworkSettings']['Networks']
        for network_name in networks:
            if network_name.endswith(DOCKER_NETWORK_NAME):
                RESOLVED_DOCKER_NETWORK = network_name
                logging.info(f"Successfully resolved Docker network to: {RESOLVED_DOCKER_NETWORK}")
                break
        if not RESOLVED_DOCKER_NETWORK:
            RESOLVED_DOCKER_NETWORK = DOCKER_NETWORK_NAME
            logging.warning(f"Could not find network ending in '{DOCKER_NETWORK_NAME}'. Falling back to base name.")
    except NotFound:
        RESOLVED_DOCKER_NETWORK = DOCKER_NETWORK_NAME
        logging.error(f"Gateway container '{GATEWAY_CONTAINER_NAME}' not found. Falling back to network '{DOCKER_NETWORK_NAME}'.")

    get_total_vram()
    load_known_footprints()
    asyncio.create_task(shutdown_inactive_containers())

async def shutdown_inactive_containers():
    """A background task that checks for inactivity and shuts down containers."""
    logging.info(f"Starting inactivity monitor with a {VLLM_INACTIVITY_TIMEOUT}s timeout.")
    while True:
        await asyncio.sleep(60)
        if VLLM_INACTIVITY_TIMEOUT <= 0:
            continue

        async with model_management_lock:
            current_time = time.time()
            inactive_containers = []
            for name, state in active_containers.items():
                if current_time - state.last_request_time > VLLM_INACTIVITY_TIMEOUT:
                    inactive_containers.append(name)
            
            for name in inactive_containers:
                logging.info(f"Container {name} has been idle. Shutting down.")
                await stop_container(name)

async def stop_container(container_name: str):
    """Stops and removes a container and updates the active_containers state."""
    try:
        container = docker_client.containers.get(container_name)
        logging.info(f"Stopping container {container_name}...")
        container.stop()
        container.remove()
        logging.info(f"Container {container_name} stopped and removed.")
    except NotFound:
        logging.warning(f"Attempted to stop container {container_name}, but it was not found.")
    except APIError as e:
        logging.error(f"Error stopping or removing container {container_name}: {e}")
    
    if container_name in active_containers:
        del active_containers[container_name]

async def get_model_max_len(model_id: str) -> int:
    """Fetches the model's config.json from Hugging Face to find its max length."""
    config_url = f"https://huggingface.co/{model_id}/raw/main/config.json"
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(config_url, follow_redirects=True)
            response.raise_for_status()
            config = response.json()
            keys_to_check = ['max_position_embeddings', 'n_positions', 'model_max_length']
            for key in keys_to_check:
                if key in config and isinstance(config[key], int):
                    return config[key]
            return 0
    except Exception as e:
        logging.error(f"An error occurred while getting max length for {model_id}: {e}")
        return 0

async def start_model_container(model_id: str, container_name: str) -> ContainerState | None:
    """Starts a new vLLM container."""
    logging.info(f"Attempting to start model {model_id} in container {container_name}")

    model_max_len = await get_model_max_len(model_id)
    final_max_len = model_max_len
    if VLLM_MAX_MODEL_LEN_GLOBAL > 0:
        final_max_len = min(model_max_len, VLLM_MAX_MODEL_LEN_GLOBAL) if model_max_len > 0 else VLLM_MAX_MODEL_LEN_GLOBAL

    command = ["--model", model_id, "--gpu-memory-utilization", VLLM_GPU_MEMORY_UTILIZATION]
    if int(VLLM_SWAP_SPACE) > 0:
        command.extend(["--swap-space", VLLM_SWAP_SPACE])
    if final_max_len > 0:
        command.extend(["--max-model-len", str(final_max_len)])

    try:
        logging.info(f"Starting container {container_name} with command: {' '.join(command)}")
        new_container = docker_client.containers.run(
            VLLM_IMAGE,
            command=command,
            name=container_name,
            hostname=container_name,
            detach=True,
            network=RESOLVED_DOCKER_NETWORK,
            environment={"HUGGING_FACE_HUB_TOKEN": HF_TOKEN},
            ipc_mode="host",
            device_requests=[DeviceRequest(count=-1, capabilities=[['gpu']])],
            volumes={os.path.expanduser("~/.cache/huggingface"): {'bind': '/root/.cache/huggingface', 'mode': 'rw'}}
        )
        new_container.reload()
        ip_address = new_container.attrs['NetworkSettings']['Networks'][RESOLVED_DOCKER_NETWORK]['IPAddress']
        
        # Health check loop
        vllm_base_url = f"http://{ip_address}:{VLLM_PORT}"
        for i in range(90): # ~3 minutes timeout
            await asyncio.sleep(2)
            try:
                response = await http_client.get(f"{vllm_base_url}/health", timeout=2)
                if response.status_code == 200:
                    logging.info(f"Model {model_id} started successfully at {vllm_base_url}.")
                    return ContainerState(
                        model_id=model_id,
                        container_name=container_name,
                        ip_address=ip_address,
                        port=VLLM_PORT,
                        last_request_time=time.time(),
                        vram_footprint=0 # Footprint is unknown until measured
                    )
            except httpx.RequestError:
                if i > 5:
                    logging.info(f"Waiting for model {model_id} to be ready... (attempt {i+1})")
        
        logging.error(f"Model {model_id} failed to start in the allocated time.")
        await stop_container(container_name)
        return None

    except APIError as e:
        logging.error(f"Error starting container {container_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start model container: {e}")

# --- Main Proxy Logic ---

@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_request(request: Request):
    body = await request.json()
    model_name = body.get("model")

    if not model_name or model_name not in ALLOWED_MODELS:
        return JSONResponse({"error": f"Model not allowed. Please choose from: {list(ALLOWED_MODELS.keys())}"}, status_code=400)
    
    target_model_id = ALLOWED_MODELS[model_name]

    async with model_management_lock:
        # 1. Check if model is already running
        target_container: ContainerState = next((c for c in active_containers.values() if c.model_id == target_model_id), None)

        if not target_container:
            # 2. Model not running, need to start it
            footprint = known_footprints.get(target_model_id)
            
            if footprint is None and TOTAL_GPU_VRAM > 0:
                # DISCOVERY RUN
                logging.info(f"Unknown footprint for {target_model_id}. Starting a discovery run.")
                if active_containers:
                    logging.info("Stopping all active containers for discovery run.")
                    for name in list(active_containers.keys()):
                        await stop_container(name)

                vram_before = get_used_vram()
                logging.info(f"VRAM usage before model load: {vram_before} MiB")
                
                container_name = f"{VLLM_CONTAINER_PREFIX}_0"
                target_container = await start_model_container(target_model_id, container_name)

                if target_container:
                    await asyncio.sleep(15) # Give model time to load
                    vram_after = get_used_vram()
                    logging.info(f"VRAM usage after model load: {vram_after} MiB")
                    
                    measured_vram = vram_after - vram_before
                    
                    if measured_vram > 256: # Sanity check for a reasonable footprint
                        logging.info(f"Measured VRAM footprint for {target_model_id}: {measured_vram} MiB")
                        target_container.vram_footprint = measured_vram
                        known_footprints[target_model_id] = measured_vram
                        save_known_footprints()
                        active_containers[container_name] = target_container
                    else:
                        logging.error(f"Could not measure footprint for {target_model_id} (calculated {measured_vram} MiB). It will not be managed dynamically.")
                        await stop_container(container_name)
                        target_container = None # Ensure it's None so error is raised
            
            elif TOTAL_GPU_VRAM > 0:
                # KNOWN FOOTPRINT RUN
                current_vram_usage = sum(c.vram_footprint for c in active_containers.values())
                if current_vram_usage + footprint > TOTAL_GPU_VRAM:
                    logging.info(f"Not enough VRAM for {target_model_id} (needs {footprint} MiB). Evicting LRU containers.")
                    # Evict LRU containers until there's space
                    sorted_containers = sorted(active_containers.values(), key=lambda c: c.last_request_time)
                    for container_to_evict in sorted_containers:
                        await stop_container(container_to_evict.container_name)
                        current_vram_usage -= container_to_evict.vram_footprint
                        if current_vram_usage + footprint <= TOTAL_GPU_VRAM:
                            break
                
                # Find a free slot index
                slot_indices = {int(name.split('_')[-1]) for name in active_containers.keys()}
                free_slot = next(i for i in range(len(slot_indices) + 1) if i not in slot_indices)
                container_name = f"{VLLM_CONTAINER_PREFIX}_{free_slot}"
                
                target_container = await start_model_container(target_model_id, container_name)
                if target_container:
                    target_container.vram_footprint = footprint
                    active_containers[container_name] = target_container
            
            else: # Fallback if VRAM management is disabled
                if active_containers:
                    lru_container_name = min(active_containers, key=lambda k: active_containers[k].last_request_time)
                    logging.info(f"No dynamic VRAM management. Evicting LRU container {lru_container_name}.")
                    await stop_container(lru_container_name)
                
                container_name = f"{VLLM_CONTAINER_PREFIX}_0"
                target_container = await start_model_container(target_model_id, container_name)
                if target_container:
                    active_containers[container_name] = target_container

        if not target_container:
            raise HTTPException(status_code=500, detail=f"Failed to start or find container for model {target_model_id}")

        # Update last request time and proxy the request
        target_container.last_request_time = time.time()
        vllm_url = f"http://{target_container.ip_address}:{target_container.port}{request.url.path}"
        
    # The lock is released before forwarding the request
    try:
        body['model'] = target_model_id
        response = await http_client.post(vllm_url, json=body, timeout=300)
        response.raise_for_status()
        return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.HTTPStatusError as e:
        return JSONResponse({"error": "Error from vLLM service", "details": str(e)}, status_code=e.response.status_code)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to vLLM service: {e}")

@app.get("/v1/models")
def list_models():
    """Lists the models allowed by the gateway, not the ones currently loaded."""
    return {"data": [{"id": name} for name in ALLOWED_MODELS.keys()], "object": "list"}

@app.get("/gateway/status")
def gateway_status():
    """Returns the current status of the gateway and its managed containers."""
    return {
        "total_gpu_vram_mib": TOTAL_GPU_VRAM,
        "known_footprints_mib": known_footprints,
        "active_containers": {name: state.__dict__ for name, state in active_containers.items()}
    }
