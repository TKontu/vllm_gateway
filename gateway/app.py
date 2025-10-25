import os
import asyncio
import httpx
import docker
import json
import time
import logging
from contextlib import asynccontextmanager
from functools import partial
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from docker.types import DeviceRequest
from docker.errors import NotFound, APIError
from dataclasses import dataclass
from huggingface_hub import hf_hub_download, list_repo_files

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
HOST_CACHE_DIR = os.getenv("HOST_CACHE_DIR", "/root/.cache/huggingface")
VLLM_PORT = 8000
VLLM_IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-openai:v0.10.2")
VLLM_GPU_MEMORY_UTILIZATION = os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
VLLM_SWAP_SPACE = os.getenv("VLLM_SWAP_SPACE", "16")
VLLM_MAX_MODEL_LEN_GLOBAL = int(os.getenv("VLLM_MAX_MODEL_LEN_GLOBAL", "0"))
VLLM_MAX_NUM_SEQS = os.getenv("VLLM_MAX_NUM_SEQS", "16")
VLLM_TENSOR_PARALLEL_SIZE = os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")
DOCKER_NETWORK_NAME = os.getenv("DOCKER_NETWORK_NAME", "vllm_network")
GATEWAY_CONTAINER_NAME = os.getenv("GATEWAY_CONTAINER_NAME", "vllm_gateway")
VLLM_INACTIVITY_TIMEOUT = int(os.getenv("VLLM_INACTIVITY_TIMEOUT", 1800))
VLLM_CONTAINER_PREFIX = "vllm_server"
NVIDIA_UTILITY_IMAGE = "nvidia/cuda:12.1.0-base-ubuntu22.04"
MEMORY_FOOTPRINT_FILE = "/app/memory_footprints.json"

# --- Model-specific vLLM Images ---
# Placeholder for future per-model image customization
MODEL_IMAGE_MAP = {}

def get_vllm_image_for_model(model_id: str) -> str:
    """Returns the appropriate vLLM image for a given model.
    All models use the same image now (v0.10.2 supports everything)."""
    return MODEL_IMAGE_MAP.get(model_id, VLLM_IMAGE)

def is_gpt_oss_model(model_id: str) -> bool:
    """Check if model is a gpt-oss model that needs special flags."""
    return model_id.startswith("openai/gpt-oss-")

# --- Global State ---
RESOLVED_DOCKER_NETWORK = None
TOTAL_GPU_VRAM = 0  # in MiB
known_footprints = {}
active_containers = {}
model_management_lock = asyncio.Lock()  # For updating active_containers dict
container_start_locks = {}  # model_id -> asyncio.Lock for preventing concurrent container starts
download_locks = {}  # model_id -> asyncio.Lock for preventing concurrent downloads

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

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan handler for startup and shutdown events."""
    # Startup
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

    logging.info("Application startup complete")

    yield

    # Shutdown
    logging.info("Shutting down application...")
    await http_client.aclose()
    logging.info("HTTP client closed")

app = FastAPI(lifespan=lifespan)

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

def is_gguf_model(model_id: str) -> bool:
    """Check if the model_id refers to a GGUF file."""
    return (model_id.endswith('.gguf') or
            model_id.startswith(('http://', 'https://')) and model_id.endswith('.gguf') or
            ('/' in model_id and model_id.split('/')[-1].endswith('.gguf')))

def extract_tokenizer_from_gguf_path(model_path: str) -> str:
    """Extract tokenizer path from GGUF model path for better compatibility."""
    if model_path.endswith('.gguf'):
        if model_path.startswith(('http://', 'https://')):
            # Extract repo from HuggingFace URL
            # https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/file.gguf
            # -> TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF
            if 'huggingface.co/' in model_path:
                parts = model_path.split('huggingface.co/')[-1].split('/')
                if len(parts) >= 2:
                    return f"{parts[0]}/{parts[1]}"
        elif '/' in model_path:
            # For repo_id/filename.gguf format, use the repo_id as tokenizer
            repo_id = model_path.rsplit('/', 1)[0]
            return repo_id
    return ""

def is_gguf_repo(model_id: str) -> bool:
    """Check if model_id is a HuggingFace repo containing GGUF files (not a local path)."""
    # Format: org/repo or user/repo, contains -gguf or -GGUF in the repo name
    return ('/' in model_id and
            not model_id.startswith('/') and
            not model_id.endswith('.gguf') and
            ('-gguf' in model_id.lower() or 'gguf' in model_id.lower()))

def infer_base_model_from_gguf_repo(gguf_repo_id: str) -> str:
    """
    Infer the base model repo from a GGUF repo name.
    Examples:
      - google/gemma-3-12b-it-qat-q4_0-gguf -> google/gemma-3-12b-it
      - TheBloke/Llama-2-7B-GGUF -> meta-llama/Llama-2-7b-hf
    """
    import re

    # Extract org/repo parts
    parts = gguf_repo_id.split('/')
    if len(parts) != 2:
        return None

    org, repo_name = parts

    # Remove common GGUF suffixes
    # Patterns: -gguf, -GGUF, -qat-q4_0-gguf, -q4_0-gguf, -int4-gguf, etc.
    base_name = re.sub(r'-?(qat-)?q\d+[_-]?[k0-9]*-?gguf$', '', repo_name, flags=re.IGNORECASE)
    base_name = re.sub(r'-?gguf$', '', base_name, flags=re.IGNORECASE)
    base_name = re.sub(r'-?int\d+-?gguf$', '', base_name, flags=re.IGNORECASE)

    # If from TheBloke or similar, try to map to original model
    # For now, just use the cleaned name with same org
    base_model = f"{org}/{base_name}"

    logging.info(f"Inferred base model '{base_model}' from GGUF repo '{gguf_repo_id}'")
    return base_model

async def download_gguf_from_repo(repo_id: str) -> tuple[str, str]:
    """
    Downloads a GGUF file from a HuggingFace repo and returns (local_path, tokenizer_repo).
    Returns the path to the downloaded GGUF file and the inferred base model repo for tokenizer.
    Uses async executor to avoid blocking the event loop during large downloads.
    """
    try:
        logging.info(f"Attempting to download GGUF file from repo: {repo_id}")

        # Prepare token (handle empty strings)
        token = HF_TOKEN if HF_TOKEN and HF_TOKEN.strip() else None

        # List all files in the repo to find .gguf files (run in thread pool)
        loop = asyncio.get_event_loop()
        files = await loop.run_in_executor(
            None,
            partial(list_repo_files, repo_id, token=token)
        )
        gguf_files = [f for f in files if f.endswith('.gguf')]

        if not gguf_files:
            raise ValueError(f"No GGUF files found in repo {repo_id}")

        # Smart GGUF file selection: prefer file matching quantization hint in repo name
        # e.g., "google/gemma-3-12b-it-qat-q4_0-gguf" -> prefer files with "q4_0"
        gguf_filename = gguf_files[0]  # Default to first file

        if len(gguf_files) > 1:
            # Extract potential quantization hint from repo name
            repo_name = repo_id.split('/')[-1]  # e.g., "gemma-3-12b-it-qat-q4_0-gguf"

            # Look for quantization patterns like q4_0, q4_1, q8_0, etc.
            import re
            quant_patterns = re.findall(r'q\d+_[k0-9]+|q\d+', repo_name.lower())

            if quant_patterns:
                quant_hint = quant_patterns[-1]  # Use last match (usually most specific)
                matching_files = [f for f in gguf_files if quant_hint in f.lower()]

                if matching_files:
                    gguf_filename = matching_files[0]
                    logging.info(f"Selected GGUF file '{gguf_filename}' based on quantization hint '{quant_hint}'")
                else:
                    logging.warning(f"No GGUF file matched quantization hint '{quant_hint}', using '{gguf_filename}'")

            logging.info(f"Found {len(gguf_files)} GGUF files, selected: {gguf_filename}")
        else:
            logging.info(f"Found GGUF file: {gguf_filename}")

        # Download the file using huggingface_hub (run in thread pool to avoid blocking)
        logging.info(f"Downloading {gguf_filename}... (this may take several minutes for large files)")
        local_path = await loop.run_in_executor(
            None,
            partial(
                hf_hub_download,
                repo_id=repo_id,
                filename=gguf_filename,
                token=token,
                cache_dir="/root/.cache/huggingface"
            )
        )

        logging.info(f"Successfully downloaded GGUF file to: {local_path}")

        # Infer base model for tokenizer
        base_model = infer_base_model_from_gguf_repo(repo_id)

        return local_path, base_model

    except Exception as e:
        logging.error(f"Failed to download GGUF file from {repo_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to download GGUF model: {e}")

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

# --- Background Tasks ---

async def shutdown_inactive_containers():
    """A background task that checks for inactivity and shuts down containers."""
    logging.info(f"Starting inactivity monitor with a {VLLM_INACTIVITY_TIMEOUT}s timeout.")
    while True:
        try:
            await asyncio.sleep(60)
            if VLLM_INACTIVITY_TIMEOUT <= 0:
                continue

            # Determine which containers are inactive with lock
            async with model_management_lock:
                current_time = time.time()
                inactive_containers = []
                for name, state in active_containers.items():
                    if current_time - state.last_request_time > VLLM_INACTIVITY_TIMEOUT:
                        inactive_containers.append(name)

            # Stop containers outside lock (I/O operation)
            for name in inactive_containers:
                logging.info(f"Container {name} has been idle. Shutting down.")
                await stop_container(name)

        except Exception as e:
            logging.error(f"Error in inactivity monitor: {e}", exc_info=True)

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

    # Update active_containers with lock to prevent race conditions
    async with model_management_lock:
        if container_name in active_containers:
            del active_containers[container_name]

async def get_model_max_len(model_id: str) -> int:
    """Fetches the model's config.json from Hugging Face to find its max length."""
    # For GGUF models, try to get config from the base model if available
    if is_gguf_model(model_id):
        tokenizer_path = extract_tokenizer_from_gguf_path(model_id)
        if tokenizer_path:
            config_url = f"https://huggingface.co/{tokenizer_path}/raw/main/config.json"
        else:
            # Skip max length detection for GGUF models without clear tokenizer path
            return 0
    else:
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

    # Clean up any existing container with the same name (from crashes or improper shutdowns)
    try:
        existing_container = docker_client.containers.get(container_name)
        logging.warning(f"Found existing container {container_name}. Removing it before starting new one.")
        try:
            existing_container.stop(timeout=10)
        except Exception as e:
            logging.warning(f"Could not stop existing container {container_name}: {e}")
        existing_container.remove(force=True)
        logging.info(f"Removed stale container {container_name}")

        # Verify container is actually removed before proceeding
        for attempt in range(10):
            try:
                docker_client.containers.get(container_name)
                # Container still exists, wait and retry
                await asyncio.sleep(0.5)
            except NotFound:
                # Container is gone, we can proceed
                break
        else:
            # After 10 attempts, container still exists
            logging.error(f"Failed to remove container {container_name} after 10 attempts")
            raise HTTPException(status_code=500, detail=f"Failed to remove existing container {container_name}")

    except NotFound:
        # No existing container, this is the expected case
        pass
    except APIError as e:
        logging.error(f"Error checking/removing existing container {container_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to handle existing container: {e}")

    # Handle GGUF repos by downloading first
    actual_model_path = model_id
    tokenizer_repo = None

    if is_gguf_repo(model_id):
        # Ensure only one download per model at a time (protect lock creation with global lock)
        async with model_management_lock:
            if model_id not in download_locks:
                download_locks[model_id] = asyncio.Lock()
            download_lock = download_locks[model_id]

        async with download_lock:
            logging.info(f"Detected GGUF repo: {model_id}. Downloading GGUF file...")
            actual_model_path, tokenizer_repo = await download_gguf_from_repo(model_id)
            logging.info(f"Will use local GGUF path: {actual_model_path}")

    model_max_len = await get_model_max_len(tokenizer_repo if tokenizer_repo else model_id)
    final_max_len = model_max_len
    if VLLM_MAX_MODEL_LEN_GLOBAL > 0:
        final_max_len = min(model_max_len, VLLM_MAX_MODEL_LEN_GLOBAL) if model_max_len > 0 else VLLM_MAX_MODEL_LEN_GLOBAL

    command = ["--model", actual_model_path, "--gpu-memory-utilization", VLLM_GPU_MEMORY_UTILIZATION]

    # Add GGUF-specific parameters if this is a GGUF model
    if is_gguf_model(actual_model_path) or tokenizer_repo:
        # If we downloaded from a repo, use that repo as tokenizer
        if tokenizer_repo:
            command.extend(["--tokenizer", tokenizer_repo])
            command.extend(["--hf-config-path", tokenizer_repo])
            logging.info(f"Using tokenizer and config from {tokenizer_repo} for GGUF model")
        else:
            tokenizer_path = extract_tokenizer_from_gguf_path(actual_model_path)
            if tokenizer_path:
                command.extend(["--tokenizer", tokenizer_path])
                command.extend(["--hf-config-path", tokenizer_path])
                logging.info(f"Using tokenizer and config from {tokenizer_path} for GGUF model {model_id}")
            else:
                logging.warning(f"No tokenizer path found for GGUF model {model_id}. Using model's embedded tokenizer.")

    if int(VLLM_SWAP_SPACE) > 0:
        command.extend(["--swap-space", VLLM_SWAP_SPACE])
    if final_max_len > 0:
        command.extend(["--max-model-len", str(final_max_len)])

    if int(VLLM_MAX_NUM_SEQS) > 0:
        command.extend(["--max-num-seqs", VLLM_MAX_NUM_SEQS])

    if int(VLLM_TENSOR_PARALLEL_SIZE) > 0:
        command.extend(["--tensor-parallel-size", VLLM_TENSOR_PARALLEL_SIZE])

    # Add gpt-oss specific optimizations for Ampere/Ada GPUs (RTX 3090, A100, etc)
    if is_gpt_oss_model(model_id):
        command.append("--async-scheduling")
        logging.info(f"Added --async-scheduling flag for gpt-oss model optimization")

    try:
        vllm_image = get_vllm_image_for_model(model_id)
        logging.info(f"Using vLLM image: {vllm_image} for model {model_id}")
        logging.info(f"Starting container {container_name} with command: {' '.join(command)}")
        new_container = docker_client.containers.run(
            vllm_image,
            command=command,
            name=container_name,
            hostname=container_name,
            detach=True,
            network=RESOLVED_DOCKER_NETWORK,
            environment={
                "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1"
            },
            ipc_mode="host",
            device_requests=[DeviceRequest(count=-1, capabilities=[['gpu']])],
            volumes={
                HOST_CACHE_DIR: {'bind': '/root/.cache/huggingface', 'mode': 'rw'},
                "/tmp": {'bind': '/tmp', 'mode': 'rw'}  # For temporary GGUF downloads
            }
        )
        new_container.reload()

        # Safely retrieve network IP address
        networks = new_container.attrs.get('NetworkSettings', {}).get('Networks', {})
        network_info = networks.get(RESOLVED_DOCKER_NETWORK)
        if not network_info or not network_info.get('IPAddress'):
            logging.error(f"Failed to get IP address for container {container_name} on network {RESOLVED_DOCKER_NETWORK}")
            logging.error(f"Available networks: {list(networks.keys())}")
            new_container.stop()
            new_container.remove()
            return None

        ip_address = network_info['IPAddress']

        # Health check loop with progress logging
        vllm_base_url = f"http://{ip_address}:{VLLM_PORT}"
        logging.info(f"Starting health checks for {model_id}. This may take several minutes for large models...")

        for i in range(1800): # ~1 hour timeout (1800 * 2s = 3600s)
            await asyncio.sleep(2)
            try:
                response = await http_client.get(f"{vllm_base_url}/health", timeout=2)
                if response.status_code == 200:
                    elapsed_time = (i + 1) * 2
                    logging.info(f"Model {model_id} started successfully at {vllm_base_url} after {elapsed_time}s.")
                    return ContainerState(
                        model_id=model_id,
                        container_name=container_name,
                        ip_address=ip_address,
                        port=VLLM_PORT,
                        last_request_time=time.time(),
                        vram_footprint=0 # Footprint is unknown until measured
                    )
            except httpx.RequestError:
                # Log progress every 30 seconds (15 attempts * 2s)
                if i > 0 and (i + 1) % 15 == 0:
                    elapsed_time = (i + 1) * 2
                    remaining_time = (1800 - i - 1) * 2
                    elapsed_min = elapsed_time // 60
                    remaining_min = remaining_time // 60
                    logging.info(f"Model {model_id} still loading... ({elapsed_min}m {elapsed_time % 60}s elapsed, {remaining_min}m {remaining_time % 60}s remaining)")
                elif i <= 5:
                    logging.info(f"Waiting for model {model_id} to initialize... (attempt {i+1})")

        elapsed_time = 1800 * 2
        elapsed_min = elapsed_time // 60
        logging.error(f"Model {model_id} failed to start after {elapsed_min}m timeout.")
        await stop_container(container_name)
        return None

    except APIError as e:
        logging.error(f"Error starting container {container_name}: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to start model container: {e}")

# --- Main Proxy Logic ---

# Specific routes MUST come before catch-all route
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

# Catch-all proxy route (must be last)
@app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"])
async def proxy_request(request: Request):
    # Handle requests with and without JSON body
    try:
        body = await request.json()
        model_name = body.get("model")
    except Exception:
        # GET requests or requests without body - can't determine model
        return JSONResponse(
            {"error": "Missing 'model' field in request body. Use POST with JSON body containing 'model' field."},
            status_code=400
        )

    if not model_name or model_name not in ALLOWED_MODELS:
        return JSONResponse({"error": f"Model not allowed. Please choose from: {list(ALLOWED_MODELS.keys())}"}, status_code=400)

    target_model_id = ALLOWED_MODELS[model_name]

    # Quick check for existing container (no lock needed for read-only check)
    target_container: ContainerState = next((c for c in active_containers.values() if c.model_id == target_model_id), None)

    if not target_container:
        # Ensure per-model lock exists (protect lock creation with global lock)
        async with model_management_lock:
            if target_model_id not in container_start_locks:
                container_start_locks[target_model_id] = asyncio.Lock()
            per_model_lock = container_start_locks[target_model_id]

        # Acquire per-model lock to prevent concurrent starts of the same model
        async with per_model_lock:
            # Double-check after acquiring lock (another request may have started it)
            target_container = next((c for c in active_containers.values() if c.model_id == target_model_id), None)

            if not target_container:
                # 2. Model not running, need to start it
                footprint = known_footprints.get(target_model_id)

                if footprint is None and TOTAL_GPU_VRAM > 0:
                    # DISCOVERY RUN
                    logging.info(f"Unknown footprint for {target_model_id}. Starting a discovery run.")

                    # Check and stop containers with lock
                    async with model_management_lock:
                        if active_containers:
                            logging.info("Stopping all active containers for discovery run.")
                            container_names_to_stop = list(active_containers.keys())
                        else:
                            container_names_to_stop = []

                    # Stop containers outside lock (I/O operation)
                    for name in container_names_to_stop:
                        await stop_container(name)

                    vram_before = get_used_vram()
                    logging.info(f"VRAM usage before model load: {vram_before} MiB")

                    container_name = f"{VLLM_CONTAINER_PREFIX}_0"
                    # Start container outside lock (long I/O operation)
                    target_container = await start_model_container(target_model_id, container_name)

                    if target_container:
                        # Wait for model to fully load into VRAM and take multiple measurements
                        logging.info("Waiting for model to fully load into VRAM (taking 3 measurements over 45 seconds)...")
                        await asyncio.sleep(15)
                        vram_samples = [get_used_vram()]

                        await asyncio.sleep(15)
                        vram_samples.append(get_used_vram())

                        await asyncio.sleep(15)
                        vram_samples.append(get_used_vram())

                        # Use maximum VRAM measurement to ensure we capture full footprint
                        vram_after = max(vram_samples)
                        logging.info(f"VRAM samples: {vram_samples} MiB, using max: {vram_after} MiB")

                        measured_vram = vram_after - vram_before

                        if measured_vram > 256: # Sanity check for a reasonable footprint
                            logging.info(f"Measured VRAM footprint for {target_model_id}: {measured_vram} MiB")
                            target_container.vram_footprint = measured_vram
                            known_footprints[target_model_id] = measured_vram
                            save_known_footprints()
                        else:
                            logging.warning(f"Could not measure accurate footprint for {target_model_id} (calculated {measured_vram} MiB). Using container without VRAM management.")
                            target_container.vram_footprint = 0  # Mark as unknown

                        # Update active_containers with lock
                        async with model_management_lock:
                            active_containers[container_name] = target_container


                elif TOTAL_GPU_VRAM > 0:
                    # KNOWN FOOTPRINT RUN
                    # Calculate VRAM and determine evictions with lock
                    async with model_management_lock:
                        current_vram_usage = sum(c.vram_footprint for c in active_containers.values())
                        containers_to_evict = []

                        if current_vram_usage + footprint > TOTAL_GPU_VRAM:
                            logging.info(f"Not enough VRAM for {target_model_id} (needs {footprint} MiB). Evicting LRU containers.")
                            # Determine which containers to evict
                            sorted_containers = sorted(active_containers.values(), key=lambda c: c.last_request_time)
                            for container_to_evict in sorted_containers:
                                containers_to_evict.append(container_to_evict.container_name)
                                current_vram_usage -= container_to_evict.vram_footprint
                                if current_vram_usage + footprint <= TOTAL_GPU_VRAM:
                                    break

                        # Find a free slot index
                        slot_indices = {int(name.split('_')[-1]) for name in active_containers.keys()}
                        free_slot = next(i for i in range(len(slot_indices) + 1) if i not in slot_indices)
                        container_name = f"{VLLM_CONTAINER_PREFIX}_{free_slot}"

                    # Evict containers outside lock (I/O operation)
                    for name in containers_to_evict:
                        await stop_container(name)

                    # Start container outside lock (long I/O operation)
                    target_container = await start_model_container(target_model_id, container_name)
                    if target_container:
                        target_container.vram_footprint = footprint
                        # Update active_containers with lock
                        async with model_management_lock:
                            active_containers[container_name] = target_container


                else: # Fallback if VRAM management is disabled
                    # Without VRAM management, only run one container at a time
                    # Stop all existing containers to prevent name conflicts
                    async with model_management_lock:
                        if active_containers:
                            logging.info(f"No dynamic VRAM management. Stopping all {len(active_containers)} active containers.")
                            container_names_to_stop = list(active_containers.keys())
                        else:
                            container_names_to_stop = []

                    # Evict all containers outside lock (I/O operation)
                    for name in container_names_to_stop:
                        await stop_container(name)

                    container_name = f"{VLLM_CONTAINER_PREFIX}_0"
                    # Start container outside lock (long I/O operation)
                    target_container = await start_model_container(target_model_id, container_name)
                    if target_container:
                        # Update active_containers with lock
                        async with model_management_lock:
                            active_containers[container_name] = target_container

    # Outside all locks - check if we have a container
    if not target_container:
        raise HTTPException(status_code=500, detail=f"Failed to start or find container for model {target_model_id}")

    # Update last request time (no lock needed - single attribute write)
    target_container.last_request_time = time.time()

    # Build full URL with query parameters
    query_string = str(request.url.query) if request.url.query else ""
    vllm_url = f"http://{target_container.ip_address}:{target_container.port}{request.url.path}"
    if query_string:
        vllm_url += f"?{query_string}"

    # Request proxying happens outside all locks
    try:
        # Update model field in body before proxying
        body['model'] = target_model_id

        # Forward relevant headers (exclude hop-by-hop headers)
        headers_to_forward = {
            k: v for k, v in request.headers.items()
            if k.lower() not in ('host', 'connection', 'content-length', 'transfer-encoding')
        }

        # Check if this is a streaming request
        is_streaming = body.get('stream', False)

        # Proxy request using the original HTTP method
        response = await http_client.request(
            method=request.method,
            url=vllm_url,
            json=body,
            headers=headers_to_forward,
            timeout=300
        )
        response.raise_for_status()

        # Handle streaming vs non-streaming responses
        if is_streaming:
            # For streaming responses, return the raw content with appropriate headers
            from fastapi.responses import StreamingResponse

            async def generate():
                async for chunk in response.aiter_bytes():
                    yield chunk

            return StreamingResponse(
                generate(),
                status_code=response.status_code,
                media_type=response.headers.get('content-type', 'text/event-stream'),
                headers={k: v for k, v in response.headers.items()
                        if k.lower() not in ('content-length', 'transfer-encoding', 'connection')}
            )
        else:
            # For non-streaming responses, return JSON as before
            return JSONResponse(content=response.json(), status_code=response.status_code)
    except httpx.HTTPStatusError as e:
        return JSONResponse({"error": "Error from vLLM service", "details": str(e)}, status_code=e.response.status_code)
    except httpx.RequestError as e:
        raise HTTPException(status_code=503, detail=f"Could not connect to vLLM service: {e}")
