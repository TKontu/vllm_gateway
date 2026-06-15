import os
import asyncio
import httpx
import docker
import json
import time
import logging
import uuid
from contextlib import asynccontextmanager
from functools import partial
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from docker.types import DeviceRequest
from docker.errors import NotFound, APIError
from dataclasses import dataclass, field
from huggingface_hub import hf_hub_download, list_repo_files
import yaml
import placement
from config_loader import (
    ModelConfig, resolve_model_configs, build_fallback_configs,
    resolve_pools, validate_model_pools, validate_tp_against_pools, validate_colocate,
    validate_pools_visible,
)

# --- Logging Configuration ---
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=LOG_LEVEL, format='%(asctime)s - %(levelname)s - %(message)s')

# --- Configuration ---
HF_TOKEN = os.getenv("HUGGING_FACE_HUB_TOKEN", "")
HOST_CACHE_DIR = os.getenv("HOST_CACHE_DIR", "/root/.cache/huggingface")
CONTAINER_CACHE_MOUNT = '/root/.cache/huggingface'  # Where HOST_CACHE_DIR is mounted inside vLLM containers
VLLM_PORT = int(os.getenv("VLLM_PORT", "8000"))
VLLM_IMAGE = os.getenv("VLLM_IMAGE", "vllm/vllm-openai:v0.10.2")
VLLM_GPU_MEMORY_UTILIZATION = os.getenv("VLLM_GPU_MEMORY_UTILIZATION", "0.90")
VLLM_MAX_MODEL_LEN_GLOBAL = int(os.getenv("VLLM_MAX_MODEL_LEN_GLOBAL", "0"))
VLLM_MAX_NUM_SEQS = os.getenv("VLLM_MAX_NUM_SEQS", "16")
VLLM_TENSOR_PARALLEL_SIZE = os.getenv("VLLM_TENSOR_PARALLEL_SIZE", "1")
VLLM_ENFORCE_EAGER = os.getenv("VLLM_ENFORCE_EAGER", "false").lower() == "true"
VLLM_NO_CUDAGRAPH = os.getenv("VLLM_NO_CUDAGRAPH", "false").lower() == "true"
DOCKER_NETWORK_NAME = os.getenv("DOCKER_NETWORK_NAME", "vllm_network")
GATEWAY_CONTAINER_NAME = os.getenv("GATEWAY_CONTAINER_NAME", "vllm_gateway")
VLLM_INACTIVITY_TIMEOUT = int(os.getenv("VLLM_INACTIVITY_TIMEOUT", 1800))
VLLM_CONTAINER_PREFIX = os.getenv("VLLM_CONTAINER_PREFIX", "vllm_server")
NVIDIA_UTILITY_IMAGE = os.getenv("NVIDIA_UTILITY_IMAGE", "nvidia/cuda:12.1.0-base-ubuntu22.04")
MEMORY_FOOTPRINT_FILE = os.getenv("MEMORY_FOOTPRINT_FILE", "/app/data/memory_footprints.json")
VLLM_TEMP_DIR = os.getenv("VLLM_TEMP_DIR", "/tmp")

# Path to the per-model YAML config. When present, it replaces ALLOWED_MODELS_JSON.
MODELS_CONFIG_FILE = os.getenv("MODELS_CONFIG_FILE", "/app/config/models.yaml")

# --- GPU Pinning ---
# Optionally pin this gateway instance (and every vLLM container it launches) to a single GPU
# by UUID. For multi-GPU, prefer a `pools:` section in the model config (which takes precedence
# over this pin). Leave unset and with no pools to use all visible GPUs (legacy behavior).
GATEWAY_GPU_UUID = os.getenv("GATEWAY_GPU_UUID", "").strip()
if GATEWAY_GPU_UUID:
    GPU_DEVICE_REQUESTS = [DeviceRequest(device_ids=[GATEWAY_GPU_UUID], capabilities=[['gpu']])]
else:
    GPU_DEVICE_REQUESTS = [DeviceRequest(count=-1, capabilities=[['gpu']])]

# Queue management configuration
GATEWAY_MAX_QUEUE_SIZE = int(os.getenv("GATEWAY_MAX_QUEUE_SIZE", "200"))  # Max requests in queue per model
GATEWAY_MAX_CONCURRENT = int(os.getenv("GATEWAY_MAX_CONCURRENT", "50"))  # Max concurrent requests to vLLM per model

# Anti-thrash: minimum seconds a freshly-loaded model is preferentially kept resident before it
# becomes a candidate for eviction-to-make-room. Eviction falls back to fresh models only if no
# older candidate frees enough VRAM (so a single-GPU swap is never blocked). 0 disables the cooldown.
GATEWAY_MIN_RESIDENT_SECONDS = int(os.getenv("GATEWAY_MIN_RESIDENT_SECONDS", "90"))

# Co-location (Phase 3): when a co-locatable model shares a card, its launch util is capped so it
# leaves COLOCATE_MARGIN_MIB free; weights must fit util*total * COLOCATE_WEIGHT_OVERHEAD; a
# colocate model whose share exceeds COLOCATE_MAX_SHARE is warned about at startup.
COLOCATE_MARGIN_MIB = int(os.getenv("COLOCATE_MARGIN_MIB", "1024"))
COLOCATE_WEIGHT_OVERHEAD = float(os.getenv("COLOCATE_WEIGHT_OVERHEAD", "1.15"))
COLOCATE_MAX_SHARE = float(os.getenv("COLOCATE_MAX_SHARE", "0.9"))

# Timeout configuration (in seconds)
GATEWAY_REQUEST_TIMEOUT = int(os.getenv("GATEWAY_REQUEST_TIMEOUT", "300"))  # Total request timeout (default 5 minutes)
GATEWAY_CONNECT_TIMEOUT = int(os.getenv("GATEWAY_CONNECT_TIMEOUT", "10"))  # Connection establishment timeout

# Validate critical configuration values
def validate_config():
    """Validates configuration values to prevent system failures."""
    if GATEWAY_MAX_QUEUE_SIZE <= 0:
        raise ValueError(f"GATEWAY_MAX_QUEUE_SIZE must be > 0, got {GATEWAY_MAX_QUEUE_SIZE}")
    if GATEWAY_MAX_CONCURRENT <= 0:
        raise ValueError(f"GATEWAY_MAX_CONCURRENT must be > 0, got {GATEWAY_MAX_CONCURRENT}")
    if GATEWAY_REQUEST_TIMEOUT <= 0:
        raise ValueError(f"GATEWAY_REQUEST_TIMEOUT must be > 0, got {GATEWAY_REQUEST_TIMEOUT}")
    if GATEWAY_CONNECT_TIMEOUT <= 0:
        raise ValueError(f"GATEWAY_CONNECT_TIMEOUT must be > 0, got {GATEWAY_CONNECT_TIMEOUT}")

    if GATEWAY_MAX_QUEUE_SIZE > 10000:
        logging.warning(f"GATEWAY_MAX_QUEUE_SIZE is very large ({GATEWAY_MAX_QUEUE_SIZE}). This may cause memory issues.")
    if GATEWAY_MAX_CONCURRENT > 500:
        logging.warning(f"GATEWAY_MAX_CONCURRENT is very large ({GATEWAY_MAX_CONCURRENT}). This may overwhelm vLLM.")
    if GATEWAY_REQUEST_TIMEOUT > 3600:
        logging.warning(f"GATEWAY_REQUEST_TIMEOUT is very large ({GATEWAY_REQUEST_TIMEOUT}s = {GATEWAY_REQUEST_TIMEOUT//60} minutes). Consider if this is intentional.")
    if GATEWAY_CONNECT_TIMEOUT > 60:
        logging.warning(f"GATEWAY_CONNECT_TIMEOUT is very large ({GATEWAY_CONNECT_TIMEOUT}s). Connection should establish quickly.")

validate_config()

# Log startup configuration for debugging
logging.info("=" * 60)
logging.info("GATEWAY TIMEOUT CONFIGURATION:")
logging.info(f"  GATEWAY_REQUEST_TIMEOUT: {GATEWAY_REQUEST_TIMEOUT}s ({GATEWAY_REQUEST_TIMEOUT//60} minutes)")
logging.info(f"  GATEWAY_CONNECT_TIMEOUT: {GATEWAY_CONNECT_TIMEOUT}s")
logging.info("=" * 60)

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
GPU_VRAM = {}  # uuid -> {"total": int MiB, "used": float MiB}; per-GPU foundation for multi-GPU placement
known_footprints = {}
active_containers = {}
model_management_lock = asyncio.Lock()  # For updating active_containers dict
container_start_locks = {}  # model_id -> asyncio.Lock for preventing concurrent container starts
download_locks = {}  # model_id -> asyncio.Lock for preventing concurrent downloads

# Multi-GPU placement state (resolved at startup; see resolve_managed_pools)
MANAGED_POOLS = {}  # pool_name -> [gpu_uuid, ...]: the GPUs this gateway manages, grouped into pools
MANAGED_GPUS = []   # flat list of managed gpu_uuids (union of MANAGED_POOLS values)
gpu_reservations = {}  # gpu_uuid -> reserved MiB for models that are loading (not yet nvidia-smi visible)
# Single global lock held ONLY for the placement decision (choose GPU + reserve), never across a
# container start. Acquire order: container_start_locks[model] -> model_management_lock -> placement_lock.
placement_lock = asyncio.Lock()

# Queue management state
model_semaphores = {}  # model_id -> asyncio.Semaphore for limiting concurrent requests
model_queue_counts = {}  # model_id -> int (number of requests waiting in queue)
queue_count_lock = asyncio.Lock()  # Protects queue counter updates

@dataclass
class ContainerState:
    model_id: str
    container_name: str
    ip_address: str
    port: int
    last_request_time: float
    vram_footprint: float # in MiB
    active_requests: int = 0  # number of requests currently being proxied to this container
    always_on: bool = False  # if True, never auto-unloaded by the inactivity monitor
    inactivity_timeout: int = VLLM_INACTIVITY_TIMEOUT  # per-model idle timeout (seconds; 0 = never unload)
    loaded_at: float = 0.0  # time.time() when the container became ready (for anti-thrash cooldown)
    gpu_uuids: list = field(default_factory=list)  # GPU UUID(s) this container is pinned to
    colocate: bool = False  # may share its GPU with other co-locatable models (Phase 3)

# --- Docker and HTTP Clients ---
docker_client = docker.from_env()

# Configure HTTP client with appropriate connection limits for high concurrency
# Connection pool must accommodate multiple concurrent models
# Formula: GATEWAY_MAX_CONCURRENT * expected number of concurrent models + buffer
# Default sizing: 50 concurrent * 3 models = 150 connections minimum
GATEWAY_MAX_MODELS_CONCURRENT = int(os.getenv("GATEWAY_MAX_MODELS_CONCURRENT", "3"))

# Validate HTTP pool configuration
if GATEWAY_MAX_MODELS_CONCURRENT <= 0:
    raise ValueError(f"GATEWAY_MAX_MODELS_CONCURRENT must be > 0, got {GATEWAY_MAX_MODELS_CONCURRENT}")
if GATEWAY_MAX_MODELS_CONCURRENT > 20:
    logging.warning(f"GATEWAY_MAX_MODELS_CONCURRENT is very large ({GATEWAY_MAX_MODELS_CONCURRENT}). Connection pool will be {GATEWAY_MAX_CONCURRENT * GATEWAY_MAX_MODELS_CONCURRENT} connections.")

http_pool_size = GATEWAY_MAX_CONCURRENT * GATEWAY_MAX_MODELS_CONCURRENT
http_keepalive_size = http_pool_size  # Keep all connections alive for best performance

http_client = httpx.AsyncClient(
    limits=httpx.Limits(
        max_connections=http_pool_size,  # 150 by default (50 * 3)
        max_keepalive_connections=http_keepalive_size,  # 150 by default (matches pool size)
    ),
    timeout=httpx.Timeout(
        connect=float(GATEWAY_CONNECT_TIMEOUT),   # Connection timeout (configurable)
        read=float(GATEWAY_REQUEST_TIMEOUT),      # Read timeout for response headers/body
        write=float(GATEWAY_REQUEST_TIMEOUT),     # Write timeout for sending request body
        pool=float(GATEWAY_CONNECT_TIMEOUT)       # Pool timeout for acquiring connection
    )
)

@asynccontextmanager
async def lifespan(_app: FastAPI):
    """Lifespan handler for startup and shutdown events."""
    # Startup
    global RESOLVED_DOCKER_NETWORK
    try:
        gateway_container = await run_in_executor(docker_client.containers.get, GATEWAY_CONTAINER_NAME)
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

    await get_total_vram()  # probes ALL GPUs -> GPU_VRAM (independent of any pin)
    resolve_managed_pools()
    # Fail fast if a configured pool/pin UUID isn't actually present (typo / moved card),
    # rather than silently treating it as a 0-VRAM, unplaceable GPU.
    if CONFIGURED_POOLS or GATEWAY_GPU_UUID:
        validate_pools_visible(MANAGED_POOLS, set(GPU_VRAM.keys()))
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

async def run_in_executor(func, *args, **kwargs):
    """Run a synchronous function in the default thread pool to avoid blocking the event loop."""
    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, partial(func, *args, **kwargs))

async def run_nvidia_smi_in_container(command: list[str]) -> str:
    """Runs an nvidia-smi command in a temporary container and returns the output.

    The probe always requests ALL GPUs (count=-1) and the caller filters to the managed set.
    This keeps per-GPU accounting complete regardless of the GATEWAY_GPU_UUID pin (so pools
    win over the pin), and avoids a single bad configured UUID erroring the whole probe."""
    probe_requests = [DeviceRequest(count=-1, capabilities=[['gpu']])]
    try:
        smi_output = await run_in_executor(
            docker_client.containers.run,
            NVIDIA_UTILITY_IMAGE,
            command=command,
            remove=True,
            device_requests=probe_requests
        )
        return smi_output.decode('utf-8').strip()
    except APIError as e:
        logging.error(f"Error running nvidia-smi container: {e}")
        return ""

def _parse_float(s: str) -> float:
    try:
        return float(s)
    except (TypeError, ValueError):
        return 0.0

async def get_gpu_vram() -> dict:
    """Returns {uuid: {"total": int MiB, "used": float MiB}} for every visible GPU.

    When the gateway is pinned (GATEWAY_GPU_UUID set), the probe container only sees that
    one GPU, so the map has a single entry. Unpinned, it lists all visible GPUs — parsing
    every CSV line instead of just the first (which read GPU 0 only)."""
    output = await run_nvidia_smi_in_container(
        ["nvidia-smi", "--query-gpu=uuid,memory.total,memory.used", "--format=csv,noheader,nounits"]
    )
    gpus = {}
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        uuid, total, used = parts
        if uuid and total.isdigit():
            gpus[uuid] = {"total": int(total), "used": _parse_float(used)}
    return gpus

def _managed_vram(gpus: dict, key: str) -> float:
    """Sum a VRAM field across the GPUs this instance manages.

    Pinned -> just that GPU; unpinned -> all visible GPUs (single-pool semantics, unchanged)."""
    if GATEWAY_GPU_UUID and GATEWAY_GPU_UUID in gpus:
        return gpus[GATEWAY_GPU_UUID][key]
    return sum(g[key] for g in gpus.values())

async def get_total_vram():
    """Measures per-GPU VRAM, stores the per-UUID map, and sets the managed total (MiB)."""
    global TOTAL_GPU_VRAM, GPU_VRAM
    logging.info("Getting total GPU VRAM...")
    GPU_VRAM = await get_gpu_vram()
    if GPU_VRAM:
        for uuid, v in GPU_VRAM.items():
            logging.info(f"GPU {uuid}: {v['total']} MiB total")
        TOTAL_GPU_VRAM = int(_managed_vram(GPU_VRAM, "total"))
        logging.info(f"Managed total GPU VRAM: {TOTAL_GPU_VRAM} MiB across {len(GPU_VRAM)} GPU(s)")
    else:
        logging.error("Could not determine total GPU VRAM. Disabling dynamic memory management.")
        TOTAL_GPU_VRAM = 0

def load_known_footprints():
    """Loads the known model memory footprints from the JSON file."""
    global known_footprints
    try:
        if os.path.exists(MEMORY_FOOTPRINT_FILE):
            if os.path.isdir(MEMORY_FOOTPRINT_FILE):
                logging.error(f"MEMORY_FOOTPRINT_FILE '{MEMORY_FOOTPRINT_FILE}' is a directory, not a file!")
                logging.error("This happens when Docker creates a missing mount target as a directory.")
                logging.error("Fix: Stop container, run 'rm -rf {0} && echo \"{{}}\" > {0}' on host, then restart.".format(MEMORY_FOOTPRINT_FILE))
                known_footprints = {}
                return
            with open(MEMORY_FOOTPRINT_FILE, 'r') as f:
                known_footprints = json.load(f)
            logging.info(f"Loaded known model footprints: {known_footprints}")
        else:
            logging.info(f"Memory footprints file not found, creating new one at {MEMORY_FOOTPRINT_FILE}")
            known_footprints = {}
            save_known_footprints()
    except (json.JSONDecodeError, IOError) as e:
        logging.error(f"Could not load memory footprints file: {e}")
        known_footprints = {}

def save_known_footprints():
    """Saves the known footprints back to the JSON file."""
    try:
        if os.path.isdir(MEMORY_FOOTPRINT_FILE):
            logging.error(f"Cannot save footprints: '{MEMORY_FOOTPRINT_FILE}' is a directory, not a file!")
            return

        # Ensure parent directory exists
        parent_dir = os.path.dirname(MEMORY_FOOTPRINT_FILE)
        if parent_dir and not os.path.exists(parent_dir):
            os.makedirs(parent_dir, exist_ok=True)

        with open(MEMORY_FOOTPRINT_FILE, 'w') as f:
            json.dump(known_footprints, f, indent=4)
    except IOError as e:
        logging.error(f"Could not save memory footprints file: {e}")

async def save_known_footprints_async():
    """Persist footprints off the event loop (blocking file I/O must not run on the loop)."""
    await run_in_executor(save_known_footprints)

async def get_used_vram() -> float:
    """Gets currently used GPU VRAM (MiB) across the managed GPU(s)."""
    gpus = await get_gpu_vram()
    if not gpus:
        logging.error("Could not determine used GPU VRAM.")
        return 0.0
    return float(_managed_vram(gpus, "used"))

DEFAULT_POOL = "_default"  # implicit pool used in the single-pool / backward-compatible cases

def resolve_managed_pools():
    """Resolve which GPUs this gateway manages and how they're grouped into pools.

    Precedence: (1) a 'pools:' section in the model config; (2) GATEWAY_GPU_UUID (one implicit
    single-GPU pool); (3) all visible GPUs (one implicit pool — today's single-pool behavior).
    Runs at startup after get_total_vram() has populated GPU_VRAM."""
    global MANAGED_POOLS, MANAGED_GPUS
    if CONFIGURED_POOLS:
        MANAGED_POOLS = {name: list(uuids) for name, uuids in CONFIGURED_POOLS.items()}
        source = f"config 'pools' ({len(MANAGED_POOLS)} pool(s))"
    elif GATEWAY_GPU_UUID:
        MANAGED_POOLS = {DEFAULT_POOL: [GATEWAY_GPU_UUID]}
        source = f"GATEWAY_GPU_UUID pin ({GATEWAY_GPU_UUID})"
    else:
        MANAGED_POOLS = {DEFAULT_POOL: list(GPU_VRAM.keys())}
        source = f"all visible GPUs ({len(GPU_VRAM)})"
    MANAGED_GPUS = [u for uuids in MANAGED_POOLS.values() for u in uuids]
    logging.info(f"Managed GPU pools resolved from {source}: "
                 f"{ {p: u for p, u in MANAGED_POOLS.items()} }")
    # Recompute the managed total from the resolved set so it reflects pools (which win over the
    # pin), not whatever _managed_vram defaulted to before resolution.
    global TOTAL_GPU_VRAM
    if GPU_VRAM:
        TOTAL_GPU_VRAM = int(sum(GPU_VRAM[u]["total"] for u in MANAGED_GPUS if u in GPU_VRAM))

def pool_for(model_cfg) -> str:
    """The pool a model belongs to: its configured pool, else the default/implicit pool."""
    return model_cfg.pool if model_cfg.pool else DEFAULT_POOL

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
            # For repo_id/filename.gguf format, use the repo_id as tokenizer.
            # Only valid if the prefix is itself a 2-part HF repo ID (org/repo).
            repo_id = model_path.rsplit('/', 1)[0]
            if '/' in repo_id:
                return repo_id
    return ""

def is_gguf_repo(model_id: str) -> bool:
    """Check if model_id is a HuggingFace repo containing GGUF files (not a local path)."""
    # Format: org/repo or user/repo, contains -gguf or -GGUF in the repo name.
    # Exclude vLLM's native repo:quant_type format (e.g. unsloth/Qwen3-0.6B-GGUF:Q4_K_M)
    # which contains a colon — vLLM handles that format natively without downloading.
    return ('/' in model_id and
            not model_id.startswith('/') and
            not model_id.endswith('.gguf') and
            ':' not in model_id and
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

async def download_gguf_from_repo(repo_id: str, quant_hint: str = "") -> tuple[str, str]:
    """
    Downloads a GGUF file from a HuggingFace repo and returns (local_path, tokenizer_repo).
    Returns the path to the downloaded GGUF file and the inferred base model repo for tokenizer.
    Uses async executor to avoid blocking the event loop during large downloads.
    quant_hint: optional quantization string (e.g. "Q4_K_M") to select a specific file.
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

        # Smart GGUF file selection: prefer file matching quantization hint.
        # Priority: explicit quant_hint arg > hint extracted from repo name.
        # e.g., "google/gemma-3-12b-it-qat-q4_0-gguf" -> prefer files with "q4_0"
        gguf_filename = gguf_files[0]  # Default to first file

        if len(gguf_files) > 1:
            # Use explicit quant_hint if provided, otherwise extract from repo name
            import re
            if quant_hint:
                resolved_hint = quant_hint.lower()
            else:
                repo_name = repo_id.split('/')[-1]  # e.g., "gemma-3-12b-it-qat-q4_0-gguf"
                quant_patterns = re.findall(r'q\d+_[k0-9]+|q\d+', repo_name.lower())
                resolved_hint = quant_patterns[-1] if quant_patterns else ""

            if resolved_hint:
                matching_files = [f for f in gguf_files if resolved_hint in f.lower()]

                if matching_files:
                    gguf_filename = matching_files[0]
                    logging.info(f"Selected GGUF file '{gguf_filename}' based on quantization hint '{resolved_hint}'")
                else:
                    logging.warning(f"No GGUF file matched quantization hint '{resolved_hint}', using '{gguf_filename}'")

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
                cache_dir=HOST_CACHE_DIR
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
    """Load allowed models from environment variable (legacy ALLOWED_MODELS_JSON fallback)."""
    models_json = os.getenv("ALLOWED_MODELS_JSON")
    if models_json:
        try:
            return json.loads(models_json)
        except json.JSONDecodeError:
            logging.warning("Invalid JSON in ALLOWED_MODELS_JSON. Using default empty set.")
    return {}

def builtin_model_defaults() -> dict:
    """Bottom-tier defaults for the config resolver, sourced from the legacy global env vars.

    These are the values used when neither a per-model entry nor the YAML 'defaults'
    block specifies a setting — preserving the exact pre-config-file behavior.
    """
    return {
        "gpu_memory_utilization": float(VLLM_GPU_MEMORY_UTILIZATION),
        "max_model_len": VLLM_MAX_MODEL_LEN_GLOBAL,
        "tensor_parallel_size": int(VLLM_TENSOR_PARALLEL_SIZE),
        "quantization": None,
        "dtype": "auto",
        "inactivity_timeout": VLLM_INACTIVITY_TIMEOUT,
        "always_on": False,
        "extra_args": [],
        "pool": None,
        "colocate": False,
    }

def load_model_configs() -> "tuple[dict, dict]":
    """Load per-model configuration and any declared GPU pools.

    Uses MODELS_CONFIG_FILE (YAML) when it exists; otherwise falls back to the legacy
    ALLOWED_MODELS_JSON env var with built-in defaults (behaves exactly as before).
    Returns (configs, pools); pools is {} unless a 'pools:' section is present.
    Raises on a malformed config file so the gateway fails fast at startup.
    """
    builtins = builtin_model_defaults()

    if MODELS_CONFIG_FILE and os.path.isfile(MODELS_CONFIG_FILE):
        logging.info(f"Loading model configuration from {MODELS_CONFIG_FILE}")
        with open(MODELS_CONFIG_FILE, 'r') as f:
            raw = yaml.safe_load(f)
        if not raw:
            raise ValueError(f"Model config file '{MODELS_CONFIG_FILE}' is empty or not valid YAML")
        configs = resolve_model_configs(raw, builtins)
        pools = resolve_pools(raw)
        validate_model_pools(configs, pools)  # fail fast on a model referencing an undeclared pool
        validate_tp_against_pools(configs, pools)  # fail fast if tensor_parallel_size > pool GPU count
        validate_colocate(configs, COLOCATE_MAX_SHARE)  # fail fast on colocate+TP; warn on high share
        logging.info(f"Loaded {len(configs)} model(s) from {MODELS_CONFIG_FILE}: {list(configs)}")
        if pools:
            logging.info(f"Declared GPU pools: { {p: len(u) for p, u in pools.items()} }")
        return configs, pools

    if MODELS_CONFIG_FILE and os.path.isdir(MODELS_CONFIG_FILE):
        # A directory at the mount target usually means Docker created a missing file mount.
        logging.warning(f"MODELS_CONFIG_FILE '{MODELS_CONFIG_FILE}' is a directory, not a file. "
                        f"Falling back to ALLOWED_MODELS_JSON.")

    allowed = load_allowed_models()
    configs = build_fallback_configs(allowed, builtins)
    logging.info(f"No model config file at '{MODELS_CONFIG_FILE}'; using ALLOWED_MODELS_JSON fallback "
                 f"({len(configs)} model(s)): {list(configs)}")
    return configs, {}

# Resolve model configuration at import time so a bad config fails fast (refuses to start).
MODEL_CONFIGS, CONFIGURED_POOLS = load_model_configs()
# name -> repo map, preserved for existing lookups throughout the app.
ALLOWED_MODELS = {name: cfg.repo for name, cfg in MODEL_CONFIGS.items()}

# --- Background Tasks ---

async def shutdown_inactive_containers():
    """A background task that checks for inactivity and shuts down containers.

    Each container uses its own resolved inactivity_timeout. Containers whose model
    is marked always_on (or whose timeout is 0) are never auto-unloaded.
    """
    logging.info("Starting inactivity monitor (per-model timeouts; always_on models are never unloaded).")
    while True:
        try:
            await asyncio.sleep(60)

            # Determine which containers are inactive with lock
            async with model_management_lock:
                current_time = time.time()
                inactive_containers = []
                for name, state in active_containers.items():
                    if state.always_on:
                        continue
                    if state.inactivity_timeout <= 0:
                        continue  # 0 = never unload
                    if current_time - state.last_request_time > state.inactivity_timeout:
                        inactive_containers.append(name)

            # Stop containers outside lock (I/O operation)
            for name in inactive_containers:
                logging.info(f"Container {name} has been idle. Shutting down.")
                await stop_container(name)

        except Exception as e:
            logging.error(f"Error in inactivity monitor: {e}", exc_info=True)

async def stop_container(container_name: str, drain_timeout: float = 30.0):
    """Stops and removes a container with graceful draining of in-flight requests."""
    # Step 1: Remove from routing so no new requests are dispatched to it
    async with model_management_lock:
        state = active_containers.pop(container_name, None)

    # Step 2: Wait for any in-flight requests to complete before killing the container
    if state and state.active_requests > 0:
        logging.info(f"Draining {state.active_requests} in-flight request(s) from {container_name} (timeout {drain_timeout}s)...")
        deadline = time.time() + drain_timeout
        while state.active_requests > 0 and time.time() < deadline:
            await asyncio.sleep(0.2)
        if state.active_requests > 0:
            logging.warning(f"Container {container_name} still had {state.active_requests} active request(s) after drain timeout; force stopping.")

    # Step 3: Stop and remove the Docker container
    try:
        container = await run_in_executor(docker_client.containers.get, container_name)
        logging.info(f"Stopping container {container_name}...")
        await run_in_executor(container.stop)
        await run_in_executor(container.remove)
        logging.info(f"Container {container_name} stopped and removed.")
    except NotFound:
        logging.warning(f"Attempted to stop container {container_name}, but it was not found.")
    except APIError as e:
        logging.error(f"Error stopping or removing container {container_name}: {e}")

async def hf_repo_exists(repo_id: str) -> bool:
    """Return True if repo_id has a readable config.json on HuggingFace."""
    if not repo_id:
        return False
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    try:
        resp = await http_client.get(url, timeout=5)
        return resp.status_code == 200
    except Exception:
        return False

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
        response = await http_client.get(config_url, follow_redirects=True, timeout=httpx.Timeout(10.0))
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

async def estimate_weight_bytes(model_id: str) -> "int | None":
    """Best-effort estimate of a model's on-disk weight size in bytes, for the TP-fallback
    decision (how many GPUs a model's weights need). Returns None when it can't be determined.

    Reads the safetensors shard index (one small JSON GET, same pattern as get_model_max_len);
    that metadata already reflects quantized sizes. GGUF / non-safetensors repos return None,
    in which case the caller degrades to the configured tensor_parallel_size."""
    if is_gguf_model(model_id) or '/' not in model_id or model_id.startswith(('http://', 'https://')):
        return None
    index_url = f"https://huggingface.co/{model_id}/raw/main/model.safetensors.index.json"
    try:
        resp = await http_client.get(index_url, follow_redirects=True, timeout=httpx.Timeout(10.0))
        if resp.status_code == 200:
            total = resp.json().get("metadata", {}).get("total_size")
            if isinstance(total, int) and total > 0:
                return total
    except Exception as e:
        logging.warning(f"Could not estimate weight size for {model_id}: {e}")
    return None

# Flags the gateway computes from placement (util cap, TP degree, model path). A model's
# extra_args must NOT override these — doing so would defeat the co-location budget or the
# pool-validated tensor-parallel degree. They are dropped (with a warning) if present.
GATEWAY_MANAGED_FLAGS = {"--gpu-memory-utilization", "--tensor-parallel-size", "--model"}

def merge_extra_args(base: "list[str]", extra: "list[str]") -> "list[str]":
    """Append raw extra_args, letting them override any NON-managed flag the base set.

    Any '--flag' present in extra_args removes the gateway-generated occurrence of that
    flag (and its following value, if any) from base, so the explicit config value wins
    and no flag is duplicated — except GATEWAY_MANAGED_FLAGS, which the gateway owns and
    which are stripped from extra_args (with a warning) so placement decisions stand.
    """
    extra = [str(a) for a in extra]

    # Strip gateway-managed flags (and their values) from extra_args.
    cleaned = []
    i = 0
    while i < len(extra):
        tok = extra[i]
        if tok in GATEWAY_MANAGED_FLAGS:
            logging.warning(f"Ignoring gateway-managed flag '{tok}' in extra_args "
                            f"(it is set from placement and cannot be overridden).")
            if i + 1 < len(extra) and not extra[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        cleaned.append(tok)
        i += 1
    extra = cleaned

    extra_flags = {tok for tok in extra if tok.startswith("--")}
    if not extra_flags:
        return base + extra

    merged: "list[str]" = []
    i = 0
    while i < len(base):
        tok = base[i]
        if tok in extra_flags:
            # Drop this flag and its value (next token, unless that token is itself a flag).
            if i + 1 < len(base) and not base[i + 1].startswith("--"):
                i += 2
            else:
                i += 1
            continue
        merged.append(tok)
        i += 1
    merged.extend(extra)
    return merged

async def start_model_container(model_id: str, container_name: str, model_cfg: ModelConfig,
                                gpu_uuids: "list[str] | None" = None,
                                effective_tp: "int | None" = None,
                                effective_util: "float | None" = None) -> ContainerState | None:
    """Starts a new vLLM container using the model's resolved configuration.

    gpu_uuids pins the container to specific GPU(s) (multi-GPU placement). When None/empty,
    falls back to GPU_DEVICE_REQUESTS (the gateway-wide pin or all GPUs) for backward compat.
    effective_tp overrides --tensor-parallel-size (for TP fallback); effective_util overrides
    --gpu-memory-utilization (for co-location). Both default to the model's configured values.
    model_cfg is never mutated (it is shared across requests)."""
    tp = effective_tp if effective_tp else model_cfg.tensor_parallel_size
    util = effective_util if effective_util is not None else model_cfg.gpu_memory_utilization
    logging.info(f"Attempting to start model {model_id} in container {container_name}")

    # Clean up any existing container with the same name (from crashes or improper shutdowns)
    try:
        existing_container = await run_in_executor(docker_client.containers.get, container_name)
        logging.warning(f"Found existing container {container_name}. Removing it before starting new one.")
        try:
            await run_in_executor(existing_container.stop, timeout=10)
        except Exception as e:
            logging.warning(f"Could not stop existing container {container_name}: {e}")
        await run_in_executor(existing_container.remove, force=True)
        logging.info(f"Removed stale container {container_name}")

        # Verify container is actually removed before proceeding
        for attempt in range(10):
            try:
                await run_in_executor(docker_client.containers.get, container_name)
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

    # Detect "repo:quant_type" format where the repo is a GGUF-only repo (no config.json).
    # vLLM's native repo:quant format requires config.json, which GGUF-only repos lack.
    # Handle these ourselves: strip the quant suffix, download via the gateway, pass a local path.
    gguf_quant_hint = ""
    resolved_model_id = model_id
    if ':' in model_id and not model_id.startswith(('http://', 'https://')):
        repo_part, quant_part = model_id.rsplit(':', 1)
        if is_gguf_repo(repo_part):
            resolved_model_id = repo_part
            gguf_quant_hint = quant_part
            logging.info(f"Detected GGUF repo:quant format '{model_id}'. "
                         f"Will download '{resolved_model_id}' with quant hint '{gguf_quant_hint}'.")

    if is_gguf_repo(resolved_model_id):
        # Ensure only one download per model at a time (protect lock creation with global lock)
        async with model_management_lock:
            if model_id not in download_locks:
                download_locks[model_id] = asyncio.Lock()
            download_lock = download_locks[model_id]

        async with download_lock:
            logging.info(f"Detected GGUF repo: {resolved_model_id}. Downloading GGUF file...")
            actual_model_path, tokenizer_repo = await download_gguf_from_repo(resolved_model_id, gguf_quant_hint)
            # Translate host path to the path as seen inside the container.
            # HOST_CACHE_DIR is mounted at CONTAINER_CACHE_MOUNT inside every vLLM container.
            if actual_model_path.startswith(HOST_CACHE_DIR):
                actual_model_path = actual_model_path.replace(HOST_CACHE_DIR, CONTAINER_CACHE_MOUNT, 1)
            logging.info(f"Container model path: {actual_model_path}")

    # Only fetch and set max_model_len when this model has a configured cap (> 0).
    # When no cap is set (0), let vLLM auto-detect the correct value for the model.
    # The configured value is capped to the model's native max to avoid startup failures.
    final_max_len = 0
    if model_cfg.max_model_len > 0:
        model_max_len = await get_model_max_len(tokenizer_repo if tokenizer_repo else model_id)
        final_max_len = min(model_max_len, model_cfg.max_model_len) if model_max_len > 0 else model_cfg.max_model_len

    # --- Build vLLM command from the resolved per-model config ---
    # Always present: --model, --gpu-memory-utilization, --tensor-parallel-size.
    command = ["--model", actual_model_path,
               "--gpu-memory-utilization", str(round(util, 4))]
    if tp > 0:
        command.extend(["--tensor-parallel-size", str(tp)])

    # Add GGUF-specific parameters if this is a GGUF model
    if is_gguf_model(actual_model_path) or tokenizer_repo:
        # Verify the inferred tokenizer repo actually exists on HuggingFace before using it.
        # Third-party GGUF hosters (e.g. TheBloke) produce inferred names that don't exist.
        if tokenizer_repo and not await hf_repo_exists(tokenizer_repo):
            logging.warning(f"Inferred tokenizer repo '{tokenizer_repo}' not found on HuggingFace. "
                            f"vLLM will use the embedded GGUF tokenizer.")
            tokenizer_repo = None

        if tokenizer_repo:
            command.extend(["--tokenizer", tokenizer_repo])
            command.extend(["--hf-config-path", tokenizer_repo])
            logging.info(f"Using tokenizer and config from {tokenizer_repo} for GGUF model")
        else:
            tokenizer_path = extract_tokenizer_from_gguf_path(actual_model_path)
            if tokenizer_path and await hf_repo_exists(tokenizer_path):
                command.extend(["--tokenizer", tokenizer_path])
                command.extend(["--hf-config-path", tokenizer_path])
                logging.info(f"Using tokenizer and config from {tokenizer_path} for GGUF model {model_id}")
            else:
                logging.warning(f"No valid tokenizer found for GGUF model {model_id}. Using model's embedded tokenizer.")

    # Conditional config-driven flags.
    if final_max_len > 0:
        command.extend(["--max-model-len", str(final_max_len)])

    if model_cfg.quantization:
        command.extend(["--quantization", model_cfg.quantization])

    if model_cfg.dtype and model_cfg.dtype != "auto":
        command.extend(["--dtype", model_cfg.dtype])

    # Preserved global flag (not part of the per-model schema; override via extra_args).
    if int(VLLM_MAX_NUM_SEQS) > 0:
        command.extend(["--max-num-seqs", VLLM_MAX_NUM_SEQS])

    # Add gpt-oss specific optimizations for Ampere/Ada GPUs (RTX 3090, A100, etc)
    if is_gpt_oss_model(model_id):
        command.append("--async-scheduling")
        logging.info(f"Added --async-scheduling flag for gpt-oss model optimization")

    if VLLM_ENFORCE_EAGER or VLLM_NO_CUDAGRAPH:
        command.append("--enforce-eager")

    # Append raw per-model extra_args verbatim; explicit values override generated flags.
    if model_cfg.extra_args:
        command = merge_extra_args(command, model_cfg.extra_args)

    try:
        vllm_image = get_vllm_image_for_model(model_id)
        logging.info(f"Using vLLM image: {vllm_image} for model {model_id}")
        # Pin to the chosen GPU(s) when placement supplied them; else gateway-wide default.
        if gpu_uuids:
            device_requests = [DeviceRequest(device_ids=list(gpu_uuids), capabilities=[['gpu']])]
        else:
            device_requests = GPU_DEVICE_REQUESTS
        logging.info(f"Starting container {container_name} on GPU(s) {gpu_uuids or 'default'} "
                     f"with command: {' '.join(command)}")
        new_container = await run_in_executor(
            docker_client.containers.run,
            vllm_image,
            command=command,
            name=container_name,
            hostname=container_name,
            detach=True,
            network=RESOLVED_DOCKER_NETWORK,
            environment={
                "HUGGING_FACE_HUB_TOKEN": HF_TOKEN,
                "VLLM_ALLOW_LONG_MAX_MODEL_LEN": "1",
                "VLLM_CACHE_BUST": str(uuid.uuid4()),
            },
            ipc_mode="host",
            device_requests=device_requests,
            volumes={
                HOST_CACHE_DIR: {'bind': '/root/.cache/huggingface', 'mode': 'rw'},
                VLLM_TEMP_DIR: {'bind': '/tmp', 'mode': 'rw'}  # For temporary GGUF downloads
            }
        )
        await run_in_executor(new_container.reload)

        # Safely retrieve network IP address
        networks = new_container.attrs.get('NetworkSettings', {}).get('Networks', {})
        network_info = networks.get(RESOLVED_DOCKER_NETWORK)
        if not network_info or not network_info.get('IPAddress'):
            logging.error(f"Failed to get IP address for container {container_name} on network {RESOLVED_DOCKER_NETWORK}")
            logging.error(f"Available networks: {list(networks.keys())}")
            await run_in_executor(new_container.stop)
            await run_in_executor(new_container.remove)
            return None

        ip_address = network_info['IPAddress']

        # Health check loop with progress logging
        vllm_base_url = f"http://{ip_address}:{VLLM_PORT}"
        logging.info(f"Starting health checks for {model_id}. This may take several minutes for large models...")

        for i in range(1800): # ~1 hour timeout (1800 * 2s = 3600s)
            await asyncio.sleep(2)

            # Check container is still running every 20 seconds to detect crashes early
            if i % 10 == 0:
                try:
                    await run_in_executor(new_container.reload)
                    status = new_container.status
                    if status not in ('running', 'created'):
                        try:
                            logs = (await run_in_executor(new_container.logs, tail=50)).decode('utf-8', errors='replace')
                        except Exception:
                            logs = "(could not retrieve logs)"
                        logging.error(f"Container {container_name} exited with status '{status}' during startup. Last logs:\n{logs}")
                        await run_in_executor(new_container.remove, force=True)
                        return None
                except NotFound:
                    logging.error(f"Container {container_name} disappeared unexpectedly during startup.")
                    return None
                except Exception as e:
                    logging.warning(f"Could not check container status for {container_name}: {e}")

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
                        vram_footprint=0, # Footprint is unknown until measured
                        always_on=model_cfg.always_on,
                        inactivity_timeout=model_cfg.inactivity_timeout,
                        loaded_at=time.time(),
                        gpu_uuids=list(gpu_uuids) if gpu_uuids else [],
                        colocate=model_cfg.colocate,
                    )
                elif response.status_code != 503:
                    # 503 is expected during vLLM initialization; anything else is worth noting
                    logging.warning(f"Unexpected health check status {response.status_code} for {model_id} (attempt {i+1})")
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
async def gateway_status():
    """Returns the current status of the gateway and its managed containers.

    Snapshots the shared state under the locks that guard it, then builds the response from
    the copies — so a concurrent mutation can't raise 'dict changed size during iteration'
    (this endpoint runs in a threadpool, truly concurrent with the event loop)."""
    async with model_management_lock:
        async with placement_lock:
            containers = {name: dict(state.__dict__) for name, state in active_containers.items()}
            reservations = dict(gpu_reservations)
        gpu_vram = {u: dict(v) for u, v in GPU_VRAM.items()}
        pools = {p: list(u) for p, u in MANAGED_POOLS.items()}
        footprints = dict(known_footprints)
        managed = list(MANAGED_GPUS)
    async with queue_count_lock:
        queue_counts = dict(model_queue_counts)

    uuid_to_pool = {u: p for p, uuids in pools.items() for u in uuids}
    gpus_status = {}
    for guid in managed:
        v = gpu_vram.get(guid, {})
        residents = [c["container_name"] for c in containers.values() if guid in c.get("gpu_uuids", [])]
        gpus_status[guid] = {
            "pool": uuid_to_pool.get(guid),
            "total_mib": v.get("total"),
            "used_mib": v.get("used"),
            "reserved_mib": reservations.get(guid, 0.0),
            "residents": residents,
        }
    model_ids = set(list(queue_counts.keys()) + [c["model_id"] for c in containers.values()])
    return {
        "total_gpu_vram_mib": TOTAL_GPU_VRAM,
        "pools": pools,
        "gpus": gpus_status,
        "known_footprints_mib": footprints,
        "active_containers": containers,
        "queue_status": {
            model_id: {
                "queue_depth": queue_counts.get(model_id, 0),
                "max_concurrent": GATEWAY_MAX_CONCURRENT,
                "max_queue_size": GATEWAY_MAX_QUEUE_SIZE
            }
            for model_id in model_ids
        }
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
    model_cfg = MODEL_CONFIGS[model_name]

    # Initialize semaphore for this model if not exists
    if target_model_id not in model_semaphores:
        async with model_management_lock:
            if target_model_id not in model_semaphores:
                model_semaphores[target_model_id] = asyncio.Semaphore(GATEWAY_MAX_CONCURRENT)
                model_queue_counts[target_model_id] = 0

    # Check if queue is full and increment counter atomically
    async with queue_count_lock:
        current_queue_depth = model_queue_counts.get(target_model_id, 0)
        if current_queue_depth >= GATEWAY_MAX_QUEUE_SIZE:
            # Queue is full - reject request with 429 Too Many Requests
            logging.warning(f"Queue full for {model_name} ({target_model_id}). Rejecting request. Queue depth: {current_queue_depth}/{GATEWAY_MAX_QUEUE_SIZE}")
            headers = {
                "X-Queue-Depth": str(current_queue_depth),
                "X-Queue-Max-Size": str(GATEWAY_MAX_QUEUE_SIZE),
                "Retry-After": "30"  # Suggest retry after 30 seconds
            }
            return JSONResponse(
                {
                    "error": "Gateway queue is full",
                    "details": f"Model {model_name} has {current_queue_depth} requests in queue (max: {GATEWAY_MAX_QUEUE_SIZE})",
                    "queue_depth": current_queue_depth,
                    "max_queue_size": GATEWAY_MAX_QUEUE_SIZE,
                    "retry_after_seconds": 30
                },
                status_code=429,
                headers=headers
            )

        # Increment queue counter atomically
        model_queue_counts[target_model_id] = current_queue_depth + 1
        logging.debug(f"Request queued for {model_name} ({target_model_id}). Queue depth: {current_queue_depth + 1}/{GATEWAY_MAX_QUEUE_SIZE}")

    # Track whether we need to decrement counter in exception handler
    # We track the INCREMENT (which always happens), not the decrement (which may fail)
    counter_needs_cleanup = True

    # Acquire semaphore manually (not via `async with`) so we can transfer ownership
    # of the release to the streaming generator. With `async with`, the slot is released
    # when proxy_request returns — before a single streaming byte is sent — making
    # GATEWAY_MAX_CONCURRENT ineffective for streaming workloads.
    sem = model_semaphores[target_model_id]
    sem_released = False  # True once the streaming generator takes ownership of sem.release()

    # Step 1: acquire — CancelledError here means the semaphore is NOT held
    try:
        await sem.acquire()
    except BaseException:
        if counter_needs_cleanup:
            async with queue_count_lock:
                model_queue_counts[target_model_id] = max(0, model_queue_counts[target_model_id] - 1)
                logging.debug(f"Exception cleanup (pre-acquire): decremented queue counter for {model_name}. Queue depth: {model_queue_counts[target_model_id]}/{GATEWAY_MAX_QUEUE_SIZE}")
        raise

    # Step 2: semaphore is held — run all logic; finally releases it unless generator took over
    try:
        # Decrement queue counter now that we have a semaphore slot
        # CRITICAL: Mark cleanup flag BEFORE decrement to prevent double-decrement
        # if logging.debug() fails after decrement but before flag is set
        async with queue_count_lock:
            counter_needs_cleanup = False  # Set flag FIRST
            model_queue_counts[target_model_id] = max(0, model_queue_counts[target_model_id] - 1)
            logging.debug(f"Request dequeued for {model_name} ({target_model_id}). Queue depth: {model_queue_counts[target_model_id]}/{GATEWAY_MAX_QUEUE_SIZE}")

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

                        if TOTAL_GPU_VRAM > 0:
                            # POOL-BASED PLACEMENT (whole-card by default; tensor-parallel and
                            # co-location handled below).
                            # known_footprints: None = never seen (discovery), 0 = seen-but-unmeasurable
                            # sentinel, >0 = measured. is_discovery only when truly unseen.
                            is_discovery = footprint is None
                            pool = pool_for(model_cfg)
                            pool_gpus = MANAGED_POOLS.get(pool, [])
                            if not pool_gpus:
                                raise HTTPException(status_code=500,
                                                    detail=f"Pool '{pool}' for model {target_model_id} has no managed GPUs")

                            # Snapshot per-GPU VRAM OUTSIDE the placement lock (it spawns a container).
                            gpus_snapshot = await get_gpu_vram()

                            pool_totals = [float(gpus_snapshot.get(u, {}).get("total", 0.0)) for u in pool_gpus]
                            max_total = max(pool_totals) if pool_totals else 0.0
                            is_colocate = model_cfg.colocate

                            # Per-card need. Co-locatable models use their intended share (so several pack
                            # onto a card); whole-card models use the learned footprint or a full-card estimate.
                            if is_colocate:
                                needed_per_gpu = model_cfg.gpu_memory_utilization * max_total
                            elif footprint and footprint > 0:
                                needed_per_gpu = float(footprint)
                            else:
                                needed_per_gpu = model_cfg.gpu_memory_utilization * max_total

                            # Effective TP: forced by config, or auto-fallback for an unseen (non-colocate)
                            # model whose weights exceed one card (only in a homogeneous, multi-GPU pool).
                            effective_tp = model_cfg.tensor_parallel_size
                            if (is_discovery and not is_colocate and model_cfg.tensor_parallel_size == 1
                                    and len(pool_gpus) >= 2
                                    and pool_totals and (max(pool_totals) - min(pool_totals)) <= 0.05 * max(pool_totals)):
                                wbytes = await estimate_weight_bytes(target_model_id)
                                min_tp = placement.minimal_tp_to_fit(
                                    wbytes, max_total, model_cfg.gpu_memory_utilization, pool_size=len(pool_gpus))
                                if min_tp > effective_tp:
                                    logging.info(f"TP fallback for {target_model_id}: weights ~{wbytes} B -> tp={min_tp}")
                                    effective_tp = min_tp

                            # Co-location needs a weight estimate to size the launch util safely (best-effort).
                            colocate_wbytes = await estimate_weight_bytes(target_model_id) if is_colocate else None

                            # --- DECIDE: choose GPU(s) + reserve, under the global placement lock ---
                            effective_util = None
                            async with model_management_lock:
                                async with placement_lock:
                                    candidates = []
                                    residents_by_gpu = {}
                                    for guid in pool_gpus:
                                        g = gpus_snapshot.get(guid)
                                        residents = [c for c in active_containers.values() if guid in c.gpu_uuids]
                                        residents_by_gpu[guid] = residents
                                        candidates.append(placement.GpuView(
                                            uuid=guid,
                                            total=float(g["total"]) if g else 0.0,
                                            used_smi=float(g["used"]) if g else 0.0,
                                            reserved=gpu_reservations.get(guid, 0.0),
                                            ready_footprint=sum(c.vram_footprint for c in residents),
                                        ))

                                    if is_colocate:
                                        uuid, evictions = placement.select_colocated(
                                            candidates, residents_by_gpu, needed_per_gpu,
                                            time.time(), GATEWAY_MIN_RESIDENT_SECONDS)
                                        chosen_uuids = [uuid] if uuid is not None else None
                                    else:
                                        chosen_uuids, evictions = placement.select_placement(
                                            candidates, residents_by_gpu, needed_per_gpu, effective_tp,
                                            time.time(), GATEWAY_MIN_RESIDENT_SECONDS)
                                    if chosen_uuids is None:
                                        raise HTTPException(
                                            status_code=503,
                                            detail=(f"Cannot place model {target_model_id} in pool '{pool}' "
                                                    f"({'colocate' if is_colocate else f'tp={effective_tp}'}, ~{int(needed_per_gpu)} MiB/GPU): "
                                                    f"no GPU available without evicting always_on/in-flight models "
                                                    f"(or pool not homogeneous / too few GPUs for TP). Retry later."))

                                    # Co-location: size the launch util from the chosen card's free space
                                    # (after this model's evictions), capped by the configured share and a margin.
                                    chosen_gv = next(c for c in candidates if c.uuid == chosen_uuids[0])
                                    if is_colocate:
                                        freed = sum(r.vram_footprint for r in residents_by_gpu[chosen_uuids[0]]
                                                    if r.container_name in evictions)
                                        post_free = chosen_gv.free + freed
                                        total = chosen_gv.total or max_total
                                        budget = max(0.0, post_free - COLOCATE_MARGIN_MIB)
                                        effective_util = min(model_cfg.gpu_memory_utilization,
                                                             (budget / total) if total > 0 else model_cfg.gpu_memory_utilization)
                                        # Weights floor: refuse a budget that can't hold the weights (would OOM).
                                        if colocate_wbytes:
                                            need_mib = (colocate_wbytes / (1024.0 * 1024.0)) * COLOCATE_WEIGHT_OVERHEAD
                                            if effective_util * total < need_mib:
                                                raise HTTPException(
                                                    status_code=503,
                                                    detail=(f"Cannot co-locate {target_model_id} on {chosen_uuids[0]}: budget "
                                                            f"~{int(effective_util * total)} MiB < weights ~{int(need_mib)} MiB. Retry later."))
                                        reserve_amt = effective_util * total
                                    else:
                                        reserve_amt = needed_per_gpu

                                    for u in chosen_uuids:
                                        gpu_reservations[u] = gpu_reservations.get(u, 0.0) + reserve_amt
                                    slot_indices = {int(name.split('_')[-1]) for name in active_containers.keys()}
                                    free_slot = next(i for i in range(len(slot_indices) + 1) if i not in slot_indices)
                                    container_name = f"{VLLM_CONTAINER_PREFIX}_{free_slot}"
                                    logging.info(f"Placing {target_model_id} on GPU(s) {chosen_uuids} (pool '{pool}', "
                                                 f"{'colocate util=%.3f' % effective_util if is_colocate else 'tp=%d' % effective_tp}, "
                                                 f"~{int(reserve_amt)} MiB/GPU); evicting {evictions or 'nothing'}; slot {container_name}.")

                            # --- Evictions + start happen OUTSIDE the locks (I/O) ---
                            for name in evictions:
                                await stop_container(name)

                            # Discovery (whole-card models only): measure the per-GPU footprint delta.
                            run_discovery = is_discovery and not is_colocate
                            before_used = gpus_snapshot.get(chosen_uuids[0], {}).get("used", 0.0) if run_discovery else 0.0

                            try:
                                target_container = await start_model_container(
                                    target_model_id, container_name, model_cfg,
                                    gpu_uuids=chosen_uuids, effective_tp=effective_tp, effective_util=effective_util)
                                if target_container:
                                    # Seed footprint so concurrent placement counts these GPU(s) as occupied.
                                    # Co-location: deterministic (vLLM grabs ~util*total). Whole-card: refined by discovery.
                                    target_container.vram_footprint = reserve_amt
                                    async with model_management_lock:
                                        active_containers[container_name] = target_container
                            finally:
                                # Always release the optimistic reservation on ALL chosen GPUs.
                                async with placement_lock:
                                    for u in chosen_uuids:
                                        gpu_reservations[u] = max(0.0, gpu_reservations.get(u, 0.0) - reserve_amt)

                            if target_container and run_discovery:
                                meas_gpu = chosen_uuids[0]
                                logging.info(f"Discovery: sampling GPU {meas_gpu} VRAM 3x over 45s...")
                                samples = []
                                for _ in range(3):
                                    await asyncio.sleep(15)
                                    samples.append((await get_gpu_vram()).get(meas_gpu, {}).get("used", 0.0))
                                measured = max(samples) - before_used
                                if measured > 256:
                                    logging.info(f"Measured per-GPU footprint for {target_model_id} on {meas_gpu}: {measured} MiB")
                                    target_container.vram_footprint = measured
                                    known_footprints[target_model_id] = measured
                                else:
                                    logging.warning(f"Could not measure footprint for {target_model_id} ({measured} MiB); "
                                                    f"keeping whole-card estimate {int(needed_per_gpu)} MiB.")
                                    known_footprints[target_model_id] = 0  # sentinel: seen but unmeasurable
                                await save_known_footprints_async()

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
                            target_container = await start_model_container(target_model_id, container_name, model_cfg)
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

            # Get current queue depth for headers
            current_queue_depth = model_queue_counts.get(target_model_id, 0)

            # Track in-flight requests for graceful container draining.
            # For streaming, the generator owns the decrement (it runs after this function returns).
            # For non-streaming, the finally block below decrements.
            target_container.active_requests += 1
            active_req_decremented = False

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

                # Add queue status headers to response
                queue_headers = {
                    "X-Queue-Depth": str(current_queue_depth),
                    "X-Max-Concurrent": str(GATEWAY_MAX_CONCURRENT),
                    "X-Max-Queue-Size": str(GATEWAY_MAX_QUEUE_SIZE)
                }

                if is_streaming:
                    # True streaming: open the connection and yield chunks as they arrive.
                    # http_client.stream() keeps the connection alive while the generator runs.
                    # The generator owns BOTH active_requests decrement AND semaphore release,
                    # so both fire after the last byte is sent (or client disconnects), not when
                    # StreamingResponse is constructed and proxy_request returns.
                    captured_container = target_container
                    captured_sem = sem

                    async def stream_generate():
                        try:
                            async with http_client.stream(
                                request.method, vllm_url,
                                json=body, headers=headers_to_forward
                            ) as r:
                                async for chunk in r.aiter_bytes():
                                    if chunk:
                                        yield chunk
                        finally:
                            captured_container.active_requests -= 1
                            captured_sem.release()  # enforce GATEWAY_MAX_CONCURRENT for full stream duration

                    sem_released = True        # generator takes ownership of sem.release()
                    active_req_decremented = True  # generator owns active_requests decrement

                    return StreamingResponse(
                        stream_generate(),
                        status_code=200,
                        media_type="text/event-stream",
                        headers=queue_headers
                    )
                else:
                    # Non-streaming: buffer full response, then return.
                    # Retry only transient connection errors; HTTP errors are not retried.
                    max_retries = 3
                    retry_delay = 1.0  # seconds

                    for retry_attempt in range(max_retries):
                        try:
                            response = await http_client.request(
                                method=request.method,
                                url=vllm_url,
                                json=body,
                                headers=headers_to_forward
                            )
                            response.raise_for_status()
                            break  # Success - exit retry loop
                        except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                            if retry_attempt < max_retries - 1:
                                logging.warning(f"Transient connection error to vLLM for {model_name} (attempt {retry_attempt + 1}/{max_retries}): {type(e).__name__}. Retrying in {retry_delay}s...")
                                await asyncio.sleep(retry_delay)
                                retry_delay *= 1.5  # Exponential backoff
                            else:
                                logging.error(f"Failed to connect to vLLM for {model_name} after {max_retries} attempts: {type(e).__name__}: {str(e)}")
                                raise  # Re-raise to trigger httpx.RequestError handler below

                    try:
                        content = response.json()
                    except Exception:
                        content = {"error": "Non-JSON response from vLLM", "raw": response.text[:500]}
                        logging.warning(f"Non-JSON response from vLLM for {model_name}: status={response.status_code}, body={response.text[:200]}")
                    return JSONResponse(
                        content=content,
                        status_code=response.status_code,
                        headers=queue_headers
                    )
            except httpx.HTTPStatusError as e:
                logging.error(f"vLLM returned HTTP error for {model_name} ({target_model_id}): {e.response.status_code} - {str(e)}")
                return JSONResponse(
                    {"error": "Error from vLLM service", "details": str(e)},
                    status_code=e.response.status_code,
                    headers={"X-Queue-Depth": str(current_queue_depth)}
                )
            except httpx.RequestError as e:
                logging.error(f"Connection error to vLLM for {model_name} ({target_model_id}) at {vllm_url}: {type(e).__name__}: {str(e)}")
                raise HTTPException(status_code=503, detail=f"Could not connect to vLLM service: {e}")
            finally:
                if not active_req_decremented:
                    target_container.active_requests -= 1
    except BaseException:
        # Exception cleanup: If the counter was incremented but never decremented
        # (because exception occurred before or during the decrement operation),
        # we need to decrement it here to avoid leaking the queue counter.
        # Uses BaseException (not Exception) to also catch asyncio.CancelledError,
        # which is raised when uvicorn cancels a task during graceful shutdown.
        if counter_needs_cleanup:
            async with queue_count_lock:
                model_queue_counts[target_model_id] = max(0, model_queue_counts[target_model_id] - 1)
                logging.debug(f"Exception cleanup: decremented queue counter for {model_name} ({target_model_id}). Queue depth: {model_queue_counts[target_model_id]}/{GATEWAY_MAX_QUEUE_SIZE}")
        raise
    finally:
        # Release semaphore for all non-streaming exits (success and error).
        # Streaming requests set sem_released=True so their generator handles release.
        if not sem_released:
            sem.release()
