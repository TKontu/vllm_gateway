import os
import asyncio
import httpx
import docker
import json
import math
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
from enum import Enum
from huggingface_hub import hf_hub_download, list_repo_files, HfApi
import yaml
import placement
from config_loader import (
    ModelConfig, resolve_model_configs, build_fallback_configs, merge_request_defaults,
    resolve_pools, validate_model_pools, validate_tp_against_pools, validate_colocate,
    validate_pools_visible, validate_budget_mode, validate_extra_args_budget, migrate_footprints,
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

# Optional LD_LIBRARY_PATH for the spawned vLLM worker containers. Leave unset for the normal
# case (the image's bundled cuda-compat libcuda is correct when the host driver is OLDER than the
# image's CUDA). Set it when the host driver is NEWER than the image's cuda-compat lib: the compat
# libcuda then can't talk to the newer kernel module and CUDA init fails with error 803
# (cudaErrorSystemDriverMismatch). Pointing the loader at the host driver libs the NVIDIA Container
# Toolkit mounts at /usr/lib/x86_64-linux-gnu makes the worker use the real (host) libcuda, e.g.:
#   WORKER_LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/nvidia/lib64:/usr/local/cuda/lib64
WORKER_LD_LIBRARY_PATH = os.getenv("WORKER_LD_LIBRARY_PATH", "").strip()

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

# A LOADING entry older than this with no completed start is treated as orphaned (the start crashed
# without running its cleanup) and reaped by the reconciler. Defaults a bit above the health-check
# budget (~1h) so a genuinely slow cold start is never reaped mid-flight.
GATEWAY_LOADING_TIMEOUT = int(os.getenv("GATEWAY_LOADING_TIMEOUT", "3900"))

# A READY container whose active_requests has been > 0 while idle for longer than
# GATEWAY_REQUEST_STALE_FACTOR x GATEWAY_REQUEST_TIMEOUT is assumed to have a leaked in-flight count
# (e.g. an undriven streaming generator) and is clamped to 0 by the reconciler so it can be evicted.
GATEWAY_REQUEST_STALE_FACTOR = int(os.getenv("GATEWAY_REQUEST_STALE_FACTOR", "2"))

# Co-location (Phase 3): when a co-locatable model shares a card, its launch util is capped so it
# leaves COLOCATE_MARGIN_MIB free; weights must fit util*total * COLOCATE_WEIGHT_OVERHEAD; a
# colocate model whose share exceeds COLOCATE_MAX_SHARE is warned about at startup.
COLOCATE_MARGIN_MIB = int(os.getenv("COLOCATE_MARGIN_MIB", "1024"))
COLOCATE_WEIGHT_OVERHEAD = float(os.getenv("COLOCATE_WEIGHT_OVERHEAD", "1.15"))
COLOCATE_MAX_SHARE = float(os.getenv("COLOCATE_MAX_SHARE", "0.9"))

# --- Placement mode (Phase 4) ---
# 'budget'     : size each model to weights+KV+overhead and pack many models per card up to a
#                per-GPU budget cap. Requires max_model_len > 0 per model (bounds the KV need).
# 'whole_card' : legacy behavior — one model fills the card at its util; swap via LRU eviction.
PLACEMENT_MODE = os.getenv("PLACEMENT_MODE", "budget").strip().lower()
# Per-GPU budget: fraction of each card the gateway may fill IN TOTAL across all its models.
# Defaults to the legacy global utilization so an existing deployment keeps the same ceiling.
GPU_BUDGET_FRACTION = float(os.getenv("GPU_BUDGET_FRACTION", VLLM_GPU_MEMORY_UTILIZATION))
# Need-estimate cushion: (weights + KV) * factor + a fixed per-card margin (CUDA context, cudagraphs).
# Bias generous — the launch util cap means an under-estimate only fails THIS model's own startup,
# never a co-resident, so erring large is safe.
BUDGET_OVERHEAD_FACTOR = float(os.getenv("BUDGET_OVERHEAD_FACTOR", "1.1"))
BUDGET_OVERHEAD_MIB = int(os.getenv("BUDGET_OVERHEAD_MIB", "1024"))

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
    if PLACEMENT_MODE not in ("budget", "whole_card"):
        raise ValueError(f"PLACEMENT_MODE must be 'budget' or 'whole_card', got {PLACEMENT_MODE!r}")
    if not (0 < GPU_BUDGET_FRACTION <= 1):
        raise ValueError(f"GPU_BUDGET_FRACTION must be in (0, 1], got {GPU_BUDGET_FRACTION}")
    if BUDGET_OVERHEAD_FACTOR < 1:
        raise ValueError(f"BUDGET_OVERHEAD_FACTOR must be >= 1, got {BUDGET_OVERHEAD_FACTOR}")
    if BUDGET_OVERHEAD_MIB < 0:
        raise ValueError(f"BUDGET_OVERHEAD_MIB must be >= 0, got {BUDGET_OVERHEAD_MIB}")

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
known_footprints = {}  # repo -> {"per_gpu_mib": float, "effective_tp": int, "measured_at": float}
active_containers = {}  # container_name -> ContainerState (entries exist while LOADING/READY/STOPPING)
# One lock guards ALL of active_containers: membership, status transitions, slot allocation, and the
# VRAM accounting derived from it (reservations now live ON the entries, not in a side dict). It is
# held ONLY for in-memory decisions/mutations — NEVER across a container start, get_gpu_vram, or
# stop_container. Acquire order: container_start_locks[model] -> state_lock.
state_lock = asyncio.Lock()
container_start_locks = {}  # model_id -> asyncio.Lock for preventing concurrent container starts
download_locks = {}  # model_id -> asyncio.Lock for preventing concurrent downloads
# container_name -> the asyncio.Task currently starting that LOADING entry. The reconciler reaps a
# LOADING entry only when its owner task is absent/done (a true orphan), never on a timer — so a
# legitimately slow start (large download) is never killed. Kept off ContainerState so /status stays
# JSON-clean.
loading_tasks = {}

# Multi-GPU placement state (resolved at startup; see resolve_managed_pools)
MANAGED_POOLS = {}  # pool_name -> [gpu_uuid, ...]: the GPUs this gateway manages, grouped into pools
MANAGED_GPUS = []   # flat list of managed gpu_uuids (union of MANAGED_POOLS values)

# Queue management state
model_semaphores = {}  # model_id -> asyncio.Semaphore for limiting concurrent requests
model_queue_counts = {}  # model_id -> int (number of requests waiting in queue)
queue_count_lock = asyncio.Lock()  # Protects ONLY the queue counter (short critical sections)

class ContainerStatus(str, Enum):
    LOADING = "loading"    # reserved + container starting; visible to placement/accounting, NOT routable
    READY = "ready"        # health-passed; routable
    STOPPING = "stopping"  # being drained/removed; excluded from fit math, NOT routable

@dataclass
class ContainerState:
    model_id: str
    container_name: str
    status: "ContainerStatus"
    gpu_uuids: list                 # GPU UUID(s) this container is pinned to (set at LOADING)
    reserved_mib: float             # VRAM reservation while LOADING (counted via GpuView.reserved). Once
                                    # READY, accounting uses vram_footprint instead; reserved_mib is retained
                                    # only as the seed/estimate and is not re-read.
    effective_tp: int = 1           # tensor-parallel degree actually launched
    effective_util: float = 0.0     # gpu-memory-utilization actually launched (0 => model default)
    colocate: bool = False          # may share its GPU with other co-locatable models (Phase 3)
    always_on: bool = False         # if True, never auto-unloaded by the inactivity monitor
    inactivity_timeout: int = VLLM_INACTIVITY_TIMEOUT  # per-model idle timeout (seconds; 0 = never unload)
    created_at: float = 0.0         # time.time() when the LOADING entry was inserted (reconciler orphan reaping)
    # Populated when READY:
    ip_address: str = ""
    port: int = 0
    vram_footprint: float = 0.0     # measured/known per-GPU footprint (meaningful once READY)
    last_request_time: float = 0.0
    active_requests: int = 0        # in-flight requests being proxied to this container
    loaded_at: float = 0.0          # time.time() when it became READY (anti-thrash cooldown)

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

async def run_nvidia_smi_in_container(command: list[str], pid_mode: "str | None" = None) -> str:
    """Runs an nvidia-smi command in a temporary container and returns the output.

    The probe always requests ALL GPUs (count=-1) and the caller filters to the managed set.
    This keeps per-GPU accounting complete regardless of the GATEWAY_GPU_UUID pin (so pools
    win over the pin), and avoids a single bad configured UUID erroring the whole probe.

    pid_mode='host' is REQUIRED for --query-compute-apps: without the host PID namespace,
    nvidia-smi inside a container cannot enumerate compute processes and returns an empty list
    (so per-process VRAM attribution silently fails). The memory.total/used queries don't need it."""
    probe_requests = [DeviceRequest(count=-1, capabilities=[['gpu']])]
    run_kwargs = dict(command=command, remove=True, device_requests=probe_requests)
    if pid_mode:
        run_kwargs["pid_mode"] = pid_mode
    try:
        smi_output = await run_in_executor(
            docker_client.containers.run,
            NVIDIA_UTILITY_IMAGE,
            **run_kwargs
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
                raw = json.load(f)
            # Normalize to the record shape {repo: {per_gpu_mib, effective_tp, measured_at}},
            # migrating the legacy {repo: number} format forward (assume tp=1).
            known_footprints = migrate_footprints(raw)
            logging.info(f"Loaded known model footprints ({len(known_footprints)} record(s)).")
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

async def get_compute_apps_vram() -> "list[tuple]":
    """Per-process GPU memory: list of (gpu_uuid, host_pid, used_mib) for every compute process.

    Same probe-container pattern as get_gpu_vram (sees all GPUs). PIDs are HOST pids, which line up
    with `docker top` / container State.Pid. Returns [] when compute-apps is unsupported/empty
    (older drivers, MIG) so the caller falls back to delta measurement."""
    output = await run_nvidia_smi_in_container(
        ["nvidia-smi", "--query-compute-apps=gpu_uuid,pid,used_memory", "--format=csv,noheader,nounits"],
        pid_mode="host",  # required so nvidia-smi can enumerate compute processes (see A2 returns nothing)
    )
    rows = []
    for line in output.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) != 3:
            continue
        guid, pid, mib = parts
        if guid and pid.isdigit():
            rows.append((guid, int(pid), _parse_float(mib)))  # mib is "[N/A]" sometimes -> 0.0
    return rows

async def container_host_pids(container) -> set:
    """Host PIDs of every process in a container (covers vLLM's EngineCore worker children).

    Uses `docker top` (host-PID view, independent of the container's PID namespace) plus the
    container's init PID. Returns an empty set on failure (caller falls back)."""
    pids = set()
    try:
        await run_in_executor(container.reload)
        init_pid = container.attrs.get("State", {}).get("Pid")
        if init_pid:
            pids.add(int(init_pid))
        top = await run_in_executor(container.top)
        titles = (top or {}).get("Titles") or []
        procs = (top or {}).get("Processes") or []
        pid_idx = titles.index("PID") if "PID" in titles else 1
        for row in procs:
            try:
                pids.add(int(row[pid_idx]))
            except (ValueError, IndexError, TypeError):
                continue
    except Exception as e:
        logging.warning(f"Could not enumerate container PIDs: {e}")
    return pids

async def measure_model_vram(container_name: str, gpu_uuids: list) -> "float | None":
    """Ground-truth VRAM (MiB) actually used by a model's container across its gpu_uuids, via
    per-process attribution (nvidia-smi compute-apps -> the container's host PIDs). Returns None when
    attribution is unavailable (no compute-apps support, PID mapping empty, or zero match) so the
    caller can fall back to delta measurement."""
    try:
        container = await run_in_executor(docker_client.containers.get, container_name)
        pids = await container_host_pids(container)
        if not pids:
            return None
        rows = await get_compute_apps_vram()
        if not rows:
            return None
        mib = placement.attribute_vram(rows, pids, gpu_uuids or [])
        return mib if mib > 0 else None
    except Exception as e:
        logging.warning(f"Per-process VRAM attribution failed for {container_name}: {e}")
        return None

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
        # A non-positive VLLM_MAX_NUM_SEQS used to mean "omit the flag"; max_num_seqs is now a real
        # per-model setting (>= 1) and bounds KV need in budget mode, so coerce 0/invalid to 16.
        "max_num_seqs": (int(VLLM_MAX_NUM_SEQS) if int(VLLM_MAX_NUM_SEQS) >= 1 else 16),
        "kv_reservation_seqs": None,  # None -> reserve KV for max_num_seqs (today's behavior)
        "quantization": None,
        "dtype": "auto",
        "inactivity_timeout": VLLM_INACTIVITY_TIMEOUT,
        "always_on": False,
        "extra_args": [],
        "pool": None,
        "colocate": False,
        "request_defaults": {},
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
        validate_budget_mode(configs, PLACEMENT_MODE)  # fail fast: budget mode needs max_model_len > 0
        validate_extra_args_budget(configs, PLACEMENT_MODE)  # fail fast: no memory flags in extra_args
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
    validate_budget_mode(configs, PLACEMENT_MODE)  # fail fast: budget mode needs max_model_len > 0
    validate_extra_args_budget(configs, PLACEMENT_MODE)  # fail fast: no memory flags in extra_args
    logging.info(f"No model config file at '{MODELS_CONFIG_FILE}'; using ALLOWED_MODELS_JSON fallback "
                 f"({len(configs)} model(s)): {list(configs)}")
    return configs, {}

# Resolve model configuration at import time so a bad config fails fast (refuses to start).
MODEL_CONFIGS, CONFIGURED_POOLS = load_model_configs()
# name -> repo map, preserved for existing lookups throughout the app.
ALLOWED_MODELS = {name: cfg.repo for name, cfg in MODEL_CONFIGS.items()}

# --- Background Tasks ---

async def shutdown_inactive_containers():
    """Background monitor + reconciler.

    Inactivity: each READY container uses its own resolved inactivity_timeout; always_on (or
    timeout 0) is never unloaded.

    Reconciliation (self-healing):
      - A LOADING entry whose owning start task is absent/done (a true orphan — the start coroutine
        died without running its cleanup) is dropped, reclaiming its reservation/slot. We use owner
        liveness, NOT elapsed time, so a legitimately slow start (large download) is never killed
        (G2). GATEWAY_LOADING_TIMEOUT is only a generous last-resort ceiling.
      - A READY container with a leaked active_requests>0 whose last_request_time is stale (older
        than GATEWAY_REQUEST_STALE_FACTOR x GATEWAY_REQUEST_TIMEOUT) is clamped to 0 so it can be
        evicted again — best-effort backstop for an undriven streaming generator (G7). The
        generator's finally remains the primary decrement.
    """
    logging.info("Starting inactivity monitor + reconciler (owner-liveness reaping; always_on never unloaded).")
    stale_after = GATEWAY_REQUEST_STALE_FACTOR * GATEWAY_REQUEST_TIMEOUT
    while True:
        try:
            await asyncio.sleep(60)
            now = time.time()
            inactive_containers = []
            orphans = []
            async with state_lock:
                for name, state in active_containers.items():
                    if state.status == ContainerStatus.LOADING:
                        task = loading_tasks.get(name)
                        # Orphan iff no live owner. Belt-and-suspenders: also reap if it has somehow
                        # outlived the absolute ceiling.
                        if (task is None or task.done()) or (now - state.created_at > GATEWAY_LOADING_TIMEOUT):
                            orphans.append(name)
                        continue
                    if state.status != ContainerStatus.READY:
                        continue  # STOPPING is owned by stop_container
                    # G7: self-heal a leaked in-flight count on an otherwise-idle container.
                    if state.active_requests > 0 and (now - state.last_request_time) > stale_after:
                        logging.warning(f"Clamping stale active_requests={state.active_requests} on {name} "
                                        f"(idle {int(now - state.last_request_time)}s) -> 0.")
                        state.active_requests = 0
                    if state.always_on or state.inactivity_timeout <= 0:
                        continue
                    if now - state.last_request_time > state.inactivity_timeout:
                        inactive_containers.append(name)
                # Reap orphaned LOADING entries in-place (no container to stop — start never finished).
                for name in orphans:
                    logging.warning(f"Reaping orphaned LOADING entry {name} (owner task absent/done); "
                                    f"reclaiming its reservation.")
                    active_containers.pop(name, None)
                    loading_tasks.pop(name, None)

            # Stop idle containers outside the lock (I/O).
            for name in inactive_containers:
                logging.info(f"Container {name} has been idle. Shutting down.")
                await stop_container(name)

        except Exception as e:
            logging.error(f"Error in inactivity monitor: {e}", exc_info=True)

async def stop_container(container_name: str, drain_timeout: float = 30.0):
    """Gracefully stop+remove a container: flip STOPPING (off routing + fit math), drain in-flight,
    docker stop/remove, then drop the entry. The entry stays present (STOPPING) during the drain so
    no new request routes to it and its VRAM isn't double-freed in accounting (still in used_smi)."""
    # Step 1: claim the teardown. If the entry is already STOPPING, another caller owns it -> return
    # (idempotent; avoids a double docker stop/remove and double drain). G8.
    async with state_lock:
        state = active_containers.get(container_name)
        if state is not None:
            if state.status == ContainerStatus.STOPPING:
                return  # already being torn down by another caller
            state.status = ContainerStatus.STOPPING

    # Step 2: wait for in-flight requests to finish before killing the container.
    if state and state.active_requests > 0:
        logging.info(f"Draining {state.active_requests} in-flight request(s) from {container_name} (timeout {drain_timeout}s)...")
        deadline = time.time() + drain_timeout
        while state.active_requests > 0 and time.time() < deadline:
            await asyncio.sleep(0.2)
        if state.active_requests > 0:
            logging.warning(f"Container {container_name} still had {state.active_requests} active request(s) after drain timeout; force stopping.")

    # Step 3: stop and remove the Docker container.
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
    finally:
        # Step 4: drop the entry (single removal site for a STOPPING entry).
        async with state_lock:
            active_containers.pop(container_name, None)

def _hf_auth_headers() -> dict:
    """Authorization header for huggingface.co requests when a token is configured.

    Gated repos (e.g. google/gemma-*) return 401 on even raw config.json without auth, which would
    make the KV-spec / max-len lookups silently fail and the model fall back to whole-card discovery.
    The token is accepted (and ignored) on public repos, so it's always safe to send."""
    tok = HF_TOKEN if HF_TOKEN and HF_TOKEN.strip() else None
    return {"Authorization": f"Bearer {tok}"} if tok else {}

async def hf_repo_exists(repo_id: str) -> bool:
    """Return True if repo_id has a readable config.json on HuggingFace."""
    if not repo_id:
        return False
    url = f"https://huggingface.co/{repo_id}/raw/main/config.json"
    try:
        resp = await http_client.get(url, timeout=5, headers=_hf_auth_headers())
        return resp.status_code == 200
    except Exception:
        return False

# Process-lifetime cache of parsed config.json by URL (a repo's config is immutable), so
# get_model_max_len and get_model_kv_spec don't double-fetch the same file per cold start.
_config_json_cache: "dict[str, dict | None]" = {}

def _config_url_for(model_id: str, tokenizer_repo: "str | None" = None) -> "str | None":
    """The config.json URL for a model (GGUF -> its base/tokenizer repo). None if unresolvable."""
    if is_gguf_model(model_id):
        path = tokenizer_repo or extract_tokenizer_from_gguf_path(model_id)
        return f"https://huggingface.co/{path}/raw/main/config.json" if path else None
    return f"https://huggingface.co/{model_id}/raw/main/config.json"

async def _fetch_config_json(config_url: str) -> "dict | None":
    """Fetch + cache a config.json. Returns the parsed dict, or None on any failure."""
    if config_url in _config_json_cache:
        return _config_json_cache[config_url]
    cfg = None
    try:
        resp = await http_client.get(config_url, follow_redirects=True, timeout=httpx.Timeout(10.0),
                                     headers=_hf_auth_headers())
        resp.raise_for_status()
        parsed = resp.json()
        cfg = parsed if isinstance(parsed, dict) else None
    except Exception as e:
        logging.warning(f"Could not fetch {config_url}: {e}")
    _config_json_cache[config_url] = cfg
    return cfg

async def get_model_max_len(model_id: str) -> int:
    """Fetches the model's config.json from Hugging Face to find its max length."""
    url = _config_url_for(model_id)
    if not url:
        return 0  # GGUF without a resolvable base repo
    config = await _fetch_config_json(url)
    if not isinstance(config, dict):
        return 0
    keys_to_check = ['max_position_embeddings', 'n_positions', 'model_max_length']
    for key in keys_to_check:
        if isinstance(config.get(key), int):
            return config[key]
    # Multimodal configs nest the LM fields under text_config.
    text = config.get("text_config")
    if isinstance(text, dict):
        for key in keys_to_check:
            if isinstance(text.get(key), int):
                return text[key]
    return 0

async def estimate_weight_bytes(model_id: str) -> "int | None":
    """Best-effort estimate of a model's on-disk weight size in bytes (quantization-aware).

    Primary: sum the sizes of all *.safetensors files from HF model metadata
    (HfApi().model_info(files_metadata=True)) — handles single-file AND sharded repos, and gated
    repos via the token. Fallback: the shard index's total_size (sharded only). Returns None when
    neither works (e.g. GGUF / non-safetensors), so the caller degrades to discovery / configured tp."""
    if is_gguf_model(model_id) or '/' not in model_id or model_id.startswith(('http://', 'https://')):
        return None
    token = HF_TOKEN if HF_TOKEN and HF_TOKEN.strip() else None
    try:
        info = await run_in_executor(partial(HfApi().model_info, model_id,
                                             files_metadata=True, token=token))
        total = 0
        for s in (getattr(info, "siblings", None) or []):
            name = getattr(s, "rfilename", "") or ""
            if name.endswith(".safetensors"):
                size = getattr(s, "size", None)
                if size:
                    total += int(size)
        if total > 0:
            return total
    except Exception as e:
        logging.warning(f"model_info weight sizing failed for {model_id}: {e}; trying shard index.")
    # Fallback: shard index total_size (present only for multi-shard models).
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

def _dtype_bytes(torch_dtype) -> int:
    """Bytes per element for a model's KV-cache dtype, inferred from config.json's torch_dtype.
    Defaults to 2 (float16/bfloat16). fp8/int8 KV -> 1; float32 -> 4."""
    s = str(torch_dtype).lower()
    if "float32" in s or "fp32" in s:
        return 4
    if "fp8" in s or "8bit" in s or "int8" in s or "float8" in s:
        return 1
    return 2

def _sliding_window_spec(cfg_text: dict, num_layers: int) -> "tuple[int, int]":
    """(sliding_window, num_sliding_layers) for a model, or (0, 0) for full attention.

    Evidence order: explicit per-layer `layer_types` (most accurate) -> Gemma-style
    `sliding_window` + `sliding_window_pattern` (every Nth layer is global/full) -> a bare
    `sliding_window` (Mistral convention: all layers sliding)."""
    window = cfg_text.get("sliding_window")
    if not isinstance(window, int) or window <= 0:
        return 0, 0
    layer_types = cfg_text.get("layer_types")
    if isinstance(layer_types, list) and layer_types:
        n_sliding = sum(1 for t in layer_types if isinstance(t, str) and "sliding" in t.lower())
        return window, n_sliding
    pattern = cfg_text.get("sliding_window_pattern")
    if isinstance(pattern, int) and pattern > 1:
        n_global = num_layers // pattern          # every `pattern`-th layer is full/global
        return window, max(0, num_layers - n_global)
    return window, num_layers                      # bare sliding_window -> assume all layers sliding

async def get_model_kv_spec(model_id: str, tokenizer_repo: "str | None" = None) -> "dict | None":
    """Fetch the KV-cache-relevant architecture fields from a model's config.json.

    Returns {num_layers, num_kv_heads, head_dim, dtype_bytes, sliding_window, num_sliding_layers}
    or None when the config can't be fetched/parsed. Multimodal repos nest the LM fields under
    'text_config'. Sliding-window layers need far less KV than full-attention ones."""
    url = _config_url_for(model_id, tokenizer_repo)
    if not url:
        return None
    cfg = await _fetch_config_json(url)
    if not isinstance(cfg, dict):
        return None
    text = cfg.get("text_config") if isinstance(cfg.get("text_config"), dict) else cfg
    num_layers = text.get("num_hidden_layers")
    n_heads = text.get("num_attention_heads")
    n_kv = text.get("num_key_value_heads", n_heads)  # GQA -> kv heads; MHA -> all heads
    hidden = text.get("hidden_size")
    head_dim = text.get("head_dim") or (hidden // n_heads if (hidden and n_heads) else None)
    if not (num_layers and n_kv and head_dim):
        logging.warning(f"Incomplete KV spec for {model_id}: layers={num_layers}, "
                        f"kv_heads={n_kv}, head_dim={head_dim}; falling back to discovery.")
        return None
    window, n_sliding = _sliding_window_spec(text, int(num_layers))
    return {"num_layers": int(num_layers), "num_kv_heads": int(n_kv),
            "head_dim": int(head_dim), "dtype_bytes": _dtype_bytes(text.get("torch_dtype", "float16")),
            "sliding_window": int(window), "num_sliding_layers": int(n_sliding)}


def kv_reservation_seqs(model_cfg: ModelConfig) -> int:
    """Sequences to RESERVE KV for in budget mode: explicit kv_reservation_seqs, else max_num_seqs.

    max_num_seqs is vLLM's scheduler width (launched as --max-num-seqs); kv_reservation_seqs lets you
    reserve VRAM for FEWER concurrent sequences than that — so a model can admit many short requests
    while packing tightly, with excess sequences queuing/paging within the reserved pool."""
    return model_cfg.kv_reservation_seqs if model_cfg.kv_reservation_seqs else model_cfg.max_num_seqs


async def estimate_model_need_mib(model_id: str, model_cfg: ModelConfig, effective_tp: int) -> "float | None":
    """Budget-mode per-card VRAM need (MiB) for a model = (weights + KV)/tp * factor + fixed margin.

    KV is sized for kv_reservation_seqs (not the full max_num_seqs scheduler width). Returns None when
    weights or the KV spec can't be determined (e.g. GGUF) — the caller then falls back to
    sole-occupant discovery to measure the real footprint instead of guessing."""
    wbytes = await estimate_weight_bytes(model_id)
    if not wbytes:
        return None
    spec = await get_model_kv_spec(model_id)
    if not spec:
        return None
    kv_total = placement.kv_cache_mib(
        max_model_len=model_cfg.max_model_len, max_num_seqs=kv_reservation_seqs(model_cfg),
        num_layers=spec["num_layers"], num_kv_heads=spec["num_kv_heads"],
        head_dim=spec["head_dim"], dtype_bytes=spec["dtype_bytes"],
        sliding_window=spec.get("sliding_window", 0), num_sliding_layers=spec.get("num_sliding_layers", 0))
    weights_mib = wbytes / (1024.0 * 1024.0)
    return placement.estimate_need_mib(weights_mib, kv_total, effective_tp,
                                       BUDGET_OVERHEAD_FACTOR, BUDGET_OVERHEAD_MIB)


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
                                effective_util: "float | None" = None) -> "tuple | None":
    """Start a vLLM container and wait until healthy. Returns (ip_address, port) on success, or
    None on failure/timeout. Does NOT touch active_containers — the caller owns that entry's
    lifecycle (it inserts a LOADING entry before calling this, and flips it READY / drops it after).

    gpu_uuids pins the container to specific GPU(s). When None/empty, falls back to
    GPU_DEVICE_REQUESTS (the gateway-wide pin or all GPUs). effective_tp/effective_util override
    --tensor-parallel-size / --gpu-memory-utilization; both default to the model's configured
    values. model_cfg is never mutated (it is shared across requests)."""
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
        async with state_lock:
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

    # Per-model concurrency cap. Must match the value used to estimate KV-cache need in budget mode.
    if model_cfg.max_num_seqs and model_cfg.max_num_seqs > 0:
        command.extend(["--max-num-seqs", str(model_cfg.max_num_seqs)])

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
                # When set, override the loader path so the worker uses the host driver's libcuda
                # instead of the image's bundled cuda-compat lib (see WORKER_LD_LIBRARY_PATH).
                **({"LD_LIBRARY_PATH": WORKER_LD_LIBRARY_PATH} if WORKER_LD_LIBRARY_PATH else {}),
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
                    # Return runtime fields only; the caller owns the active_containers entry lifecycle.
                    return (ip_address, VLLM_PORT)
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
    async with state_lock:
        containers = {name: dict(state.__dict__) for name, state in active_containers.items()}
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
        # reserved = LOADING entries on this GPU; residents = LOADING/READY (exclude STOPPING).
        reserved = sum(c.get("reserved_mib", 0.0) for c in containers.values()
                       if guid in c.get("gpu_uuids", []) and c.get("status") == ContainerStatus.LOADING)
        residents = [c["container_name"] for c in containers.values()
                     if guid in c.get("gpu_uuids", []) and c.get("status") != ContainerStatus.STOPPING]
        total_mib = v.get("total")
        gpus_status[guid] = {
            "pool": uuid_to_pool.get(guid),
            "total_mib": total_mib,
            "used_mib": v.get("used"),
            "reserved_mib": reserved,
            # Budget mode: the per-GPU cap on total gateway VRAM (None in whole_card).
            "budget_mib": (GPU_BUDGET_FRACTION * total_mib) if (PLACEMENT_MODE == "budget" and total_mib) else None,
            "residents": residents,
        }
    model_ids = set(list(queue_counts.keys()) + [c["model_id"] for c in containers.values()])
    return {
        "total_gpu_vram_mib": TOTAL_GPU_VRAM,
        "placement_mode": PLACEMENT_MODE,
        "gpu_budget_fraction": GPU_BUDGET_FRACTION if PLACEMENT_MODE == "budget" else None,
        "pools": pools,
        # Per-model need/measured/util are in active_containers (reserved_mib / vram_footprint /
        # effective_util) and known_footprints_mib (signature-stamped measured footprints).
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

# --- Placement helpers (the LOADING/READY/STOPPING lifecycle) ---

async def _release_active(container):
    """Decrement a container's in-flight count under state_lock (single guarded owner of the count)."""
    async with state_lock:
        if container.active_requests > 0:
            container.active_requests -= 1

async def _settled_used(uuid: str, poll: float = 2.0, max_polls: int = 5) -> float:
    """Return a GPU's `used` MiB after it has settled — poll until it stops dropping (freed VRAM from
    a just-evicted container is released asynchronously) or `max_polls` elapse. Used as the discovery
    baseline so an eviction's not-yet-released VRAM doesn't understate the measured footprint (G4)."""
    last = (await get_gpu_vram()).get(uuid, {}).get("used", 0.0)
    for _ in range(max_polls):
        await asyncio.sleep(poll)
        cur = (await get_gpu_vram()).get(uuid, {}).get("used", 0.0)
        if cur >= last - 64:  # stopped dropping (within noise)
            return cur
        last = cur
    return last

def _build_gpu_views(pool_gpus, gpus_snapshot):
    """Build (candidates, residents_by_gpu, blocked_gpus) for placement. MUST be called under
    state_lock (reads active_containers).

    Accounting is conservative: a STOPPING model's VRAM is still resident until its container is
    actually removed, so STOPPING counts toward ready_footprint and blocks co-location just like
    READY (G6) — it just isn't an eviction candidate (it's already leaving). LOADING folds into
    GpuView.reserved. A GPU with any non-colocate LOADING/READY/STOPPING occupant is 'blocked'
    for co-location (a colocate model never shares with a whole-card model)."""
    candidates = []
    residents_by_gpu = {}
    blocked = set()
    for guid in pool_gpus:
        g = gpus_snapshot.get(guid)
        total = float(g["total"]) if g else 0.0
        on_gpu = [c for c in active_containers.values() if guid in c.gpu_uuids]
        ready = [c for c in on_gpu if c.status == ContainerStatus.READY]
        loading = [c for c in on_gpu if c.status == ContainerStatus.LOADING]
        stopping = [c for c in on_gpu if c.status == ContainerStatus.STOPPING]
        residents_by_gpu[guid] = ready  # eviction candidates are READY only
        candidates.append(placement.GpuView(
            uuid=guid,
            total=total,
            used_smi=float(g["used"]) if g else 0.0,
            reserved=sum(c.reserved_mib for c in loading),
            # READY + STOPPING footprints both still occupy the card.
            ready_footprint=sum(c.vram_footprint for c in (ready + stopping)),
            # Budget mode caps total gateway VRAM per card; whole_card leaves it uncapped (inf).
            budget=(GPU_BUDGET_FRACTION * total) if PLACEMENT_MODE == "budget" else math.inf,
        ))
        if any(not c.colocate for c in on_gpu):  # incl. STOPPING — its VRAM is still resident
            blocked.add(guid)
    return candidates, residents_by_gpu, blocked

async def _start_and_finalize(entry, target_model_id, model_cfg, *, gpu_uuids, effective_tp,
                              effective_util, run_discovery, before_used, meas_gpu,
                              signature=None):
    """Start the container for an already-inserted LOADING `entry`, then flip it READY (or drop it
    on failure — the single cleanup site). `target_model_id` is the gateway IDENTITY (config name);
    the vLLM `--model` is `model_cfg.repo`, so several named profiles can share one repo.

    After READY the footprint is set to GROUND TRUTH: the model's actual per-process VRAM
    (`measure_model_vram`). If attribution is unavailable (old driver / no compute-apps) the
    sole-occupant `run_discovery` delta is used; failing that, the reserved estimate stands. The
    result is persisted stamped with `signature` (when provided) so it's reused only in a matching
    sizing context. `signature=None` (degraded mode) skips measurement + persistence."""
    try:
        try:
            runtime = await start_model_container(
                model_cfg.repo, entry.container_name, model_cfg,
                gpu_uuids=gpu_uuids, effective_tp=effective_tp, effective_util=effective_util)
        except BaseException:
            async with state_lock:
                active_containers.pop(entry.container_name, None)  # drop LOADING on any failure/cancel
            raise
        if not runtime:
            async with state_lock:
                active_containers.pop(entry.container_name, None)
            return None

        ip_address, port = runtime
        # G1: the container is up. Re-check the LOADING entry is still OURS (the reconciler or an
        # eviction could have removed it during the start). If it's gone, do NOT resurrect it as a
        # forgotten zombie — stop the container we just started and report failure.
        async with state_lock:
            if active_containers.get(entry.container_name) is not entry:
                stale = True
            else:
                stale = False
                entry.ip_address = ip_address
                entry.port = port
                entry.status = ContainerStatus.READY
                entry.loaded_at = time.time()
                entry.last_request_time = time.time()
                entry.vram_footprint = entry.reserved_mib  # seeded estimate; discovery may refine
        if stale:
            logging.warning(f"LOADING entry {entry.container_name} disappeared during start "
                            f"(reaped/evicted); stopping the orphaned container.")
            try:
                c = await run_in_executor(docker_client.containers.get, entry.container_name)
                await run_in_executor(c.stop)
                await run_in_executor(c.remove)
            except (NotFound, APIError) as e:
                logging.error(f"Could not clean up orphaned container {entry.container_name}: {e}")
            return None

        # Footprint = ground truth. Skip entirely in degraded mode (signature is None).
        if signature is not None:
            # 1) Primary: actual per-process VRAM for THIS model's container (works packed or alone).
            measured = await measure_model_vram(entry.container_name, gpu_uuids or entry.gpu_uuids)
            source = "compute-apps"
            # 2) Fallback: sole-occupant whole-GPU delta (only meaningful when run_discovery / alone).
            if measured is None and run_discovery and meas_gpu:
                logging.info(f"compute-apps attribution unavailable; sampling GPU {meas_gpu} VRAM 3x over 45s...")
                samples = []
                for _ in range(3):
                    await asyncio.sleep(15)
                    samples.append((await get_gpu_vram()).get(meas_gpu, {}).get("used", 0.0))
                delta = max(samples) - before_used
                measured = delta if delta > 256 else None
                source = "gpu-delta"
            # 3) Last resort: keep the reserved estimate (already seeded into vram_footprint).
            if measured is not None and measured > 256:
                entry.vram_footprint = measured
                footprint_mib = measured
            else:
                footprint_mib = entry.reserved_mib
                source = (source or "") + "/estimate-fallback"
            known_footprints[target_model_id] = {
                "per_gpu_mib": float(footprint_mib), "effective_tp": effective_tp,
                "effective_util": float(effective_util or 0.0), "measured_at": time.time(),
                "signature": signature}
            await save_known_footprints_async()
            logging.info(f"Footprint for {target_model_id}: {int(footprint_mib)} MiB (via {source}).")
        return entry
    finally:
        loading_tasks.pop(entry.container_name, None)  # release owner-liveness handle on every exit

async def _ensure_started(target_model_id, model_cfg):
    """Ensure a READY container exists for the model and return it (or None on failure).

    The caller holds container_start_locks[target_model_id], so no other start for THIS model runs
    concurrently. Inserts a LOADING entry under state_lock (reserving VRAM + slot atomically), then
    starts + finalizes outside the lock."""
    # Became READY since the fast-path check?
    async with state_lock:
        ready = next((c for c in active_containers.values()
                      if c.model_id == target_model_id and c.status == ContainerStatus.READY), None)
    if ready is not None:
        return ready

    # record shape: {per_gpu_mib, effective_tp, effective_util, measured_at, signature}. prior_tp is a
    # deterministic optimization (reused regardless of signature). learned_mib / is_discovery are
    # gated on a signature match below (once effective_tp is known) so stale / cross-context footprints
    # are NOT reused.
    record = known_footprints.get(target_model_id)
    prior_tp = record.get("effective_tp") if record else None
    raw_learned_mib = float(record.get("per_gpu_mib", 0.0)) if record else 0.0
    pool = pool_for(model_cfg)
    pool_gpus = MANAGED_POOLS.get(pool, [])
    budget_mode = (PLACEMENT_MODE == "budget")
    # Co-location is a whole_card-mode concept; in budget mode every model packs by VRAM accounting,
    # so the colocate flag is moot and ignored.
    is_colocate = model_cfg.colocate and not budget_mode
    util = model_cfg.gpu_memory_utilization

    # Degraded mode: VRAM accounting unavailable. One container at a time, but still record a
    # proper entry and honor gpu_uuids (pin/pool) so device pinning + /status stay consistent (F5).
    if TOTAL_GPU_VRAM <= 0:
        async with state_lock:
            to_stop = [n for n, c in active_containers.items() if c.status != ContainerStatus.STOPPING]
        for n in to_stop:
            await stop_container(n)
        gpu_uuids = list(pool_gpus) if pool_gpus else list(MANAGED_GPUS)
        async with state_lock:
            slot_indices = {int(n.split('_')[-1]) for n in active_containers.keys()}
            free_slot = next(i for i in range(len(slot_indices) + 1) if i not in slot_indices)
            entry = ContainerState(
                model_id=target_model_id, container_name=f"{VLLM_CONTAINER_PREFIX}_{free_slot}",
                status=ContainerStatus.LOADING, gpu_uuids=gpu_uuids, reserved_mib=0.0,
                effective_tp=model_cfg.tensor_parallel_size, colocate=is_colocate,
                always_on=model_cfg.always_on, inactivity_timeout=model_cfg.inactivity_timeout,
                created_at=time.time())
            active_containers[entry.container_name] = entry
            loading_tasks[entry.container_name] = asyncio.current_task()  # owner-liveness handle (G2)
        return await _start_and_finalize(
            entry, target_model_id, model_cfg, gpu_uuids=gpu_uuids or None,
            effective_tp=model_cfg.tensor_parallel_size, effective_util=None,
            run_discovery=False, before_used=0.0, meas_gpu=None)

    if not pool_gpus:
        raise HTTPException(status_code=500,
                            detail=f"Pool '{pool}' for model {target_model_id} has no managed GPUs")

    gpus_snapshot = await get_gpu_vram()  # OUTSIDE state_lock (spawns a container)
    pool_totals = [float(gpus_snapshot.get(u, {}).get("total", 0.0)) for u in pool_gpus]
    max_total = max(pool_totals) if pool_totals else 0.0

    # Effective TP — deterministic every request (F2). The HF weight estimate is fetched ONLY for an
    # unseen model (no prior record) that is TP-eligible; a seen model reuses its persisted decision.
    if is_colocate:
        effective_tp = 1
    elif model_cfg.tensor_parallel_size > 1:
        effective_tp = model_cfg.tensor_parallel_size
    elif prior_tp is not None:
        effective_tp = max(1, prior_tp)              # seen before -> reuse persisted tp, no HF call
    else:
        wbytes = None
        if (len(pool_gpus) >= 2 and max_total > 0
                and (max(pool_totals) - min(pool_totals)) <= 0.05 * max_total):
            wbytes = await estimate_weight_bytes(model_cfg.repo)
        effective_tp = placement.compute_effective_tp(wbytes, model_cfg.tensor_parallel_size,
                                                      prior_tp, pool_totals, util)

    # Reuse a persisted footprint only when its signature matches the current sizing context;
    # otherwise treat the model as unseen and re-measure (fixes cross-mode / config-change staleness).
    # Sizing depends on kv_reservation_seqs (the reserved KV), not max_num_seqs (scheduler width).
    current_sig = placement.footprint_signature(
        PLACEMENT_MODE, model_cfg.max_model_len, kv_reservation_seqs(model_cfg), effective_tp, util_basis=util)
    seen = record is not None and placement.signature_matches(record, current_sig)
    is_discovery = not seen
    learned_mib = raw_learned_mib if seen else 0.0

    # Per-card need.
    #  - budget mode: weights + bounded KV + overhead, recomputed fresh from config each cold start
    #    (so a max_model_len/max_num_seqs change takes effect). A model whose need can't be estimated
    #    (GGUF / unknown arch) reuses a previously measured footprint, else falls back to discovery.
    #  - whole_card mode (F9): learned per-GPU footprint when known(>0), else util * that card's total.
    budget_need = None
    budget_discovery = False
    if budget_mode:
        budget_need = await estimate_model_need_mib(model_cfg.repo, model_cfg, effective_tp)
        if budget_need is None:                 # GGUF / unknown arch -> can't estimate
            if learned_mib > 0:
                budget_need = learned_mib       # reuse a previously measured footprint
            else:
                budget_discovery = True         # measure as sole occupant on first load

    if budget_mode and budget_need is not None:
        need_fn = (lambda g, n=budget_need: n)              # constant per-card need
    elif learned_mib > 0:
        need_fn = (lambda g: learned_mib)
    else:
        need_fn = (lambda g: util * g.total)                # whole-card / discovery fallback
    colocate_wbytes = await estimate_weight_bytes(model_cfg.repo) if is_colocate else None

    # DECIDE + reserve + insert LOADING — under state_lock (no I/O inside).
    async with state_lock:
        candidates, residents_by_gpu, blocked = _build_gpu_views(pool_gpus, gpus_snapshot)
        if is_colocate:
            uuid, evictions = placement.select_colocated(
                candidates, residents_by_gpu, need_fn, blocked, time.time(), GATEWAY_MIN_RESIDENT_SECONDS)
            chosen_uuids = [uuid] if uuid is not None else None
        else:
            chosen_uuids, evictions = placement.select_placement(
                candidates, residents_by_gpu, need_fn, effective_tp, time.time(), GATEWAY_MIN_RESIDENT_SECONDS)
        if chosen_uuids is None:
            raise HTTPException(
                status_code=503,
                detail=(f"Cannot place model {target_model_id} in pool '{pool}' "
                        f"({'colocate' if is_colocate else f'tp={effective_tp}'}): no GPU available without "
                        f"evicting always_on/in-flight models (or pool not homogeneous / too few GPUs for TP)."))

        chosen_gv = next(c for c in candidates if c.uuid == chosen_uuids[0])
        if is_colocate:
            freed = sum(r.vram_footprint for r in residents_by_gpu[chosen_uuids[0]]
                        if r.container_name in evictions)
            total = chosen_gv.total or max_total
            budget = max(0.0, chosen_gv.free + freed - COLOCATE_MARGIN_MIB)
            effective_util = min(util, (budget / total) if total > 0 else util)
            if colocate_wbytes:
                need_mib = (colocate_wbytes / (1024.0 * 1024.0)) * COLOCATE_WEIGHT_OVERHEAD
                if effective_util * total < need_mib:
                    raise HTTPException(
                        status_code=503,
                        detail=(f"Cannot co-locate {target_model_id} on {chosen_uuids[0]}: budget "
                                f"~{int(effective_util * total)} MiB < weights ~{int(need_mib)} MiB."))
            reserve_amt = effective_util * total
        elif budget_mode and budget_need is not None:
            # Size the launch util so vLLM takes ONLY this need; the budget cap makes it physically
            # unable to exceed it, so an under-estimate fails only this model's startup, not a peer.
            total = chosen_gv.total or max_total
            effective_util = min(GPU_BUDGET_FRACTION,
                                 (budget_need / total) if total > 0 else GPU_BUDGET_FRACTION)
            reserve_amt = budget_need
        else:
            effective_util = None
            reserve_amt = need_fn(chosen_gv)

        slot_indices = {int(n.split('_')[-1]) for n in active_containers.keys()}
        free_slot = next(i for i in range(len(slot_indices) + 1) if i not in slot_indices)
        entry = ContainerState(
            model_id=target_model_id, container_name=f"{VLLM_CONTAINER_PREFIX}_{free_slot}",
            status=ContainerStatus.LOADING, gpu_uuids=list(chosen_uuids), reserved_mib=reserve_amt,
            effective_tp=effective_tp, effective_util=effective_util or 0.0, colocate=is_colocate,
            always_on=model_cfg.always_on, inactivity_timeout=model_cfg.inactivity_timeout,
            created_at=time.time())
        active_containers[entry.container_name] = entry
        loading_tasks[entry.container_name] = asyncio.current_task()  # owner-liveness handle (G2)
        if is_colocate:
            placed_desc = 'colocate util=%.3f' % effective_util
        elif budget_mode and effective_util is not None:
            placed_desc = 'budget util=%.3f tp=%d' % (effective_util, effective_tp)
        else:
            placed_desc = 'tp=%d' % effective_tp
        logging.info(f"Placing {target_model_id} on GPU(s) {chosen_uuids} (pool '{pool}', {placed_desc}, "
                     f"~{int(reserve_amt)} MiB/GPU); evicting {evictions or 'nothing'}; slot {entry.container_name}.")

    # Evict + start OUTSIDE the lock.
    for name in evictions:
        await stop_container(name)
    # Per-process attribution (in _start_and_finalize) is the primary footprint source. `run_discovery`
    # marks a sole-occupant load (unseen whole_card model, or a budget model whose need couldn't be
    # estimated) — for those the whole-GPU delta is a valid FALLBACK if attribution is unavailable.
    run_discovery = budget_discovery if budget_mode else (is_discovery and not is_colocate)
    # G4: take the discovery baseline AFTER evictions have actually released VRAM (stop_container
    # returns before the driver frees it), not from the pre-decision snapshot. Settle by polling the
    # chosen GPU's used until it stops dropping (or a short ceiling).
    before_used = 0.0
    if run_discovery:
        before_used = await _settled_used(chosen_uuids[0]) if evictions else \
            gpus_snapshot.get(chosen_uuids[0], {}).get("used", 0.0)
    return await _start_and_finalize(
        entry, target_model_id, model_cfg, gpu_uuids=chosen_uuids, effective_tp=effective_tp,
        effective_util=effective_util, run_discovery=run_discovery,
        before_used=before_used, meas_gpu=chosen_uuids[0], signature=current_sig)

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

    # Identity is the config NAME (so several named profiles can share one repo, each its own
    # container / footprint / queue). The repo (model_cfg.repo) is what vLLM serves under.
    target_model_id = model_name
    model_cfg = MODEL_CONFIGS[model_name]

    # Initialize semaphore for this model if not exists
    if target_model_id not in model_semaphores:
        async with state_lock:
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

    # Step 2: semaphore is held — run all logic; the finally releases it unless the streaming
    # generator takes ownership. queue_count_lock now guards ONLY the counter (never held across
    # placement, container start, or proxying — fixes the global-serialization bug F1).
    try:
        async with queue_count_lock:
            counter_needs_cleanup = False  # set flag FIRST (guard against double-decrement)
            model_queue_counts[target_model_id] = max(0, model_queue_counts[target_model_id] - 1)
            logging.debug(f"Request dequeued for {model_name} ({target_model_id}). Queue depth: {model_queue_counts[target_model_id]}/{GATEWAY_MAX_QUEUE_SIZE}")

        # Fast path: a READY container for this model already exists -> route to it.
        async with state_lock:
            target_container = next((c for c in active_containers.values()
                                     if c.model_id == target_model_id
                                     and c.status == ContainerStatus.READY), None)

        # Slow path: ensure a container is started (serialized per model via the start lock).
        if target_container is None:
            async with state_lock:
                if target_model_id not in container_start_locks:
                    container_start_locks[target_model_id] = asyncio.Lock()
                per_model_lock = container_start_locks[target_model_id]
            async with per_model_lock:
                target_container = await _ensure_started(target_model_id, model_cfg)

        if target_container is None:
            raise HTTPException(status_code=500, detail=f"Failed to start or find container for model {target_model_id}")

        # Claim an in-flight slot under state_lock with a status RE-CHECK (G3): the container could
        # have flipped STOPPING (eviction/idle-unload) between resolution and now. The increment and
        # the READY check are atomic, so stop_container's drain can't miss this request and we never
        # route to a container being torn down.
        async with state_lock:
            if target_container.status != ContainerStatus.READY:
                raise HTTPException(status_code=503,
                                    detail=f"Model {target_model_id} container became unavailable; retry.")
            target_container.last_request_time = time.time()
            target_container.active_requests += 1
        active_req_decremented = False

        query_string = str(request.url.query) if request.url.query else ""
        vllm_url = f"http://{target_container.ip_address}:{target_container.port}{request.url.path}"
        if query_string:
            vllm_url += f"?{query_string}"
        current_queue_depth = model_queue_counts.get(target_model_id, 0)

        try:
            # Inject per-model request defaults UNDER the caller's body (caller wins). Must run
            # before body['model'] is set (so repo always wins) and before is_streaming is read.
            if model_cfg.request_defaults:
                body = merge_request_defaults(model_cfg.request_defaults, body)
            body['model'] = model_cfg.repo  # vLLM serves under the repo it was launched with (--model)
            headers_to_forward = {
                k: v for k, v in request.headers.items()
                if k.lower() not in ('host', 'connection', 'content-length', 'transfer-encoding')
            }
            is_streaming = body.get('stream', False)
            queue_headers = {
                "X-Queue-Depth": str(current_queue_depth),
                "X-Max-Concurrent": str(GATEWAY_MAX_CONCURRENT),
                "X-Max-Queue-Size": str(GATEWAY_MAX_QUEUE_SIZE)
            }
            if is_streaming:
                # The generator owns BOTH the active_requests decrement AND the semaphore release,
                # so both fire after the last byte (or client disconnect), enforcing MAX_CONCURRENT
                # for the full stream duration.
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
                        await _release_active(captured_container)
                        captured_sem.release()

                sem_released = True
                active_req_decremented = True
                return StreamingResponse(
                    stream_generate(), status_code=200,
                    media_type="text/event-stream", headers=queue_headers)
            else:
                # Non-streaming: buffer the full response. Retry only transient connection errors.
                max_retries = 3
                retry_delay = 1.0
                for retry_attempt in range(max_retries):
                    try:
                        response = await http_client.request(
                            method=request.method, url=vllm_url, json=body, headers=headers_to_forward)
                        response.raise_for_status()
                        break
                    except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as e:
                        if retry_attempt < max_retries - 1:
                            logging.warning(f"Transient connection error to vLLM for {model_name} (attempt {retry_attempt + 1}/{max_retries}): {type(e).__name__}. Retrying in {retry_delay}s...")
                            await asyncio.sleep(retry_delay)
                            retry_delay *= 1.5
                        else:
                            logging.error(f"Failed to connect to vLLM for {model_name} after {max_retries} attempts: {type(e).__name__}: {str(e)}")
                            raise
                try:
                    content = response.json()
                except Exception:
                    content = {"error": "Non-JSON response from vLLM", "raw": response.text[:500]}
                    logging.warning(f"Non-JSON response from vLLM for {model_name}: status={response.status_code}, body={response.text[:200]}")
                return JSONResponse(content=content, status_code=response.status_code, headers=queue_headers)
        except httpx.HTTPStatusError as e:
            logging.error(f"vLLM returned HTTP error for {model_name} ({target_model_id}): {e.response.status_code} - {str(e)}")
            return JSONResponse(
                {"error": "Error from vLLM service", "details": str(e)},
                status_code=e.response.status_code, headers={"X-Queue-Depth": str(current_queue_depth)})
        except httpx.RequestError as e:
            logging.error(f"Connection error to vLLM for {model_name} ({target_model_id}) at {vllm_url}: {type(e).__name__}: {str(e)}")
            raise HTTPException(status_code=503, detail=f"Could not connect to vLLM service: {e}")
        finally:
            if not active_req_decremented:
                await _release_active(target_container)
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
