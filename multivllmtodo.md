# Multi-vLLM Implementation Plan (Learning-Based)

This document outlines the steps to implement a resource-aware, learning-based gateway for managing multiple concurrent vLLM containers.

## 1. Configuration (`docker-compose.yml`)

- [ ] Add a volume mount to the `gateway` service to persist the learned model memory data.
  ```yaml
  volumes:
    - ./memory_footprints.json:/app/memory_footprints.json
  ```
- [ ] The `MAX_CONCURRENT_MODELS` variable is no longer needed, as the gateway will now use dynamic memory calculations.

## 2. Gateway Application (`gateway/app.py`)

### 2.1. Core Components & State

- [ ] **`memory_footprints.json`:** The gateway will read from and write to `/app/memory_footprints.json` to persist learned VRAM requirements for each model.
- [ ] **`nvidia-smi` Orchestration:** The gateway will not have `nvidia-smi` locally. It will launch temporary, privileged `nvidia/cuda` containers to run `nvidia-smi` commands and capture their output.
- [ ] **Global State:**
    - `TOTAL_GPU_VRAM`: Determined at startup by running an `nvidia-smi` utility container.
    - `known_footprints`: A dictionary holding the learned memory requirements, loaded from the JSON file.
    - `active_containers`: A dictionary tracking running containers, their model, and their known memory footprint.

### 2.2. Startup Logic (`startup_event`)

- [ ] On startup, execute a temporary `nvidia-smi` container to get the total GPU VRAM. If this fails, log an error and disable the dynamic memory feature.
- [ ] Load `/app/memory_footprints.json`. If it doesn't exist, create an empty dictionary.

### 2.3. Core Request Logic (`proxy_request`)

The logic will be executed under a lock to ensure thread safety.

1.  A request for `model_A` arrives.
2.  Check if `model_A` is already running. If yes, update its last-used time and proxy the request.
3.  If not running, look up its footprint in `known_footprints`.

#### Scenario A: The model's footprint is KNOWN.
4.  Calculate `current_vram_usage` from `active_containers`.
5.  If `current_vram_usage + model_A_footprint <= TOTAL_GPU_VRAM`:
    - Start a new container for `model_A`.
    - Add it to `active_containers`.
6.  If it does not fit:
    - Evict one or more LRU containers until there is enough space.
    - Start the new container for `model_A`.

#### Scenario B: The model's footprint is UNKNOWN (Discovery Run).
7.  To ensure a clean measurement, **stop and remove all other running vLLM containers**.
8.  Start a new container for `model_A` in isolation.
9.  Wait for the model to be fully loaded (by polling its health/readiness endpoint).
10. Launch a temporary `nvidia-smi` utility container to measure the GPU memory used by the new container's process.
11. Save the measured footprint to `known_footprints` (in memory) and persist it to `/app/memory_footprints.json`.
12. Add the new container to `active_containers` with its now-known footprint.

13. Proxy the request to the container.

### 2.4. Helper Functions

- [ ] **`get_total_vram()`**: Executes the `nvidia-smi` utility container to get total VRAM.
- [ ] **`measure_model_footprint(container)`**: Executes `nvidia-smi` to get the VRAM usage of a given container's process.
- [ ] **`start_model_container(model_id, container_name)`**: Starts a vLLM container.
- [ ] **`stop_container(container_name)`**: Stops and removes a container.

### 2.5. Background Tasks

- [ ] The `shutdown_if_inactive` task will remain, shutting down individual containers that have been idle for the timeout period and freeing up their VRAM from the `current_vram_usage` calculation.

## 3. Testing

- [ ] **First Run (Discovery):** Test requesting a model for the first time. Verify that other containers are stopped and that a `memory_footprints.json` file is created with the correct measurement.
- [ ] **Second Run (Known Model):** Request the same model again. Verify it starts without a discovery run.
- [ ] **Parallel Run (Fit):** Test requesting a second, smaller model. Verify it starts in parallel if it fits in the remaining VRAM.
- [ ] **Eviction Run (No Fit):** Test requesting a model that doesn't fit. Verify that the LRU container is evicted to make space.
- [ ] **Inactivity Shutdown:** Verify that an idle container is shut down and its VRAM is freed.