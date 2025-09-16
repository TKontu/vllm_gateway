# TODO: Implement Parallel Model Containers & Load Balancing

**Objective:** Refactor `gateway/app.py` to allow running multiple container instances of the same model and load balance incoming requests between them. The initial target is to support two instances of the `gemma3-4B` model.

---

### Step 1: Add Configuration for Parallel Models

1.  **Introduce New Environment Variable:** The system will be configured via a new environment variable named `PARALLEL_MODELS_JSON`.
    -   **Format:** A JSON string mapping a user-facing model name to the desired number of parallel instances.
    -   **Example:** `'''{"gemma3-4B": 2, "qwen2.5": 1}'''`
2.  **Add Global Config Variable:** Create a new global dictionary `PARALLEL_MODELS_CONFIG` to hold this configuration.
3.  **Create a Loader Function:** Implement `load_parallel_models_config()` to:
    -   Read the `PARALLEL_MODELS_JSON` environment variable.
    -   Parse the JSON into the `PARALLEL_MODELS_CONFIG` dictionary.
    -   Include a default value of `'''{"gemma3-4B": 2}'''` to satisfy the immediate requirement.
    -   Log the loaded configuration for debugging.
4.  **Integrate into Startup:** Call `load_parallel_models_config()` within the `startup_event` function.

### Step 2: Redesign State Management Data Structures

The current one-to-one model-to-container mapping must be updated.

1.  **`model_to_containers`:** Introduce a new global dictionary:
    -   `model_to_containers: dict[str, list[ContainerState]]`
    -   This will be the primary structure for looking up all running containers associated with a given `model_id`.
2.  **`round_robin_counters`:** Introduce a new global dictionary:
    -   `round_robin_counters: dict[str, int]`
    -   This will store a counter for each `model_id` to enable round-robin load balancing.
3.  **`active_containers`:** This existing dictionary will be kept. It will continue to serve as a direct, name-based lookup for managing the lifecycle of every container, regardless of the model it runs.

### Step 3: Update Container Shutdown Logic

The `stop_container()` function must be updated to clean up the new state structures.

1.  **Modify `stop_container()`:** When a container is stopped, it must be removed from:
    -   The `active_containers` dictionary (as it is now).
    -   The corresponding list within the `model_to_containers` dictionary.
    -   If its removal leaves an empty list for a `model_id` in `model_to_containers`, that `model_id` key should be deleted.

### Step 4: Refactor Core Request Logic (`proxy_request`)

This is the most significant change. The function will be re-architected to handle the new one-to-many logic.

1.  **Determine Required Instances:**
    -   On receiving a request, the code will acquire the `model_management_lock`.
    -   It will check `model_to_containers` to find the number of currently running instances for the requested `model_id`.
    -   It will compare this to the number specified in `PARALLEL_MODELS_CONFIG` to determine if new instances need to be started.
2.  **Implement Instance Startup Loop:**
    -   If `instances_to_start > 0`, the code will loop that many times.
    -   **Inside the loop, for each new instance:**
        -   **VRAM & Eviction:** Perform the VRAM availability check and LRU eviction logic *before* starting the new container. This ensures there is space for each new instance.
        -   **Container Naming:** Generate a new unique and descriptive container name. The proposed scheme is `{VLLM_CONTAINER_PREFIX}_{safe_model_name}_instance_{instance_num}` (e.g., `vllm_server_gemma3-4B_instance_1`).
        -   **Footprint Discovery:** If the model's footprint is unknown, the first instance will trigger the discovery run. The container from this run will be kept and registered as the first active instance.
        -   **Start Container:** Call `start_model_container()` with the new name.
        -   **Update State:** Upon successful startup, add the new `ContainerState` object to both the `active_containers` dictionary and the `model_to_containers` list for that model.
3.  **Implement Load Balancing:**
    -   After the startup logic is complete (and still inside the `proxy_request` function), select a container to handle the current request.
    -   Use the `round_robin_counters` to pick an instance from the `model_to_containers` list.
    -   Increment the counter for that model.
    -   The lock will be released *before* the request is forwarded to the selected container.

### Step 5: Update Status Endpoint

The `/gateway/status` endpoint must be updated to provide visibility into the new state.

1.  **Modify `gateway_status()`:** The returned JSON should include:
    -   `parallel_model_config`: The parsed configuration.
    -   `active_models_by_id`: A dictionary showing each model and the list of its active container instances (from `model_to_containers`).
    -   `active_containers_by_name`: The existing flat list of all containers, which is useful for debugging.
