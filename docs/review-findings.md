# vLLM Gateway — Implementation Review Findings

Diligent review of the placement engine across Phases 1a–3 (PRs #4–#7). **Documentation only —
nothing here is fixed.** Each finding was verified against the source (file:line confirmed) and
tagged:

- **[pre-existing]** — predates the placement phase work (originates in the earlier streaming/queue code).
- **[introduced]** — added or materially worsened by Phases 1a/1b/2/3.

Scope: `gateway/app.py`, `gateway/placement.py`, `gateway/config_loader.py`, `config/models.yaml.example`,
`docker-compose.yml`. Line numbers are as of branch `placement-phase-3` and will drift.

## Fix status (branch `placement-hardening`)

**Stage 1 — DONE** (block-independent fixes; no concurrency-model change):
- **F6** ✅ `merge_extra_args` now drops + warns on gateway-managed flags (`GATEWAY_MANAGED_FLAGS`).
- **F8/F11** ✅ probe is now `count=-1` (sees all GPUs; pin no longer constrains it) + post-probe
  `validate_pools_visible` fail-fast; `TOTAL_GPU_VRAM` recomputed from the managed set.
- **F10** ✅ `gateway_status` is `async` and snapshots shared state under locks.
- **F15** ✅ `save_known_footprints_async` runs the write off the event loop.
- **F16** ✅ stale GPU-pin and "POOL-BASED (no TP/colocation)" comments corrected.

**Stage 2 — DONE** (branch `placement-hardening-stage2`; the concurrency-core refactor):
- **F1** ✅ `queue_count_lock` now guards only the counter; placement/start/proxy run outside it.
- **F3** ✅ reservation is the LOADING entry's `reserved_mib`; the entry is dropped in one place
  (`_start_and_finalize`'s failure path) — no side-dict to leak.
- **F4** ✅ slot + VRAM claimed atomically by inserting the LOADING entry under `state_lock`.
- **F7** ✅ LOADING models are visible to placement; `select_colocated` takes `blocked_gpus`.
- **F13** ✅ reconciler in the inactivity monitor reaps orphaned LOADING entries (self-healing).
- **F2** ✅ `placement.compute_effective_tp` runs every request; footprints re-keyed to
  `{per_gpu_mib, effective_tp, measured_at}` (legacy migrated) and the tp decision is persisted.
- **F9** ✅ `select_*` take a per-card `need_fn(GpuView)`.
- **F5** ✅ the `TOTAL_GPU_VRAM==0` path is folded into the unified flow (records an entry, honors
  `gpu_uuids`).

Architecture: `model_management_lock`+`placement_lock`→one `state_lock`; `gpu_reservations` deleted
(reservation lives on the LOADING entry); `ContainerState` gains `status`/`reserved_mib`/
`effective_tp`/`effective_util`/`created_at`; `start_model_container` returns `(ip, port)` and the
caller owns the entry lifecycle (`_ensure_started` → `_reserve`/insert → `_start_and_finalize`).

*(Sequencing note: the approved plan placed F2/F9/F5 in Stage 1, but they edit the same placement
block that the Stage 2 rewrite replaces, so they were moved to Stage 2 to avoid editing it twice.
Same fixes, fewer rewrites.)*


---

## ⚠️ Read this first: the lock interaction

The over-broad `queue_count_lock` (**F1**) currently **masks most of the concurrency hazards below**.
Because one global lock is held across the entire placement+proxy path, only one request is ever in
that region at a time, so the slot-collision (F4), reserved-but-not-resident (F7), and double-booking
races **cannot fire today**. They go live the moment F1 is fixed.

➡️ **F1 must be fixed together with F3, F4, F7 — not before them.** Fixing F1 alone would expose the
masked races.

---

## Critical

### F1 — `queue_count_lock` held across the whole request lifecycle  `[pre-existing, amplified]`
`app.py:1115` opens `async with queue_count_lock:` and the block (indent 12) extends through
placement, `start_model_container` (health-check loop up to ~1 h), the 45 s discovery sleeps, and the
entire non-streaming proxy — closing only at the outer `finally` (~`app.py:1432`). The comments
"Outside all locks" at `app.py:1313` and `1335` are **false**.

- **Effect:** the admission/queue-counter lock is effectively global. A second request *for any model*
  blocks behind the first request's cold-start/proxy at the admission `async with queue_count_lock`
  (~`app.py:1063`). `GATEWAY_MAX_CONCURRENT` and the per-model semaphores are defeated; the gateway
  serializes globally.
- **Phase amplification:** the phases added, *inside* this lock, per-GPU `get_gpu_vram()` probes (each
  spawns a container), `estimate_weight_bytes()` HTTP calls, and 3×15 s discovery sleeps — greatly
  extending the hold time.
- **Fix shape:** the block should close right after the counter decrement (~`app.py:1118`); everything
  below belongs outside it.

---

## High

### F2 — Auto-TP-fallback footprint reused for a later single-GPU placement → OOM  `[introduced: Phase 2]`
For a `tensor_parallel_size: 1` model whose weights exceed one card:
1. First request: `is_discovery=True` → TP fallback sets `effective_tp=2` → discovery stores a
   **per-GPU** footprint under `known_footprints[model_id]` (~`app.py:1283`).
2. Next request: `footprint>0` ⇒ `is_discovery=False`, so the fallback block (guarded by `is_discovery`)
   is skipped → `effective_tp` reverts to the configured `1`, and `needed_per_gpu` = the half-size
   per-GPU footprint → the model is placed on **one** GPU with `--tensor-parallel-size 1` → OOM at load.

The fallback *decision* is never persisted. Forced-TP models (config `tp≥2`) are unaffected.
**Root cause is architectural:** `known_footprints` is keyed by `model_id` only (see Architecture §A1).

### F3 — Reservation leak if eviction throws between reserve and the `try/finally`  `[introduced: 1b]` *(masked by F1)*
Reservations are added under the lock (~`app.py:1240`), then the eviction loop
`for name in evictions: await stop_container(name)` runs **before** the `try:` whose `finally` frees
them (~`app.py:1257`/`1268`). `stop_container` only catches `NotFound`/`APIError`; any other error — or a
`CancelledError` (client disconnect) anywhere in that gap — escapes before the `finally`, permanently
leaking `reserve_amt` on the chosen GPU(s) and progressively shrinking their `free` (→ eventual 503s).
The reserve-add and the eviction loop should sit inside the same `try`.

### F4 — Slot-name collision clobbers a freshly-started container  `[pre-existing for known-footprint path; now all paths]` *(masked by F1)*
`free_slot`/`container_name` are computed from `active_containers` under the lock (~`app.py:1242`) but
the name is not inserted until after `start_model_container` returns (~`app.py:1266`) — minutes later,
after the locks release. Two requests for **different** models can compute the same `vllm_server_N`;
the second's stale-container cleanup (`start_model_container` ~`app.py:756–785`, `remove(force=True)`)
then destroys the first's container. Only safe today because F1 serializes the whole region.

### F5 — VRAM-management-disabled fallback bypasses pools and device pinning  `[introduced: not updated in 1b/2/3]`
The `else:` branch when `TOTAL_GPU_VRAM == 0` (~`app.py:1305`) calls
`start_model_container(target_model_id, container_name, model_cfg)` with **no `gpu_uuids` /
`effective_tp` / `effective_util`** → device selection falls back to `GPU_DEVICE_REQUESTS` (all GPUs or
the instance pin), `ContainerState.gpu_uuids=[]`. So a `pool: util` model is **not** confined to the
A2000 and never appears in per-GPU accounting / `/gateway/status`. Triggers whenever the startup VRAM
probe fails on a host that declares `pools:`.

---

## Medium

### F6 — `extra_args` silently overrides gateway-computed `--gpu-memory-utilization` / `--tensor-parallel-size`  `[introduced: interaction]`
`merge_extra_args` (`app.py:712`) drops the gateway's flag whenever the user supplies the same one. So
`extra_args: ["--gpu-memory-utilization","0.95"]` defeats the co-location `effective_util` cap (OOM
risk), and a `--tensor-parallel-size` in `extra_args` bypasses `validate_tp_against_pools`. No warning;
`models.yaml.example` even advertises "override generated flags." The gateway-managed flags should be
protected or at least warned about.

### F7 — `select_colocated` ignores a reserved-but-not-yet-resident whole-card model  `[introduced: Phase 3]` *(partly masked by F1 + reservation)*
Eligibility (`placement.py:~218`) is computed from `active_containers` only. A whole-card model
mid-launch is in `gpu_reservations` but not yet a "resident", so the GPU looks empty/eligible and a
colocate model could land on a card a whole-card model is loading onto. The reservation reduces
`GpuView.free` (partial protection) and F1 serializes starts today, but the eligibility predicate
itself is unsound.

### F8 — Pool with zero *visible* GPUs isn't caught; config validation runs before the GPU probe  `[introduced: 1b]`
`validate_model_pools` / `validate_tp_against_pools` / `validate_colocate` run at import (before
`get_total_vram` / `resolve_managed_pools`). A pool whose UUIDs are typos passes validation; at runtime
those candidates get `total=0` → `free<0` → never chosen, and for TP a `0`-total breaks `_homogeneous`
→ permanent 503 with only a startup WARNING (~`app.py:373`). Medium-confidence cascade:
`run_nvidia_smi_in_container` requests `device_ids=MANAGED_GPUS`; if one UUID is invalid Docker may
error the **whole** probe → all VRAM reads return 0.

### F9 — `needed_per_gpu = util × max(pool_totals)` is wrong on a mixed-size pool  `[introduced: 1b]`
`app.py:~1154` uses the pool's largest card for every candidate, but vLLM targets `util × that card's
total`. Harmless for homogeneous pools (the intended use), but a pool mixing card sizes (not prevented
by config) mis-sizes the fit check per card. Should be `util × g.total` per candidate.

### F10 — `/gateway/status` iterates shared dicts with no lock  `[introduced: status reads grew in 1b]`
`gateway_status` (sync `def`, runs in a threadpool) iterates `active_containers` / `gpu_reservations` /
`model_queue_counts` / `GPU_VRAM` (~`app.py:1003–1031`) while async paths mutate them under locks →
possible `RuntimeError: dictionary changed size during iteration` (500). Field serialization itself is
fine (`gpu_uuids` list, `colocate` bool are JSON-clean).

### F11 — `get_total_vram()` runs before `resolve_managed_pools()`; pin constrains the initial probe  `[introduced: 1b/3 ordering]`
At startup the probe uses `GPU_DEVICE_REQUESTS`. With both `pools:` **and** `GATEWAY_GPU_UUID` set, only
the pinned GPU is probed, so `GPU_VRAM` has one entry and every other pool GPU becomes "unseen" (→ F8).
`pools:` is meant to win over the pin, but the pin still constrains the initial probe — inconsistent
precedence.

---

## Low

### F12 — Co-located footprint is the configured *share*, not actual use; never re-measured  `[introduced: Phase 3 — design limitation, not a bug]`
`vram_footprint = effective_util×total` for colocate models and discovery is skipped, so packing is by
*configured share*, conservatively. **Note (correcting an over-statement from the review):** this does
**not** drive `free` negative or block legitimate packing — the fit-check guarantees
`Σ resident footprints ≤ total`. It just means shares must be tuned; the gateway won't auto-discover
that a model uses less than its declared share.

### F13 — Streaming generator owns its own `active_requests` / semaphore release  `[pre-existing]`
If the ASGI layer ever builds the `StreamingResponse` but doesn't drive the generator to its `finally`
(~`app.py:1374`), both the semaphore slot and `active_requests` leak (container becomes permanently
non-evictable). Normally Starlette closes the generator on disconnect, so this is an edge case.

### F14 — Weight estimate is dtype-blind; GGUF returns `None` → no TP fallback  `[introduced: Phase 2]`
`estimate_weight_bytes` reads the safetensors index `total_size`; vLLM may upcast (e.g. fp8→bf16) so the
`1.2`/`1.15` overhead factors can under-count, and GGUF / non-safetensors models get no TP-fallback
sizing (silently attempt single-card → OOM). The degradation is "fail loud at launch," acceptable but
undocumented.

### F15 — Blocking `save_known_footprints()` on the event loop, under locks  `[pre-existing, now under more locks]`
`app.py:~1289` calls synchronous `json.dump` directly (not via `run_in_executor`) inside the per-model
lock and F1's global lock.

### F16 — Stale comments / docstrings  `[introduced]`
- `app.py:~1140` "POOL-BASED PLACEMENT (whole-card model; no TP, no co-location)" sits above code that
  now does both TP and co-location.
- `app.py:51–53` GPU-pin comment doesn't mention that `pools:` overrides it (docker-compose.yml does).
- "Outside all locks" at `app.py:1313`/`1335` is false (F1).

---

## Architectural observations

**A1 — Footprint registry keyed by `model_id` only.** The real VRAM need depends on placement (TP
degree, card size, util share). F2 and F9 are both symptoms. A footprint key should encode
`(repo, effective_tp, util, card_total)`, or footprints should be per-(model, gpu-class). The TP
fallback decision should also be persisted, not re-derived only during discovery.

**A2 — Two sources of truth for VRAM.** nvidia-smi `used_smi` vs app-tracked `ready_footprint` /
`reserved`, reconciled by `max()` + subtraction (`placement.py:30`). It works but is subtle.
External-process VRAM on a shared card (the A2000 embedder) is only ever *reacted to*; an
over-subscribed external process yields a clean 503 on that card (expected — **not** a needless-eviction
bug: `_gpu_fit_cost` returns `None` without evicting when nothing fits).

**A3 — Overloaded lock model.** `queue_count_lock` doubles as a de-facto global request lock (F1). A
clean design wants: a short counter lock, the already-added briefly-held `placement_lock`, and the
reservation as the only cross-request VRAM guard (with the reserve+evict made atomic per F3).

---

## Verified NOT bugs (recorded so they are not re-raised)

- `select_placement` TP picks **distinct** GPUs (one `GpuView` per pool GPU); no same-GPU-twice.
- `minimal_tp_to_fit` correctly caps at `pool_size`.
- Unit handling (bytes vs MiB) is consistent across `estimate_weight_bytes`, `minimal_tp_to_fit`, the
  co-location floor, and `GpuView`.
- The documented lock order (`container_start_lock → model_management_lock → placement_lock`) is
  followed; `get_gpu_vram()` is correctly snapshotted *outside* `placement_lock`.
- Queue-counter / semaphore cleanup on the new 503/500 paths does **not** leak (the
  `counter_needs_cleanup` flag + outer `finally` handle it).

---

## Suggested triage order (when fixing later)

1. **F1 + F3 + F4 + F7 together** (lock scope + reservation atomicity + slot allocation + colocate
   eligibility) — they are coupled; fixing F1 alone exposes the others.
2. **F2** (persist TP-fallback / footprint keying) — silent OOM on the second request.
3. **F5** (VRAM-disabled fallback honors pools) and **F6** (protect managed flags) — correctness/safety.
4. **F8/F11** (fail-fast on 0 visible GPUs; probe ordering) and **F10** (lock the status reads).
5. Low / docs: F12–F16.

*Generated by an implementation review; findings verified against source on branch `placement-phase-3`.*
