# Stage 2 Review — concurrency-core refactor (branch `placement-hardening-stage2`)

Diligent review of the Stage 2 rewrite (PR #9: unified `LOADING/READY/STOPPING` lifecycle, single
`state_lock`, reservation-as-entry, reconciler). **Documentation only — nothing fixed.** Each item
verified against source (file:line confirmed). New findings are numbered **G1…** to distinguish from
the original F1–F16 in `review-findings.md`. Line numbers are as of `placement-hardening-stage2`.

## Fix status (branch `placement-hardening-stage3`) — ALL RESOLVED
- **G1** ✅ `_start_and_finalize` re-checks `active_containers.get(name) is entry` before flipping READY;
  if the entry was reaped/evicted mid-start it stops the orphaned container and returns None (no zombie).
- **G2** ✅ reconciler reaps a LOADING entry only when its **owner task** (tracked in `loading_tasks`)
  is absent/`done()`, not on a timer — a slow start is never killed. `GATEWAY_LOADING_TIMEOUT` kept as a
  last-resort ceiling.
- **G3/G9** ✅ `active_requests` is incremented under `state_lock` with a `status == READY` re-check
  (else 503), and decremented under `state_lock` via `_release_active` (non-streaming + streaming).
- **G7** ✅ reconciler clamps a stale READY `active_requests>0` (idle past
  `GATEWAY_REQUEST_STALE_FACTOR × GATEWAY_REQUEST_TIMEOUT`) to 0 so it's evictable again (best-effort).
- **G4** ✅ discovery baseline taken from `_settled_used(chosen)` AFTER evictions release VRAM.
- **G5** ✅ unmeasurable samples persist the `util×card` estimate, never a `0` sentinel.
- **G6** ✅ `_build_gpu_views` counts STOPPING toward `ready_footprint` and `blocked` (eviction candidates stay READY-only).
- **G8** ✅ `stop_container` claims STOPPING and returns early if already STOPPING (idempotent).
- **G10** ✅ stale comments corrected.

Verified: unit suites 29/29 + 15/15; docker-stubbed E2E for G1/G2/G3/G7/G8 (and G4/G5 via the start
flow). Not yet run on real hardware.

---

First, the good news (verified still-correct): **F1** (no global lock — `queue_count_lock` is
counter-only), **F4** (slot + entry inserted atomically under `state_lock`, no release between),
**F7** (`_build_gpu_views` blocks GPUs with a non-colocate LOADING/READY occupant), the
reserved→READY accounting handoff (atomic under one lock — no double/zero count), the colocation
two-model over-subscribe race (closed by in-lock reservation insert), `need_fn` closures (no
late-binding), and the footprint save/load round-trip (`migrate_footprints` idempotent; every reader
uses the record shape). The proxy counter/semaphore bookkeeping is also correct (decremented exactly
once; no NameError on `sem_released`/`active_req_decremented`).

---

## High

### G1 — Zombie container: `_start_and_finalize` flips a *popped* LOADING entry to READY without re-checking membership
`gateway/app.py:1157-1164`. After `start_model_container` returns, the entry is mutated to READY
under `state_lock` but the code **never verifies the entry is still in `active_containers`** and
never re-inserts:
```python
async with state_lock:
    entry.ip_address = ip_address
    entry.status = ContainerStatus.READY
    ...
    entry.vram_footprint = entry.reserved_mib
```
`start_model_container` is awaited holding **no lock**, and the reconciler (G2) can `pop` the LOADING
entry during that window (`app.py:651-655`). If it does, this block sets `status=READY` on a
**detached object** that is no longer in the dict → a running vLLM container the gateway has
forgotten: never routed to (fast-path scans the dict), never idle-reaped, VRAM leaked, Docker
container runs forever. **Severity: High. Confidence: high (mechanism), medium (trigger gated on G2).**
Fix shape (not applied): under the lock, check `active_containers.get(entry.container_name) is entry`
before flipping; otherwise `stop_container` the orphan.

### G2 — `GATEWAY_LOADING_TIMEOUT` budget ignores pre-health work, so the reconciler can reap a *legitimately* slow start (the trigger for G1)
`app.py:73-75` (default 3900 s) vs. the health loop `range(1800)*2 = 3600 s` (`app.py` health loop).
`created_at` is stamped when the LOADING entry is **inserted** (`app.py:1224`/`1302`), *before*
`start_model_container` runs its stale-container cleanup and **GGUF download**
(`download_gguf_from_repo`, many minutes for large repos) and before the 3600 s health loop. Eviction
drains (up to 30 s each) also elapse between insert and start. So a model's LOADING age can exceed
3900 s while it is still legitimately initializing → reconciler reaps it (`app.py:655`) → G1 fires.
The comment "Defaults a bit above the health-check budget … so a genuinely slow cold start is never
reaped" is **wrong** — it omits download/eviction time. **Severity: Med→High (couples with G1).
Confidence: high that the budget reasoning is unsound; trigger needs a slow/large start.**

---

## Medium

### G3 — A request can be routed to a container flipped STOPPING after placement; graceful-drain not airtight
`app.py:1403-1429`. The fast path resolves a READY `target_container` under `state_lock`, releases
the lock, then **increments `active_requests` with no lock and no status re-check**:
```python
async with state_lock:
    target_container = next(... status == READY ...)
...
target_container.active_requests += 1   # no lock; status may now be STOPPING
```
Between the lookup and the increment, the inactivity monitor or an eviction can call `stop_container`,
which flips the entry STOPPING and snapshots `active_requests` for the drain (`app.py:670-679`). Races:
(a) `stop_container` reads `active_requests==0`, drains nothing, stops the container; then this request
proxies to a dying container (→ connection error / 503-retry); (b) the increment is invisible to the
drain loop. Failure mode is a failed request (handled), not corruption — but it defeats the stated
graceful-drain guarantee. **Severity: Med. Confidence: high.**

### G4 — Discovery `before_used` is captured *before* evictions free VRAM → understated/negative measurement
`app.py:1314` (`before_used = gpus_snapshot.get(chosen,{})["used"]`). `gpus_snapshot` was taken at
`app.py:1235`, **before** the placement decision and before `stop_container` ran for any evictions.
If discovery's chosen GPU required evicting a resident, `before_used` still includes that resident's
VRAM (freed only asynchronously after stop). So `measured = max(samples) - before_used` is too low —
possibly `<= 256` (→ G5 sentinel) or a too-small `per_gpu_mib` that under-counts the model in all
future placements (over-commit/OOM risk). **Severity: Med. Confidence: med-high. Trigger: unseen
whole-card model placed onto a GPU that needed an eviction.**

### G5 — A single sub-256 sample writes a `per_gpu_mib: 0.0` sentinel that *permanently* disables both re-measurement and learned-footprint use
`app.py:1179-1182`. On `measured <= 256` the record is `{"per_gpu_mib": 0.0, "effective_tp": …}`.
Next request: `record is not None` → `is_discovery = False` (never re-measures) and `learned_mib = 0.0`
→ `need_fn = util*g.total` **forever** (`app.py:1252`). A transient bad sample (often caused by G4, or
nvidia-smi lag) locks the model onto the coarse estimate for the life of the footprints file. Partly
by-design (sentinel = "seen, unmeasurable") but the permanence + the G4 coupling make it a trap.
**Severity: Med. Confidence: high.**

### G6 — STOPPING non-colocate model is excluded from `blocked`, so co-location can land on a still-draining card
`app.py:1123-1136`. `_build_gpu_views` filters `on_gpu` to `status != STOPPING`, so a draining
non-colocate model neither contributes to `reserved`/`ready_footprint` nor blocks its GPU for
co-location. Its weights are still resident until the container actually dies; the only thing
accounting for it is `used_smi` from a snapshot taken once at `app.py:1235`. If that snapshot postdates
the process exit but the entry is still STOPPING (Docker cleanup lingering), the GPU looks free and a
colocate (or whole-card) model can be placed on top → transient over-commit/OOM. **Severity: Med.
Confidence: med (window depends on smi vs. snapshot timing).**

### G7 — Streaming `active_requests`/semaphore leak (the old F13) is live and NOT backstopped; a leaked READY count is permanently non-evictable
`app.py:1461-1468` transfers the decrement + `sem.release()` to the generator's `finally`; if the
ASGI layer never drives the generator (client disconnect before iteration), both leak. The reconciler
(`app.py:640-655`) only reaps **LOADING** orphans and idle-times-out READY — it does **not** reconcile
a leaked `active_requests` on a READY entry. Worse, `placement._evictable_lru` excludes any resident
with `active_requests != 0` (`placement.py:39-44`), so a leaked-count READY container can **never be
evicted to make room** (only its own idle-timeout can stop it; an `always_on` one never). The plan
claimed the reconciler reconciles `active_requests` best-effort; the implementation does not.
**Severity: Med. Confidence: high. Pre-existing mechanism, but the reconciler claim is unmet.**

---

## Low

### G8 — Concurrent `stop_container` on the same name is "safe by exception-swallowing", not by a claim
`app.py:665-698`. Two callers (inactivity monitor + an eviction) can both flip STOPPING, both drain
the same `state` ref (read once, used lock-free), and both run docker `stop`/`remove` — the second
relying on `NotFound`/`APIError` being caught. The final `pop(name, None)` is by **name, not
identity**. No crash, but it leans on docker raising rather than a "already STOPPING → return" guard.
**Severity: Low. Confidence: high.**

### G9 — `active_requests += 1` sits just outside the proxy `try` (no-raise window today, fragile)
`app.py:1429-1432`. Leak-free only because nothing between the increment and the `try:` can raise. A
future statement inserted there (or a cancellation at exactly that point) would leak the count with no
reclamation (see G7). Worth moving the increment to the first line inside the `try`. **Severity: Low.**

### G10 — Stale/misleading comments & docstrings
- `app.py:1163` "vram_footprint already seeded with reserved_mib at insert" — it is **not**; the insert
  leaves the dataclass default `0.0`, and the seeding actually happens on the next line (1164).
- `shutdown_inactive_containers` docstring (`app.py:624-631`) says it reconciles orphaned state and
  leaks are "self-healing" — true only for LOADING orphans, not READY `active_requests` (G7).
- `ContainerState`'s "one source of truth" note (`app.py:165`) is no longer literally true post-discovery:
  a READY entry's `reserved_mib` is a stale estimate that disagrees with the refined `vram_footprint`
  (harmless — `reserved_mib` is only read for LOADING entries). **Severity: Low.**

---

## Corrected (agent over-statements I checked and downgraded)
- **Slot-index reuse race** (proposed mid-review): NOT a real bug. STOPPING entries remain in
  `active_containers`, so their slot index is excluded from `slot_indices` (`app.py:1295`) until
  `stop_container` pops them *after* `container.remove()` — the Docker name is free by then. Contained.
- **Reserved→READY double/zero count**: NOT a bug — the flip and `vram_footprint = reserved_mib` are in
  the same `state_lock` block, and `_build_gpu_views` only runs under that lock, so no view is built
  mid-transition.
- **Degraded-mode zero-footprint GPU looking infinitely free**: NOT reachable — the `TOTAL_GPU_VRAM==0`
  path returns before `_build_gpu_views` and force-stops others first.

## Architectural note
The root of G1/G2 is that a LOADING entry's lifecycle is owned by **two** actors without a handshake:
the in-flight `_start_and_finalize` and the reconciler. A robust design gives the entry an explicit
"owned by an in-progress start" marker (or has the reconciler only reap entries with no live owning
task), and makes `_start_and_finalize` re-validate identity before the READY flip. G3/G7 share a root:
`active_requests` is mutated outside `state_lock` and is the one piece of per-entry state without a
single guarded owner.

## Suggested triage (when fixing later)
1. **G1 + G2** (zombie) — identity re-check in `_start_and_finalize` + reconciler that doesn't reap an
   entry with a live start (or a much larger/elapsed-aware LOADING budget).
2. **G4 + G5** (discovery baseline → sentinel trap) — measure `before_used` after evictions settle;
   don't persist a `0` sentinel from a single sample.
3. **G3 + G7** (`active_requests` ownership) — increment under `state_lock` with a status re-check;
   reconcile/clamp leaked READY counts.
4. **G6** (STOPPING vs colocation), then low/docs **G8–G10**.

*Verified against source on branch `placement-hardening-stage2`.*
