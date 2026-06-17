# Review findings — budget placement mode (Phase 4) — RE-VERIFIED

> **Status (Phase 5 hardening applied).** All confirmed findings below are now addressed:
> - **B1** weights via `HfApi().model_info` (single-file + sharded + gated) — `gateway/app.py:estimate_weight_bytes`.
> - **B4** sliding-window-aware KV (`_sliding_window_spec` + extended `placement.kv_cache_mib`) — ~5× tighter for Gemma 3.
> - **B2** footprints stamped with a `signature` (mode + max_model_len + max_num_seqs + tp + util); reused only on match (`placement.signature_matches`) → no cross-mode / stale reuse.
> - **B3** memory-affecting `extra_args` rejected at startup in budget mode (`config_loader.validate_extra_args_budget`).
> - **B5** footprint is now the **measured** per-process VRAM (`measure_model_vram` / `placement.attribute_vram`), not the reserve estimate → accounting is ground-truth.
> - **B6** `VLLM_MAX_NUM_SEQS=0` coerced to 16 (no startup break).
> - **B7** un-estimable models load sole-occupant once, are measured, then pack at the measured size.
> - **B15** `/gateway/status` now exposes `placement_mode`, per-GPU `budget_mib`, and per-model measured footprints.
> - **B8/B16** config.json fetched once (cached); stale comment fixed.
> See the updated plan and `tests/test_placement.py` / `tests/test_config_resolution.py`.


Re-review of `PLACEMENT_MODE=budget`. Every finding below was re-checked against the actual code;
several from the first pass were **overstated and are corrected here**. Documentation only — nothing
is fixed. The focus is on what can actually happen in this deployment (single-GPU pin, AWQ/safetensors
`cyankiwi/*` + gemma models, budget mode), not theoretical corners.

## The key safety property (verified, and it holds)

`--gpu-memory-utilization` is in `GATEWAY_MANAGED_FLAGS` (app.py:894), so `extra_args` cannot override
it. In budget mode each model launches at `effective_util = min(GPU_BUDGET_FRACTION, budget_need/total)`,
and `effective_util × total ≤ budget_need = reserve_amt`. vLLM physically cannot exceed
`effective_util × total`, so **it always takes ≤ what the gateway reserved.**

Consequence: **in pure budget mode a neighbor model can never be OOM'd**, no matter how wrong the need
estimate is or what's in `extra_args`. A bad estimate only starves or fails *that* model's own startup.
This is the strong guarantee, and it is real. The OOM-flavored findings from the first pass survive
**only** on the non-budget launch paths (whole_card, and the budget-discovery fallback), where
`effective_util = None` and vLLM instead uses the configured `gpu_memory_utilization` uncapped.

Concurrency/lifecycle invariants (G1–G10/F1): re-confirmed intact (no I/O under `state_lock`, clean
LOADING→READY→STOPPING, no new orphan path).

---

## CONFIRMED real and relevant to this deployment

### B1 — Single-file `model.safetensors` (no shard index) → model loads whole-card, never packs
**Verified:** `estimate_weight_bytes` (app.py:814) fetches **only** `model.safetensors.index.json`. A model
shipped as one `model.safetensors` (no index — common for small/quantized models) gets a 404 there and
returns `None`. `estimate_model_need_mib` then returns `None`, so `_ensure_started` takes the
`budget_discovery` path: `effective_util=None`, `need_fn = util*g.total` → the model loads **whole-card**
and is discovery-measured at its configured `gpu_memory_utilization`. On reload the measured (~whole-card)
value is reused, so it **never packs**.

**Why it matters here:** your smaller AWQ targets (e.g. `gemma-4-E4B`, ~2–3 GB) are plausibly single-file
→ this silently defeats packing for exactly the model the feature was meant to co-locate. This is the
single most likely finding to bite in practice.

**Partial workaround that exists today (undocumented):** set a low per-model `gpu_memory_utilization`
(e.g. 0.45) on such a model; the discovery measurement is then taken at that util, so it packs partially.

### B4 — KV formula ignores sliding-window attention → over-sizes Gemma 3, hurts packing
**Verified:** `kv_cache_mib` (placement.py) multiplies over **all** layers at **full** `max_model_len`;
`get_model_kv_spec` never reads `sliding_window` or the local/global layer pattern. Gemma 3 uses a
~1024 sliding window on ~5 of every 6 layers, so the true resident KV is a small fraction of the full
figure — the estimate can be ~5–6× too large.

**Why it matters here:** you run gemma models, and `google/gemma-3-4b-it` is the first example in the
config. The over-estimate is **safe** (it only over-reserves, never OOMs), but it directly shrinks how
many models pack per card — undercutting the feature for these exact models. Mistral sliding-window
variants are affected the same way.

### B2(a) — Stale footprints reused without invalidation (deterministic)
**Verified:** `known_footprints` is keyed by **repo only** and stores `effective_util`/`measured_at` but
**not the placement mode or the `max_model_len`/`max_num_seqs` it was sized for**. `learned_mib` is read
unconditionally and reused as the GGUF/un-estimable need (app.py ~1467) and as the whole_card need
(~1475).

**Real, deterministic effects (no timing window needed):**
- On **upgrade** from a whole_card-era deployment, `memory_footprints.json` already holds whole-card
  measurements. For estimable sharded-safetensors models budget mode recomputes and ignores them (fine),
  but for **single-file (B1) or GGUF** models it reuses the stale whole-card value → they never pack.
- Changing a model's `max_model_len`/`max_num_seqs` does **not** invalidate a persisted record for the
  un-estimable models, so their size won't track the new config.

This is the existing review note **A1** (footprints not keyed by placement context), now load-bearing.

---

## CONFIRMED but conditional (real, narrower trigger)

### B2(b) — whole_card after budget can over-commit (the one real neighbor-OOM vector)
If you run budget mode (persisting *tight* footprints), then switch to `PLACEMENT_MODE=whole_card`
without clearing the file, model A is sized for placement at its tight `learned_mib` yet launched
uncapped (`effective_util=None` → configured 0.90 util → ~whole card). Its `vram_footprint` (= the tight
reserve) under-floors `ready_footprint`. If a second model's placement snapshot is taken while A is still
LOADING (so nvidia-smi hasn't yet caught up to A's real usage), the gateway can believe the card has room
and place B → genuine **OOM** when both finish allocating.

**Honest scope:** requires (1) a budget→whole_card switch without clearing footprints, and (2) two models
loading concurrently in that window. Not a steady-state hazard (once A is READY, `used_smi` reflects
reality and the floor is moot). Real, but conditional and avoidable by clearing footprints on a mode
switch. This is the **only** confirmed neighbor-OOM path.

### B7 — GGUF models behave whole-card in budget mode (and evict peers to self-measure)
Same `None`-estimate path as B1. First load reserves ~whole card and `select_placement` **evicts
co-resident packed models** just to measure one GGUF model; on reload it stays whole-card-sized. Only
relevant if you actually run GGUF models (your current set is AWQ safetensors, so N/A today — but the
gateway advertises GGUF support, so worth knowing).

### B3 — `extra_args` memory flags diverge from the estimate → model's own startup may fail (NOT a neighbor OOM)
**Corrected from the first pass.** `--max-model-len`/`--max-num-seqs`/`--kv-cache-dtype` in `extra_args`
aren't managed, so the launched config can differ from the structured values used in the estimate. But
because the util cap **is** managed, the consequence is bounded: e.g. `max_model_len: 4096` (tight util)
+ `extra_args: ["--max-model-len","32768"]` → vLLM is given the small KV budget but asked for 32k context
and **fails its own startup** with the "KV cache insufficient" error you already saw — it cannot OOM a
neighbor. `--enable-prefix-caching` (shipped for `qwen2.5-coder`) does **not** increase VRAM beyond the
util budget, so it's a non-issue. Real footgun (silent own-startup failure / mis-size), Medium at most,
no OOM.

### B6 — `VLLM_MAX_NUM_SEQS=0` now fails startup
**Verified** real: the env default feeds `max_num_seqs` whose validator requires `>= 1`, so `0`
(previously the "omit the flag" sentinel) now raises at startup. **But** the default is 16 and nothing
documents 0 as meaningful, so it's unlikely anyone set it. Clean fail-fast, not silent. Low likelihood;
note in the upgrade docs and move on.

---

## Real but minor (keep in mind, low impact)

- **B5** — `reserve_amt = budget_need` vs `effective_util` clamped at the cap: when `budget_need/total ≥
  GPU_BUDGET_FRACTION` the reservation over-counts what vLLM took (under-packs). Conservative, no harm;
  the code comment "vLLM takes ONLY this need" is inexact at the cap.
- **B8** — `config.json` is fetched twice per cold start (`get_model_max_len` + `get_model_kv_spec`) and
  weights up to twice. A few redundant GETs; negligible vs a multi-minute load.
- **B9** — KV estimate uses configured `max_model_len`, not capped to native; over-reserves when
  configured > native (safe).
- **B12** — `get_model_kv_spec`'s `text_config` heuristic is best-effort; multimodal configs that nest the
  LM under a different key, or omit the fields, return `None` → discovery → whole-card (compounds B1).
  Worth confirming the gemma multimodal configs expose the fields under `text_config` as expected.
- **B15** — `/gateway/status` doesn't expose `PLACEMENT_MODE`, the per-GPU budget cap, or each model's
  effective need/util — budget-mode packing is hard to observe operationally.
- **B16** — stale comment in `_ensure_started` (record shape omits `effective_util`).
- **B17** — design note: for estimable models the need is recomputed every cold start, so the persisted
  `per_gpu_mib` is display-only; `learned_mib` is load-bearing only for the un-estimable (B1/GGUF) path.

---

## Considered and dropped as theoretical (won't happen in practice)

- `head_dim` fallback wrong (B10): only if `config.json` omits `head_dim` AND the model uses a
  non-standard one — modern configs include it.
- KV-dtype guessed as 2 bytes when fp8 KV set via `extra_args` (B11): rare config, and it only
  over-reserves (safe).
- Mamba/SSM/hybrid KV formula invalid (B13): not in this model set.
- `budget_free` negative on mid-flight budget reduction (B18): handled by the eviction math; not a real
  workflow.
- `round(util,4)` → 0 for a sub-2.4 MiB need (B19): impossible — `BUDGET_OVERHEAD_MIB` (1024) floors
  every need far above that.

---

## Bottom line

- **No neighbor-OOM risk in pure budget mode** — the util cap guarantees it. The one real OOM path (B2b)
  needs a budget→whole_card switch with a stale footprints file and a concurrent load.
- **The findings that will actually bite you:** **B1** (single-file safetensors → no packing) and **B4**
  (sliding-window over-sizing on gemma) — both quietly defeat packing for your exact models, and both
  fail *safe* (over-reserve / load-alone), so you'd see "models still swapping / not co-residing" rather
  than a crash. **B2(a)** matters the moment you reuse an existing `memory_footprints.json` across the
  mode change.
- Everything else is either conditional on workflows you may not hit (B2b, B6, B7) or genuinely minor.
