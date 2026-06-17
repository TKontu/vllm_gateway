"""Pure placement/eviction policy helpers.

Docker-free and side-effect-free so the policy can be unit-tested without a Docker
daemon (importing app.py triggers docker.from_env()). Functions operate on any object
exposing the ContainerState attributes used below (duck-typed).
"""

import math
from dataclasses import dataclass


@dataclass
class GpuView:
    """A point-in-time view of one GPU's VRAM for a placement decision (all MiB).

    - used_smi: actual used VRAM from nvidia-smi (includes external processes such as an
      out-of-band embedder, plus gateway models already visible to the driver).
    - reserved: optimistic reservation for gateway models still loading (not yet smi-visible).
    - ready_footprint: sum of known footprints of ready gateway models on this GPU; floors
      `used_smi` so a just-loaded model isn't under-counted while nvidia-smi catches up.
    """
    uuid: str
    total: float
    used_smi: float = 0.0
    reserved: float = 0.0
    ready_footprint: float = 0.0
    # Per-GPU cap on the TOTAL VRAM this gateway may fill across ALL its models on this card
    # (budget placement mode). math.inf = uncapped, i.e. the legacy whole-card behavior is unchanged.
    budget: float = math.inf

    @property
    def free(self) -> float:
        physical_free = self.total - max(self.used_smi, self.ready_footprint) - self.reserved
        if math.isinf(self.budget):
            return physical_free
        # In budget mode also cap by the gateway's own remaining budget. External (non-gateway)
        # VRAM is still respected through used_smi in physical_free; the budget caps only OUR models.
        gateway_used = self.ready_footprint + self.reserved
        budget_free = self.budget - gateway_used
        return min(physical_free, budget_free)


def _evictable_lru(residents, now, min_resident_seconds, allow_fresh):
    """Eviction candidates, least-recently-used first.

    Never includes an always_on model or one with in-flight requests. Unless allow_fresh,
    also excludes models still within their anti-thrash cooldown (min_resident_seconds).
    """
    candidates = [
        r for r in residents
        if not r.always_on
        and r.active_requests == 0
        and (allow_fresh or (now - r.loaded_at) >= min_resident_seconds)
    ]
    return sorted(candidates, key=lambda r: r.last_request_time)


def select_evictions(residents, needed, total_vram, current_usage, now, min_resident_seconds):
    """Pick which resident containers to evict to fit `needed` MiB.

    Returns a list of container_name strings to stop, LRU order.

    Policy:
      - If the model already fits (current_usage + needed <= total_vram), evict nothing.
      - Otherwise evict idle (in_flight == 0), non-always_on models, least-recently-used first.
      - Two passes: first only models past their min_resident_seconds cooldown (anti-thrash);
        if that can't free enough, a second pass also allows freshly-loaded models so a
        single-GPU swap is never blocked by the cooldown.
      - Best-effort: if even the second pass can't fit (remaining VRAM is held by always_on
        or in-flight models), returns the full evictable set; the caller decides what to do
        (today: start anyway; Phase 1b: queue instead of over-committing).
    """
    if current_usage + needed <= total_vram:
        return []

    best = []
    for allow_fresh in (False, True):
        chosen = []
        usage = current_usage
        for r in _evictable_lru(residents, now, min_resident_seconds, allow_fresh):
            chosen.append(r.container_name)
            usage -= r.vram_footprint
            if usage + needed <= total_vram:
                return chosen
        best = chosen  # remember the most we could free (widest set is the allow_fresh pass)
    return best


def _need(need_fn, g):
    """Per-GPU need: need_fn may be a callable(GpuView)->mib or a plain scalar (back-compat)."""
    return need_fn(g) if callable(need_fn) else need_fn


def select_gpu(candidates, residents_by_gpu, need_fn, now, min_resident_seconds):
    """Choose a GPU in a pool for a model (whole-card, no TP).

    candidates       : list[GpuView] — the pool's GPUs.
    residents_by_gpu : {uuid: [resident, ...]} — gateway containers currently on each GPU.
    need_fn          : callable(GpuView)->MiB (per-card need; e.g. util*g.total) or a scalar.

    Returns (chosen_uuid, eviction_container_names). chosen_uuid is None when no GPU can fit
    the model even after evicting every idle/non-always_on model (remaining VRAM is held by
    always_on or in-flight models, or external processes) — the caller should reject (503)
    rather than over-commit and risk an OOM.

    1. Direct fit: among GPUs with free >= need, pick the MOST free (spreads load).
    2. Otherwise guarded per-GPU LRU eviction (accept only GPUs that genuinely fit); choose the
       GPU needing the fewest evictions, tie-broken by most resulting free space.
    """
    # 1. Direct fit, most-free first.
    direct = [g for g in candidates if g.free >= _need(need_fn, g)]
    if direct:
        return max(direct, key=lambda g: g.free).uuid, []

    # 2. Eviction required — evaluate each GPU.
    options = []  # (num_evictions, -resulting_free, uuid, eviction_names)
    for g in candidates:
        cost = _gpu_fit_cost(g, residents_by_gpu.get(g.uuid, []), _need(need_fn, g), now, min_resident_seconds)
        if cost is not None:
            num_ev, resulting_free, ev = cost
            options.append((num_ev, -resulting_free, g.uuid, ev))

    if not options:
        return None, []
    options.sort()
    _, _, uuid, evictions = options[0]
    return uuid, evictions


def _homogeneous(candidates, tol=0.05) -> bool:
    """True iff all candidate GPUs have ~equal total VRAM (within `tol` of the largest).

    Tensor-parallel requires equal-VRAM cards; this decisively rejects a mixed pool
    (e.g. a 24 GB 3090 + a 6 GB A2000) while tolerating minor driver/ECC reporting drift."""
    totals = [c.total for c in candidates]
    if not totals:
        return False
    hi = max(totals)
    return hi > 0 and (hi - min(totals)) <= tol * hi


def minimal_tp_to_fit(weight_bytes, card_total_mib, util, overhead=1.2, pool_size=None) -> int:
    """Minimal tensor-parallel degree so a model's weights fit across homogeneous cards.

    weight_bytes : estimated total model weight size in bytes (None/0 -> 1, i.e. no split).
    Per card, vLLM can use ~util*card_total_mib; `overhead` (~1.2) allows for activations /
    CUDA context / fragmentation on top of raw weights. Result is capped at pool_size."""
    if not weight_bytes or weight_bytes <= 0:
        return 1
    usable_per_card = util * card_total_mib
    if usable_per_card <= 0:
        return 1
    weight_mib = (weight_bytes / (1024.0 * 1024.0)) * overhead
    tp = max(1, math.ceil(weight_mib / usable_per_card))
    if pool_size is not None:
        tp = min(tp, pool_size)
    return tp


def kv_cache_mib(max_model_len, max_num_seqs, num_layers, num_kv_heads, head_dim, dtype_bytes,
                 sliding_window=0, num_sliding_layers=0) -> float:
    """Estimated TOTAL KV-cache size (MiB) for a model held resident at the given context/concurrency.

    Per (full-attention) layer a sequence needs `max_model_len` tokens of KV; a sliding-window layer
    only needs `min(sliding_window, max_model_len)` (Gemma 3, Mistral interleave these). So:
        token-layers/seq = full_layers * L + sliding_layers * min(window, L)        (L = max_model_len)
        KV bytes = 2 (key+value) * num_kv_heads * head_dim * dtype_bytes * token-layers/seq * seqs
    With `sliding_window<=0` or `num_sliding_layers<=0` every layer is full (legacy behavior). This is
    the whole-model KV; under tensor parallel the caller divides by tp. Returns 0.0 if any core input
    is non-positive (unknown -> caller falls back to discovery).
    """
    vals = (max_model_len, max_num_seqs, num_layers, num_kv_heads, head_dim, dtype_bytes)
    if any((v is None or v <= 0) for v in vals):
        return 0.0
    if sliding_window and sliding_window > 0 and num_sliding_layers and num_sliding_layers > 0:
        sliding = min(int(num_sliding_layers), int(num_layers))
        full = int(num_layers) - sliding
        eff_sliding_len = min(int(sliding_window), int(max_model_len))
    else:
        full, sliding, eff_sliding_len = int(num_layers), 0, 0
    token_layers_per_seq = full * int(max_model_len) + sliding * eff_sliding_len
    total_bytes = 2 * int(num_kv_heads) * int(head_dim) * int(dtype_bytes) \
        * token_layers_per_seq * int(max_num_seqs)
    return float(total_bytes) / (1024.0 * 1024.0)


# Canonical keys of a footprint "signature" — the inputs that determine a model's measured size.
# A persisted footprint is reused only when its stored signature matches the current request's, so a
# mode switch or a config change (context length / concurrency / TP / util basis) forces re-measure.
_SIGNATURE_KEYS = ("mode", "max_model_len", "max_num_seqs", "effective_tp", "util_basis")


def footprint_signature(mode, max_model_len, max_num_seqs, effective_tp, util_basis=0.0) -> dict:
    """Build the signature stamped onto a footprint record (and compared on reuse)."""
    return {
        "mode": mode,
        "max_model_len": int(max_model_len),
        "max_num_seqs": int(max_num_seqs),
        "effective_tp": int(effective_tp),
        "util_basis": round(float(util_basis), 4),
    }


def signature_matches(record, sig) -> bool:
    """True iff `record` carries a signature equal to `sig` across all canonical keys.

    Legacy records (no signature, or `{}`) never match -> they are treated as unseen and re-measured,
    which is the safe default after a schema/behavior change."""
    stored = record.get("signature") if isinstance(record, dict) else None
    if not isinstance(stored, dict) or not stored:
        return False
    return all(stored.get(k) == sig.get(k) for k in _SIGNATURE_KEYS)


def attribute_vram(compute_app_rows, host_pids, gpu_uuids) -> float:
    """Sum the GPU memory (MiB) of compute processes that belong to `host_pids` on `gpu_uuids`.

    `compute_app_rows`: iterable of (gpu_uuid, pid, used_mib) from
    `nvidia-smi --query-compute-apps`. Sums across all of a model's `gpu_uuids` (so a tensor-parallel
    model's footprint is its total across cards). Returns 0.0 when nothing matches — the caller treats
    that as "attribution unavailable" and falls back to delta measurement.
    """
    pids = {int(p) for p in host_pids}
    uuids = set(gpu_uuids)
    total = 0.0
    for row in compute_app_rows:
        try:
            guid, pid, mib = row
            if int(pid) in pids and guid in uuids:
                total += float(mib)
        except (TypeError, ValueError):
            continue
    return total


def estimate_need_mib(weights_mib, kv_mib_total, tp, overhead_factor, fixed_overhead_mib) -> float:
    """Per-card VRAM need (MiB) for a model in budget mode.

    Weights and KV both shard across `tp` cards; the result is scaled by `overhead_factor`
    (activations / fragmentation) and a per-card `fixed_overhead_mib` (CUDA context, cudagraphs)
    that does NOT shard. Bias these generous: the launch util cap makes vLLM physically unable to
    exceed this, so an under-estimate only fails THIS model's own startup, never a co-resident.
    """
    tp = max(1, int(tp or 1))
    shardable = (max(0.0, weights_mib) + max(0.0, kv_mib_total)) / tp
    return shardable * overhead_factor + fixed_overhead_mib


def _gpu_fit_cost(g, residents, needed, now, min_resident_seconds):
    """Can GPU `g` host `needed` MiB? Returns (num_evictions, resulting_free, eviction_names)
    if it can (directly or after guarded eviction), else None."""
    if g.free >= needed:
        return (0, g.free, [])
    evictions = select_evictions(
        residents=residents, needed=needed, total_vram=g.total,
        current_usage=g.total - g.free, now=now, min_resident_seconds=min_resident_seconds,
    )
    footprint = {r.container_name: r.vram_footprint for r in residents}
    freed = sum(footprint.get(n, 0.0) for n in evictions)
    if g.free + freed >= needed:
        return (len(evictions), g.free + freed, evictions)
    return None


def compute_effective_tp(weight_bytes, configured_tp, prior_tp, pool_totals, util, overhead=1.2):
    """Deterministic tensor-parallel degree for a model — computed every request (not gated on
    discovery), so the decision never reverts between requests (fixes F2).

    Precedence: forced config tp (>1) → a persisted prior tp (>1) → minimal TP to fit the weights
    on a HOMOGENEOUS, multi-GPU pool → 1. `pool_totals` is the list of per-GPU totals (MiB) for the
    model's pool. `weight_bytes` may be None (unknown) → no auto-split.
    """
    if configured_tp and configured_tp > 1:
        return configured_tp
    if prior_tp and prior_tp > 1:
        return prior_tp
    if len(pool_totals) >= 2 and max(pool_totals) > 0 and \
            (max(pool_totals) - min(pool_totals)) <= 0.05 * max(pool_totals):
        return minimal_tp_to_fit(weight_bytes, max(pool_totals), util,
                                 overhead=overhead, pool_size=len(pool_totals))
    return 1


def select_placement(candidates, residents_by_gpu, need_fn, tp, now, min_resident_seconds):
    """Choose GPU(s) for a model. Returns (chosen_uuids: list | None, eviction_names: list).

    need_fn : callable(GpuView)->MiB (per-card need) or a scalar.
    tp == 1 : single-GPU placement (delegates to select_gpu — identical to Phase 1b).
    tp >= 2 : tensor-parallel across exactly `tp` GPUs of a HOMOGENEOUS pool. Each chosen GPU must
              host need_fn(g) (vLLM applies gpu_memory_utilization to every card, so the per-card
              need does NOT shrink with tp). Picks the `tp` GPUs needing the fewest evictions
              (tie-break: most resulting free). None if the pool is heterogeneous, has fewer than
              `tp` GPUs, or `tp` GPUs can't fit.
    """
    if tp <= 1:
        uuid, evictions = select_gpu(candidates, residents_by_gpu, need_fn, now, min_resident_seconds)
        return ([uuid], evictions) if uuid is not None else (None, [])

    if not _homogeneous(candidates) or len(candidates) < tp:
        return None, []

    options = []  # (num_evictions, -resulting_free, uuid, eviction_names)
    for g in candidates:
        cost = _gpu_fit_cost(g, residents_by_gpu.get(g.uuid, []), _need(need_fn, g), now, min_resident_seconds)
        if cost is not None:
            num_ev, resulting_free, ev = cost
            options.append((num_ev, -resulting_free, g.uuid, ev))

    if len(options) < tp:
        return None, []
    options.sort()
    chosen = options[:tp]
    chosen_uuids = [o[2] for o in chosen]
    # Dedupe eviction names (a multi-GPU/TP resident can appear on several chosen GPUs).
    evictions = list(dict.fromkeys(n for o in chosen for n in o[3]))
    return chosen_uuids, evictions


def select_colocated(candidates, residents_by_gpu, need_fn, blocked_gpus, now, min_resident_seconds):
    """Choose a GPU for a co-locatable model that may SHARE a card with other co-locatable models.

    Returns (chosen_uuid | None, eviction_names). Single GPU.

    A GPU is eligible only if it is NOT in `blocked_gpus` (GPUs hosting a non-colocate LOADING or
    READY model — closes F7) and every resident on it is itself co-locatable. Among eligible GPUs:
    direct fit (free >= need_fn(g)) most-free first; else evict idle co-residents guarded-LRU only
    as much as needed; else None (caller 503s).
    """
    blocked = blocked_gpus or set()
    eligible = [
        g for g in candidates
        if g.uuid not in blocked
        and all(getattr(r, "colocate", False) for r in residents_by_gpu.get(g.uuid, []))
    ]

    # 1. Direct fit, most-free first.
    direct = [g for g in eligible if g.free >= _need(need_fn, g)]
    if direct:
        return max(direct, key=lambda g: g.free).uuid, []

    # 2. Partial guarded eviction of idle co-residents.
    options = []  # (num_evictions, -resulting_free, uuid, eviction_names)
    for g in eligible:
        cost = _gpu_fit_cost(g, residents_by_gpu.get(g.uuid, []), _need(need_fn, g), now, min_resident_seconds)
        if cost is not None:
            num_ev, resulting_free, ev = cost
            options.append((num_ev, -resulting_free, g.uuid, ev))

    if not options:
        return None, []
    options.sort()
    _, _, uuid, evictions = options[0]
    return uuid, evictions
