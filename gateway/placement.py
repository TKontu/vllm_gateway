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

    @property
    def free(self) -> float:
        return self.total - max(self.used_smi, self.ready_footprint) - self.reserved


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


def select_gpu(candidates, residents_by_gpu, needed, now, min_resident_seconds):
    """Choose a GPU in a pool for a model needing `needed` MiB (whole-card, no TP).

    candidates       : list[GpuView] — the pool's GPUs.
    residents_by_gpu : {uuid: [resident, ...]} — gateway containers currently on each GPU.

    Returns (chosen_uuid, eviction_container_names). chosen_uuid is None when no GPU can fit
    the model even after evicting every idle/non-always_on model (remaining VRAM is held by
    always_on or in-flight models, or external processes) — the caller should reject (503)
    rather than over-commit and risk an OOM.

    1. Direct fit: among GPUs with free >= needed, pick the MOST free (spreads load).
    2. Otherwise, for each GPU compute a guarded LRU eviction set (select_evictions, scoped to
       that GPU) and accept it only if it actually frees enough. Choose the GPU needing the
       fewest evictions, tie-broken by most resulting free space.
    """
    # 1. Direct fit, most-free first.
    direct = [g for g in candidates if g.free >= needed]
    if direct:
        best = max(direct, key=lambda g: g.free)
        return best.uuid, []

    # 2. Eviction required — evaluate each GPU.
    options = []  # (num_evictions, -resulting_free, uuid, eviction_names)
    for g in candidates:
        residents = residents_by_gpu.get(g.uuid, [])
        used_eff = g.total - g.free  # effective used (smi/floor + reservation)
        evictions = select_evictions(
            residents=residents,
            needed=needed,
            total_vram=g.total,
            current_usage=used_eff,
            now=now,
            min_resident_seconds=min_resident_seconds,
        )
        footprint = {r.container_name: r.vram_footprint for r in residents}
        freed = sum(footprint.get(n, 0.0) for n in evictions)
        if g.free + freed >= needed:  # only accept GPUs that genuinely fit after eviction
            resulting_free = g.free + freed
            options.append((len(evictions), -resulting_free, g.uuid, evictions))

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


def select_placement(candidates, residents_by_gpu, needed_per_gpu, tp, now, min_resident_seconds):
    """Choose GPU(s) for a model. Returns (chosen_uuids: list | None, eviction_names: list).

    tp == 1 : single-GPU placement (delegates to select_gpu — identical to Phase 1b).
    tp >= 2 : tensor-parallel across exactly `tp` GPUs of a HOMOGENEOUS pool. Each chosen GPU
              must host `needed_per_gpu` (vLLM applies gpu_memory_utilization to every card, so
              the per-card need does NOT shrink with tp). Picks the `tp` GPUs needing the fewest
              evictions (tie-break: most resulting free), evicting guarded-LRU per GPU. Returns
              None if the pool is heterogeneous, has fewer than `tp` GPUs, or `tp` GPUs can't fit.
    """
    if tp <= 1:
        uuid, evictions = select_gpu(candidates, residents_by_gpu, needed_per_gpu, now, min_resident_seconds)
        return ([uuid], evictions) if uuid is not None else (None, [])

    if not _homogeneous(candidates) or len(candidates) < tp:
        return None, []

    options = []  # (num_evictions, -resulting_free, uuid, eviction_names)
    for g in candidates:
        cost = _gpu_fit_cost(g, residents_by_gpu.get(g.uuid, []), needed_per_gpu, now, min_resident_seconds)
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


def select_colocated(candidates, residents_by_gpu, needed, now, min_resident_seconds):
    """Choose a GPU for a co-locatable model that may SHARE a card with other co-locatable models.

    Returns (chosen_uuid | None, eviction_names). Single GPU.

    A GPU is eligible only if it is empty or every resident on it is itself co-locatable
    (a co-locatable model never shares with a whole-card model). Among eligible GPUs: direct fit
    (free >= needed) most-free first; else evict idle co-residents guarded-LRU *only as much as
    needed* (reuse _gpu_fit_cost — it stops once `needed` fits, so co-residents are kept when
    possible); else None (caller 503s).
    """
    eligible = [
        g for g in candidates
        if all(getattr(r, "colocate", False) for r in residents_by_gpu.get(g.uuid, []))
    ]

    # 1. Direct fit, most-free first.
    direct = [g for g in eligible if g.free >= needed]
    if direct:
        return max(direct, key=lambda g: g.free).uuid, []

    # 2. Partial guarded eviction of idle co-residents.
    options = []  # (num_evictions, -resulting_free, uuid, eviction_names)
    for g in eligible:
        cost = _gpu_fit_cost(g, residents_by_gpu.get(g.uuid, []), needed, now, min_resident_seconds)
        if cost is not None:
            num_ev, resulting_free, ev = cost
            options.append((num_ev, -resulting_free, g.uuid, ev))

    if not options:
        return None, []
    options.sort()
    _, _, uuid, evictions = options[0]
    return uuid, evictions
