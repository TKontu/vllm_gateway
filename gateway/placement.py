"""Pure placement/eviction policy helpers.

Docker-free and side-effect-free so the policy can be unit-tested without a Docker
daemon (importing app.py triggers docker.from_env()). Functions operate on any object
exposing the ContainerState attributes used below (duck-typed).
"""

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
