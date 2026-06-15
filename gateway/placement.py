"""Pure placement/eviction policy helpers.

Docker-free and side-effect-free so the policy can be unit-tested without a Docker
daemon (importing app.py triggers docker.from_env()). Functions operate on any object
exposing the ContainerState attributes used below (duck-typed).
"""


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
