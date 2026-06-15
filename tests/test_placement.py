"""Unit tests for the eviction/placement policy (placement.select_evictions).

Runnable directly (no pytest required):  python tests/test_placement.py
Also discoverable by pytest. Imports only placement (docker-free).
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway"))

from placement import (  # noqa: E402
    select_evictions, select_gpu, GpuView, select_placement, minimal_tp_to_fit, _homogeneous,
    select_colocated, compute_effective_tp,
)


@dataclass
class R:  # minimal stand-in for ContainerState
    container_name: str
    vram_footprint: float
    last_request_time: float
    loaded_at: float = 0.0
    active_requests: int = 0
    always_on: bool = False
    colocate: bool = False


NOW = 1_000_000.0
OLD = NOW - 10_000  # well past any cooldown


def test_no_eviction_when_it_fits():
    residents = [R("a", 8000, OLD, loaded_at=OLD)]
    assert select_evictions(residents, needed=8000, total_vram=24000,
                            current_usage=8000, now=NOW, min_resident_seconds=90) == []


def test_lru_order_and_minimal_set():
    # total 24000, already 24000 used by three 8000 models; need 8000 -> evict just the oldest.
    residents = [
        R("new", 8000, NOW - 1, loaded_at=OLD),
        R("old", 8000, NOW - 100, loaded_at=OLD),
        R("mid", 8000, NOW - 50, loaded_at=OLD),
    ]
    out = select_evictions(residents, needed=8000, total_vram=24000,
                           current_usage=24000, now=NOW, min_resident_seconds=90)
    assert out == ["old"], out  # LRU (smallest last_request_time), and only as many as needed


def test_never_evicts_always_on():
    residents = [R("keep", 20000, OLD, loaded_at=OLD, always_on=True)]
    # Needs room but the only resident is always_on -> nothing evictable (best-effort empty).
    out = select_evictions(residents, needed=8000, total_vram=24000,
                           current_usage=20000, now=NOW, min_resident_seconds=90)
    assert out == [], out


def test_never_evicts_in_flight():
    residents = [R("busy", 20000, OLD, loaded_at=OLD, active_requests=3)]
    out = select_evictions(residents, needed=8000, total_vram=24000,
                           current_usage=20000, now=NOW, min_resident_seconds=90)
    assert out == [], out


def test_cooldown_prefers_old_then_falls_back_to_fresh():
    # One fresh model fully occupies the card. A swap must still happen (single-GPU),
    # so the fallback pass allows evicting the fresh model rather than blocking forever.
    residents = [R("fresh", 24000, NOW - 5, loaded_at=NOW - 5)]  # within 90s cooldown
    out = select_evictions(residents, needed=24000, total_vram=24000,
                           current_usage=24000, now=NOW, min_resident_seconds=90)
    assert out == ["fresh"], out


def test_cooldown_respected_when_an_older_candidate_suffices():
    # An older model alone frees enough -> the fresh one is preserved (anti-thrash).
    residents = [
        R("fresh", 8000, NOW - 5, loaded_at=NOW - 5),     # in cooldown
        R("older", 8000, NOW - 500, loaded_at=OLD),       # past cooldown, LRU
        R("oldest", 8000, NOW - 900, loaded_at=OLD),      # past cooldown, most-LRU
    ]
    # 24000 used, need 8000 -> evict one. Should pick the most-LRU past-cooldown model only.
    out = select_evictions(residents, needed=8000, total_vram=24000,
                           current_usage=24000, now=NOW, min_resident_seconds=90)
    assert out == ["oldest"], out


def test_best_effort_when_cannot_fit():
    # Remaining capacity held by an always_on model; the one evictable can't free enough.
    residents = [
        R("pinned", 20000, OLD, loaded_at=OLD, always_on=True),
        R("small", 2000, OLD, loaded_at=OLD),
    ]
    out = select_evictions(residents, needed=10000, total_vram=24000,
                           current_usage=22000, now=NOW, min_resident_seconds=90)
    # Best effort: evict what we can (the small one); caller logs/starts best-effort.
    assert out == ["small"], out


# --- select_gpu (multi-GPU placement) ---

def test_gpuview_free_floors_used_by_ready_footprint():
    # A just-loaded model not yet in nvidia-smi: ready_footprint floors used_smi.
    g = GpuView("A", total=24000, used_smi=500, reserved=0, ready_footprint=20000)
    assert g.free == 24000 - 20000  # floored by ready_footprint, not the lagging 500
    g2 = GpuView("B", total=24000, used_smi=21000, reserved=1000, ready_footprint=0)
    assert g2.free == 24000 - 21000 - 1000


def test_select_gpu_direct_fit_picks_most_free():
    a = GpuView("A", total=24000, used_smi=10000)   # free 14000
    b = GpuView("B", total=24000, used_smi=2000)     # free 22000
    chosen, ev = select_gpu([a, b], {"A": [], "B": []}, 8000, NOW, 90)
    assert chosen == "B" and ev == [], (chosen, ev)


def test_select_gpu_external_process_excludes_gpu():
    # A2000 (small, busy with an external embedder) can't fit; 3090 can.
    a2000 = GpuView("A2000", total=6000, used_smi=4000)   # free 2000
    rtx = GpuView("3090", total=24000, used_smi=0)        # free 24000
    chosen, ev = select_gpu([a2000, rtx], {"A2000": [], "3090": []}, 8000, NOW, 90)
    assert chosen == "3090" and ev == []


def test_select_gpu_evicts_when_needed():
    # GPU full with one old idle resident; eviction frees enough.
    g = GpuView("A", total=24000, used_smi=20000, ready_footprint=20000)
    resident = R("old", vram_footprint=20000, last_request_time=OLD, loaded_at=OLD)
    chosen, ev = select_gpu([g], {"A": [resident]}, 18000, NOW, 90)
    assert chosen == "A" and ev == ["old"], (chosen, ev)


def test_select_gpu_no_fit_returns_none():
    # Only resident is always_on and occupies the card -> cannot fit -> None (caller 503s).
    g = GpuView("A", total=24000, used_smi=22000, ready_footprint=22000)
    pinned = R("pinned", vram_footprint=22000, last_request_time=OLD, loaded_at=OLD, always_on=True)
    chosen, ev = select_gpu([g], {"A": [pinned]}, 8000, NOW, 90)
    assert chosen is None and ev == [], (chosen, ev)


def test_select_gpu_prefers_fewer_evictions():
    # Two GPUs can fit after eviction; pick the one needing fewer evictions.
    g1 = GpuView("G1", total=24000, used_smi=24000, ready_footprint=24000)  # needs 2 evictions
    g1r = [R("g1a", 12000, OLD, loaded_at=OLD), R("g1b", 12000, OLD - 1, loaded_at=OLD)]
    g2 = GpuView("G2", total=24000, used_smi=24000, ready_footprint=24000)  # needs 1 eviction
    g2r = [R("g2a", 24000, OLD, loaded_at=OLD)]
    chosen, ev = select_gpu([g1, g2], {"G1": g1r, "G2": g2r}, 20000, NOW, 90)
    assert chosen == "G2" and ev == ["g2a"], (chosen, ev)


# --- select_placement / TP (Phase 2) ---

def test_homogeneous_helper():
    assert _homogeneous([GpuView("A", 24000), GpuView("B", 24000)])
    assert _homogeneous([GpuView("A", 24000), GpuView("B", 23500)])   # within 5%
    assert not _homogeneous([GpuView("A", 24000), GpuView("B", 6000)])  # 3090 + A2000


def test_minimal_tp_to_fit():
    # weights fit one card -> 1
    assert minimal_tp_to_fit(10 * 1024**3, 24000, 0.9) == 1
    # ~30 GiB weights * 1.2 overhead = 36 GiB; usable/card = 0.9*24000 MiB ≈ 21.6 GiB -> 2
    assert minimal_tp_to_fit(30 * 1024**3, 24000, 0.9) == 2
    # huge weights capped at pool size
    assert minimal_tp_to_fit(500 * 1024**3, 24000, 0.9, pool_size=2) == 2
    # unknown size -> no split
    assert minimal_tp_to_fit(None, 24000, 0.9) == 1
    assert minimal_tp_to_fit(0, 24000, 0.9) == 1


def test_select_placement_tp1_matches_select_gpu():
    a = GpuView("A", 24000, used_smi=10000)
    b = GpuView("B", 24000, used_smi=2000)
    rbg = {"A": [], "B": []}
    sp = select_placement([a, b], rbg, 8000, 1, NOW, 90)
    sg = select_gpu([a, b], rbg, 8000, NOW, 90)
    assert sp == ([sg[0]], sg[1]), (sp, sg)   # tp==1 wraps select_gpu identically


def test_select_placement_forced_tp2_picks_two_most_free():
    a = GpuView("A", 24000, used_smi=20000)  # free 4000
    b = GpuView("B", 24000, used_smi=0)       # free 24000
    c = GpuView("C", 24000, used_smi=1000)    # free 23000
    chosen, ev = select_placement([a, b, c], {"A": [], "B": [], "C": []}, 8000, 2, NOW, 90)
    assert sorted(chosen) == ["B", "C"] and ev == [], (chosen, ev)


def test_select_placement_tp_rejects_heterogeneous():
    a = GpuView("3090", 24000, used_smi=0)
    b = GpuView("A2000", 6000, used_smi=0)
    chosen, ev = select_placement([a, b], {"3090": [], "A2000": []}, 4000, 2, NOW, 90)
    assert chosen is None, chosen


def test_select_placement_tp_insufficient_gpus():
    # forced tp=2 with only one GPU visible (2nd 3090 not installed) -> None, graceful
    a = GpuView("A", 24000, used_smi=0)
    chosen, ev = select_placement([a], {"A": []}, 8000, 2, NOW, 90)
    assert chosen is None, chosen


def test_select_placement_tp_evicts_and_dedupes():
    # Both GPUs full; each has an idle resident; tp=2 must evict on both, deduped.
    a = GpuView("A", 24000, used_smi=24000, ready_footprint=24000)
    b = GpuView("B", 24000, used_smi=24000, ready_footprint=24000)
    ra = [R("ra", 24000, OLD, loaded_at=OLD)]
    rb = [R("rb", 24000, OLD, loaded_at=OLD)]
    chosen, ev = select_placement([a, b], {"A": ra, "B": rb}, 20000, 2, NOW, 90)
    assert sorted(chosen) == ["A", "B"]
    assert sorted(ev) == ["ra", "rb"] and len(ev) == len(set(ev)), ev


# --- select_colocated (Phase 3) ---

def test_colocated_empty_gpu_direct_fit():
    g = GpuView("A", 24000, used_smi=0)
    uuid, ev = select_colocated([g], {"A": []}, 10800, set(), NOW, 90)
    assert uuid == "A" and ev == []


def test_colocated_shares_with_colocate_resident():
    # A already holds a colocate model (10800); a second colocate model (10800) fits alongside.
    a = GpuView("A", 24000, used_smi=10800, ready_footprint=10800)
    res = [R("a1", 10800, OLD, loaded_at=OLD, colocate=True)]
    uuid, ev = select_colocated([a], {"A": res}, 10800, set(), NOW, 90)
    assert uuid == "A" and ev == [], (uuid, ev)


def test_colocated_rejects_wholecard_resident():
    # A holds a non-colocate (whole-card) model -> not eligible for co-location.
    a = GpuView("A", 24000, used_smi=12000, ready_footprint=12000)
    res = [R("wc", 12000, OLD, loaded_at=OLD, colocate=False)]
    uuid, ev = select_colocated([a], {"A": res}, 10800, set(), NOW, 90)
    assert uuid is None, (uuid, ev)


def test_colocated_evicts_only_as_needed():
    # Card full of two idle colocate models; need room for one more -> evict the single LRU.
    a = GpuView("A", 24000, used_smi=24000, ready_footprint=24000)
    res = [R("old", 12000, NOW - 900, loaded_at=OLD, colocate=True),
           R("new", 12000, NOW - 5, loaded_at=OLD, colocate=True)]
    uuid, ev = select_colocated([a], {"A": res}, 10000, set(), NOW, 90)
    assert uuid == "A" and ev == ["old"], (uuid, ev)  # evict just the LRU, keep the other


def test_colocated_all_alwayson_returns_none():
    a = GpuView("A", 24000, used_smi=24000, ready_footprint=24000)
    res = [R("p", 24000, OLD, loaded_at=OLD, colocate=True, always_on=True)]
    uuid, ev = select_colocated([a], {"A": res}, 10000, set(), NOW, 90)
    assert uuid is None, (uuid, ev)


def test_colocated_picks_most_free_eligible():
    a = GpuView("A", 24000, used_smi=10800, ready_footprint=10800)  # free 13200
    b = GpuView("B", 24000, used_smi=2000, ready_footprint=2000)     # free 22000
    rbg = {"A": [R("a1", 10800, OLD, loaded_at=OLD, colocate=True)],
           "B": [R("b1", 2000, OLD, loaded_at=OLD, colocate=True)]}
    uuid, ev = select_colocated([a, b], rbg, 8000, set(), NOW, 90)
    assert uuid == "B" and ev == []


# --- Stage 2: compute_effective_tp, per-card need_fn, blocked_gpus ---

def test_compute_effective_tp():
    totals2 = [24000, 24000]
    # forced config tp wins
    assert compute_effective_tp(None, 2, None, totals2, 0.9) == 2
    # persisted prior tp reused (fixes F2 — 2nd request doesn't revert to 1)
    assert compute_effective_tp(None, 1, 2, totals2, 0.9) == 2
    # unseen oversized weights on homogeneous 2-GPU pool -> 2
    assert compute_effective_tp(30 * 1024**3, 1, None, totals2, 0.9) == 2
    # fits one card -> 1
    assert compute_effective_tp(10 * 1024**3, 1, None, totals2, 0.9) == 1
    # single-GPU pool -> 1 regardless
    assert compute_effective_tp(99 * 1024**3, 1, None, [24000], 0.9) == 1
    # heterogeneous pool -> no auto-split -> 1
    assert compute_effective_tp(99 * 1024**3, 1, None, [24000, 6000], 0.9) == 1
    # unknown weights -> 1
    assert compute_effective_tp(None, 1, None, totals2, 0.9) == 1


def test_need_fn_per_card_on_mixed_pool():
    # need_fn = util*g.total differs per card; a model "fits" the big card but not the small one.
    big = GpuView("big", total=24000, used_smi=0)
    small = GpuView("small", total=8000, used_smi=0)
    need_fn = lambda g: 0.9 * g.total  # noqa: E731
    chosen, ev = select_gpu([small, big], {"small": [], "big": []}, need_fn, NOW, 90)
    assert chosen == "big" and ev == []   # 0.9*8000=7200 fits small too, but big is most-free
    # A need that exceeds the small card's util budget only fits the big card.
    need_big = lambda g: 20000  # noqa: E731
    chosen2, _ = select_gpu([small, big], {"small": [], "big": []}, need_big, NOW, 90)
    assert chosen2 == "big"


def test_select_colocated_blocked_gpus():
    # GPU "A" hosts a non-colocate model (blocked); "B" is free -> colocate lands on B.
    a = GpuView("A", total=24000, used_smi=12000, ready_footprint=12000)
    b = GpuView("B", total=24000, used_smi=0)
    rbg = {"A": [R("wc", 12000, OLD, loaded_at=OLD, colocate=True)], "B": []}
    # Even though A's resident is marked colocate here, blocked_gpus forces exclusion of A.
    uuid, ev = select_colocated([a, b], rbg, lambda g: 8000, {"A"}, NOW, 90)
    assert uuid == "B", (uuid, ev)
    # If both are blocked -> None.
    uuid2, _ = select_colocated([a, b], rbg, lambda g: 8000, {"A", "B"}, NOW, 90)
    assert uuid2 is None


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok:", t.__name__)
    print(f"\nAll {len(tests)} placement tests passed.")
