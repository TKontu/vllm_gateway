"""Unit tests for the eviction/placement policy (placement.select_evictions).

Runnable directly (no pytest required):  python tests/test_placement.py
Also discoverable by pytest. Imports only placement (docker-free).
"""

import os
import sys
from dataclasses import dataclass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway"))

from placement import select_evictions  # noqa: E402


@dataclass
class R:  # minimal stand-in for ContainerState
    container_name: str
    vram_footprint: float
    last_request_time: float
    loaded_at: float = 0.0
    active_requests: int = 0
    always_on: bool = False


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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
        print("ok:", t.__name__)
    print(f"\nAll {len(tests)} placement tests passed.")
