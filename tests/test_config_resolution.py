"""Unit tests for the per-model config resolution.

Runnable directly (no pytest required):  python tests/test_config_resolution.py
Also discoverable by pytest (test_* functions).

These import only config_loader (docker-free) so they don't trigger docker.from_env().
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway"))

from config_loader import (  # noqa: E402
    resolve_model_configs, build_fallback_configs, resolve_pools,
    validate_model_pools, validate_tp_against_pools, validate_colocate,
    validate_pools_visible, validate_budget_mode, validate_extra_args_budget, migrate_footprints,
)

# Mirrors app.builtin_model_defaults() with stock env-var defaults.
BUILTINS = {
    "gpu_memory_utilization": 0.90,
    "max_model_len": 0,
    "tensor_parallel_size": 1,
    "max_num_seqs": 16,
    "kv_reservation_seqs": None,
    "quantization": None,
    "dtype": "auto",
    "inactivity_timeout": 1800,
    "always_on": False,
    "extra_args": [],
    "pool": None,
    "colocate": False,
}


def test_precedence_per_model_over_defaults_over_builtin():
    """per-model entry > defaults block > built-in default, verified per field."""
    raw = {
        "defaults": {
            "gpu_memory_utilization": 0.80,   # overrides builtin 0.90
            "max_model_len": 8192,            # overrides builtin 0
            "dtype": "bfloat16",              # overrides builtin "auto"
        },
        "models": {
            "model-a": {
                "repo": "org/a",
                "gpu_memory_utilization": 0.50,  # per-model overrides defaults
                "always_on": True,               # per-model overrides builtin
            },
            "model-b": {
                "repo": "org/b",                 # inherits everything from defaults/builtin
            },
        },
    }

    cfgs = resolve_model_configs(raw, BUILTINS)

    a, b = cfgs["model-a"], cfgs["model-b"]

    # per-model wins
    assert a.gpu_memory_utilization == 0.50, a.gpu_memory_utilization
    assert a.always_on is True

    # defaults win over builtin
    assert b.gpu_memory_utilization == 0.80, b.gpu_memory_utilization
    assert b.max_model_len == 8192
    assert b.dtype == "bfloat16"

    # builtin used when neither defaults nor per-model set it
    assert b.tensor_parallel_size == 1
    assert b.inactivity_timeout == 1800
    assert b.always_on is False
    assert b.quantization is None
    assert b.extra_args == []

    # per-model inherits defaults where it doesn't override
    assert a.max_model_len == 8192
    assert a.dtype == "bfloat16"

    print("ok: precedence per-model > defaults > builtin")


def test_repo_required_and_preserved():
    cfgs = resolve_model_configs({"models": {"m": {"repo": "org/r"}}}, BUILTINS)
    assert cfgs["m"].repo == "org/r"
    print("ok: repo preserved")


def test_extra_args_not_shared_between_models():
    """Each model must get its own extra_args list (no shared-mutable aliasing)."""
    cfgs = resolve_model_configs(
        {"models": {"x": {"repo": "o/x"}, "y": {"repo": "o/y"}}}, BUILTINS
    )
    cfgs["x"].extra_args.append("--foo")
    assert cfgs["y"].extra_args == [], "extra_args leaked across models"
    assert BUILTINS["extra_args"] == [], "builtins list was mutated"
    print("ok: extra_args isolated per model")


def _expect_error(raw, needle):
    try:
        resolve_model_configs(raw, BUILTINS)
    except ValueError as e:
        assert needle in str(e), f"expected '{needle}' in: {e}"
        return
    raise AssertionError(f"expected ValueError containing '{needle}'")


def test_validation_errors():
    # missing repo
    _expect_error({"models": {"m": {"max_model_len": 10}}}, "repo")
    # empty repo
    _expect_error({"models": {"m": {"repo": "  "}}}, "repo")
    # unknown per-model key
    _expect_error({"models": {"m": {"repo": "o/r", "bogus": 1}}}, "unknown key")
    # unknown defaults key
    _expect_error({"defaults": {"bogus": 1}, "models": {"m": {"repo": "o/r"}}}, "unknown key")
    # unknown top-level key
    _expect_error({"junk": 1, "models": {"m": {"repo": "o/r"}}}, "unknown top-level")
    # empty models
    _expect_error({"models": {}}, "non-empty")
    # missing models
    _expect_error({"defaults": {}}, "models")
    # bad gpu_memory_utilization range
    _expect_error({"models": {"m": {"repo": "o/r", "gpu_memory_utilization": 1.5}}}, "gpu_memory_utilization")
    # bad int type (string where int expected)
    _expect_error({"models": {"m": {"repo": "o/r", "max_model_len": "lots"}}}, "max_model_len")
    # bool must not satisfy int field
    _expect_error({"models": {"m": {"repo": "o/r", "tensor_parallel_size": True}}}, "tensor_parallel_size")
    # extra_args must be a list
    _expect_error({"models": {"m": {"repo": "o/r", "extra_args": "--foo"}}}, "extra_args")
    print("ok: validation errors")


def test_fallback_matches_builtins():
    cfgs = build_fallback_configs({"name1": "org/repo1", "name2": "org/repo2"}, BUILTINS)
    assert set(cfgs) == {"name1", "name2"}
    c = cfgs["name1"]
    assert c.repo == "org/repo1"
    assert c.gpu_memory_utilization == 0.90
    assert c.max_model_len == 0
    assert c.tensor_parallel_size == 1
    assert c.inactivity_timeout == 1800
    assert c.always_on is False
    print("ok: fallback configs match builtins")


# --- pools (multi-GPU) ---

def test_resolve_pools_absent_and_valid():
    assert resolve_pools({"models": {}}) == {}
    pools = resolve_pools({"pools": {"llm": ["GPU-a", "GPU-b"], "util": ["GPU-c"]}, "models": {}})
    assert pools == {"llm": ["GPU-a", "GPU-b"], "util": ["GPU-c"]}
    print("ok: resolve_pools absent + valid")


def test_resolve_pools_errors():
    for raw, needle in [
        ({"pools": {}}, "non-empty"),
        ({"pools": {"llm": []}}, "non-empty list"),
        ({"pools": {"llm": ["GPU-a", "  "]}}, "invalid GPU UUID"),
        ({"pools": {"llm": ["GPU-a"], "util": ["GPU-a"]}}, "both pools"),
    ]:
        try:
            resolve_pools(raw)
        except ValueError as e:
            assert needle in str(e), f"{needle!r} not in {e}"
        else:
            raise AssertionError(f"expected ValueError containing {needle!r}")
    print("ok: resolve_pools errors")


def test_pool_precedence_and_field():
    raw = {
        "defaults": {"pool": "llm"},
        "models": {
            "a": {"repo": "o/a"},                 # inherits defaults.pool
            "b": {"repo": "o/b", "pool": "util"}, # per-model overrides
        },
    }
    cfgs = resolve_model_configs(raw, BUILTINS)
    assert cfgs["a"].pool == "llm"
    assert cfgs["b"].pool == "util"
    print("ok: pool precedence + field")


def test_validate_model_pools():
    pools = {"llm": ["GPU-a"], "util": ["GPU-b"]}
    cfgs = resolve_model_configs({"models": {"a": {"repo": "o/a", "pool": "llm"}}}, BUILTINS)
    validate_model_pools(cfgs, pools)  # ok
    bad = resolve_model_configs({"models": {"a": {"repo": "o/a", "pool": "nope"}}}, BUILTINS)
    try:
        validate_model_pools(bad, pools)
    except ValueError as e:
        assert "not a declared pool" in str(e), e
    else:
        raise AssertionError("expected ValueError for undeclared pool")
    # pool set but no pools declared -> ignored (no raise)
    validate_model_pools(bad, {})
    print("ok: validate_model_pools")


def test_poolless_model_rejected_when_pools_declared():
    pools = {"llm": ["GPU-a"]}
    cfgs = resolve_model_configs({"models": {"a": {"repo": "o/a"}}}, BUILTINS)  # no pool, no default
    try:
        validate_model_pools(cfgs, pools)
    except ValueError as e:
        assert "no pool set" in str(e), e
    else:
        raise AssertionError("expected ValueError for pool-less model under declared pools")
    # but with a defaults.pool it resolves and passes
    ok = resolve_model_configs({"defaults": {"pool": "llm"}, "models": {"a": {"repo": "o/a"}}}, BUILTINS)
    validate_model_pools(ok, pools)
    print("ok: pool-less model rejected when pools declared")


def test_validate_tp_against_pools():
    pools = {"llm": ["GPU-a", "GPU-b"], "util": ["GPU-c"]}
    # tp=2 with 2-GPU pool -> ok
    ok = resolve_model_configs({"models": {"m": {"repo": "o/m", "pool": "llm", "tensor_parallel_size": 2}}}, BUILTINS)
    validate_tp_against_pools(ok, pools)
    # tp=2 in a 1-GPU pool -> raises
    bad = resolve_model_configs({"models": {"m": {"repo": "o/m", "pool": "util", "tensor_parallel_size": 2}}}, BUILTINS)
    try:
        validate_tp_against_pools(bad, pools)
    except ValueError as e:
        assert "exceeds" in str(e), e
    else:
        raise AssertionError("expected ValueError for tp > pool size")
    # no pools declared -> never raises on tp
    validate_tp_against_pools(bad, {})
    print("ok: validate_tp_against_pools")


def test_colocate_field_and_validation():
    # default false; override true; precedence via defaults
    cfgs = resolve_model_configs(
        {"defaults": {"colocate": True}, "models": {"a": {"repo": "o/a"}, "b": {"repo": "o/b", "colocate": False}}},
        BUILTINS)
    assert cfgs["a"].colocate is True and cfgs["b"].colocate is False
    # non-bool rejected
    try:
        resolve_model_configs({"models": {"m": {"repo": "o/m", "colocate": "yes"}}}, BUILTINS)
    except ValueError as e:
        assert "colocate" in str(e), e
    else:
        raise AssertionError("expected ValueError for non-bool colocate")
    print("ok: colocate field + type validation")


def test_validate_colocate_forbids_tp():
    ok = resolve_model_configs({"models": {"m": {"repo": "o/m", "colocate": True}}}, BUILTINS)
    validate_colocate(ok)  # tp defaults to 1 -> fine
    bad = resolve_model_configs(
        {"models": {"m": {"repo": "o/m", "colocate": True, "tensor_parallel_size": 2}}}, BUILTINS)
    try:
        validate_colocate(bad)
    except ValueError as e:
        assert "incompatible" in str(e), e
    else:
        raise AssertionError("expected ValueError for colocate + tp>1")
    print("ok: validate_colocate forbids TP")


def test_validate_pools_visible():
    pools = {"llm": ["GPU-a", "GPU-b"], "util": ["GPU-c"]}
    validate_pools_visible(pools, {"GPU-a", "GPU-b", "GPU-c", "GPU-x"})  # all present -> ok
    validate_pools_visible({}, set())  # no pools -> ok
    try:
        validate_pools_visible(pools, {"GPU-a", "GPU-c"})  # GPU-b missing
    except ValueError as e:
        assert "GPU-b" in str(e) and "not visible" in str(e), e
    else:
        raise AssertionError("expected ValueError for a missing UUID")
    print("ok: validate_pools_visible")


def test_migrate_footprints():
    # legacy {repo: number} -> record with tp=1; new shape passes through; corrupt dropped.
    out = migrate_footprints({
        "org/a": 20000,                                              # legacy scalar
        "org/b": {"per_gpu_mib": 13000, "effective_tp": 2, "measured_at": 5.0},  # new shape
        "org/c": 0,                                                  # legacy sentinel
        "org/bad": "nonsense",                                       # corrupt -> dropped
    })
    assert out["org/a"] == {"per_gpu_mib": 20000.0, "effective_tp": 1,
                            "effective_util": 0.0, "measured_at": 0.0, "signature": {}}
    assert out["org/b"]["effective_tp"] == 2 and out["org/b"]["per_gpu_mib"] == 13000.0
    assert out["org/b"]["effective_util"] == 0.0  # absent in old new-shape record -> defaulted
    assert out["org/b"]["signature"] == {}        # legacy record -> empty signature (never reused)
    assert out["org/c"] == {"per_gpu_mib": 0.0, "effective_tp": 1,
                            "effective_util": 0.0, "measured_at": 0.0, "signature": {}}
    assert "org/bad" not in out
    assert migrate_footprints({}) == {} and migrate_footprints(None) == {}
    print("ok: migrate_footprints")


def test_max_num_seqs_precedence_and_validation():
    raw = {
        "defaults": {"max_num_seqs": 32},
        "models": {
            "a": {"repo": "org/a", "max_num_seqs": 4},   # per-model wins
            "b": {"repo": "org/b"},                        # inherits defaults
            "c": {"repo": "org/c"},
        },
    }
    cfgs = resolve_model_configs(raw, BUILTINS)
    assert cfgs["a"].max_num_seqs == 4
    assert cfgs["b"].max_num_seqs == 32
    # builtin fallback when neither sets it
    cfgs2 = resolve_model_configs({"models": {"m": {"repo": "o/r"}}}, BUILTINS)
    assert cfgs2["m"].max_num_seqs == BUILTINS["max_num_seqs"]
    # type/range validation
    _expect_error({"models": {"m": {"repo": "o/r", "max_num_seqs": 0}}}, "max_num_seqs")
    _expect_error({"models": {"m": {"repo": "o/r", "max_num_seqs": True}}}, "max_num_seqs")
    print("ok: max_num_seqs precedence + validation")


def test_validate_budget_mode():
    bounded = resolve_model_configs(
        {"defaults": {"max_model_len": 8192}, "models": {"m": {"repo": "o/r"}}}, BUILTINS)
    unbounded = resolve_model_configs({"models": {"m": {"repo": "o/r"}}}, BUILTINS)  # max_model_len 0
    # budget mode: bounded passes, unbounded fails fast naming the offending model.
    validate_budget_mode(bounded, "budget")          # no raise
    try:
        validate_budget_mode(unbounded, "budget")
    except ValueError as e:
        assert "max_model_len" in str(e) and "'m'" in str(e), e
    else:
        raise AssertionError("expected ValueError for unbounded model in budget mode")
    # whole_card mode: no requirement, both pass.
    validate_budget_mode(unbounded, "whole_card")
    print("ok: validate_budget_mode")


def test_kv_reservation_seqs():
    cfgs = resolve_model_configs({"models": {
        "a": {"repo": "o/a", "max_num_seqs": 32, "kv_reservation_seqs": 4},
        "b": {"repo": "o/b"},  # unset -> None (falls back to max_num_seqs at runtime)
    }}, BUILTINS)
    assert cfgs["a"].kv_reservation_seqs == 4 and cfgs["a"].max_num_seqs == 32
    assert cfgs["b"].kv_reservation_seqs is None
    # null is explicitly allowed; <1 and bool are rejected
    assert resolve_model_configs({"models": {"m": {"repo": "o/m", "kv_reservation_seqs": None}}},
                                 BUILTINS)["m"].kv_reservation_seqs is None
    _expect_error({"models": {"m": {"repo": "o/m", "kv_reservation_seqs": 0}}}, "kv_reservation_seqs")
    _expect_error({"models": {"m": {"repo": "o/m", "kv_reservation_seqs": True}}}, "kv_reservation_seqs")
    print("ok: kv_reservation_seqs")


def test_same_repo_multiple_profiles():
    # Two named profiles may share one repo (distinct identities, distinct configs).
    cfgs = resolve_model_configs({"defaults": {"max_model_len": 8192}, "models": {
        "fast": {"repo": "org/m", "max_num_seqs": 32, "kv_reservation_seqs": 4},
        "long": {"repo": "org/m", "max_num_seqs": 2, "max_model_len": 32768},
    }}, BUILTINS)
    assert cfgs["fast"].repo == cfgs["long"].repo == "org/m"
    assert cfgs["fast"].max_model_len == 8192 and cfgs["long"].max_model_len == 32768
    print("ok: same repo, multiple profiles")


def test_validate_extra_args_budget():
    raw = {
        "defaults": {"max_model_len": 8192},
        "models": {
            "ok": {"repo": "o/ok", "extra_args": ["--enable-prefix-caching"]},
            "bad": {"repo": "o/bad", "extra_args": ["--max-model-len", "65536"]},
        },
    }
    cfgs = resolve_model_configs(raw, BUILTINS)
    # budget mode: a memory-affecting flag in extra_args is rejected, naming the model.
    try:
        validate_extra_args_budget(cfgs, "budget")
    except ValueError as e:
        assert "bad" in str(e) and "extra_args" in str(e), e
    else:
        raise AssertionError("expected ValueError for --max-model-len in extra_args (budget)")
    # The '=' form is caught too.
    cfgs2 = resolve_model_configs(
        {"defaults": {"max_model_len": 8192},
         "models": {"m": {"repo": "o/m", "extra_args": ["--kv-cache-dtype=fp8"]}}}, BUILTINS)
    try:
        validate_extra_args_budget(cfgs2, "budget")
        raise AssertionError("expected ValueError for --kv-cache-dtype= in extra_args")
    except ValueError:
        pass
    # whole_card mode: no restriction.
    validate_extra_args_budget(cfgs, "whole_card")
    # A model with only harmless extra_args passes in budget mode.
    cfgs3 = resolve_model_configs(
        {"defaults": {"max_model_len": 8192},
         "models": {"m": {"repo": "o/m", "extra_args": ["--enable-prefix-caching"]}}}, BUILTINS)
    validate_extra_args_budget(cfgs3, "budget")
    print("ok: validate_extra_args_budget")


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test functions passed.")
