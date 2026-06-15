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
)

# Mirrors app.builtin_model_defaults() with stock env-var defaults.
BUILTINS = {
    "gpu_memory_utilization": 0.90,
    "max_model_len": 0,
    "tensor_parallel_size": 1,
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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test functions passed.")
