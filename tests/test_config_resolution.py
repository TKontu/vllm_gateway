"""Unit tests for the per-model config resolution.

Runnable directly (no pytest required):  python tests/test_config_resolution.py
Also discoverable by pytest (test_* functions).

These import only config_loader (docker-free) so they don't trigger docker.from_env().
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "gateway"))

from config_loader import resolve_model_configs, build_fallback_configs  # noqa: E402

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


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t()
    print(f"\nAll {len(tests)} test functions passed.")
