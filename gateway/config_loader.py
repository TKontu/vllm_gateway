"""File-based per-model configuration loader and resolver.

This module is intentionally free of any Docker / network / FastAPI dependency so
the resolution logic can be unit-tested in isolation (importing app.py triggers
docker.from_env(), which requires a live Docker socket).

Resolution precedence per setting:  per-model entry  >  defaults block  >  built-in
default (the corresponding existing env var, supplied by the caller as ``builtins``).
"""

from dataclasses import dataclass
from typing import Optional

# Keys allowed in the ``defaults`` block and as per-model overrides.
ALLOWED_CONFIG_KEYS = {
    "gpu_memory_utilization",
    "max_model_len",
    "tensor_parallel_size",
    "quantization",
    "dtype",
    "inactivity_timeout",
    "always_on",
    "extra_args",
}

# Per-model entries additionally require/allow "repo".
ALLOWED_MODEL_KEYS = ALLOWED_CONFIG_KEYS | {"repo"}

ALLOWED_TOP_LEVEL_KEYS = {"defaults", "models"}


@dataclass
class ModelConfig:
    """Fully-resolved configuration for a single model."""
    name: str
    repo: str
    gpu_memory_utilization: float
    max_model_len: int
    tensor_parallel_size: int
    quantization: Optional[str]
    dtype: str
    inactivity_timeout: int
    always_on: bool
    extra_args: list


def _require_int(field: str, model: str, value):
    # bool is a subclass of int; reject it explicitly so always_on/foo can't masquerade.
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"model '{model}': '{field}' must be an integer, got {value!r}")
    return value


def _construct(name: str, repo: str, merged: dict) -> ModelConfig:
    """Validate a merged settings dict and build a ModelConfig. Raises ValueError on any problem."""
    # gpu_memory_utilization: number in (0, 1]
    gmu = merged["gpu_memory_utilization"]
    if isinstance(gmu, bool) or not isinstance(gmu, (int, float)):
        raise ValueError(f"model '{name}': 'gpu_memory_utilization' must be a number, got {gmu!r}")
    gmu = float(gmu)
    if not (0 < gmu <= 1):
        raise ValueError(f"model '{name}': 'gpu_memory_utilization' must be in (0, 1], got {gmu}")

    max_model_len = _require_int("max_model_len", name, merged["max_model_len"])
    if max_model_len < 0:
        raise ValueError(f"model '{name}': 'max_model_len' must be >= 0, got {max_model_len}")

    tensor_parallel_size = _require_int("tensor_parallel_size", name, merged["tensor_parallel_size"])
    if tensor_parallel_size < 1:
        raise ValueError(f"model '{name}': 'tensor_parallel_size' must be >= 1, got {tensor_parallel_size}")

    inactivity_timeout = _require_int("inactivity_timeout", name, merged["inactivity_timeout"])
    if inactivity_timeout < 0:
        raise ValueError(f"model '{name}': 'inactivity_timeout' must be >= 0, got {inactivity_timeout}")

    quantization = merged["quantization"]
    if quantization is not None and (not isinstance(quantization, str) or not quantization.strip()):
        raise ValueError(f"model '{name}': 'quantization' must be a non-empty string or null, got {quantization!r}")

    dtype = merged["dtype"]
    if not isinstance(dtype, str) or not dtype.strip():
        raise ValueError(f"model '{name}': 'dtype' must be a non-empty string, got {dtype!r}")

    always_on = merged["always_on"]
    if not isinstance(always_on, bool):
        raise ValueError(f"model '{name}': 'always_on' must be a boolean, got {always_on!r}")

    extra_args = merged["extra_args"]
    if not isinstance(extra_args, list):
        raise ValueError(f"model '{name}': 'extra_args' must be a list, got {extra_args!r}")

    return ModelConfig(
        name=name,
        repo=repo,
        gpu_memory_utilization=gmu,
        max_model_len=int(max_model_len),
        tensor_parallel_size=int(tensor_parallel_size),
        quantization=quantization,
        dtype=dtype,
        inactivity_timeout=int(inactivity_timeout),
        always_on=always_on,
        # Fresh list of strings; never share the builtins/defaults list across models.
        extra_args=[str(a) for a in extra_args],
    )


def resolve_model_configs(raw: dict, builtins: dict) -> "dict[str, ModelConfig]":
    """Resolve a parsed YAML config into per-model ModelConfig objects.

    ``raw``      : the parsed YAML mapping ({defaults?, models}).
    ``builtins`` : bottom-tier defaults (one value per ALLOWED_CONFIG_KEYS key),
                   typically sourced from the legacy global env vars by the caller.

    Raises ValueError on any structural, key, or type/range violation (fail fast).
    """
    if not isinstance(raw, dict):
        raise ValueError("config root must be a mapping with a 'models' section")

    unknown_top = set(raw) - ALLOWED_TOP_LEVEL_KEYS
    if unknown_top:
        raise ValueError(f"unknown top-level key(s): {sorted(unknown_top)}; allowed: {sorted(ALLOWED_TOP_LEVEL_KEYS)}")

    defaults_block = raw.get("defaults") or {}
    if not isinstance(defaults_block, dict):
        raise ValueError("'defaults' must be a mapping")
    unknown_defaults = set(defaults_block) - ALLOWED_CONFIG_KEYS
    if unknown_defaults:
        raise ValueError(f"unknown key(s) in 'defaults': {sorted(unknown_defaults)}; allowed: {sorted(ALLOWED_CONFIG_KEYS)}")

    models = raw.get("models")
    if not isinstance(models, dict) or not models:
        raise ValueError("'models' is required and must be a non-empty mapping of name -> settings")

    # Sanity: builtins must cover every configurable key.
    missing_builtins = ALLOWED_CONFIG_KEYS - set(builtins)
    if missing_builtins:
        raise ValueError(f"internal error: builtins missing key(s) {sorted(missing_builtins)}")

    resolved: "dict[str, ModelConfig]" = {}
    for name, entry in models.items():
        if not isinstance(entry, dict):
            raise ValueError(f"model '{name}': entry must be a mapping")

        unknown = set(entry) - ALLOWED_MODEL_KEYS
        if unknown:
            raise ValueError(f"model '{name}': unknown key(s) {sorted(unknown)}; allowed: {sorted(ALLOWED_MODEL_KEYS)}")

        repo = entry.get("repo")
        if not isinstance(repo, str) or not repo.strip():
            raise ValueError(f"model '{name}': 'repo' is required and must be a non-empty string")

        # precedence: builtins < defaults block < per-model entry
        merged = dict(builtins)
        merged.update(defaults_block)
        merged.update({k: v for k, v in entry.items() if k != "repo"})

        resolved[name] = _construct(name, repo.strip(), merged)

    return resolved


def build_fallback_configs(allowed_models: dict, builtins: dict) -> "dict[str, ModelConfig]":
    """Build ModelConfigs for the legacy ALLOWED_MODELS_JSON path (name -> repo).

    Every model gets the built-in defaults verbatim, so behavior matches the
    pre-config-file gateway exactly.
    """
    out: "dict[str, ModelConfig]" = {}
    for name, repo in allowed_models.items():
        out[name] = _construct(name, repo, dict(builtins))
    return out
