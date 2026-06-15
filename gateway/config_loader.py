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
    "pool",  # name of the GPU pool this model is placed in (multi-GPU); None = default pool
    "colocate",  # if true, may share a GPU with other co-locatable models (Phase 3)
}

# Per-model entries additionally require/allow "repo".
ALLOWED_MODEL_KEYS = ALLOWED_CONFIG_KEYS | {"repo"}

ALLOWED_TOP_LEVEL_KEYS = {"defaults", "models", "pools"}


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
    pool: Optional[str]  # GPU pool name (multi-GPU placement); None = the default/implicit pool
    colocate: bool  # may share a GPU with other co-locatable models (Phase 3)


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

    pool = merged["pool"]
    if pool is not None and (not isinstance(pool, str) or not pool.strip()):
        raise ValueError(f"model '{name}': 'pool' must be a non-empty string or null, got {pool!r}")

    colocate = merged["colocate"]
    if not isinstance(colocate, bool):
        raise ValueError(f"model '{name}': 'colocate' must be a boolean, got {colocate!r}")

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
        pool=pool.strip() if isinstance(pool, str) else None,
        colocate=colocate,
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


def resolve_pools(raw: dict) -> "dict[str, list]":
    """Parse and validate the optional top-level ``pools`` section.

    Returns {pool_name: [gpu_uuid, ...]}, or {} when no ``pools`` key is present
    (single-pool / backward-compatible mode). Raises ValueError (fail fast) on a
    malformed declaration so the gateway never starts with an ambiguous GPU topology.
    """
    if not isinstance(raw, dict) or "pools" not in raw:
        return {}

    pools = raw["pools"]
    if not isinstance(pools, dict) or not pools:
        raise ValueError("'pools' must be a non-empty mapping of pool-name -> [gpu-uuid, ...]")

    seen = {}  # uuid -> pool name, to catch a UUID assigned to two pools
    resolved: "dict[str, list]" = {}
    for pool_name, uuids in pools.items():
        if not isinstance(uuids, list) or not uuids:
            raise ValueError(f"pool '{pool_name}' must be a non-empty list of GPU UUIDs")
        clean = []
        for u in uuids:
            if not isinstance(u, str) or not u.strip():
                raise ValueError(f"pool '{pool_name}' contains an invalid GPU UUID: {u!r}")
            u = u.strip()
            if u in seen:
                raise ValueError(f"GPU UUID {u!r} is in both pools '{seen[u]}' and '{pool_name}'")
            seen[u] = pool_name
            clean.append(u)
        resolved[pool_name] = clean
    return resolved


def validate_model_pools(configs: "dict[str, ModelConfig]", pools: "dict[str, list]") -> None:
    """Ensure each model's resolved pool names a declared pool. Raises ValueError on mismatch.

    When ``pools`` is empty (no multi-GPU topology), a stray per-model ``pool`` is meaningless
    and ignored by the caller; we don't raise so adding ``pool:`` can't break the fallback path.
    When ``pools`` is declared, every model must name one (an explicit ``pool`` or a
    ``defaults.pool``) — a pool-less model would otherwise have no GPUs to place onto.
    """
    if not pools:
        return
    declared = set(pools)
    for name, cfg in configs.items():
        if cfg.pool is None:
            raise ValueError(
                f"model '{name}': no pool set, but pools are declared. Set 'pool' on the model "
                f"or a 'defaults.pool'; declared pools: {sorted(declared)}"
            )
        if cfg.pool not in declared:
            raise ValueError(
                f"model '{name}': pool '{cfg.pool}' is not a declared pool; declared: {sorted(declared)}"
            )


def validate_tp_against_pools(configs: "dict[str, ModelConfig]", pools: "dict[str, list]") -> None:
    """Fail fast when a model's tensor_parallel_size exceeds the GPUs declared in its pool.

    Only meaningful when ``pools`` is declared (the GPU count per pool is then known at load
    time). Homogeneity (equal VRAM) can only be checked at runtime; placement enforces that.
    """
    if not pools:
        return
    for name, cfg in configs.items():
        if cfg.tensor_parallel_size > 1 and cfg.pool in pools:
            available = len(pools[cfg.pool])
            if cfg.tensor_parallel_size > available:
                raise ValueError(
                    f"model '{name}': tensor_parallel_size={cfg.tensor_parallel_size} exceeds the "
                    f"{available} GPU(s) declared in pool '{cfg.pool}'"
                )


def validate_pools_visible(pools: "dict[str, list]", visible_uuids: "set") -> None:
    """Fail fast if any configured GPU UUID isn't visible to nvidia-smi.

    Runs at startup AFTER the GPU probe (the visible set is a runtime fact). Catches the common
    typo in a long `GPU-xxxx...` UUID, instead of silently treating that GPU as having 0 VRAM
    (which would make its pool unplaceable with only a log warning).
    """
    if not pools:
        return
    missing = []
    for pool_name, uuids in pools.items():
        for u in uuids:
            if u not in visible_uuids:
                missing.append(f"{u} (pool '{pool_name}')")
    if missing:
        raise ValueError(
            "Configured GPU UUID(s) not visible to nvidia-smi: " + "; ".join(missing) +
            f". Visible: {sorted(visible_uuids)}. Check pools/GATEWAY_GPU_UUID against `nvidia-smi -L`."
        )


def validate_colocate(configs: "dict[str, ModelConfig]", max_share: float = 0.9) -> None:
    """Validate co-location settings. Raises on a hard conflict; warns on a soft smell.

    Co-location is single-GPU only, so it is incompatible with tensor parallel. A co-locatable
    model whose share (gpu_memory_utilization) is near a whole card won't actually co-locate —
    that's a config smell, logged as a warning, not an error.
    """
    import logging
    for name, cfg in configs.items():
        if not cfg.colocate:
            continue
        if cfg.tensor_parallel_size > 1:
            raise ValueError(
                f"model '{name}': colocate=true is incompatible with tensor_parallel_size="
                f"{cfg.tensor_parallel_size} (co-location is single-GPU only)"
            )
        if cfg.gpu_memory_utilization > max_share:
            logging.warning(
                f"model '{name}': colocate=true but gpu_memory_utilization="
                f"{cfg.gpu_memory_utilization} > {max_share}; it will rarely fit alongside another model."
            )
