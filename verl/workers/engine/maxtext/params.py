"""Resolve a Primus MaxText experiment YAML into a flat parameter dict."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import importlib
import importlib.util
import sys

_META_KEYS = {"name", "framework", "config", "model", "modules", "overrides", "stage"}
_PRIMUS_ONLY = ("file_sink_level", "stderr_sink_level", "sink_level", "trainable", "model")


def ensure_primus_importable(path: str | None, turbo_path: str | None = None) -> None:
    """Put a Primus checkout on sys.path.

    This is not imported from ``verl.workers.engine.primus``: that package
    imports Megatron, which the MaxText image does not ship.
    """
    del turbo_path
    candidate = path or os.environ.get("PRIMUS_PATH")
    if candidate:
        resolved = os.path.abspath(candidate)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"Primus checkout not found: {resolved}")
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    importlib.invalidate_caches()
    if importlib.util.find_spec("primus") is None:
        raise ModuleNotFoundError("Install Primus or set primus_path/PRIMUS_PATH to a Primus checkout")


def load_maxtext_params(config) -> dict[str, Any]:
    """Return the selected module's resolved parameters, with Primus-only keys removed."""
    ensure_primus_importable(config.primus_path, None)
    from primus.core.config.merge_utils import deep_merge
    from primus.core.config.primus_config import get_module_config, get_module_names, load_primus_config
    from primus.core.utils.arg_utils import parse_cli_overrides
    from primus.core.utils.yaml_utils import nested_namespace_to_dict

    path = Path(config.primus_config_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Primus config not found: {path}")
    experiment = load_primus_config(path)
    module = get_module_config(experiment, config.primus_module)
    if module is None:
        names = ", ".join(get_module_names(experiment)) or "none"
        raise ValueError(f"{path} has no Primus module {config.primus_module!r}; available: {names}")
    if getattr(module, "framework", None) != "maxtext":
        raise ValueError(
            f"Primus module {config.primus_module!r} uses framework={getattr(module, 'framework', None)!r}; "
            "MaxTextEngine requires framework='maxtext'"
        )
    params = nested_namespace_to_dict(module.params)
    if config.primus_overrides:
        params = deep_merge(params, parse_cli_overrides(config.primus_overrides))
    for key in _META_KEYS | set(_PRIMUS_ONLY):
        params.pop(key, None)
    params.pop("override_model", None)
    # A zero eval batch makes the log-prob step slice every sequence away.
    per_device = params.get("per_device_batch_size")
    if per_device is not None and float(params.get("eval_per_device_batch_size") or 0) <= 0:
        params["eval_per_device_batch_size"] = per_device
    params.setdefault("gradient_accumulation_steps", 1)
    return params


def prepare_jax_env(config) -> None:
    """Apply Primus' MaxText environment, keeping memory free for the torch export."""
    fraction = str(config.mem_fraction)
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = fraction
    ensure_primus_importable(config.primus_path, None)
    from primus.backends.maxtext.maxtext_adapter import MaxTextAdapter

    MaxTextAdapter().apply_env_defaults()
    # apply_env_defaults uses setdefault, so the fraction above wins.
    os.environ["XLA_PYTHON_CLIENT_MEM_FRACTION"] = fraction
