"""Resolve a Primus experiment YAML for the veRL Primus engine."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path
from typing import Any

_META_KEYS = {"name", "framework", "config", "model", "modules", "overrides", "stage"}


def ensure_primus_importable(path: str | None, turbo_path: str | None = None) -> None:
    for candidate in (path or os.environ.get("PRIMUS_PATH"), turbo_path or os.environ.get("PRIMUS_TURBO_PATH")):
        if not candidate:
            continue
        resolved = os.path.abspath(candidate)
        if not os.path.isdir(resolved):
            raise FileNotFoundError(f"Primus dependency not found: {resolved}")
        if resolved not in sys.path:
            sys.path.insert(0, resolved)
    importlib.invalidate_caches()
    if importlib.util.find_spec("primus") is None:
        raise ModuleNotFoundError("Install Primus or set primus_path/PRIMUS_PATH to a Primus checkout")


def load_primus_params(config) -> dict[str, Any]:
    """Return the selected Primus module's fully resolved flat parameter mapping."""
    ensure_primus_importable(config.primus_path, config.primus_turbo_path)
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
    if getattr(module, "framework", None) != "megatron":
        raise ValueError(f"Primus module {config.primus_module!r} must use framework='megatron'")
    params = nested_namespace_to_dict(module.params)
    if config.primus_overrides:
        params = deep_merge(params, parse_cli_overrides(config.primus_overrides))
    return {key: value for key, value in params.items() if key not in _META_KEYS}


_ENGINE_FIELDS = {
    "tensor_model_parallel_size",
    "expert_model_parallel_size",
    "expert_tensor_parallel_size",
    "pipeline_model_parallel_size",
    "virtual_pipeline_model_parallel_size",
    "context_parallel_size",
    "sequence_parallel",
    "use_distributed_optimizer",
    "use_remove_padding",
    "param_offload",
    "optimizer_offload",
    "dtype",
}

_TRANSFORMER_FIELDS = {
    "recompute_granularity",
    "recompute_method",
    "recompute_num_layers",
    "apply_rope_fusion",
    "masked_softmax_fusion",
    "bias_activation_fusion",
    "bias_dropout_fusion",
    "gradient_accumulation_fusion",
    "moe_grouped_gemm",
    "moe_permute_fusion",
    "moe_router_dtype",
    "moe_router_topk",
    "moe_aux_loss_coeff",
    "attention_backend",
}

_OPTIMIZER_FIELDS = {
    "lr",
    "min_lr",
    "weight_decay",
    "lr_decay_style",
    "lr_warmup_iters",
    "clip_grad",
    "adam_beta1",
    "adam_beta2",
    "use_precision_aware_optimizer",
    "main_grads_dtype",
    "exp_avg_dtype",
    "exp_avg_sq_dtype",
}

_PRIMUS_FLAGS = {
    "bf16",
    "fp16",
    "enable_packed_sequences",
    "use_packed_attention",
    "enable_primus_turbo",
    "use_turbo_attention",
    "use_turbo_rms_norm",
    "use_turbo_gemm",
    "use_turbo_parallel_linear",
    "use_turbo_grouped_gemm",
    "use_turbo_grouped_mlp",
    "use_turbo_deepep",
    "hf_path",
    "tokenizer_model",
    "untie_embeddings_and_output_weights",
    "seq_length",
    "max_position_embeddings",
    "micro_batch_size",
    "stage",
}


def apply_primus_params(config, params: dict[str, Any]) -> None:
    """Project authoritative Primus YAML values into veRL's mcore config."""
    for key in _ENGINE_FIELDS:
        if key in params and params[key] is not None:
            object.__setattr__(config, key, params[key])
    if params.get("bf16") is True:
        object.__setattr__(config, "dtype", "bfloat16")
    elif params.get("fp16") is True:
        object.__setattr__(config, "dtype", "float16")
    overrides = dict(config.override_transformer_config)
    for key in _TRANSFORMER_FIELDS:
        if key in params and params[key] is not None:
            overrides[key] = params[key]
    if params.get("use_packed_attention") and overrides.get("attention_backend", "auto") in (None, "auto"):
        overrides["attention_backend"] = "flash"
    object.__setattr__(config, "override_transformer_config", overrides)
    object.__setattr__(config, "_primus_params", dict(params))
    _validate_overrides(config, params)


def apply_primus_optimizer_params(optimizer_config, params: dict[str, Any]) -> None:
    """Apply optimizer/scheduler values owned by the Primus YAML."""
    direct = {
        "lr": "lr",
        "min_lr": "min_lr",
        "weight_decay": "weight_decay",
        "lr_decay_style": "lr_decay_style",
        "lr_warmup_iters": "lr_warmup_steps",
        "clip_grad": "clip_grad",
        "use_precision_aware_optimizer": "use_precision_aware_optimizer",
        "main_grads_dtype": "main_grads_dtype",
        "exp_avg_dtype": "exp_avg_dtype",
        "exp_avg_sq_dtype": "exp_avg_sq_dtype",
    }
    for source, target in direct.items():
        if source in params and params[source] is not None and hasattr(optimizer_config, target):
            object.__setattr__(optimizer_config, target, params[source])
    if "adam_beta1" in params or "adam_beta2" in params:
        beta1, beta2 = optimizer_config.betas
        object.__setattr__(
            optimizer_config,
            "betas",
            (params.get("adam_beta1", beta1), params.get("adam_beta2", beta2)),
        )


def _validate_overrides(config, params: dict[str, Any]) -> None:
    if not config.primus_strict_config:
        return
    known = _ENGINE_FIELDS | _TRANSFORMER_FIELDS | _OPTIMIZER_FIELDS | _PRIMUS_FLAGS | set(params)
    unknown = []
    for item in config.primus_overrides:
        key = item.split("=", 1)[0].strip().split(".")[-1]
        if key not in known:
            unknown.append(key)
    if unknown:
        raise ValueError("Unsupported Primus overrides: " + ", ".join(unknown))


def dynamic_token_budget(max_tokens_per_gpu: int, context_parallel_size: int) -> int:
    """Global packed tokens owned by one data-parallel rank at this CP width."""
    if max_tokens_per_gpu <= 0 or context_parallel_size <= 0:
        raise ValueError("dynamic token budget requires positive token and CP sizes")
    return max_tokens_per_gpu * context_parallel_size


def validate_primus_capabilities(config, params: dict[str, Any], model_config) -> None:
    """Reject configurations that would silently lose THD/CP or use unsafe Turbo paths."""
    hf_config = getattr(model_config, "hf_config", None)
    architecture = " ".join(getattr(hf_config, "architectures", None) or getattr(model_config, "architectures", []) or [])
    if "Qwen3Moe" in architecture:
        if not config.use_remove_padding or not params.get("use_packed_attention", False):
            raise ValueError("Qwen3 MoE Primus training requires strict THD: use_remove_padding/use_packed_attention")
        if config.context_parallel_size < 2:
            raise ValueError("The Qwen3-30B Primus recipe requires context_parallel_size >= 2")
    backend = str(params.get("attention_backend", config.override_transformer_config.get("attention_backend", "flash")))
    if params.get("use_packed_attention") and "aiter" in backend.lower():
        raise RuntimeError("Qwen packed THD must use TransformerEngine flash attention, not AITER")
    if params.get("use_turbo_attention", False):
        raise RuntimeError("Primus Turbo attention is disabled: its training-gradient parity is not validated")
    tp1_only = (
        "use_turbo_gemm",
        "use_turbo_parallel_linear",
        "use_turbo_grouped_gemm",
        "use_turbo_grouped_mlp",
    )
    enabled = [name for name in tp1_only if params.get(name, False)]
    if config.tensor_model_parallel_size > 1 and enabled:
        raise ValueError(f"Primus Turbo linear kernels require TP=1; enabled: {', '.join(enabled)}")
