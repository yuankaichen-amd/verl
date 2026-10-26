"""Standalone Primus backend implemented on veRL's Megatron engine."""

from __future__ import annotations

from verl.workers.config import PrimusEngineConfig
from verl.workers.engine.base import EngineRegistry
from verl.workers.engine.megatron.transformer_impl import (
    MegatronEngine,
    MegatronEngineWithLMHead,
    MegatronEngineWithValueHead,
)

from .config import (
    apply_primus_optimizer_params,
    apply_primus_params,
    ensure_primus_importable,
    load_primus_params,
    validate_primus_capabilities,
)
from .export import normalize_export_name
from .patches import PrimusPatchRunner


class _PrimusInitializationMixin:
    def __init__(self, model_config, engine_config: PrimusEngineConfig, optimizer_config, checkpoint_config):
        ensure_primus_importable(engine_config.primus_path, engine_config.primus_turbo_path)
        params = load_primus_params(engine_config)
        apply_primus_params(engine_config, params)
        apply_primus_optimizer_params(optimizer_config, params)
        validate_primus_capabilities(engine_config, params, model_config)
        super().__init__(model_config, engine_config, optimizer_config, checkpoint_config)
        self.primus_patch_runner = PrimusPatchRunner(engine_config)
        self.primus_patch_runner.run("setup")
        self.primus_patch_runner.run("build_args")
        self.primus_patch_runner.bind_megatron_globals()
        self.primus_patch_runner.run("before_train")

    def get_per_tensor_param(self, *args, **kwargs):
        tensors, peft_config = super().get_per_tensor_param(*args, **kwargs)
        return ((normalize_export_name(name), tensor) for name, tensor in tensors), peft_config


class PrimusEngine(_PrimusInitializationMixin, MegatronEngine):
    """Base Primus engine retaining Megatron's optimizer/checkpoint/runtime."""


@EngineRegistry.register(model_type="language_model", backend="primus", vendor="amd")
@EngineRegistry.register(model_type="language_model", backend="primus")
class PrimusEngineWithLMHead(_PrimusInitializationMixin, MegatronEngineWithLMHead):
    """Primus language-model engine."""


@EngineRegistry.register(model_type="value_model", backend="primus", vendor="amd")
@EngineRegistry.register(model_type="value_model", backend="primus")
class PrimusEngineWithValueHead(_PrimusInitializationMixin, MegatronEngineWithValueHead):
    """Primus value-model engine."""
