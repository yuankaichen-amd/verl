from pathlib import Path

import pytest

import importlib.util

from verl.workers.config import PrimusEngineConfig


def _load(module_name, relative_path):
    path = Path(__file__).parents[4] / relative_path
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


_config = _load("primus_config_under_test", "verl/workers/engine/primus/config.py")
_export = _load("primus_export_under_test", "verl/workers/engine/primus/export.py")
apply_primus_params = _config.apply_primus_params
dynamic_token_budget = _config.dynamic_token_budget
load_primus_params = _config.load_primus_params
normalize_export_name = _export.normalize_export_name

PRIMUS_ROOT = Path("/home/yuankach/Miles-Prims-backup/Primus")
TURBO_ROOT = Path("/home/yuankach/Miles-Prims-backup/Primus-Turbo")
RECIPE = Path(__file__).parents[4] / "examples/primus/configs/qwen3_30b_a3b_grpo.yaml"


@pytest.mark.skipif(not PRIMUS_ROOT.exists(), reason="local Primus checkout is unavailable")
def test_qwen_recipe_drives_parallelism_and_thd():
    config = PrimusEngineConfig(
        primus_config_path=str(RECIPE),
        primus_path=str(PRIMUS_ROOT),
        primus_turbo_path=str(TURBO_ROOT),
    )
    params = load_primus_params(config)
    apply_primus_params(config, params)

    assert config.strategy == "primus"
    assert config.tensor_model_parallel_size == 1
    assert config.pipeline_model_parallel_size == 1
    assert config.context_parallel_size == 2
    assert config.expert_model_parallel_size == 4
    assert config.use_remove_padding is True
    assert config.dtype == "bfloat16"
    assert config.override_transformer_config["attention_backend"] == "flash"
    assert params["use_packed_attention"] is True


def test_primus_config_requires_yaml():
    with pytest.raises(ValueError, match="primus_config_path"):
        PrimusEngineConfig()


def test_dynamic_budget_scales_with_context_parallel():
    assert dynamic_token_budget(4096, 2) == 8192


def test_qwen_export_names_pass_through_and_packed_turbo_weights_fail():
    assert normalize_export_name("model.layers.0.self_attn.q_proj.weight") == "model.layers.0.self_attn.q_proj.weight"
    with pytest.raises(RuntimeError, match="packed expert"):
        normalize_export_name("model.layers.0.mlp.experts.linear_fc1.weights")


def test_unknown_override_is_rejected():
    config = PrimusEngineConfig(primus_config_path="unused.yaml", primus_overrides=["not_a_real_knob=1"])
    with pytest.raises(ValueError, match="Unsupported Primus overrides"):
        apply_primus_params(config, {"bf16": True, "use_packed_attention": True, "attention_backend": "flash"})
