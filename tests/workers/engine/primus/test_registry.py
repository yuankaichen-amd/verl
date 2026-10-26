import pytest

try:
    import megatron.core  # noqa: F401
    from verl.workers.engine.base import EngineRegistry
    from verl.workers.engine.megatron import MegatronEngineWithLMHead, MegatronEngineWithValueHead
    from verl.workers.engine.primus import PrimusEngineWithLMHead, PrimusEngineWithValueHead
except OSError as error:
    pytest.skip(f"Megatron runtime is unavailable on this host: {error}", allow_module_level=True)


def test_primus_engine_is_registered_and_inherits_megatron():
    assert issubclass(PrimusEngineWithLMHead, MegatronEngineWithLMHead)
    assert issubclass(PrimusEngineWithValueHead, MegatronEngineWithValueHead)
    registry = EngineRegistry._engines
    assert registry["language_model"]["primus"]["cuda"] is PrimusEngineWithLMHead
    assert registry["language_model"]["primus"][("cuda", "amd")] is PrimusEngineWithLMHead
    assert registry["value_model"]["primus"]["cuda"] is PrimusEngineWithValueHead
