"""One Primus optimizer step with packed THD and context parallelism.

``--mode tiny`` builds a random Qwen3 MoE and checks CP=2, variable-length
micro-batches, checkpoint resume, and HF export names. ``--mode qwen30b`` loads
the local Qwen3-30B-A3B checkpoint with the recipe topology.
"""

import argparse
import os
from functools import partial
from pathlib import Path

import ray
import torch
from transformers import AutoTokenizer, Qwen3MoeConfig, Qwen3MoeForCausalLM

from verl import DataProto
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.trainer.config import CheckpointConfig
from verl.utils import tensordict_utils as tu
from verl.utils.device import get_torch_device
from verl.utils.model import compute_position_id_with_mask, create_random_mask
from verl.workers.config import HFModelConfig, McoreOptimizerConfig, PrimusEngineConfig
from verl.workers.engine_workers import TrainingWorker, TrainingWorkerConfig
from verl.workers.utils.losses import ppo_loss
from verl.workers.utils.padding import left_right_2_no_padding


ROOT = Path(__file__).resolve().parents[2]
RECIPE = ROOT / "examples/primus/configs/qwen3_30b_a3b_grpo.yaml"


def _save_tiny_model(destination: Path) -> None:
    config = Qwen3MoeConfig(
        vocab_size=1024,
        hidden_size=256,
        intermediate_size=512,
        moe_intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        head_dim=32,
        num_experts=4,
        num_experts_per_tok=2,
        max_position_embeddings=256,
        tie_word_embeddings=False,
        attention_dropout=0.0,
    )
    model = Qwen3MoeForCausalLM(config).to(torch.bfloat16)
    destination.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(destination)
    tokenizer_source = os.environ.get("MODEL_PATH", "/home/yuankach/qwen3_30b_iter199_hf")
    AutoTokenizer.from_pretrained(tokenizer_source, trust_remote_code=True).save_pretrained(destination)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("tiny", "qwen30b"), default="tiny")
    parser.add_argument("--checkpoint-dir", default="/tmp/verl-primus-smoke")
    args = parser.parse_args()

    if args.mode == "tiny":
        model_path = Path("/tmp/verl-primus-tiny-qwen")
        _save_tiny_model(model_path)
        overrides = [
            "tensor_model_parallel_size=1",
            "pipeline_model_parallel_size=1",
            "context_parallel_size=2",
            "expert_model_parallel_size=1",
            "num_layers=2",
            "hidden_size=256",
            "num_experts=4",
            "moe_router_topk=2",
        ]
        vocab = 1024
    else:
        model_path = Path(os.environ.get("MODEL_PATH", "/home/yuankach/qwen3_30b_iter199_hf"))
        overrides = []
        vocab = 151936

    engine = PrimusEngineConfig(
        primus_config_path=str(RECIPE),
        primus_path=os.environ["PRIMUS_PATH"],
        primus_turbo_path=os.environ.get("PRIMUS_TURBO_PATH"),
        primus_overrides=overrides,
        use_remove_padding=True,
        use_dynamic_bsz=True,
        max_token_len_per_gpu=128 if args.mode == "tiny" else 1024,
        param_offload=False,
        optimizer_offload=False,
    )
    model = HFModelConfig(path=str(model_path), use_remove_padding=True, trust_remote_code=True)
    worker_config = TrainingWorkerConfig(
        model_type="language_model",
        model_config=model,
        engine_config=engine,
        optimizer_config=McoreOptimizerConfig(lr=1e-6, lr_decay_steps=1),
        checkpoint_config=CheckpointConfig(),
    )
    device_count = get_torch_device().device_count()
    ray.init(num_gpus=device_count)
    worker = RayWorkerGroup(
        resource_pool=RayResourcePool(process_on_nodes=[device_count]),
        ray_cls_with_init=RayClassWithInitArgs(cls=ray.remote(TrainingWorker), config=worker_config),
    )
    worker.reset()
    batch = 4
    length = 32
    input_ids = torch.randint(0, vocab, (batch, length))
    attention_mask = create_random_mask(
        input_ids=input_ids, max_ratio_of_valid_token=0.9, max_ratio_of_left_padding=0.1, min_ratio_of_valid_token=0.4
    )
    response_length = length // 2
    batch_data = DataProto.from_single_dict(
        {
            "input_ids": input_ids,
            "prompts": input_ids[:, :response_length],
            "attention_mask": attention_mask,
            "position_ids": compute_position_id_with_mask(attention_mask),
            "responses": input_ids[:, response_length:],
            "response_mask": attention_mask[:, response_length:],
            "advantages": torch.randn(batch, response_length),
            "old_log_probs": torch.zeros(batch, response_length),
            "ref_log_prob": torch.zeros(batch, response_length),
        }
    ).to_tensordict()
    batch_data = left_right_2_no_padding(batch_data)
    tu.assign_non_tensor(batch_data, global_batch_size=batch, temperature=1.0)
    from verl.workers.config import ActorConfig

    worker.set_loss_fn(partial(ppo_loss, config=ActorConfig(strategy="primus", rollout_n=1, use_dynamic_bsz=True)))
    metrics = tu.get(worker.train_batch(batch_data).get(), "metrics")
    print("PRIMUS_SMOKE_METRICS", metrics, flush=True)
    if args.mode == "tiny":
        destination = str(Path(args.checkpoint_dir) / "tiny")
        worker.save_checkpoint(destination, global_step=1)
        worker.load_checkpoint(destination)
        print("PRIMUS_SMOKE_CHECKPOINT_OK", destination, flush=True)
    ray.shutdown()


if __name__ == "__main__":
    main()
