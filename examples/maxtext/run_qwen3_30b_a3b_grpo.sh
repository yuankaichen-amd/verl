#!/usr/bin/env bash
# GRPO with a MaxText trainer. The trainer is one Ray actor that owns every
# GPU on its node. Rollout is vLLM by default (ROLLOUT_NAME), one process per GPU, and receives
# Hugging Face torch tensors through the existing NCCL checkpoint engine.
# KL against a reference model is off: a second MaxText process cannot share
# the trainer's GPUs.
#
# The compiled step consumes num_devices * per_device_batch_size sequences.
# examples/maxtext/qwen3_30b_a3b.yaml uses 1, so an 8-GPU node trains 8
# sequences at a time. That matches ppo_mini_batch_size * rollout.n below.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PRIMUS_PATH="${PRIMUS_PATH:-/home/yuankach/Miles-Prims-backup/Primus}"
export MAXTEXT_PATH="${MAXTEXT_PATH:-${PRIMUS_PATH}/third_party/maxtext}"
export PYTHONPATH="${ROOT}:${PRIMUS_PATH}:${MAXTEXT_PATH}/src:${PYTHONPATH:-}"

MODEL_PATH="${MODEL_PATH:-/home/yuankach/qwen3_30b_iter199_hf}"
PRIMUS_CONFIG="${PRIMUS_CONFIG:-${ROOT}/examples/maxtext/qwen3_30b_a3b.yaml}"
TRAIN_FILES="${TRAIN_FILES:-${ROOT}/data/primus_smoke/prompts.parquet}"
VAL_FILES="${VAL_FILES:-${TRAIN_FILES}}"

# REWARD_FN=builtin uses veRL's rule-based reward for the data source (GSM8K, MATH, ...).
REWARD_ARGS=()
if [ "${REWARD_FN:-}" != "builtin" ]; then
  REWARD_ARGS=(
    reward.custom_reward_function.path="${REWARD_FN:-${ROOT}/examples/primus/reward.py}"
    reward.custom_reward_function.name=compute_score
  )
fi

python3 -m verl.trainer.main_ppo \
  model_engine=maxtext \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.parameter_sync_step=1 \
  trainer.v1.separate_async.num_warmup_batches=1 \
  actor_rollout_ref.hybrid_engine=false \
  actor_rollout_ref.actor.maxtext.primus_config_path="${PRIMUS_CONFIG}" \
  actor_rollout_ref.actor.use_kl_loss=false \
  actor_rollout_ref.actor.use_dynamic_bsz=false \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-4}" \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu="${PPO_MICRO_BATCH_SIZE:-4}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.rollout.name="${ROLLOUT_NAME:-vllm}" \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-2}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-8}" \
  actor_rollout_ref.rollout.nnodes="${ROLLOUT_NNODES:-1}" \
  actor_rollout_ref.rollout.n_gpus_per_node="${ROLLOUT_GPUS_PER_NODE:-8}" \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu="${ROLLOUT_LOGPROB_MICRO_BATCH:-4}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.8}" \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-4}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-64}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-64}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  trainer.nnodes="${NNODES:-1}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE:-8}" \
  trainer.project_name="${PROJECT_NAME:-verl_maxtext}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_30b_a3b_maxtext}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-2}" \
  trainer.test_freq="${TEST_FREQ:-0}" \
  trainer.save_freq="${SAVE_FREQ:--1}" \
  trainer.default_local_dir="${CKPT_DIR:-checkpoints/${PROJECT_NAME:-verl_maxtext}/${EXPERIMENT_NAME:-qwen3_30b_a3b_maxtext}}" \
  trainer.resume_mode="${RESUME_MODE:-disable}" \
  trainer.max_actor_ckpt_to_keep="${MAX_CKPT_KEEP:-2}" \
  trainer.val_before_train=false \
  trainer.logger="${TRAINER_LOGGER:-[console]}" \
  "${REWARD_ARGS[@]}" \
  "$@"
