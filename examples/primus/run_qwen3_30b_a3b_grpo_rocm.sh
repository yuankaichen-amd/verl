#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
export PRIMUS_PATH="${PRIMUS_PATH:-/home/yuankach/Miles-Prims-backup/Primus}"
export PRIMUS_TURBO_PATH="${PRIMUS_TURBO_PATH:-/home/yuankach/Miles-Prims-backup/Primus-Turbo}"
export PYTHONPATH="${ROOT}:${PRIMUS_PATH}:${PRIMUS_TURBO_PATH}:${PYTHONPATH:-}"
export CUDA_DEVICE_MAX_CONNECTIONS=1

MODEL_PATH="${MODEL_PATH:-/home/yuankach/qwen3_30b_iter199_hf}"
PRIMUS_CONFIG="${PRIMUS_CONFIG:-${ROOT}/examples/primus/configs/qwen3_30b_a3b_grpo.yaml}"
TRAIN_FILES="${TRAIN_FILES:-${ROOT}/data/DAPO-Math-17k/data/dapo-math-17k.parquet}"
VAL_FILES="${VAL_FILES:-${ROOT}/data/AIME-2024/data/aime-2024.parquet}"
NNODES="${NNODES:-1}"
GPUS_PER_NODE="${GPUS_PER_NODE:-8}"

python3 -m verl.trainer.main_ppo \
  model_engine=primus \
  actor_rollout_ref.actor.primus.primus_config_path="${PRIMUS_CONFIG}" \
  actor_rollout_ref.ref.primus.primus_config_path="${PRIMUS_CONFIG}" \
  actor_rollout_ref.model.path="${MODEL_PATH}" \
  actor_rollout_ref.model.use_remove_padding=true \
  actor_rollout_ref.model.use_fused_kernels=false \
  actor_rollout_ref.actor.use_dynamic_bsz=true \
  actor_rollout_ref.actor.ppo_max_token_len_per_gpu="${PPO_MAX_TOKEN_LEN_PER_GPU:-4096}" \
  actor_rollout_ref.actor.ppo_mini_batch_size="${PPO_MINI_BATCH_SIZE:-32}" \
  actor_rollout_ref.actor.optim.lr="${ACTOR_LR:-1e-6}" \
  actor_rollout_ref.ref.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.ref.log_prob_max_token_len_per_gpu="${REF_MAX_TOKEN_LEN_PER_GPU:-4096}" \
  actor_rollout_ref.rollout.name=sglang \
  actor_rollout_ref.rollout.mode=async \
  actor_rollout_ref.rollout.n="${ROLLOUT_N:-5}" \
  actor_rollout_ref.rollout.tensor_model_parallel_size="${ROLLOUT_TP:-8}" \
  actor_rollout_ref.rollout.gpu_memory_utilization="${ROLLOUT_GPU_MEMORY_UTILIZATION:-0.65}" \
  actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=true \
  actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu="${ROLLOUT_LOGPROB_MAX_TOKENS:-4096}" \
  data.train_files="${TRAIN_FILES}" \
  data.val_files="${VAL_FILES}" \
  data.train_batch_size="${TRAIN_BATCH_SIZE:-128}" \
  data.max_prompt_length="${MAX_PROMPT_LENGTH:-2048}" \
  data.max_response_length="${MAX_RESPONSE_LENGTH:-8192}" \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=false \
  trainer.nnodes="${NNODES}" \
  trainer.n_gpus_per_node="${GPUS_PER_NODE}" \
  trainer.project_name="${PROJECT_NAME:-verl_primus}" \
  trainer.experiment_name="${EXPERIMENT_NAME:-qwen3_30b_a3b_cp2_thd}" \
  trainer.total_epochs="${TOTAL_EPOCHS:-1}" \
  trainer.total_training_steps="${TOTAL_TRAINING_STEPS:-1}" \
  trainer.test_freq="${TEST_FREQ:-5}" \
  trainer.save_freq="${SAVE_FREQ:-50}" \
  trainer.val_before_train=false \
  reward.custom_reward_function.path="${REWARD_FN:-${ROOT}/examples/primus/reward.py}" \
  reward.custom_reward_function.name=compute_score \
  "$@"
