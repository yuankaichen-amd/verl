#!/usr/bin/env bash
# 100-step DAPO-Math-17k GRPO run on 2 nodes, metrics to Weights & Biases.
# Each step samples fresh prompts (100 x 8 of 17k), so the train reward is an unbiased estimate.
set -euo pipefail
ROOT=/home/yuankach/verl-primus
export WANDB_API_KEY="$(cat "${HOME}/.wandb_key")"
export WANDB_MODE=online
export PRIMUS_CONFIG="${ROOT}/examples/maxtext/qwen3_30b_a3b_dapo.yaml"
export TRAIN_FILES=/shared_inference/yuankach/verl-primus/data/dapo/train.parquet
export VAL_FILES=/shared_inference/yuankach/verl-primus/data/dapo/val.parquet
export REWARD_FN=builtin
export TRAINER_LOGGER='[console,wandb]'
export PROJECT_NAME=verl_maxtext
export EXPERIMENT_NAME="qwen3_30b_a3b_dapo_$(date +%m%d_%H%M)"
export TRAIN_BATCH_SIZE=8 ROLLOUT_N=8 PPO_MINI_BATCH_SIZE=8
export MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=2048
export EXTRA_ARGS="data.truncation=left data.filter_overlong_prompts=False actor_rollout_ref.actor.clip_ratio_high=0.28"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
exec sbatch -p "${PARTITION:-amd-arad}" --qos="${QOS:-normal}" --exclude="${EXTRA_EXCLUDE:+${EXTRA_EXCLUDE},}useocpm2m-097-162,useocpm2m-097-163,useocpm2m-097-164" --export=ALL "${ROOT}/examples/maxtext/run_grpo_async.sbatch"
