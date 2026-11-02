#!/usr/bin/env bash
# 100-step GSM8K GRPO run on 2 nodes, metrics to Weights & Biases.
# The API key is read from ~/.wandb_key and never printed.
set -euo pipefail
ROOT=/home/yuankach/verl-primus
export WANDB_API_KEY="$(cat "${HOME}/.wandb_key")"
export WANDB_MODE=online
export PRIMUS_CONFIG="${ROOT}/examples/maxtext/qwen3_30b_a3b_gsm8k.yaml"
export TRAIN_FILES=/shared_inference/yuankach/verl-primus/data/gsm8k/train.parquet
export VAL_FILES=/shared_inference/yuankach/verl-primus/data/gsm8k/test.parquet
export REWARD_FN=builtin
export TRAINER_LOGGER='[console,wandb]'
export PROJECT_NAME=verl_maxtext
export EXPERIMENT_NAME="qwen3_30b_a3b_gsm8k_$(date +%m%d_%H%M)"
export TRAIN_BATCH_SIZE=8 ROLLOUT_N=4 PPO_MINI_BATCH_SIZE=8
export MAX_PROMPT_LENGTH=256 MAX_RESPONSE_LENGTH=384
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-100}"
exec sbatch --exclude="${EXTRA_EXCLUDE:+${EXTRA_EXCLUDE},}useocpm2m-097-162,useocpm2m-097-163,useocpm2m-097-164" --export=ALL "${ROOT}/examples/maxtext/run_grpo_async.sbatch"
