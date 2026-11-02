#!/usr/bin/env bash
# 150-step DAPO-Math-17k run, 8k responses, DAPO overlong penalty, lr 1e-6.
# Each step samples fresh prompts (100 x 8 of 17k), so the train reward is an unbiased estimate.
set -euo pipefail
ROOT=/home/yuankach/verl-primus
export WANDB_API_KEY="$(cat "${HOME}/.wandb_key")"
export WANDB_MODE=online
export PRIMUS_CONFIG="${ROOT}/examples/maxtext/qwen3_30b_a3b_dapo8k.yaml"
export TRAIN_FILES=/shared_inference/yuankach/verl-primus/data/dapo/train.parquet
export VAL_FILES=/shared_inference/yuankach/verl-primus/data/dapo/val.parquet
export REWARD_FN=builtin
export TRAINER_LOGGER='[console,wandb]'
export PROJECT_NAME=verl_maxtext
export EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_30b_a3b_dapo8k_$(date +%m%d_%H%M)}"
# Weights-only checkpoints (fp32, no optimizer state). A requeued or resubmitted job with the
# same EXPERIMENT_NAME resumes from the latest one and continues the same W&B run.
export CKPT_DIR="${CKPT_DIR:-/shared_inference/yuankach/verl-primus/ckpt/${EXPERIMENT_NAME}}"
export SAVE_FREQ="${SAVE_FREQ:-10}" RESUME_MODE="${RESUME_MODE:-auto}" MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}"
export WANDB_RUN_ID="${WANDB_RUN_ID:-${EXPERIMENT_NAME}}" WANDB_RESUME=allow
export TRAIN_BATCH_SIZE=8 ROLLOUT_N=8 PPO_MINI_BATCH_SIZE=8
export MAX_PROMPT_LENGTH=512 MAX_RESPONSE_LENGTH=8192
# DAPO overlong penalty: linear 0 to -1 over the last 1024 tokens of the cap.
export EXTRA_ARGS="data.truncation=left data.filter_overlong_prompts=False actor_rollout_ref.actor.clip_ratio_high=0.28 reward.reward_manager.name=dapo +reward.reward_kwargs.max_resp_len=8192 +reward.reward_kwargs.overlong_buffer_cfg.enable=True +reward.reward_kwargs.overlong_buffer_cfg.len=1024 +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=1.0 +reward.reward_kwargs.overlong_buffer_cfg.log=True"
export TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-150}"
exec sbatch --time="${TIME_LIMIT:-12:00:00}" -p "${PARTITION:-amd-arad}" --qos="${QOS:-normal}" --exclude="${EXTRA_EXCLUDE:+${EXTRA_EXCLUDE},}useocpm2m-097-162,useocpm2m-097-163,useocpm2m-097-164" --export=ALL "${ROOT}/examples/maxtext/run_grpo_async.sbatch"
