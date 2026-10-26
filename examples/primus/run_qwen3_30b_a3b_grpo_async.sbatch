#!/bin/bash
#SBATCH --job-name=verl-primus-async
#SBATCH --account=amd-rccl-guest
#SBATCH --partition=amd-arad-burst
#SBATCH --qos=low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi300x:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/shared_inference/yuankach/verl-primus/logs/%x-%j.out
# Fully async GRPO: node 0 trains Primus, node 1 serves SGLang. Dataset is the
# 8-row synthetic parquet at data/primus_smoke/prompts.parquet.
set -euo pipefail
ROOT=/home/yuankach/verl-primus
LOGDIR=/shared_inference/yuankach/verl-primus/logs/${SLURM_JOB_ID}
mkdir -p "${LOGDIR}"
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD="${NODES[0]}"
WORKER="${NODES[1]}"
CNAME="verl_primus_async_${SLURM_JOB_ID}"
RAY_PORT=26379
export PRIMUS_PATH="${PRIMUS_PATH:-/home/yuankach/Miles-Prims-backup/Primus}"
export PRIMUS_TURBO_PATH="${PRIMUS_TURBO_PATH:-/home/yuankach/Miles-Prims-backup/Primus-Turbo}"
export MODEL_PATH="${MODEL_PATH:-/home/yuankach/qwen3_30b_iter199_hf}"
export WANDB_MODE=disabled
export TRAIN_FILES="${ROOT}/data/primus_smoke/prompts.parquet"
export VAL_FILES="${ROOT}/data/primus_smoke/prompts.parquet"
export TRAIN_BATCH_SIZE=4
export ROLLOUT_N=2
export MAX_PROMPT_LENGTH=128
export MAX_RESPONSE_LENGTH=64
export PPO_MINI_BATCH_SIZE=4
export PPO_MAX_TOKEN_LEN_PER_GPU=512
export REF_MAX_TOKEN_LEN_PER_GPU=512
export ROLLOUT_LOGPROB_MAX_TOKENS=512
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.8
export TOTAL_TRAINING_STEPS=100
export TOTAL_EPOCHS=100
export TEST_FREQ=0
export SAVE_FREQ=-1
export NNODES=1
export GPUS_PER_NODE=8
export EXPERIMENT_NAME=qwen3_30b_a3b_async
echo "ASYNC_NODES head=${HEAD} worker=${WORKER} dataset=${TRAIN_FILES}" | tee "${LOGDIR}/nodes.txt"
cleanup() {
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 docker rm -f "${CNAME}" >/dev/null 2>&1 || true
}
trap cleanup EXIT

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --export=ALL \
  bash "${ROOT}/examples/primus/async_node_setup.sh" "${CNAME}" | tee "${LOGDIR}/setup.log"

HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD}" docker exec "${CNAME}" hostname -I | awk '{print $1}')
echo "HEAD_IP ${HEAD_IP}" | tee -a "${LOGDIR}/nodes.txt"
srun --nodes=1 --ntasks=1 -w "${HEAD}" docker exec "${CNAME}" \
  ray start --head --port="${RAY_PORT}" --num-gpus=8 --disable-usage-stats --dashboard-host=127.0.0.1
srun --nodes=1 --ntasks=1 -w "${WORKER}" docker exec "${CNAME}" \
  ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus=8 --disable-usage-stats
for _ in 1 2 3 4 5 6; do
  srun --nodes=1 --ntasks=1 -w "${HEAD}" docker exec -e RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" "${CNAME}" \
    ray status | tee "${LOGDIR}/ray_status.txt"
  if grep -q '16.0 GPU' "${LOGDIR}/ray_status.txt"; then
    break
  fi
  sleep 5
done
grep -q '16.0 GPU' "${LOGDIR}/ray_status.txt"

srun --nodes=1 --ntasks=1 -w "${HEAD}" docker exec \
  -e PYTHONPATH="${ROOT}:${PRIMUS_PATH}:${PRIMUS_TURBO_PATH}" \
  -e WANDB_MODE=disabled \
  -e PRIMUS_PATH -e PRIMUS_TURBO_PATH -e MODEL_PATH \
  -e TRAIN_FILES -e VAL_FILES -e TRAIN_BATCH_SIZE -e ROLLOUT_N \
  -e MAX_PROMPT_LENGTH -e MAX_RESPONSE_LENGTH -e PPO_MINI_BATCH_SIZE \
  -e PPO_MAX_TOKEN_LEN_PER_GPU -e REF_MAX_TOKEN_LEN_PER_GPU \
  -e ROLLOUT_LOGPROB_MAX_TOKENS -e ROLLOUT_GPU_MEMORY_UTILIZATION \
  -e TOTAL_TRAINING_STEPS -e TOTAL_EPOCHS -e TEST_FREQ -e SAVE_FREQ \
  -e NNODES -e GPUS_PER_NODE -e EXPERIMENT_NAME \
  -e RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" \
  "${CNAME}" \
  bash "${ROOT}/examples/primus/run_qwen3_30b_a3b_grpo_rocm.sh" \
  trainer.v1.trainer_mode=separate_async \
  trainer.v1.separate_async.parameter_sync_step=1 \
  trainer.v1.separate_async.num_warmup_batches=1 \
  actor_rollout_ref.hybrid_engine=false \
  actor_rollout_ref.rollout.nnodes=1 \
  actor_rollout_ref.rollout.n_gpus_per_node=8 \
  actor_rollout_ref.rollout.checkpoint_engine.backend=nccl \
  actor_rollout_ref.actor.primus.param_offload=false \
  actor_rollout_ref.actor.primus.optimizer_offload=false \
  actor_rollout_ref.actor.use_kl_loss=false \
  trainer.logger='[console]' \
  trainer.val_before_train=false \
  +ray_kwargs.ray_init.address="${HEAD_IP}:${RAY_PORT}" \
  | tee "${LOGDIR}/grpo.log"
echo PRIMUS_ASYNC_DONE | tee "${LOGDIR}/done.txt"
