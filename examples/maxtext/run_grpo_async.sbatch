#!/bin/bash
#SBATCH --job-name=verl-maxtext-grpo
#SBATCH --account=amd-rccl-guest
#SBATCH --partition=amd-arad-burst
#SBATCH --qos=low
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi300x:8
#SBATCH --exclusive
#SBATCH --exclude=useocpm2m-097-162,useocpm2m-097-163,useocpm2m-097-164
#SBATCH --time=04:00:00
#SBATCH --output=/shared_inference/yuankach/verl-primus/logs/%x-%j.out
# Async GRPO: node 0 runs the MaxText trainer (one actor, 8 GPUs) and node 1
# serves vLLM. Weights move through the NCCL checkpoint engine.
set -euo pipefail
ROOT=/home/yuankach/verl-primus
LOGDIR=/shared_inference/yuankach/verl-primus/logs/${SLURM_JOB_ID}
mkdir -p "${LOGDIR}"
NODES=($(scontrol show hostnames "${SLURM_JOB_NODELIST}"))
HEAD="${NODES[0]}"
WORKER="${NODES[1]}"
CNAME="verl_maxtext_${SLURM_JOB_ID}"
RAY_PORT=26379
echo "NODES head=${HEAD} worker=${WORKER}" | tee "${LOGDIR}/nodes.txt"
cleanup() {
  # Keep Ray's per-worker logs: a hung actor leaves no trace on the driver.
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 bash -c \
    "docker cp ${CNAME}:/tmp/ray/session_latest/logs ${LOGDIR}/ray_logs_\$(hostname) >/dev/null 2>&1 || true"
  srun --nodes=2 --ntasks=2 --ntasks-per-node=1 docker rm -f "${CNAME}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

srun --nodes=2 --ntasks=2 --ntasks-per-node=1 --export=ALL \
  bash "${ROOT}/examples/maxtext/node_setup.sh" "${CNAME}" | tee "${LOGDIR}/setup.log"

HEAD_IP=$(srun --nodes=1 --ntasks=1 -w "${HEAD}" docker exec "${CNAME}" hostname -I | awk '{print $1}')
echo "HEAD_IP ${HEAD_IP}" | tee -a "${LOGDIR}/nodes.txt"
srun --nodes=1 --ntasks=1 -w "${HEAD}" docker exec "${CNAME}" \
  ray start --head --port="${RAY_PORT}" --num-gpus=8 --disable-usage-stats --dashboard-host=127.0.0.1
srun --nodes=1 --ntasks=1 -w "${WORKER}" docker exec "${CNAME}" \
  ray start --address="${HEAD_IP}:${RAY_PORT}" --num-gpus=8 --disable-usage-stats
for _ in 1 2 3 4 5 6; do
  srun --nodes=1 --ntasks=1 -w "${HEAD}" docker exec -e RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" "${CNAME}" \
    ray status | tee "${LOGDIR}/ray_status.txt"
  grep -q '16.0 GPU' "${LOGDIR}/ray_status.txt" && break
  sleep 5
done
grep -q '16.0 GPU' "${LOGDIR}/ray_status.txt"

# Host memory every 20 s on the trainer node, to explain any SIGKILL.
srun --overlap --nodes=1 --ntasks=1 -w "${HEAD}" bash -c \
  'while true; do echo "$(date +%T) $(free -m | awk "/Mem:/{print \"used_mb=\"\$3\" avail_mb=\"\$7}")"; sleep 20; done' \
  > "${LOGDIR}/hostmem.log" 2>&1 &
MEMPID=$!
trap 'kill ${MEMPID} 2>/dev/null || true; cleanup' EXIT

srun --nodes=1 --ntasks=1 -w "${HEAD}" docker exec \
  -e RAY_ADDRESS="${HEAD_IP}:${RAY_PORT}" \
  -e PRIMUS_CONFIG="${PRIMUS_CONFIG:-${ROOT}/examples/maxtext/qwen3_30b_a3b.yaml}" \
  -e TRAIN_FILES="${TRAIN_FILES:-${ROOT}/data/primus_smoke/prompts.parquet}" \
  -e VAL_FILES="${VAL_FILES:-${TRAIN_FILES:-${ROOT}/data/primus_smoke/prompts.parquet}}" \
  -e REWARD_FN="${REWARD_FN:-}" \
  -e TRAINER_LOGGER="${TRAINER_LOGGER:-[console]}" \
  -e PROJECT_NAME="${PROJECT_NAME:-verl_maxtext}" -e EXPERIMENT_NAME="${EXPERIMENT_NAME:-qwen3_30b_a3b_maxtext}" \
  -e SAVE_FREQ="${SAVE_FREQ:--1}" -e CKPT_DIR="${CKPT_DIR:-}" -e RESUME_MODE="${RESUME_MODE:-disable}" \
  -e MAX_CKPT_KEEP="${MAX_CKPT_KEEP:-2}" ${WANDB_RUN_ID:+-e WANDB_RUN_ID=${WANDB_RUN_ID} -e WANDB_RESUME=${WANDB_RESUME:-allow}} \
  -e WANDB_API_KEY="${WANDB_API_KEY:-}" -e WANDB_MODE="${WANDB_MODE:-disabled}" \
  -e TRAIN_BATCH_SIZE="${TRAIN_BATCH_SIZE:-4}" -e ROLLOUT_N="${ROLLOUT_N:-2}" \
  -e PPO_MINI_BATCH_SIZE="${PPO_MINI_BATCH_SIZE:-4}" \
  -e MAX_PROMPT_LENGTH="${MAX_PROMPT_LENGTH:-128}" -e MAX_RESPONSE_LENGTH="${MAX_RESPONSE_LENGTH:-64}" \
  -e TOTAL_TRAINING_STEPS="${TOTAL_TRAINING_STEPS:-4}" -e TOTAL_EPOCHS=100 \
  "${CNAME}" \
  bash "${ROOT}/examples/maxtext/run_qwen3_30b_a3b_grpo.sh" \
  trainer.nnodes=1 trainer.n_gpus_per_node=8 ${EXTRA_ARGS:-} \
  +ray_kwargs.ray_init.address="${HEAD_IP}:${RAY_PORT}" \
  2>&1 | tee "${LOGDIR}/grpo.log"
echo MAXTEXT_ASYNC_DONE | tee "${LOGDIR}/done.txt"
