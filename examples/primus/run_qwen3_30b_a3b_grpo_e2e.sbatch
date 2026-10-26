#!/bin/bash
#SBATCH --job-name=verl-primus-grpo
#SBATCH --account=amd-rccl-guest
#SBATCH --partition=amd-arad-burst
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi300x:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/shared_inference/yuankach/verl-primus/logs/%x-%j.out
# One colocated GRPO step: SGLang rollout, weight sync, Primus optimizer step.
set -euo pipefail
ROOT=/home/yuankach/verl-primus
LOGDIR=/shared_inference/yuankach/verl-primus/logs/${SLURM_JOB_ID}
mkdir -p "${LOGDIR}"
IMAGE="${IMAGE:-verlai/verl:rocm7.14_torch2.12_release_0724}"
FALLBACK_IMAGE="${FALLBACK_IMAGE:-rocm/sgl-dev:miles-rocm10-mi30x-20261001}"
CNAME="verl_primus_grpo_${SLURM_JOB_ID}"
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
export ROLLOUT_GPU_MEMORY_UTILIZATION=0.5
export TOTAL_TRAINING_STEPS=2
export TEST_FREQ=0
export SAVE_FREQ=-1

docker rm -f "${CNAME}" >/dev/null 2>&1 || true
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  docker pull "${IMAGE}" || true
fi
if ! docker image inspect "${IMAGE}" >/dev/null 2>&1; then
  if docker image inspect "${FALLBACK_IMAGE}" >/dev/null 2>&1; then
    IMAGE="${FALLBACK_IMAGE}"
  else
    echo "NO_LOCAL_IMAGE on $(hostname)"
    docker images
    exit 1
  fi
fi
echo "USING_IMAGE ${IMAGE}" | tee "${LOGDIR}/image.txt"
docker run -d --name "${CNAME}" --user root \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --network=host --shm-size=64G \
  -v /shared_inference:/shared_inference -v /home/yuankach:/home/yuankach \
  -e PRIMUS_PATH -e PRIMUS_TURBO_PATH -e MODEL_PATH -e WANDB_MODE \
  -e USE_ROCM_AITER_ROPE_BACKEND=0 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
  -e PYTHONPATH="${ROOT}:${PRIMUS_PATH}:${PRIMUS_TURBO_PATH}" \
  "${IMAGE}" sleep infinity >/dev/null
cleanup() { docker rm -f "${CNAME}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

run() {
  docker exec \
    -e PYTHONPATH="${ROOT}:${PRIMUS_PATH}:${PRIMUS_TURBO_PATH}" \
    -e WANDB_MODE=disabled \
    -e PRIMUS_PATH -e PRIMUS_TURBO_PATH -e MODEL_PATH \
    -e TRAIN_FILES -e VAL_FILES -e TRAIN_BATCH_SIZE -e ROLLOUT_N \
    -e MAX_PROMPT_LENGTH -e MAX_RESPONSE_LENGTH -e PPO_MINI_BATCH_SIZE \
    -e PPO_MAX_TOKEN_LEN_PER_GPU -e REF_MAX_TOKEN_LEN_PER_GPU \
    -e ROLLOUT_LOGPROB_MAX_TOKENS -e ROLLOUT_GPU_MEMORY_UTILIZATION \
    -e TOTAL_TRAINING_STEPS -e TEST_FREQ -e SAVE_FREQ \
    "${CNAME}" "$@"
}
run python3 -m pip install --disable-pip-version-check --no-deps \
  'tensordict>=0.8.0,<=0.10.0,!=0.9.0' pyvers cloudpickle hydra-core omegaconf antlr4-python3-runtime \
  codetiming dill pyarrow pandas peft pylatexenc accelerate datasets packaging wandb \
  'TransferQueue==0.1.11' pyzmq msgspec 'prometheus_client>=0.20.0' \
  cachetools einops fastapi uvicorn torchdata \
  | tee "${LOGDIR}/pip.log"
run bash "${ROOT}/examples/primus/run_qwen3_30b_a3b_grpo_rocm.sh" \
  actor_rollout_ref.actor.primus.param_offload=true \
  actor_rollout_ref.actor.primus.optimizer_offload=true \
  actor_rollout_ref.actor.use_kl_loss=false \
  trainer.logger='[console]' \
  trainer.val_before_train=false \
  | tee "${LOGDIR}/grpo.log"
echo PRIMUS_GRPO_DONE | tee "${LOGDIR}/done.txt"
