#!/bin/bash
#SBATCH --job-name=verl-primus-qwen
#SBATCH --account=amd-rccl-guest
#SBATCH --partition=amd-arad-burst
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi300x:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/shared_inference/yuankach/verl-primus/logs/%x-%j.out
# Primus Qwen3-30B-A3B: tiny CP/THD step, then one real 30B optimizer step.
set -euo pipefail
ROOT=/home/yuankach/verl-primus
LOGDIR=/shared_inference/yuankach/verl-primus/logs/${SLURM_JOB_ID}
mkdir -p "${LOGDIR}"
IMAGE="${IMAGE:-verlai/verl:rocm7.14_torch2.12_release_0724}"
FALLBACK_IMAGE="${FALLBACK_IMAGE:-rocm/sgl-dev:miles-rocm10-mi30x-20261001}"
CNAME="verl_primus_${SLURM_JOB_ID}"
export PRIMUS_PATH="${PRIMUS_PATH:-/home/yuankach/Miles-Prims-backup/Primus}"
export PRIMUS_TURBO_PATH="${PRIMUS_TURBO_PATH:-/home/yuankach/Miles-Prims-backup/Primus-Turbo}"
export MODEL_PATH="${MODEL_PATH:-/home/yuankach/qwen3_30b_iter199_hf}"

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
  -e PRIMUS_PATH -e PRIMUS_TURBO_PATH -e MODEL_PATH \
  -e USE_ROCM_AITER_ROPE_BACKEND=0 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e PYTHONPATH="${ROOT}:${PRIMUS_PATH}:${PRIMUS_TURBO_PATH}" \
  "${IMAGE}" sleep infinity >/dev/null
cleanup() { docker rm -f "${CNAME}" >/dev/null 2>&1 || true; }
trap cleanup EXIT

run() { docker exec -e PYTHONPATH="${ROOT}:${PRIMUS_PATH}:${PRIMUS_TURBO_PATH}" "${CNAME}" "$@"; }
run python3 -m pip install --disable-pip-version-check --no-deps \
  'tensordict>=0.8.0,<=0.10.0,!=0.9.0' pyvers cloudpickle hydra-core omegaconf antlr4-python3-runtime \
  codetiming dill pyarrow pandas peft pylatexenc accelerate datasets packaging wandb \
  | tee "${LOGDIR}/pip.log"
run python3 -m pytest -q "${ROOT}/tests/workers/engine/primus" --tb=short | tee "${LOGDIR}/pytest.log"
run python3 "${ROOT}/examples/primus/smoke_qwen_thd_cp.py" --mode tiny | tee "${LOGDIR}/tiny.log"
run python3 "${ROOT}/examples/primus/smoke_qwen_thd_cp.py" --mode qwen30b | tee "${LOGDIR}/qwen30b.log"
echo PRIMUS_SMOKE_DONE | tee "${LOGDIR}/done.txt"
