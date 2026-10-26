#!/bin/bash
# Start the Miles container and install the pure-Python veRL extras on this node.
set -euo pipefail
CNAME="${1:?container name}"
ROOT=/home/yuankach/verl-primus
IMAGE="${IMAGE:-verlai/verl:rocm7.14_torch2.12_release_0724}"
FALLBACK_IMAGE="${FALLBACK_IMAGE:-rocm/sgl-dev:miles-rocm10-mi30x-20261001}"
PRIMUS_PATH="${PRIMUS_PATH:-/home/yuankach/Miles-Prims-backup/Primus}"
PRIMUS_TURBO_PATH="${PRIMUS_TURBO_PATH:-/home/yuankach/Miles-Prims-backup/Primus-Turbo}"

docker rm -f "${CNAME}" >/dev/null 2>&1 || true
docker ps -aq --filter name=verl_primus_async_ | xargs -r docker rm -f >/dev/null 2>&1 || true
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
echo "USING_IMAGE ${IMAGE} on $(hostname)"
docker run -d --name "${CNAME}" --user root \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --network=host --shm-size=64G \
  -v /shared_inference:/shared_inference -v /home/yuankach:/home/yuankach \
  -e PRIMUS_PATH -e PRIMUS_TURBO_PATH -e MODEL_PATH -e WANDB_MODE \
  -e USE_ROCM_AITER_ROPE_BACKEND=0 \
  -e CUDA_DEVICE_MAX_CONNECTIONS=1 \
  -e SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0 \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
  -e ROCM_HOME=/opt/rocm \
  -e PYTHONPATH="${ROOT}:${PRIMUS_PATH}:${PRIMUS_TURBO_PATH}" \
  "${IMAGE}" sleep infinity >/dev/null
docker exec "${CNAME}" python3 -m pip install --disable-pip-version-check --no-deps \
  'tensordict>=0.8.0,<=0.10.0,!=0.9.0' pyvers cloudpickle hydra-core omegaconf antlr4-python3-runtime \
  codetiming dill pyarrow pandas peft pylatexenc accelerate datasets packaging wandb \
  'TransferQueue==0.1.11' pyzmq msgspec 'prometheus_client>=0.20.0' \
  cachetools einops fastapi uvicorn torchdata cupy-rocm-7-0
docker exec -e ROCM_HOME=/opt/rocm "${CNAME}" python3 -c 'import cupy; import ray.util.collective.collective_group.nccl_collective_group as nccl; print("CUPY_NCCL_OK", cupy.__version__, nccl.NCCLGroup)'
