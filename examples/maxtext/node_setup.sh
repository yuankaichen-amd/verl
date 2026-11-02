#!/bin/bash
# Load verl-maxtext:v26.7 if needed and start a long-lived container on this node.
set -euo pipefail
CNAME="${1:?container name}"
TAG="${TAG:-verl-maxtext:v26.7}"
TAR="${TAR:-/shared_inference/yuankach/verl-primus/verl-maxtext-v26.7.tar}"
ROOT=/home/yuankach/verl-primus
PRIMUS="${PRIMUS_PATH:-/home/yuankach/Miles-Prims-backup/Primus}"

docker image inspect "${TAG}" >/dev/null 2>&1 || docker load -i "${TAR}"
# vLLM finds ROCm through the amdsmi python module. The image only carries it
# inside the rocm-sdk tree, so expose that package and just its shared library.
SDK=/opt/venv/lib/python3.12/site-packages/_rocm_sdk_devel
IMAGE_LD="$(docker image inspect -f '{{range .Config.Env}}{{println .}}{{end}}' "${TAG}" | sed -n 's/^LD_LIBRARY_PATH=//p')"
docker rm -f "${CNAME}" >/dev/null 2>&1 || true
docker run -d --name "${CNAME}" --user root \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  --ipc=host --network=host --shm-size=64G \
  --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
  -v /shared_inference:/shared_inference -v /home/yuankach:/home/yuankach \
  -e PRIMUS_PATH="${PRIMUS}" \
  -e MAXTEXT_PATH=/workspace/maxtext \
  -e PRIMUS_WORKSPACE=/tmp/verl-maxtext \
  -e MODEL_PATH="${MODEL_PATH:-/home/yuankach/qwen3_30b_iter199_hf}" \
  -e HF_HUB_OFFLINE=1 \
  -e WANDB_MODE="${WANDB_MODE:-disabled}" -e WANDB_API_KEY="${WANDB_API_KEY:-}" -e WANDB_ENTITY="${WANDB_ENTITY:-}" \
  -e MAXTEXT_STACK_DUMP_SECONDS="${MAXTEXT_STACK_DUMP_SECONDS:-0}" \
  -e PYTORCH_ALLOC_CONF=expandable_segments:False \
  -e PYTORCH_HIP_ALLOC_CONF=expandable_segments:False \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:False \
  -e PYTHONPATH="${ROOT}:${PRIMUS}:/workspace/maxtext/src:${SDK}/share/amd_smi" \
  -e LD_LIBRARY_PATH="/opt/amdsmi_lib${IMAGE_LD:+:${IMAGE_LD}}" \
  "${TAG}" sleep infinity >/dev/null
docker exec "${CNAME}" bash -c "mkdir -p /opt/amdsmi_lib && ln -sf ${SDK}/lib/libamd_smi.so* /opt/amdsmi_lib/"
echo "NODE_READY $(hostname)"

# The MaxText image ships libtpu for JAX. vLLM then activates its TPU platform
# next to ROCm and refuses to pick one. JAX on ROCm does not use libtpu.
docker exec "${CNAME}" bash -c 'pip list 2>/dev/null | grep -iE "tpu" || true'
docker exec "${CNAME}" bash -c 'pip uninstall -y libtpu libtpu-nightly 2>&1 | tail -2 || true'
docker exec "${CNAME}" python -c "from vllm.platforms import current_platform; print('VLLM_PLATFORM', current_platform.device_name, flush=True)" 2>&1 | grep -v UserWarning | tail -3

# Weights & Biases logging needs the client in the image.
if [ "${WANDB_MODE:-disabled}" != "disabled" ]; then
  docker exec "${CNAME}" bash -c 'python -c "import wandb" 2>/dev/null || pip install -q wandb 2>&1 | tail -2'
  docker exec "${CNAME}" python -c "import wandb; print('WANDB_VERSION', wandb.__version__)"
fi
