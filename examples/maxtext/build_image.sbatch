#!/bin/bash
#SBATCH --job-name=verl-maxtext-image
#SBATCH --account=amd-rccl-guest
#SBATCH --partition=amd-arad-burst
#SBATCH --qos=low
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:mi300x:8
#SBATCH --exclusive
#SBATCH --time=04:00:00
#SBATCH --output=/shared_inference/yuankach/verl-primus/logs/%x-%j.out
# Pull the public MaxText image and build the vLLM/AITER layer on this node.
# The login node has no room for a 44GB image, and a stored Docker Hub token
# makes anonymous pulls of public repositories fail until logout.
set -euo pipefail

ROOT=/home/yuankach/verl-primus
PRIMUS=/home/yuankach/Miles-Prims-backup/Primus
LOGDIR=/shared_inference/yuankach/verl-primus/logs/${SLURM_JOB_ID}
TAG=verl-maxtext:v26.7
mkdir -p "${LOGDIR}"

echo "HOST $(hostname) $(date -Is)" | tee "${LOGDIR}/build.txt"
df -h / /var/lib/docker | tee -a "${LOGDIR}/build.txt"
AVAIL_GB=$(df -BG --output=avail / | tail -1 | tr -dc '0-9')
if [[ "${AVAIL_GB}" -lt 80 ]]; then
  echo "DISK_TOO_SMALL ${AVAIL_GB}GB" | tee -a "${LOGDIR}/build.txt"
  exit 1
fi

# Expired credentials block public pulls. This only drops the Docker Hub login.
docker logout https://index.docker.io/v1/ || true
docker pull rocm/jax-training:maxtext-v26.7 | tee -a "${LOGDIR}/build.txt"

docker build \
  -f "${ROOT}/docker/Dockerfile.maxtext-vllm" \
  -t "${TAG}" \
  "${ROOT}/docker" | tee -a "${LOGDIR}/build.txt"

# Primus-Turbo is the Megatron kernel package. The MaxText adapter does not
# import it, so it is not installed here.
CNAME="verl_maxtext_build_${SLURM_JOB_ID}"
docker rm -f "${CNAME}" >/dev/null 2>&1 || true
docker run -d --name "${CNAME}" --user root \
  --device=/dev/kfd --device=/dev/dri --group-add video --group-add render \
  -v /home/yuankach:/home/yuankach \
  "${TAG}" sleep infinity >/dev/null
docker exec "${CNAME}" python -m pip install --no-cache-dir --no-deps "${PRIMUS}" "${ROOT}" \
  | tee -a "${LOGDIR}/build.txt"
docker exec "${CNAME}" python - <<'PY' | tee "${LOGDIR}/imports.txt"
import jax, torch, vllm, aiter, primus, verl
print("jax", jax.__version__)
print("torch", torch.__version__)
print("vllm", vllm.__version__)
print("aiter", getattr(aiter, "__version__", "ok"))
print("primus", primus.__file__)
print("verl", verl.__file__)
PY
docker commit "${CNAME}" "${TAG}" | tee -a "${LOGDIR}/build.txt"
docker rm -f "${CNAME}" >/dev/null
docker save "${TAG}" -o /shared_inference/yuankach/verl-primus/verl-maxtext-v26.7.tar
echo "SAVED /shared_inference/yuankach/verl-primus/verl-maxtext-v26.7.tar" | tee -a "${LOGDIR}/build.txt"
ls -lh /shared_inference/yuankach/verl-primus/verl-maxtext-v26.7.tar | tee -a "${LOGDIR}/build.txt"
