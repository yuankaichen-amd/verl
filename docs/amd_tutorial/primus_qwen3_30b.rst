Primus trainer for Qwen3-30B-A3B
=================================

``strategy: primus`` is a standalone veRL engine. ``PrimusEngine`` inherits the
Megatron engine, and the Primus experiment YAML is the trainer configuration.

Pinned revisions
----------------

- veRL: ``3c6d9b43b67c74ec343d77874cd8fa6e841523da``
- Primus: ``8779b86cd7218fb3b503159c9380f532ee44fa7d``
- Primus-Turbo: ``8cda13c79adca2f8fd4aef7cda7ed19597e03b3b``
- Intended container: ``verlai/verl:rocm7.14_torch2.12_release_0724``. The registry token on this cluster was expired, so the measured runs used the local image ``rocm/sgl-dev:miles-rocm10-mi30x-20261001`` (ROCm 10, Megatron-LM under ``/root/Megatron-LM``, SGLang from ``/sgl-workspace/sglang``).
- Model: ``/home/yuankach/qwen3_30b_iter199_hf`` (Qwen3-30B-A3B)

Recipe
------

``examples/primus/configs/qwen3_30b_a3b_grpo.yaml`` selects TP=1, PP=1, CP=2,
EP=4, BF16, packed THD, and TransformerEngine flash attention. Dynamic batching
uses the actor token budget; veRL multiplies it by the context-parallel size
before packing. Turbo attention stays off because its gradient parity is not
validated. AITER is rejected for this packed path.

Launch
------

.. code-block:: bash

    export PRIMUS_PATH=/home/yuankach/Miles-Prims-backup/Primus
    export PRIMUS_TURBO_PATH=/home/yuankach/Miles-Prims-backup/Primus-Turbo
    export MODEL_PATH=/home/yuankach/qwen3_30b_iter199_hf
    bash examples/primus/run_qwen3_30b_a3b_grpo_rocm.sh

The trainer-only job is ``examples/primus/run_qwen3_30b_a3b_grpo_rocm.sbatch``.
The colocated GRPO job is ``examples/primus/run_qwen3_30b_a3b_grpo_e2e.sbatch``.
The fully async job is ``examples/primus/run_qwen3_30b_a3b_grpo_async.sbatch``
(one trainer node, one SGLang node). All three need nodes that already have the
Miles image cached. Pass those nodes with ``sbatch -w <node>`` or
``sbatch -w <node0>,<node1>``.

The prompts are the 8-row file ``data/primus_smoke/prompts.parquet``. Each row
asks the model to reply with the digit 2. The reward is the completion's UTF-8
byte sum modulo 5, divided by 4, so scores land on 0, 0.25, 0.5, 0.75, or 1.
DAPO-Math-17k and AIME-2024 are not on this machine.

Measured on one MI300X node (8 GPUs)
-------------------------------------

Job 499526 on ``useocpm2m-097-121`` ran the container unit tests (7 passed) and
two optimizer steps.

- Tiny Qwen3 MoE, TP=1 PP=1 CP=2 EP=1, packed THD, dynamic batches, zigzag CP:
  loss 0.194, grad norm 0.00511, checkpoint save and HF-bridge reload succeeded.
  The provider log recorded ``context_parallel_size=2``,
  ``cp_partition_mode='zigzag'``, ``variable_seq_lengths=True``, and flash attention.
- Qwen3-30B-A3B, recipe TP=1 PP=1 CP=2 EP=4: 8.79B parameters on rank (0, 0),
  loss 0.226, grad norm 2.77e-5, 92 GB allocated / 95 GB reserved. Context-parallel
  packing logged alignment of variable lengths 18 and 22.

Job 499615 on ``useocpm2m-097-078`` ran one colocated GRPO step with SGLang
rollout TP=8, actor parameter and optimizer offload, 4 prompts, 2 samples,
response length 64. Weight sync took 19.5 s, generation 19.9 s, and the actor
update 61.0 s. Actor-to-rollout KL was 0.00749 and the probability correlation
was 0.911. That smoke reward was 1 for every non-empty completion, so the GRPO
advantage on that step was 0.

Job 499856 on ``useocpm2m-097-071`` repeated the colocated run for two steps
with a response-dependent reward (scores 0.25 to 1.0). Step 1 advantages spanned
-0.707 to 0.707 and the grad norm was 3.54. The mean policy loss stayed at 0
because GRPO advantages are centered and the importance ratio was still 1.
Weight sync was 44.6 s on the initial copy and 16.9 s on the second copy, after
the optimizer step. Actor-to-rollout KL moved from 0.0158 to 0.0249. Step 2
happened to tie inside every group, so its advantage and grad norm were 0.
Peak actor memory stayed 92 GB.

Fully async (1 trainer node + 1 rollout node)
----------------------------------------------

``trainer.v1.trainer_mode=separate_async`` with ``hybrid_engine=false``,
``parameter_sync_step=1``, and ``checkpoint_engine.backend=nccl``. The trainer
keeps the recipe (TP=1 PP=1 CP=2 EP=4 on 8 GPUs). SGLang uses TP=8 on the other
node. Ray's NCCL collective backend imports CuPy; this image has no ROCm 10
wheel, and ``cupy-rocm-7-0`` 14.2.0 loads and allocates on the ROCm 10 Miles
image. ``total_epochs`` is 100 because eight prompts and a train batch of 4
make only two steps per epoch, and the loop stops at ``total_epochs`` even when
``total_training_steps`` is larger.

Job 500146 on ``useocpm2m-097-028`` (SGLang) and ``useocpm2m-097-159`` (Primus)
finished all 100 steps in 1 h 21 min (exit 0). Rank (0, 0) held 8.79B parameters
and actor memory stayed at 92.7 GB. Mean step time was 45 s and mean cross-node
weight sync was 26 s. Step 1 grad norm was 5.55, advantages spanned -0.707 to
0.707, reward mean was 0.5625, and actor-to-rollout KL was 0.0107. Entropy fell
from 0.076 to 0.00072 by step 100. From the middle of the run the 8-prompt set
often tied inside every group, so 72 of 100 steps had grad norm 0 and reward
mean 1.0. Step 100 KL was 6.3e-6.

Known limits
------------

- Qwen packed training requires ``use_remove_padding`` and ``use_packed_attention``.
- Context parallel below 2 is rejected for Qwen3 MoE.
- Primus-Turbo linear kernels require tensor parallel size 1.
- Router replay with context parallel is unchanged from the Megatron engine and
  remains limited to the modes that engine supports.
- The Miles SGLang build rejects ``update_weights_from_tensor`` until
  ``begin_weight_update`` opens a session. The rollout client opens and closes
  that session, and ignores a 404 from older servers.
- Colocated rollout needs actor parameter and optimizer offload. The actor cache
  is released before SGLang starts, and
  ``SGLANG_ENABLE_TP_MEMORY_INBALANCE_CHECK=0`` is set because one rank can
  still report less free memory than the others.
- Separate-async weight sync uses Ray NCCL collectives, which import CuPy.
  ``cupy-rocm-7-0`` is installed in the container; a source build is not required
  on this image.
- The 100-step async run cycles the same 8 smoke prompts. It does not train on
  DAPO-Math-17k.
