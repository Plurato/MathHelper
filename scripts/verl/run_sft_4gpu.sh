#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${RWRL_ROOT:-${REPO_ROOT}}"
ENV_TORCHRUN="${ENV_TORCHRUN:-torchrun}"
VERL_PATH="${VERL_PATH:-${ROOT}/verl}"

export PYTHONPATH="${VERL_PATH}:${PYTHONPATH:-}"
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-lo}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

mkdir -p "${ROOT}/result/log" "${ROOT}/outputs/sft_grpo_qwen1p7b/sft"

LOG_FILE="${ROOT}/result/log/sft_stage_4gpu.log"

"${ENV_TORCHRUN}" --standalone --nnodes=1 --nproc_per_node=4 -m verl.trainer.fsdp_sft_trainer \
  data.train_files="${ROOT}/data/gsm8k_parquet/train.parquet" \
  data.val_files="${ROOT}/data/gsm8k_parquet/test.parquet" \
  data.prompt_key=extra_info \
  +data.prompt_dict_keys='[question]' \
  data.response_key=extra_info \
  +data.response_dict_keys='[answer]' \
  data.train_batch_size=16 \
  data.micro_batch_size_per_gpu=2 \
  data.max_length=2048 \
  model.partial_pretrain="${ROOT}/model/Qwen3-1.7B-Base" \
  model.enable_gradient_checkpointing=True \
  model.lora_rank=64 \
  model.lora_alpha=128 \
  model.target_modules=all-linear \
  model.strategy=fsdp \
  model.fsdp_config.model_dtype=bf16 \
  optim.lr=2e-5 \
  trainer.total_epochs=1 \
  trainer.logger='["console"]' \
  trainer.project_name=sft_grpo_qwen1p7b \
  trainer.experiment_name=sft_stage_4gpu \
  trainer.default_local_dir="${ROOT}/outputs/sft_grpo_qwen1p7b/sft" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.resume_mode=disable \
  2>&1 | tee "${LOG_FILE}"
