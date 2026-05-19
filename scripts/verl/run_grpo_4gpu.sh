#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ROOT="${RWRL_ROOT:-${REPO_ROOT}}"
ENV_PY="${ENV_PY:-python}"
VERL_PATH="${VERL_PATH:-${ROOT}/verl}"

export PYTHONPATH="${VERL_PATH}:${PYTHONPATH:-}"
if [ -n "${CONDA_PREFIX:-}" ]; then
  export LD_LIBRARY_PATH="${CONDA_PREFIX}/lib:${LD_LIBRARY_PATH:-}"
fi
export DISABLE_VERSION_CHECK=1
export TOKENIZERS_PARALLELISM=false
export RAY_DEDUP_LOGS=0
export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=0
export NCCL_SOCKET_IFNAME="${NCCL_SOCKET_IFNAME:-eno1}"
export GLOO_SOCKET_IFNAME="${GLOO_SOCKET_IFNAME:-eno1}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"

# 先确保 SFT 合并模型存在（默认用最终 step_467；可通过 SFT_STEP 覆盖）
SFT_STEP="${SFT_STEP:-467}"
SFT_MERGED_MODEL="${SFT_MERGED_MODEL:-${ROOT}/outputs/sft_grpo_qwen1p7b/sft_merged/step_${SFT_STEP}_hf_lora_merged}"
if [ ! -d "${SFT_MERGED_MODEL}" ]; then
  echo "SFT merged model not found: ${SFT_MERGED_MODEL}"
  echo "请先运行 merge_fsdp_ckpt.sh 合并一个SFT checkpoint。"
  exit 1
fi

mkdir -p "${ROOT}/outputs/sft_grpo_qwen1p7b/rl_grpo" "${ROOT}/outputs/sft_grpo_qwen1p7b/rollout_data" "${ROOT}/result/log" "${ROOT}/result/tensorboard/sft_grpo_qwen1p7b/grpo_stage_4gpu"
export TENSORBOARD_DIR="${ROOT}/result/tensorboard/sft_grpo_qwen1p7b/grpo_stage_4gpu"

LOG_FILE="${ROOT}/result/log/grpo_stage_4gpu.log"

"${ENV_PY}" -m verl.trainer.main_ppo \
  algorithm.adv_estimator=grpo \
  algorithm.use_kl_in_reward=False \
  algorithm.kl_ctrl.kl_coef=0.02 \
  data.train_files="${ROOT}/data/gsm8k_parquet/train.parquet" \
  data.val_files="${ROOT}/data/gsm8k_parquet/test.parquet" \
  data.train_batch_size=8 \
  data.val_batch_size=64 \
  data.max_prompt_length=512 \
  data.max_response_length=512 \
  data.filter_overlong_prompts=True \
  data.truncation=error \
  data.shuffle=False \
  actor_rollout_ref.model.path="${SFT_MERGED_MODEL}" \
  actor_rollout_ref.model.use_shm=False \
  actor_rollout_ref.model.lora_rank=0 \
  actor_rollout_ref.model.lora_alpha=128 \
  actor_rollout_ref.model.target_modules=all-linear \
  actor_rollout_ref.model.enable_gradient_checkpointing=True \
  actor_rollout_ref.actor.optim.lr=5e-6 \
  actor_rollout_ref.actor.ppo_mini_batch_size=8 \
  actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.actor.use_kl_loss=True \
  actor_rollout_ref.actor.kl_loss_coef=0.02 \
  actor_rollout_ref.actor.kl_loss_type=low_var_kl \
  actor_rollout_ref.actor.entropy_coeff=0 \
  actor_rollout_ref.actor.fsdp_config.param_offload=False \
  actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
  actor_rollout_ref.rollout.name=vllm \
  actor_rollout_ref.rollout.n=4 \
  actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
  actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
  actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.rollout.load_format=safetensors \
  actor_rollout_ref.rollout.layered_summon=True \
  actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=2 \
  actor_rollout_ref.ref.fsdp_config.param_offload=False \
  custom_reward_function.path="${ROOT}/scripts/reward_gsm8k_grpo.py" \
  custom_reward_function.name=compute_score \
  trainer.critic_warmup=0 \
  trainer.logger='["console","tensorboard"]' \
  trainer.project_name=sft_grpo_qwen1p7b \
  trainer.experiment_name=grpo_stage_4gpu \
  trainer.default_local_dir="${ROOT}/outputs/sft_grpo_qwen1p7b/rl_grpo" \
  trainer.rollout_data_dir="${ROOT}/outputs/sft_grpo_qwen1p7b/rollout_data" \
  trainer.n_gpus_per_node=4 \
  trainer.nnodes=1 \
  trainer.total_training_steps=200 \
  trainer.total_epochs=1 \
  trainer.save_freq=100 \
  trainer.test_freq=100 \
  trainer.val_before_train=False \
  trainer.resume_mode=disable \
  2>&1 | tee "${LOG_FILE}"
