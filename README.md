# MathHelper

MathHelper 是一个用于数学推理后训练实验的小型代码库，当前主要覆盖 GSM8K 上的 SFT、GRPO/DAPO 风格强化学习，以及 GSM8K / MATH-500 评测。

## 目录结构

```text
.
├── src/mathhelper/              # 公共工具函数：数据加载、prompt、答案抽取与判分
├── scripts/
│   ├── train_gsm8k_sft_qlora.py # GSM8K QLoRA SFT
│   ├── train_gsm8k_sft_dapo.py  # TRL 两阶段 SFT -> DAPO/GRPO
│   ├── evaluate_math_accuracy.py # GSM8K + MATH-500 评测
│   ├── reward_gsm8k_grpo.py     # verl 自定义 GSM8K reward
│   ├── verl/                    # verl/FSDP 4 卡运行脚本
│   └── legacy/                  # 保留的旧版独立评测脚本
├── pyproject.toml
├── requirements.txt
└── README.md
```

## 安装

建议使用 Python 3.10+ 和 CUDA 环境：

```bash
pip install -e ".[train,eval]"
```

如果只想阅读或跑非 4-bit 流程，可以先安装基础依赖：

```bash
pip install -e .
```

## 数据

默认会从 Hugging Face 加载：

- `openai/gsm8k`
- `HuggingFaceH4/MATH-500`

也可以传入本地 `.json` / `.jsonl` / `.parquet`。GSM8K 需要 `question`、`answer` 字段；MATH-500 需要 `problem`、`answer` 字段。

## 常用命令

QLoRA SFT：

```bash
python scripts/train_gsm8k_sft_qlora.py \
  --model_path Qwen/Qwen3-1.7B-Base \
  --output_dir outputs/qwen3-1.7b-gsm8k-sft-qlora
```

小样本 smoke test：

```bash
python scripts/train_gsm8k_sft_qlora.py \
  --max_train_samples 64 \
  --max_eval_samples 16 \
  --output_dir outputs/smoke-sft
```

评测 SFT/LoRA adapter：

```bash
python scripts/evaluate_math_accuracy.py \
  --base_model_path Qwen/Qwen3-1.7B-Base \
  --adapter_path outputs/qwen3-1.7b-gsm8k-sft-qlora/final \
  --output_dir outputs/qwen3-1.7b-gsm8k-sft-qlora/eval
```

TRL 两阶段 SFT -> DAPO/GRPO：

```bash
python scripts/train_gsm8k_sft_dapo.py \
  --base_model Qwen/Qwen3-1.7B-Base \
  --gsm8k_path openai/gsm8k \
  --bf16
```

verl 4 卡脚本在 `scripts/verl/` 下。默认假设仓库根目录或 `$RWRL_ROOT` 下有 `verl/`、`data/`、`model/` 等目录；也可以用环境变量覆盖：

```bash
RWRL_ROOT=/path/to/workspace VERL_PATH=/path/to/verl bash scripts/verl/run_sft_4gpu.sh
RWRL_ROOT=/path/to/workspace VERL_PATH=/path/to/verl bash scripts/verl/run_grpo_4gpu.sh
```

## 输出

训练和评测结果默认写入 `outputs/`。该目录已加入 `.gitignore`，避免把 checkpoint、模型权重和大文件误传到 GitHub。
