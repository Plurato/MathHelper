"""
GSM8K two-stage: SFT -> online RL.

DAPO: Hugging Face TRL implements the DAPO objective as [`GRPOTrainer`] with
`loss_type="dapo"` (length-unbiased aggregation + recommended clip settings).
There is no separate `DAPOTrainer` class in TRL 1.x.

KL term: when `beta > 0`, GRPO adds `beta * per_token_kl` to the clipped policy
loss and keeps a frozen reference (full weights, or PEFT `"ref"` adapter copy).
"""

import argparse
import glob
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Sequence

import torch
from datasets import Dataset, load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import trl
from trl import GRPOConfig, GRPOTrainer, SFTConfig, SFTTrainer


def _require_peft() -> Any:
    try:
        from peft import LoraConfig, PeftModel, TaskType

        return LoraConfig, PeftModel, TaskType
    except ImportError as e:
        raise ImportError(
            "LoRA training requires `peft`. Install with: pip install peft"
        ) from e


@dataclass
class StageOutputs:
    sft_model_dir: str
    rl_model_dir: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Two-stage GSM8K: SFT (1 epoch, LoRA by default) -> DAPO RL (TRL GRPOTrainer, loss_type=dapo, KL via beta)"
    )

    parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--gsm8k_path", type=str, default="openai/gsm8k")
    parser.add_argument("--gsm8k_name", type=str, default="main")

    parser.add_argument("--sft_output", type=str, default="./outputs/sft_qwen3_1p7b_gsm8k")
    parser.add_argument("--rl_output", type=str, default="./outputs/rl_dapo_qwen3_1p7b_gsm8k")

    parser.add_argument("--max_seq_len", type=int, default=2048)
    parser.add_argument("--sft_batch_size", type=int, default=2)
    parser.add_argument("--sft_grad_accum", type=int, default=8)
    parser.add_argument("--sft_lr", type=float, default=2e-5)

    parser.add_argument("--rl_batch_size", type=int, default=2)
    parser.add_argument("--rl_grad_accum", type=int, default=4)
    parser.add_argument("--rl_lr", type=float, default=5e-6)
    parser.add_argument("--rl_max_steps", type=int, default=200)
    parser.add_argument("--num_generations", type=int, default=4)
    parser.add_argument("--max_completion_len", type=int, default=512)

    # KL penalty in GRPO/DAPO: per_token_loss += beta * per_token_kl (see TRL GRPOTrainer).
    parser.add_argument(
        "--beta",
        type=float,
        default=0.02,
        help="KL loss coefficient β: adds β * KL(π_θ || π_ref) per token. Set 0 to disable (no ref model).",
    )
    parser.add_argument(
        "--use_bias_correction_kl",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Use TRL's bias-corrected KL estimate (importance-sampling corrected). Optional stability tweak.",
    )

    # DAPO (via TRL): loss aggregation + clipping hyperparams from the paper.
    parser.add_argument(
        "--loss_type",
        type=str,
        default="dapo",
        choices=["dapo", "grpo", "dr_grpo", "bnpo", "cispo", "sapo", "luspo", "vespo"],
        help="TRL GRPO loss aggregation; 'dapo' matches the DAPO paper's length-unbiased normalization.",
    )
    parser.add_argument(
        "--epsilon_high",
        type=float,
        default=0.28,
        help="Upper PPO-style clip bound (DAPO paper recommends 0.28). Lower bound uses --epsilon.",
    )
    parser.add_argument(
        "--epsilon",
        type=float,
        default=0.2,
        help="Lower clip epsilon (symmetric with upper when epsilon_high is set).",
    )
    parser.add_argument(
        "--mask_truncated_completions",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exclude truncated completions from the loss (recommended in DAPO).",
    )

    parser.add_argument(
        "--no_lora",
        action="store_true",
        help="Full fine-tune (no PEFT). Uses more VRAM; RL loads a separate reference model when beta>0.",
    )
    parser.add_argument("--lora_r", type=int, default=64)
    parser.add_argument("--lora_alpha", type=int, default=128)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument(
        "--lora_target_modules",
        type=str,
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
        help="Comma-separated module names for LoRA (Qwen/Llama-style MLP + attention).",
    )

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sft_only", action="store_true", help="Run only Stage-1 SFT and skip RL.")
    parser.add_argument("--rl_only", action="store_true", help="Skip SFT and run RL only from --sft_model_dir.")
    parser.add_argument(
        "--sft_model_dir",
        type=str,
        default="",
        help="Existing SFT model/adapter directory to start RL from (used with --rl_only).",
    )
    parser.add_argument("--bf16", action="store_true", help="Enable bf16")
    parser.add_argument("--fp16", action="store_true", help="Enable fp16")

    return parser.parse_args()


def _lora_target_modules(s: str) -> List[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def build_lora_config(args: argparse.Namespace) -> Any:
    LoraConfig, _, TaskType = _require_peft()
    return LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=_lora_target_modules(args.lora_target_modules),
        bias="none",
    )


def extract_gsm8k_answer(answer_text: str) -> str:
    """Extract GSM8K final answer from canonical '#### <ans>' pattern."""
    match = re.search(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)", answer_text)
    if not match:
        return ""
    return match.group(1).replace(",", "").strip()


def format_sft_completion(answer_text: str) -> str:
    """Align SFT target with eval style by ensuring a final boxed answer."""
    ans = extract_gsm8k_answer(answer_text)
    cleaned = answer_text.strip()
    if ans:
        # Keep original rationale, but add an explicit boxed final answer.
        if "\\boxed{" not in cleaned:
            cleaned = f"{cleaned}\n\nFinal answer: \\boxed{{{ans}}}"
    return cleaned


def extract_answer_from_response(response: str) -> str:
    boxed = re.search(r"\\boxed\{\s*([^{}]+)\s*\}", response)
    if boxed:
        return boxed.group(1).replace(",", "").strip()

    ans = re.search(
        r"(?:answer|result|final answer)\s*(?:is|=|:)?\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)",
        response,
        re.IGNORECASE,
    )
    if ans:
        return ans.group(1).replace(",", "").strip()

    nums = re.findall(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?", response)
    if nums:
        return nums[-1].replace(",", "").strip()

    return ""


def math_reward_fn(completions: List[str], answer: List[str], **kwargs) -> List[float]:
    rewards: List[float] = []
    for completion, ans_text in zip(completions, answer):
        gt = extract_gsm8k_answer(ans_text)
        pred = extract_answer_from_response(completion)

        try:
            correct = abs(float(pred) - float(gt)) < 1e-5
        except Exception:
            correct = pred.strip() == gt.strip()

        rewards.append(1.0 if correct else 0.0)
    return rewards


def load_gsm8k_train(path: str, name: str) -> Dataset:
    train_files = sorted(glob.glob(os.path.join(path, name, "train-*.parquet")))
    if train_files:
        data_files: Dict[str, Any] = {"train": train_files}
        test_files = sorted(glob.glob(os.path.join(path, name, "test-*.parquet")))
        if test_files:
            data_files["test"] = test_files
        ds = load_dataset("parquet", data_files=data_files)
    else:
        ds = load_dataset(path, name)
    if "train" not in ds:
        raise ValueError("Expected a train split in GSM8K dataset.")
    return ds["train"]


def build_sft_dataset(train_ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def to_prompt_completion(example: Dict[str, Any]) -> Dict[str, str]:
        question = example["question"]
        answer = example["answer"]

        user_prompt = (
            "Solve the following math problem step by step. "
            "Put your final answer in \\boxed{} format.\n\n"
            f"Problem: {question}"
        )
        completion = format_sft_completion(answer)

        if hasattr(tokenizer, "apply_chat_template"):
            prompt = tokenizer.apply_chat_template(
                [{"role": "user", "content": user_prompt}],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"User: {user_prompt}\nAssistant: "

        return {"prompt": prompt, "completion": completion}

    return train_ds.map(to_prompt_completion, remove_columns=train_ds.column_names)


def build_rl_dataset(train_ds: Dataset, tokenizer: AutoTokenizer) -> Dataset:
    def to_prompt(example: Dict[str, Any]) -> Dict[str, str]:
        question = example["question"]
        answer = example["answer"]

        user_prompt = (
            "Solve the following math problem step by step. "
            "Put your final answer in \\boxed{} format.\n\n"
            f"Problem: {question}\n\nSolution:"
        )

        if hasattr(tokenizer, "apply_chat_template"):
            messages = [{"role": "user", "content": user_prompt}]
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = user_prompt

        return {"prompt": prompt, "answer": answer}

    return train_ds.map(to_prompt, remove_columns=train_ds.column_names)


def _model_dtype(args: argparse.Namespace) -> torch.dtype:
    if args.bf16:
        return torch.bfloat16
    if args.fp16:
        return torch.float16
    return torch.float32


def train_sft(args: argparse.Namespace, tokenizer: AutoTokenizer, train_ds: Dataset) -> str:
    os.makedirs(args.sft_output, exist_ok=True)

    dtype = _model_dtype(args)
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=dtype,
        trust_remote_code=True,
    )

    peft_cfg = None if args.no_lora else build_lora_config(args)

    sft_dataset = build_sft_dataset(train_ds, tokenizer)

    sft_cfg = SFTConfig(
        output_dir=args.sft_output,
        num_train_epochs=1,
        learning_rate=args.sft_lr,
        per_device_train_batch_size=args.sft_batch_size,
        gradient_accumulation_steps=args.sft_grad_accum,
        max_length=args.max_seq_len,
        logging_steps=10,
        save_strategy="epoch",
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        optim="adamw_torch",
        seed=args.seed,
        report_to="none",
        completion_only_loss=True,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_cfg,
        train_dataset=sft_dataset,
        processing_class=tokenizer,
        peft_config=peft_cfg,
    )
    trainer.train()
    trainer.save_model(args.sft_output)
    tokenizer.save_pretrained(args.sft_output)

    meta = {
        "base_model": args.base_model,
        "use_lora": not args.no_lora,
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "lora_target_modules": _lora_target_modules(args.lora_target_modules),
    }
    with open(os.path.join(args.sft_output, "hw_training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    return args.sft_output


def train_rl(args: argparse.Namespace, tokenizer: AutoTokenizer, train_ds: Dataset, sft_model_dir: str) -> str:
    os.makedirs(args.rl_output, exist_ok=True)

    rl_dataset = build_rl_dataset(train_ds, tokenizer)

    dtype = _model_dtype(args)

    dtype_name = "bfloat16" if args.bf16 else "float16" if args.fp16 else "float32"
    rl_cfg = GRPOConfig(
        output_dir=args.rl_output,
        learning_rate=args.rl_lr,
        per_device_train_batch_size=args.rl_batch_size,
        gradient_accumulation_steps=args.rl_grad_accum,
        num_generations=args.num_generations,
        max_completion_length=args.max_completion_len,
        max_steps=args.rl_max_steps,
        beta=args.beta,
        epsilon=args.epsilon,
        epsilon_high=args.epsilon_high,
        loss_type=args.loss_type,
        mask_truncated_completions=args.mask_truncated_completions,
        use_bias_correction_kl=args.use_bias_correction_kl,
        logging_steps=1,
        save_steps=max(args.rl_max_steps // 2, 50),
        save_total_limit=2,
        bf16=args.bf16,
        fp16=args.fp16,
        gradient_checkpointing=True,
        seed=args.seed,
        report_to="none",
        model_init_kwargs={"dtype": dtype_name, "trust_remote_code": True} if args.no_lora else None,
    )

    if args.no_lora:
        trainer = GRPOTrainer(
            model=sft_model_dir,
            args=rl_cfg,
            train_dataset=rl_dataset,
            reward_funcs=math_reward_fn,
            processing_class=tokenizer,
        )
    else:
        _, PeftModel, _ = _require_peft()
        base = AutoModelForCausalLM.from_pretrained(
            args.base_model,
            torch_dtype=dtype,
            trust_remote_code=True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base, sft_model_dir, is_trainable=True)
        trainer = GRPOTrainer(
            model=model,
            args=rl_cfg,
            train_dataset=rl_dataset,
            reward_funcs=math_reward_fn,
            processing_class=tokenizer,
        )

    trainer.train()
    trainer.save_model(args.rl_output)
    tokenizer.save_pretrained(args.rl_output)

    with open(os.path.join(args.rl_output, "hw_training_meta.json"), "w", encoding="utf-8") as f:
        json.dump(
            {
                "base_model": args.base_model,
                "sft_adapter_dir": sft_model_dir,
                "use_lora": not args.no_lora,
                "dapo_impl": "TRL.GRPOTrainer+loss_type=dapo",
                "trl_version": getattr(trl, "__version__", "unknown"),
                "loss_type": args.loss_type,
                "beta_kl": args.beta,
                "use_bias_correction_kl": args.use_bias_correction_kl,
                "epsilon": args.epsilon,
                "epsilon_high": args.epsilon_high,
                "mask_truncated_completions": args.mask_truncated_completions,
            },
            f,
            indent=2,
        )

    return args.rl_output


def run() -> StageOutputs:
    args = parse_args()
    torch.manual_seed(args.seed)
    if args.sft_only and args.rl_only:
        raise ValueError("--sft_only and --rl_only are mutually exclusive.")
    if args.rl_only and not args.sft_model_dir:
        raise ValueError("--rl_only requires --sft_model_dir <existing_sft_dir>.")

    print(f"[hw] trl version: {getattr(trl, '__version__', 'unknown')}")
    if args.loss_type != "dapo":
        print(f"[hw] note: loss_type={args.loss_type} (DAPO paper uses 'dapo')")
    if args.beta == 0.0:
        print("[hw] warning: beta=0 — KL term disabled; reference model not used (higher collapse risk).")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    train_ds = load_gsm8k_train(args.gsm8k_path, args.gsm8k_name)

    sft_dir = args.sft_model_dir if args.rl_only else ""
    if args.rl_only:
        print(f"[hw] RL-only mode: reuse SFT dir {sft_dir}")
    else:
        print("=" * 80)
        print("Stage 1/2: SFT on GSM8K train for 1 epoch ({})".format("full weights" if args.no_lora else "LoRA"))
        print("=" * 80)
        sft_dir = train_sft(args, tokenizer, train_ds)

    rl_dir = ""
    if args.sft_only:
        print("[hw] SFT-only mode enabled: skip Stage 2/2 RL.")
    else:
        print("=" * 80)
        print(
            "Stage 2/2: DAPO (TRL GRPOTrainer, loss_type={}) | KL: beta={} bias_corrected_kl={} | "
            "clip eps_low={} eps_high={} | mask_truncated={}".format(
                args.loss_type,
                args.beta,
                args.use_bias_correction_kl,
                args.epsilon,
                args.epsilon_high,
                args.mask_truncated_completions,
            )
        )
        print("=" * 80)
        rl_dir = train_rl(args, tokenizer, train_ds, sft_dir)

    print("\nDone.")
    print(f"SFT model used/saved at: {sft_dir}")
    if rl_dir:
        print(f"RL model saved to:  {rl_dir}")
    if not args.no_lora and rl_dir:
        print("Inference: load base_model + PeftModel.from_pretrained(base, rl_dir) (or sft_dir for SFT-only).")

    return StageOutputs(sft_model_dir=sft_dir, rl_model_dir=rl_dir)


if __name__ == "__main__":
    run()
