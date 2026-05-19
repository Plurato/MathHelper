import argparse
import os
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="QLoRA SFT on GSM8K for Qwen3-1.7B-Base.")
    parser.add_argument("--model_path", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output_dir", default="outputs/qwen3-1.7b-gsm8k-sft-qlora")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--gsm8k_path", default=None, help="Optional local .json/.jsonl/.parquet GSM8K file.")
    parser.add_argument("--local_files_only", action="store_true", help="Load model/tokenizer from local cache only.")
    parser.add_argument("--num_train_epochs", type=float, default=3.0)
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=2e-4)
    parser.add_argument("--warmup_ratio", type=float, default=0.1)
    parser.add_argument("--max_length", type=int, default=2048)
    parser.add_argument("--eval_ratio", type=float, default=0.03)
    parser.add_argument("--logging_steps", type=int, default=10)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--save_steps", type=int, default=200)
    parser.add_argument("--save_total_limit", type=int, default=2)
    parser.add_argument("--lora_r", type=int, default=32)
    parser.add_argument("--lora_alpha", type=int, default=64)
    parser.add_argument("--lora_dropout", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max_train_samples", type=int, default=None)
    parser.add_argument("--max_eval_samples", type=int, default=None)
    parser.add_argument("--gradient_checkpointing", action=argparse.BooleanOptionalAction, default=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.cache_dir:
        os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import torch
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from trl import SFTConfig, SFTTrainer

    from mathhelper.eval_utils import dump_json, ensure_columns, load_gsm8k, make_gsm8k_sft_text

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        quantization_config=bnb_config,
        device_map={"": 0},
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    model.config.use_cache = False
    model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=args.gradient_checkpointing)

    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=args.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    gsm8k = load_gsm8k(args.gsm8k_path, args.cache_dir)
    ensure_columns(gsm8k["train"], {"question", "answer"}, "GSM8K train")

    train_raw = gsm8k["train"]
    if args.max_train_samples:
        train_raw = train_raw.shuffle(seed=args.seed).select(range(min(args.max_train_samples, len(train_raw))))

    split = train_raw.train_test_split(test_size=args.eval_ratio, seed=args.seed)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    if args.max_eval_samples:
        eval_dataset = eval_dataset.select(range(min(args.max_eval_samples, len(eval_dataset))))

    eos = tokenizer.eos_token or ""

    def format_example(example):
        return {"text": make_gsm8k_sft_text(example["question"], example["answer"], eos)}

    train_dataset = train_dataset.map(format_example, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(format_example, remove_columns=eval_dataset.column_names)

    sft_config = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        warmup_ratio=args.warmup_ratio,
        bf16=True,
        gradient_checkpointing=args.gradient_checkpointing,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        max_length=args.max_length,
        packing=False,
        dataset_text_field="text",
        logging_steps=args.logging_steps,
        eval_strategy="steps",
        eval_steps=args.eval_steps,
        save_strategy="steps",
        save_steps=args.save_steps,
        save_total_limit=args.save_total_limit,
        report_to="none",
        seed=args.seed,
    )

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    train_result = trainer.train()

    final_dir = output_dir / "final"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    metrics = dict(train_result.metrics)
    metrics["train_loss"] = train_result.training_loss
    trainer.log_metrics("train", metrics)
    trainer.save_metrics("train", metrics)
    trainer.save_state()

    log_history = trainer.state.log_history
    train_losses = [(x["step"], x["loss"]) for x in log_history if "loss" in x]
    eval_losses = [(x["step"], x["eval_loss"]) for x in log_history if "eval_loss" in x]
    if train_losses:
        fig, ax = plt.subplots(1, 1, figsize=(10, 5))
        ax.plot([x for x, _ in train_losses], [y for _, y in train_losses], label="Train Loss", alpha=0.8)
        if eval_losses:
            ax.plot([x for x, _ in eval_losses], [y for _, y in eval_losses], label="Eval Loss", marker="o")
        ax.set_xlabel("Step")
        ax.set_ylabel("Loss")
        ax.set_title("GSM8K SFT Training Loss")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(output_dir / "training_loss.png", dpi=150)
        plt.close(fig)

    dump_json(
        output_dir / "run_config.json",
        {
            "model_path": args.model_path,
            "output_dir": str(output_dir),
            "train_samples": len(train_dataset),
            "eval_samples": len(eval_dataset),
            "args": vars(args),
        },
    )
    print(f"Training complete. Final adapter saved to {final_dir}")


if __name__ == "__main__":
    main()
