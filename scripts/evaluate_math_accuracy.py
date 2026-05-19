import argparse
import gc
import json
import os
from collections import defaultdict
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a GSM8K SFT adapter on GSM8K and MATH-500.")
    parser.add_argument("--base_model_path", default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--adapter_path", default="outputs/qwen3-1.7b-gsm8k-sft-qlora/final")
    parser.add_argument("--model_path", default=None, help="Optional full model path. If set, adapter_path is ignored.")
    parser.add_argument("--output_dir", default="outputs/qwen3-1.7b-gsm8k-sft-qlora/eval")
    parser.add_argument("--cache_dir", default=None)
    parser.add_argument("--gsm8k_path", default=None)
    parser.add_argument("--math500_path", default=None)
    parser.add_argument("--local_files_only", action="store_true", help="Load model/tokenizer from local cache only.")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--limit", type=int, default=None, help="Optional per-dataset limit for smoke tests.")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--load_in_4bit", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--no_math_verify", action="store_true", help="Disable optional math_verify use.")
    parser.add_argument(
        "--rescore_existing",
        action="store_true",
        help="Only rescore existing prediction JSONL files in output_dir; do not load model or generate.",
    )
    return parser.parse_args()


def load_model_and_tokenizer(args):
    import torch
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer_path = args.model_path or args.base_model_path
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        trust_remote_code=True,
        local_files_only=args.local_files_only,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model_load_path = args.model_path or args.base_model_path
    quantization_config = None
    model_kwargs = {
        "device_map": {"": 0} if torch.cuda.is_available() else None,
        "trust_remote_code": True,
        "local_files_only": args.local_files_only,
    }
    if args.load_in_4bit and torch.cuda.is_available():
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        model_kwargs["quantization_config"] = quantization_config
    else:
        model_kwargs["torch_dtype"] = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    model = AutoModelForCausalLM.from_pretrained(model_load_path, **model_kwargs)
    if args.model_path is None and args.adapter_path:
        model = PeftModel.from_pretrained(model, args.adapter_path, is_trainable=False)
    model.eval()
    model.config.use_cache = True
    return model, tokenizer


def generate_batch(model, tokenizer, prompts, args):
    import torch

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=False).to(model.device)
    generation_kwargs = {
        "max_new_tokens": args.max_new_tokens,
        "do_sample": args.do_sample,
        "pad_token_id": tokenizer.pad_token_id,
        "eos_token_id": tokenizer.eos_token_id,
    }
    if args.do_sample:
        if args.temperature is not None:
            generation_kwargs["temperature"] = args.temperature
        generation_kwargs["top_p"] = args.top_p
    with torch.no_grad():
        outputs = model.generate(**inputs, **generation_kwargs)
    prompt_width = inputs["input_ids"].shape[1]
    responses = []
    for row in outputs:
        responses.append(tokenizer.decode(row[prompt_width:], skip_special_tokens=True))
    return responses


def evaluate_dataset(samples, make_prompt, get_answer, output_path, model, tokenizer, args, meta_fn=None):
    from mathhelper.eval_utils import answers_match, extract_answer_from_response

    total = 0
    correct = 0
    breakdown = defaultdict(lambda: {"correct": 0, "total": 0})
    output_path.parent.mkdir(parents=True, exist_ok=True)
    prefer_math_verify = not args.no_math_verify

    with output_path.open("w", encoding="utf-8") as f:
        for start in range(0, len(samples), args.batch_size):
            batch = samples.select(range(start, min(start + args.batch_size, len(samples))))
            prompts = [make_prompt(sample) for sample in batch]
            responses = generate_batch(model, tokenizer, prompts, args)
            for sample, prompt, response in zip(batch, prompts, responses):
                prediction = extract_answer_from_response(response)
                ground_truth = get_answer(sample)
                is_correct = answers_match(prediction, ground_truth, prefer_math_verify=prefer_math_verify)
                total += 1
                correct += int(is_correct)

                item = {
                    "id": sample.get("unique_id", total - 1),
                    "prompt": prompt,
                    "ground_truth": ground_truth,
                    "prediction": prediction,
                    "correct": is_correct,
                    "response": response,
                }
                if meta_fn:
                    meta = meta_fn(sample)
                    item.update(meta)
                    for key, value in meta.items():
                        if value is not None:
                            tag = f"{key}:{value}"
                            breakdown[tag]["total"] += 1
                            breakdown[tag]["correct"] += int(is_correct)
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

            print(f"Evaluated {total}/{len(samples)} for {output_path.name}")

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "breakdown": {
            key: {
                "correct": value["correct"],
                "total": value["total"],
                "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
            }
            for key, value in sorted(breakdown.items())
        },
    }


def rescore_prediction_file(path, args):
    from mathhelper.eval_utils import answers_match

    total = 0
    correct = 0
    breakdown = defaultdict(lambda: {"correct": 0, "total": 0})
    prefer_math_verify = not args.no_math_verify
    with path.open(encoding="utf-8") as f:
        for line in f:
            item = json.loads(line)
            is_correct = answers_match(
                item.get("prediction", ""),
                item.get("ground_truth", ""),
                prefer_math_verify=prefer_math_verify,
            )
            total += 1
            correct += int(is_correct)
            for key in ("subject", "level"):
                value = item.get(key)
                if value is not None:
                    tag = f"{key}:{value}"
                    breakdown[tag]["total"] += 1
                    breakdown[tag]["correct"] += int(is_correct)

    return {
        "correct": correct,
        "total": total,
        "accuracy": correct / total if total else 0.0,
        "breakdown": {
            key: {
                "correct": value["correct"],
                "total": value["total"],
                "accuracy": value["correct"] / value["total"] if value["total"] else 0.0,
            }
            for key, value in sorted(breakdown.items())
        },
    }


def main() -> None:
    args = parse_args()
    if args.cache_dir:
        os.environ.setdefault("HF_DATASETS_CACHE", args.cache_dir)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    from mathhelper.eval_utils import (
        dump_json,
        ensure_columns,
        extract_gsm8k_answer,
        format_gsm8k_prompt,
        format_math500_prompt,
        load_gsm8k,
        load_math500,
    )

    torch.manual_seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    math_verify_available = False
    if not args.no_math_verify:
        try:
            import math_verify  # noqa: F401

            math_verify_available = True
        except ImportError:
            print(
                "Warning: math_verify is not installed. "
                "Falling back to exact/numeric/sympy heuristics; MATH-500 accuracy may be conservative."
            )

    if args.rescore_existing:
        gsm8k_path = output_dir / "gsm8k_predictions.jsonl"
        math500_path = output_dir / "math500_predictions.jsonl"
        if not gsm8k_path.exists() or not math500_path.exists():
            raise FileNotFoundError(
                f"Expected existing prediction files at {gsm8k_path} and {math500_path}"
            )
        summary = {
            "gsm8k": rescore_prediction_file(gsm8k_path, args),
            "math500": rescore_prediction_file(math500_path, args),
            "config": {**vars(args), "math_verify_available": math_verify_available},
        }
        dump_json(output_dir / "summary.json", summary)
        print(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"Rescoring complete. Summary saved to {output_dir / 'summary.json'}")
        return

    gsm8k = load_gsm8k(args.gsm8k_path, args.cache_dir)
    math500 = load_math500(args.math500_path, args.cache_dir)
    ensure_columns(gsm8k["test"], {"question", "answer"}, "GSM8K test")
    ensure_columns(math500, {"problem", "answer"}, "MATH-500")

    gsm8k_test = gsm8k["test"]
    if args.limit:
        gsm8k_test = gsm8k_test.select(range(min(args.limit, len(gsm8k_test))))
        math500 = math500.select(range(min(args.limit, len(math500))))

    model, tokenizer = load_model_and_tokenizer(args)

    gsm8k_metrics = evaluate_dataset(
        gsm8k_test,
        lambda sample: format_gsm8k_prompt(sample["question"]),
        lambda sample: extract_gsm8k_answer(sample["answer"]),
        output_dir / "gsm8k_predictions.jsonl",
        model,
        tokenizer,
        args,
    )

    math500_metrics = evaluate_dataset(
        math500,
        lambda sample: format_math500_prompt(sample["problem"]),
        lambda sample: sample["answer"],
        output_dir / "math500_predictions.jsonl",
        model,
        tokenizer,
        args,
        meta_fn=lambda sample: {"subject": sample.get("subject"), "level": sample.get("level")},
    )

    summary = {
        "gsm8k": gsm8k_metrics,
        "math500": math500_metrics,
        "config": {**vars(args), "math_verify_available": math_verify_available},
    }
    dump_json(output_dir / "summary.json", summary)

    del model
    del tokenizer
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Evaluation complete. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
