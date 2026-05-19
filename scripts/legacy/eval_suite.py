#!/usr/bin/env python3
import argparse
import json
import os
import re
from fractions import Fraction
from pathlib import Path

"""
Note: keep heavy ML imports lazy so `--help` is fast and
the script doesn't require GPU libraries unless evaluation runs.
"""


def _default_root() -> Path:
    """
    Prefer MATHHELPER_ROOT/RWRL_ROOT when provided; otherwise use the repository root.
    """
    script_root = Path(__file__).resolve().parents[2]
    return Path(os.environ.get("MATHHELPER_ROOT", os.environ.get("RWRL_ROOT", str(script_root))))


def build_prompt(question: str, dataset: str) -> str:
    if dataset == "gsm8k":
        return f"{question} Let's think step by step and output the final answer after \"####\"."
    return (
        "Solve the following math problem step by step.\n"
        "Put your final answer in \\boxed{} format.\n\n"
        f"Problem: {question}"
    )


def extract_gsm8k_gt(answer: str) -> str:
    marker = "####"
    if marker in answer:
        return answer.split(marker)[-1].strip()
    return answer.strip()


def normalize_text(x: str) -> str:
    x = x.strip()
    x = x.replace("$", "")
    x = x.replace(",", "")
    x = x.replace("\\left", "").replace("\\right", "")
    x = re.sub(r"\\text\{([^}]*)\}", r"\1", x)
    x = re.sub(r"\s+", " ", x).strip()
    return x


def latex_to_float(x: str):
    x = normalize_text(x)
    frac = re.fullmatch(r"-?\\frac\{(-?\d+)\}\{(-?\d+)\}", x)
    if frac:
        a, b = int(frac.group(1)), int(frac.group(2))
        if b != 0:
            return float(Fraction(a, b))
    if re.fullmatch(r"-?\d+(?:\.\d+)?", x):
        return float(x)
    return None


def extract_answer(pred: str):
    boxed = re.findall(r"\\boxed\{([^{}]+)\}", pred)
    if boxed:
        return normalize_text(boxed[-1])

    pats = [
        r"final answer is\s*[:：]?\s*([^\n\.]+)",
        r"answer is\s*[:：]?\s*([^\n\.]+)",
    ]
    for p in pats:
        m = re.findall(p, pred, flags=re.IGNORECASE)
        if m:
            return normalize_text(m[-1])

    nums = re.findall(r"-?\d+(?:\.\d+)?", pred.replace(",", ""))
    if nums:
        return normalize_text(nums[-1])

    return normalize_text(pred)


def is_correct(pred_ans: str, gt_ans: str) -> bool:
    p = normalize_text(pred_ans)
    g = normalize_text(gt_ans)

    pf = latex_to_float(p)
    gf = latex_to_float(g)
    if pf is not None and gf is not None:
        return abs(pf - gf) < 1e-5

    return p == g


def load_eval_items(root: Path, dataset_name: str):
    from datasets import load_from_disk

    if dataset_name == "gsm8k":
        ds = load_from_disk(str(root / "data/gsm8k"))["test"]
        return [{"question": x["question"], "answer": extract_gsm8k_gt(x["answer"])} for x in ds]
    if dataset_name == "math500":
        lines = (root / "data/math500/test.jsonl").read_text(encoding="utf-8").splitlines()
        data = [json.loads(x) for x in lines if x.strip()]
        return [{"question": x["problem"], "answer": x["answer"]} for x in data]
    raise ValueError(f"Unknown dataset: {dataset_name}")


def _load_model(model_path: str, base_model_path: str = ""):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    try:
        from peft import PeftModel
    except Exception:
        PeftModel = None

    model_dir = Path(model_path)
    model_is_adapter = (model_dir / "adapter_config.json").exists() and (
        model_dir / "adapter_model.safetensors"
    ).exists()

    tokenizer_path = base_model_path if (model_is_adapter and base_model_path) else model_path
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    if model_is_adapter:
        if not base_model_path:
            raise ValueError("model_path points to a LoRA adapter; please provide --base_model_path")
        if PeftModel is None:
            raise RuntimeError("peft is required to load LoRA adapter for evaluation")
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )
        model = PeftModel.from_pretrained(base_model, model_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            device_map="auto",
        )

    model.eval()
    return model, tokenizer


def evaluate_one_dataset(
    root: Path,
    model_path: str,
    dataset: str,
    output_json: Path,
    max_new_tokens: int = 512,
    batch_size: int = 16,
    limit: int = 0,
    base_model_path: str = "",
):
    import torch

    model, tokenizer = _load_model(model_path=model_path, base_model_path=base_model_path)

    items = load_eval_items(root, dataset)
    if limit > 0:
        items = items[:limit]

    total = len(items)
    correct = 0

    for i in range(0, total, batch_size):
        batch = items[i : i + batch_size]
        prompts = [build_prompt(x["question"], dataset) for x in batch]
        inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(model.device)
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )
        gen_ids = outputs[:, inputs["input_ids"].shape[1] :]
        texts = tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

        for ex, pred in zip(batch, texts):
            pa = extract_answer(pred)
            if is_correct(pa, ex["answer"]):
                correct += 1

    result = {
        "dataset": dataset,
        "model_path": model_path,
        "total": total,
        "correct": correct,
        "accuracy": (correct / total) if total > 0 else 0.0,
        "decode_config": {"do_sample": False, "max_new_tokens": max_new_tokens},
    }

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result


def main():
    p = argparse.ArgumentParser(
        description="Single-file evaluation suite for GSM8K + MATH500 (greedy decoding)."
    )
    p.add_argument("--root", default="", help="Project root (defaults to env MATHHELPER_ROOT/RWRL_ROOT)")
    p.add_argument("--model_path", required=True, help="HF model dir or LoRA adapter dir")
    p.add_argument("--base_model_path", default="", help="Required only when model_path is a LoRA adapter dir")
    p.add_argument("--tag", default="", help="Tag used in output filenames")
    p.add_argument("--out_dir", default="", help="Output dir (default: <root>/results/hw_eval)")
    p.add_argument("--datasets", nargs="+", default=["gsm8k", "math500"], choices=["gsm8k", "math500"])
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    root = Path(args.root).resolve() if args.root else _default_root().resolve()
    out_dir = Path(args.out_dir).resolve() if args.out_dir else (root / "results" / "hw_eval")

    tag = args.tag.strip()
    if not tag:
        tag = Path(args.model_path).name

    all_results = {}
    for ds in args.datasets:
        out_json = out_dir / f"eval_{tag}_{ds}.json"
        r = evaluate_one_dataset(
            root=root,
            model_path=args.model_path,
            dataset=ds,
            output_json=out_json,
            max_new_tokens=args.max_new_tokens,
            batch_size=args.batch_size,
            limit=args.limit,
            base_model_path=args.base_model_path,
        )
        all_results[ds] = r

    print(json.dumps(all_results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
