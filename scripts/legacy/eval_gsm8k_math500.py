import argparse
import glob
import json
import os
import re
import sys
import shutil
import tempfile
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


@dataclass
class EvalItem:
    dataset: str
    idx: int
    question: str
    prediction_raw: str
    prediction_extracted: str
    ground_truth_raw: str
    ground_truth_extracted: str
    correct: bool


@dataclass
class EvalSummary:
    dataset: str
    total: int
    correct: int
    accuracy: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate model on GSM8K(test) and MATH500")

    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument(
        "--base_model_path",
        type=str,
        default=None,
        help="Optional base model path for LoRA adapter evaluation.",
    )
    parser.add_argument(
        "--lora_adapter_path",
        type=str,
        default=None,
        help="Optional LoRA adapter path. When set, model_path is ignored for weight loading.",
    )
    parser.add_argument(
        "--tokenizer_path",
        type=str,
        default=None,
        help="Optional: load tokenizer from this path (e.g. base model) instead of model_path.",
    )

    parser.add_argument("--gsm8k_path", type=str, default="openai/gsm8k")
    parser.add_argument("--gsm8k_name", type=str, default="main")

    parser.add_argument("--math500_path", type=str, default="HuggingFaceH4/MATH-500")
    parser.add_argument("--math500_name", type=str, default=None)
    parser.add_argument("--math500_split", type=str, default="test")

    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)

    parser.add_argument("--limit_gsm8k", type=int, default=0, help="0 means full test split")
    parser.add_argument("--limit_math500", type=int, default=0, help="0 means full split")

    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=4,
        help="Number of problems per model.generate call (GPU batch). Lower if OOM; 1 disables batching.",
    )

    parser.add_argument("--output_json", type=str, default="./outputs/eval_gsm8k_math500.json")
    parser.add_argument("--seed", type=int, default=42)

    return parser.parse_args()


def load_gsm8k_dataset(path: str, name: str):
    """Avoid broken local dataset_infos.json (Value without dtype) by loading parquet shards."""
    train_files = sorted(glob.glob(os.path.join(path, name, "train-*.parquet")))
    test_files = sorted(glob.glob(os.path.join(path, name, "test-*.parquet")))
    if train_files or test_files:
        data_files: Dict[str, Any] = {}
        if train_files:
            data_files["train"] = train_files
        if test_files:
            data_files["test"] = test_files
        return load_dataset("parquet", data_files=data_files)
    return load_dataset(path, name)


def extract_gsm8k_answer(answer_text: str) -> str:
    match = re.search(r"####\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)", answer_text)
    if not match:
        return ""
    return match.group(1).replace(",", "").strip()


def extract_answer_from_text(text: str) -> str:
    boxed = re.search(r"\\boxed\{\s*([^{}]+)\s*\}", text)
    if boxed:
        return boxed.group(1).replace(",", "").strip()

    ans = re.search(
        r"(?:answer|result|final answer)\s*(?:is|=|:)?\s*([-+]?[0-9][0-9,]*(?:\.[0-9]+)?)",
        text,
        re.IGNORECASE,
    )
    if ans:
        return ans.group(1).replace(",", "").strip()

    nums = re.findall(r"[-+]?[0-9][0-9,]*(?:\.[0-9]+)?", text)
    if nums:
        return nums[-1].replace(",", "").strip()

    # fallback: compact final line
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    if lines:
        return lines[-1]

    return ""


def norm_text(s: str) -> str:
    s = s.strip().lower()
    s = s.replace("$", "")
    s = re.sub(r"\\left|\\right", "", s)
    s = re.sub(r"\s+", "", s)
    return s


def is_correct(pred: str, gt: str) -> bool:
    try:
        return abs(float(pred) - float(gt)) < 1e-5
    except Exception:
        return norm_text(pred) == norm_text(gt)


def load_tokenizer_for_eval(model_path: str, tokenizer_path: Optional[str]) -> AutoTokenizer:
    """Load tokenizer; fix TRL-saved configs where extra_special_tokens is a list (breaks fast tokenizer)."""
    if tokenizer_path:
        return AutoTokenizer.from_pretrained(tokenizer_path, trust_remote_code=True)

    cfg_path = os.path.join(model_path, "tokenizer_config.json")
    if os.path.isfile(cfg_path):
        with open(cfg_path, encoding="utf-8") as f:
            cfg = json.load(f)
        extra = cfg.get("extra_special_tokens")
        if isinstance(extra, list):
            add = cfg.get("additional_special_tokens")
            if not isinstance(add, list):
                add = []
            merged: List[str] = []
            seen: set[str] = set()
            for t in add + extra:
                if t not in seen:
                    merged.append(t)
                    seen.add(t)
            cfg["additional_special_tokens"] = merged
            del cfg["extra_special_tokens"]

            tmp = tempfile.mkdtemp(prefix="eval_tok_")
            try:
                with open(os.path.join(tmp, "tokenizer_config.json"), "w", encoding="utf-8") as wf:
                    json.dump(cfg, wf, ensure_ascii=False, indent=2)
                for name in ("tokenizer.json", "chat_template.jinja"):
                    src = os.path.join(model_path, name)
                    if os.path.isfile(src):
                        shutil.copy2(src, os.path.join(tmp, name))
                return AutoTokenizer.from_pretrained(tmp, trust_remote_code=True)
            finally:
                shutil.rmtree(tmp, ignore_errors=True)

    return AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)


def build_prompt(question: str) -> str:
    return (
        "Solve the following math problem step by step. "
        "Put your final answer in \\boxed{} format.\n\n"
        f"Problem: {question}\n\nSolution:"
    )


def format_with_chat_template(tokenizer: AutoTokenizer, prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        messages = [{"role": "user", "content": prompt}]
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
    return prompt


def _model_device(model: AutoModelForCausalLM) -> torch.device:
    dev = getattr(model, "device", None)
    if dev is not None:
        return dev
    return next(model.parameters()).device


def generate_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_texts: List[str],
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> List[str]:
    """Batched greedy/sampled generation (single GPU forward per batch)."""
    if not prompt_texts:
        return []

    device = _model_device(model)
    # Causal LM batched generation requires left padding so positions align.
    pad_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        enc = tokenizer(
            prompt_texts,
            return_tensors="pt",
            padding=True,
            truncation=False,
        )
    finally:
        tokenizer.padding_side = pad_side

    enc = {k: v.to(device) for k, v in enc.items()}
    in_len = enc["input_ids"].shape[1]

    do_sample = temperature > 0
    gen_kwargs: Dict[str, Any] = {
        "max_new_tokens": max_new_tokens,
        "do_sample": do_sample,
        "eos_token_id": tokenizer.eos_token_id,
        "pad_token_id": tokenizer.pad_token_id,
    }
    if do_sample:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = top_p

    with torch.no_grad():
        output_ids = model.generate(**enc, **gen_kwargs)

    out: List[str] = []
    for i in range(len(prompt_texts)):
        gen_ids = output_ids[i, in_len:]
        out.append(tokenizer.decode(gen_ids, skip_special_tokens=True))
    return out


def generate_one(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompt_text: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
) -> str:
    return generate_batch(model, tokenizer, [prompt_text], max_new_tokens, temperature, top_p)[0]


def pick_math500_fields(example: Dict[str, Any]) -> Tuple[str, str]:
    """Support Hub-style columns and Verl/RLHF-style MATH-500 (chat prompt + reward_model)."""
    question = ""
    prompt_raw = example.get("prompt")
    if isinstance(prompt_raw, list):
        chunks: List[str] = []
        for msg in prompt_raw:
            if isinstance(msg, dict) and msg.get("content") is not None:
                chunks.append(str(msg["content"]).strip())
        question = "\n".join(chunks).strip()
    elif prompt_raw is not None:
        question = str(prompt_raw).strip()

    if not question:
        for k in ("problem", "question", "input"):
            if k in example and example[k] is not None:
                question = str(example[k]).strip()
                break

    gt_raw = ""
    rm = example.get("reward_model")
    if isinstance(rm, dict) and rm.get("ground_truth") is not None:
        gt_raw = str(rm["ground_truth"])
    if not gt_raw:
        for k in ("answer", "final_answer", "solution", "output", "target", "ground_truth"):
            if k in example and example[k] is not None:
                gt_raw = str(example[k])
                break

    if not question or not gt_raw:
        raise ValueError(
            f"Cannot infer MATH500 fields from keys={list(example.keys())}. "
            "Please edit pick_math500_fields() mapping."
        )

    return question, gt_raw


def maybe_limit(ds: Dataset, n: int) -> Dataset:
    if n and n > 0:
        return ds.select(range(min(n, len(ds))))
    return ds


def eval_gsm8k(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> Tuple[EvalSummary, List[EvalItem]]:
    ds = load_gsm8k_dataset(args.gsm8k_path, args.gsm8k_name)["test"]
    ds = maybe_limit(ds, args.limit_gsm8k)

    details: List[EvalItem] = []
    correct = 0

    bs = max(1, int(args.eval_batch_size))
    pbar = tqdm(
        total=len(ds),
        desc="gsm8k test",
        ascii=True,
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=5.0,
    )
    for start in range(0, len(ds), bs):
        end = min(start + bs, len(ds))
        batch = ds.select(range(start, end))
        prompts: List[str] = []
        metas: List[Tuple[int, str, str, str]] = []
        for j in range(len(batch)):
            ex = batch[j]
            i = start + j
            question = str(ex["question"])
            gt_raw = str(ex["answer"])
            gt = extract_gsm8k_answer(gt_raw)
            prompt = build_prompt(question)
            full_prompt = format_with_chat_template(tokenizer, prompt)
            prompts.append(full_prompt)
            metas.append((i, question, gt_raw, gt))

        pred_raws = generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for (i, question, gt_raw, gt), pred_raw in zip(metas, pred_raws):
            pred = extract_answer_from_text(pred_raw)
            ok = is_correct(pred, gt)
            correct += int(ok)
            details.append(
                EvalItem(
                    dataset="gsm8k_test",
                    idx=i,
                    question=question,
                    prediction_raw=pred_raw,
                    prediction_extracted=pred,
                    ground_truth_raw=gt_raw,
                    ground_truth_extracted=gt,
                    correct=ok,
                )
            )

        pbar.update(len(metas))
        done = start + len(metas)
        pbar.set_postfix(acc=f"{correct / done:.4f}")

    summary = EvalSummary(
        dataset="gsm8k_test",
        total=len(ds),
        correct=correct,
        accuracy=(correct / len(ds)) if len(ds) else 0.0,
    )
    return summary, details


def eval_math500(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    args: argparse.Namespace,
) -> Tuple[EvalSummary, List[EvalItem]]:
    if args.math500_name is None:
        ds_all = load_dataset(args.math500_path)
    else:
        ds_all = load_dataset(args.math500_path, args.math500_name)

    if args.math500_split not in ds_all:
        raise ValueError(f"Split '{args.math500_split}' not found. Available={list(ds_all.keys())}")

    ds = maybe_limit(ds_all[args.math500_split], args.limit_math500)

    details: List[EvalItem] = []
    correct = 0

    bs = max(1, int(args.eval_batch_size))
    pbar = tqdm(
        total=len(ds),
        desc="math500",
        ascii=True,
        file=sys.stdout,
        dynamic_ncols=True,
        mininterval=5.0,
    )
    for start in range(0, len(ds), bs):
        end = min(start + bs, len(ds))
        batch = ds.select(range(start, end))
        prompts: List[str] = []
        metas: List[Tuple[int, str, str, str]] = []
        for j in range(len(batch)):
            ex = batch[j]
            i = start + j
            question, gt_raw = pick_math500_fields(ex)
            gt = extract_answer_from_text(gt_raw)
            if not gt:
                gt = extract_gsm8k_answer(gt_raw)
            prompt = build_prompt(question)
            full_prompt = format_with_chat_template(tokenizer, prompt)
            prompts.append(full_prompt)
            metas.append((i, question, gt_raw, gt))

        pred_raws = generate_batch(
            model,
            tokenizer,
            prompts,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
        )
        for (i, question, gt_raw, gt), pred_raw in zip(metas, pred_raws):
            pred = extract_answer_from_text(pred_raw)
            ok = is_correct(pred, gt)
            correct += int(ok)
            details.append(
                EvalItem(
                    dataset="math500",
                    idx=i,
                    question=question,
                    prediction_raw=pred_raw,
                    prediction_extracted=pred,
                    ground_truth_raw=gt_raw,
                    ground_truth_extracted=gt,
                    correct=ok,
                )
            )

        pbar.update(len(metas))
        done = start + len(metas)
        pbar.set_postfix(acc=f"{correct / done:.4f}")

    summary = EvalSummary(
        dataset="math500",
        total=len(ds),
        correct=correct,
        accuracy=(correct / len(ds)) if len(ds) else 0.0,
    )
    return summary, details


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    model_for_tokenizer = args.base_model_path or args.model_path
    print("[eval] loading tokenizer ...", flush=True)
    tokenizer = load_tokenizer_for_eval(model_for_tokenizer, args.tokenizer_path)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[eval] loading model weights (may take several minutes on GPFS) ...", flush=True)
    if args.lora_adapter_path:
        if not args.base_model_path:
            raise ValueError("--base_model_path is required when --lora_adapter_path is set.")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
        model = PeftModel.from_pretrained(base_model, args.lora_adapter_path)
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            trust_remote_code=True,
        )
    model.eval()
    print(
        f"[eval] model ready; eval_batch_size={args.eval_batch_size}; running GSM8K test ...",
        flush=True,
    )

    gsm8k_summary, gsm8k_details = eval_gsm8k(model, tokenizer, args)
    print("[eval] GSM8K done; running MATH-500 ...", flush=True)
    math500_summary, math500_details = eval_math500(model, tokenizer, args)

    payload = {
        "model_path": args.model_path,
        "base_model_path": args.base_model_path,
        "lora_adapter_path": args.lora_adapter_path,
        "args": vars(args),
        "summary": [asdict(gsm8k_summary), asdict(math500_summary)],
        "details": [asdict(x) for x in (gsm8k_details + math500_details)],
    }

    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    print("=" * 80)
    print(f"gsm8k_test accuracy: {gsm8k_summary.accuracy:.4f} ({gsm8k_summary.correct}/{gsm8k_summary.total})")
    print(f"math500   accuracy: {math500_summary.accuracy:.4f} ({math500_summary.correct}/{math500_summary.total})")
    print(f"Saved to: {args.output_json}")


if __name__ == "__main__":
    main()
