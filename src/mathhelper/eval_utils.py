import json
import math
import re
from pathlib import Path
from typing import Any

from datasets import Dataset, DatasetDict, load_dataset


GSM8K_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {question}\n\n"
    "Solution:"
)

MATH500_PROMPT_TEMPLATE = (
    "Solve the following math problem step by step. "
    "Put your final answer in \\boxed{{}}.\n\n"
    "Problem: {problem}\n\n"
    "Solution:"
)


def format_gsm8k_prompt(question: str) -> str:
    return GSM8K_PROMPT_TEMPLATE.format(question=question)


def format_math500_prompt(problem: str) -> str:
    return MATH500_PROMPT_TEMPLATE.format(problem=problem)


def extract_gsm8k_answer(answer_text: str) -> str:
    match = re.search(r"####\s*(.+)", answer_text)
    if match:
        return clean_answer(match.group(1))
    return clean_answer(answer_text)


def make_gsm8k_sft_text(question: str, answer_text: str, eos_token: str = "") -> str:
    reasoning = re.sub(r"\s*####\s*.+\s*$", "", answer_text, flags=re.DOTALL).strip()
    final_answer = extract_gsm8k_answer(answer_text)
    solution = reasoning
    if solution:
        solution = f"{solution}\n\n"
    solution += f"Therefore, the final answer is \\boxed{{{final_answer}}}."
    return f"{format_gsm8k_prompt(question)}\n{solution}{eos_token}"


def extract_boxed(text: str) -> str | None:
    marker = r"\boxed{"
    start = text.rfind(marker)
    if start < 0:
        return None
    i = start + len(marker)
    depth = 1
    out = []
    while i < len(text):
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return "".join(out).strip()
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return None


def extract_answer_from_response(response: str) -> str:
    boxed = extract_boxed(response)
    if boxed:
        return clean_answer(boxed)

    hash_match = re.search(r"####\s*([^\n]+)", response)
    if hash_match:
        return clean_answer(hash_match.group(1))

    answer_patterns = [
        r"(?:final answer|answer|result)\s*(?:is|=|:)\s*([^\n\.]+)",
        r"(?:答案|结果)\s*(?:是|为|=|:)\s*([^\n。]+)",
    ]
    for pattern in answer_patterns:
        matches = re.findall(pattern, response, flags=re.IGNORECASE)
        if matches:
            return clean_answer(matches[-1])

    latex_numbers = re.findall(r"\\frac\{[^{}]+\}\{[^{}]+\}|-?\d[\d,]*(?:\.\d+)?", response)
    if latex_numbers:
        return clean_answer(latex_numbers[-1])

    return ""


def clean_answer(answer: str) -> str:
    answer = str(answer).strip()
    answer = answer.replace("\n", " ")
    answer = re.sub(r"\s+", " ", answer)
    answer = answer.strip(" .,$")
    return re.sub(r"(?<=\d),(?=\d{3}\b)", "", answer)


def _strip_latex_wrappers(text: str) -> str:
    text = clean_answer(text)
    text = text.replace("$", "")
    text = text.replace("\\left", "").replace("\\right", "")
    text = text.replace("\\,", "").replace("\\!", "")
    text = text.replace("\\cdot", "*").replace("\\times", "*")
    text = text.replace("^\\circ", "").replace("\\circ", "")
    text = text.replace("\\%", "%")
    text = re.sub(r"\\text\{([^{}]*)\}", r"\1", text)
    text = re.sub(r"\\mathrm\{([^{}]*)\}", r"\1", text)
    return text.strip()


def normalize_for_exact_match(text: str) -> str:
    text = _strip_latex_wrappers(text).lower()
    text = text.replace(" ", "")
    text = text.replace("\\", "")
    return text.strip("{}")


def latex_to_sympyish(text: str) -> str:
    text = _strip_latex_wrappers(text)
    text = re.sub(r"\\frac\{([^{}]+)\}\{([^{}]+)\}", r"(\1)/(\2)", text)
    text = re.sub(r"\\sqrt\{([^{}]+)\}", r"sqrt(\1)", text)
    text = text.replace("\\pi", "pi")
    text = text.replace("{", "(").replace("}", ")")
    text = text.replace("^", "**")
    if text.endswith("%"):
        text = f"({text[:-1]})/100"
    return text


def _try_float(text: str) -> float | None:
    text = latex_to_sympyish(text)
    try:
        return float(text)
    except Exception:
        return None


def _try_sympy_equal(predicted: str, ground_truth: str) -> bool | None:
    try:
        import sympy as sp

        pred_expr = sp.sympify(latex_to_sympyish(predicted))
        gt_expr = sp.sympify(latex_to_sympyish(ground_truth))
        diff = sp.simplify(pred_expr - gt_expr)
        return bool(diff == 0)
    except Exception:
        return None


def _try_math_verify(predicted: str, ground_truth: str) -> bool | None:
    try:
        from math_verify import LatexExtractionConfig, parse, verify

        gold = parse(ground_truth, extraction_config=[LatexExtractionConfig()])
        pred = parse(predicted, extraction_config=[LatexExtractionConfig()])
        return bool(verify(gold, pred))
    except ImportError:
        return None
    except Exception:
        return None


def answers_match(predicted: str, ground_truth: str, prefer_math_verify: bool = True) -> bool:
    predicted = clean_answer(predicted)
    ground_truth = clean_answer(ground_truth)
    if not predicted or not ground_truth:
        return False

    if normalize_for_exact_match(predicted) == normalize_for_exact_match(ground_truth):
        return True

    pred_float = _try_float(predicted)
    gt_float = _try_float(ground_truth)
    if pred_float is not None and gt_float is not None:
        return math.isclose(pred_float, gt_float, rel_tol=1e-6, abs_tol=1e-6)

    sympy_equal = _try_sympy_equal(predicted, ground_truth)
    if sympy_equal is not None:
        return sympy_equal

    if prefer_math_verify:
        verified = _try_math_verify(predicted, ground_truth)
        if verified is True:
            return True

    return False


def load_table_file(path: str | Path) -> Dataset:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = [
            json.loads(line)
            for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        return Dataset.from_list(rows)
    if suffix == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            for key in ("data", "test", "train"):
                if key in data and isinstance(data[key], list):
                    data = data[key]
                    break
        return Dataset.from_list(data)
    if suffix == ".parquet":
        import pandas as pd

        return Dataset.from_pandas(pd.read_parquet(path), preserve_index=False)
    raise ValueError(f"Unsupported dataset file type for {path}. Use .json, .jsonl, or .parquet.")


def load_gsm8k(gsm8k_path: str | None = None, cache_dir: str | None = None) -> DatasetDict:
    if gsm8k_path:
        ds = load_table_file(gsm8k_path)
        if "split" in ds.column_names:
            train = ds.filter(lambda x: x["split"] == "train")
            test = ds.filter(lambda x: x["split"] == "test")
            return DatasetDict({"train": train, "test": test})
        return DatasetDict({"train": ds, "test": ds})
    return load_dataset("openai/gsm8k", "main", cache_dir=cache_dir)


def load_math500(math500_path: str | None = None, cache_dir: str | None = None) -> Dataset:
    if math500_path:
        return load_table_file(math500_path)
    return load_dataset("HuggingFaceH4/MATH-500", split="test", cache_dir=cache_dir)


def ensure_columns(dataset: Dataset, required: set[str], name: str) -> None:
    missing = required - set(dataset.column_names)
    if missing:
        raise ValueError(f"{name} is missing required columns: {sorted(missing)}")


def dump_json(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
