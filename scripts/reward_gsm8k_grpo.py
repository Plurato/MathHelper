#!/usr/bin/env python3
import re
from fractions import Fraction

_BOXED_RE = re.compile(r"\\boxed\{([^{}]+)\}")
# e.g. "final answer is 42", "the answer is: 42"
_KEYWORD_RE = re.compile(
    r"(?:final\s+answer\s+is|the\s+answer\s+is|answer\s+is)\s*[:：]?\s*([^\n\r\.;,]+)",
    flags=re.IGNORECASE,
)
# number / decimal / optional sign / simple fraction
# supports both plain integers (57500) and comma style (57,500)
_NUMBER_RE = re.compile(r"[-+]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?(?:/\d+(?:\.\d+)?)?")


def _normalize_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _try_parse_number(s: str):
    if not isinstance(s, str):
        return None
    c = s.strip().replace(",", "").replace("$", "")
    c = c.replace("，", "").replace("￥", "")
    if not c:
        return None

    # fraction first
    if "/" in c and re.fullmatch(r"[-+]?\d+(?:\.\d+)?/\d+(?:\.\d+)?", c):
        try:
            a, b = c.split("/", 1)
            return float(Fraction(a) / Fraction(b))
        except Exception:
            return None

    try:
        return float(c)
    except Exception:
        return None


def _extract_candidate(text: str):
    if not isinstance(text, str) or not text.strip():
        return None

    # 1) \boxed{...}
    boxed = _BOXED_RE.findall(text)
    if boxed:
        return boxed[-1].strip()

    # 2) keyword pattern
    kw = _KEYWORD_RE.findall(text)
    if kw:
        cand = kw[-1].strip()
        # If keyword span contains numbers with units (e.g. "$57,500/year"),
        # prefer the last numeric token.
        nums = _NUMBER_RE.findall(cand)
        if nums:
            return nums[-1].strip()
        return cand

    # 3) last number
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1].strip()

    return None


def _default_compute_score_local(data_source, solution_str, ground_truth):
    """Minimal fallback for non-GSM8K without importing verl package."""
    if data_source in ["lighteval/MATH", "DigitalLearningGmbH/MATH-lighteval", "HuggingFaceH4/MATH-500"]:
        # preserve previous behavior for non-gsm8k in this training: binary exact string
        return 1.0 if str(solution_str).strip() == str(ground_truth).strip() else 0.0
    return 0.0


def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs):
    """DAPO-style robust binary reward for GSM8K-like math tasks.

    Priority:
      1) \boxed{...}
      2) keyword pattern (final answer is ...)
      3) last number in response

    Compare:
      - numeric: abs(pred - gt) < 1e-5
      - non-numeric: normalized string exact match
    """
    # Keep default behavior for non-GSM8K tasks.
    if data_source != "openai/gsm8k":
        return _default_compute_score_local(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
        )

    pred = _extract_candidate(solution_str)
    gt = _extract_candidate(ground_truth) if isinstance(ground_truth, str) else None
    if gt is None and isinstance(ground_truth, str):
        gt = ground_truth.strip()

    if pred is None or gt is None:
        return 0.0

    p_num = _try_parse_number(pred)
    g_num = _try_parse_number(gt)

    if p_num is not None and g_num is not None:
        return 1.0 if abs(p_num - g_num) < 1e-5 else 0.0

    return 1.0 if _normalize_text(pred) == _normalize_text(gt) else 0.0

