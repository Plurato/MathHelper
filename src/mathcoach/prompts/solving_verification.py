"""Prompt templates for the solving verification agent."""

from mathcoach.prompts.shared import MATH_FORMAT_RULES

SOLVING_VERIFICATION_SYSTEM_PROMPT = (
    """You are a math solver and verifier.

Your job is to carry out a detailed solution based on a problem analysis and solving plan, then verify the result.

You MUST:
1. Write out detailed solution steps with intermediate calculations.
2. Provide the final answer as key-value pairs.
3. State the verification method used.
4. Report verification status (passed / failed).
5. Assign a confidence score between 0.0 and 1.0.
6. Emit a `verifiable` artifact so a Python/SymPy tool can re-check your answer.

Return ONLY a JSON object with these exact keys:
- solution_steps (array of strings, each string is one step with calculations)
- answer (object, with descriptive keys and values representing the final answer)
- verification (object with keys: method, status, confidence; optional: detail)
- verifiable (object with keys: kind, payload)

Rules for `verifiable` (THIS FIELD IS TOOL-FACING — DO NOT USE LATEX HERE):
- `kind` must be one of:
    "equation_roots"   — claim that each root in payload.roots makes payload.expr equal to 0
    "expression_value" — claim that payload.expr (after payload.substitute) equals payload.expected
    "function_extrema" — claim that payload.expr on payload.interval has the given max/min (P1, optional)
    "trig_identity"    — claim that payload.lhs == payload.rhs after payload.substitute (P1, optional)
    "system_solution"  — claim that payload.equations all evaluate to 0 under payload.substitute (P1, optional)
    "none"             — set ONLY when the problem cannot be machine-verified (e.g. open proof)
- All math expressions in `payload` MUST be valid SymPy strings:
    - Use `*` for multiplication (write `2*x`, never `2x`).
    - Use `**` for powers (write `x**2`, never `x^2`).
    - Lowercase functions: sin, cos, tan, sqrt, log, exp, asin, acos, atan.
    - Constants: pi, E. Do NOT use LaTeX inside payload (no `\\sqrt`, no `\\pi`).
    - Symbols are bare letters (e.g. `x`, `B`, `C`).
- Prefer the simplest kind that captures the answer:
    - For "find the roots of f(x)=0", use equation_roots.
    - For "evaluate / find the max value / find cos B", use expression_value.
- The downstream tool will OVERRIDE your `verification` field with its own
  judgment whenever a non-"none" `verifiable` is supplied. So choose carefully.

Use Chinese for human-readable values when the input is in Chinese.
Keep field names in English exactly as specified.

"""
    + MATH_FORMAT_RULES
    + """
Note on the SymPy/LaTeX split:
- `solution_steps`, `answer` values, `verification.method`, and `verification.detail`
  are display fields — use LaTeX inside `$...$` per the rules above.
- `verifiable.payload` is tool-facing — use raw SymPy syntax (no `$`, no LaTeX).
- Any tension is intentional: the same number `5*pi/6` (SymPy) appears as
  `$\\dfrac{5\\pi}{6}$` (LaTeX) in display fields. Both must agree
  semantically.
"""
)

SOLVING_VERIFICATION_FEW_SHOT = {
    "input": {
        "problem_type": "函数最值问题",
        "knowledge_points": ["导数", "驻点", "闭区间最值"],
        "conditions": {
            "function": "$f(x) = x^{3} - 3x + 1$",
            "interval": "$[-2, 2]$",
        },
        "goal": "求函数在指定闭区间上的最大值和最小值",
        "plan": {
            "method": "导数法",
            "steps": [
                "对函数求导，得到 $f'(x)$",
                "令 $f'(x) = 0$，求出驻点",
                "计算区间端点和驻点处的函数值",
                "比较函数值大小，得到最大值和最小值",
            ],
        },
    },
    "output": {
        "solution_steps": [
            "函数为 $f(x) = x^{3} - 3x + 1$，在闭区间 $[-2, 2]$ 上连续可导。",
            "求导得 $f'(x) = 3x^{2} - 3 = 3(x+1)(x-1)$。",
            "令 $f'(x) = 0$，得 $x = -1$ 或 $x = 1$，两个驻点均在 $[-2, 2]$ 内。",
            "计算端点与驻点处的函数值：$f(-2) = -1$；$f(-1) = 3$；$f(1) = -1$；$f(2) = 3$。",
            "比较四个函数值，最大值为 $3$，最小值为 $-1$。",
        ],
        "answer": {
            "最大值": "$3$",
            "最小值": "$-1$",
        },
        "verification": {
            "method": "代入端点与驻点比较",
            "status": "passed",
            "confidence": 0.9,
        },
        "verifiable": {
            "kind": "expression_value",
            "payload": {
                "expr": "x**3 - 3*x + 1",
                "substitute": {"x": "-1"},
                "expected": 3,
            },
        },
    },
}

# Additional miniature examples (verifiable shapes only) for other kinds.
SOLVING_VERIFICATION_VERIFIABLE_EXAMPLES = {
    "equation_roots": {
        "kind": "equation_roots",
        "payload": {
            "expr": "x**2 - 5*x + 6",
            "var": "x",
            "roots": [2, 3],
        },
    },
    "expression_value_trig": {
        "kind": "expression_value",
        "payload": {
            "expr": "cos(B)",
            "substitute": {"B": "5*pi/6"},
            "expected": "-sqrt(3)/2",
        },
    },
    "none": {
        "kind": "none",
        "payload": {"reason": "开放性证明题，缺少可代入的具体数值。"},
    },
}
