"""Cross-agent prompt fragments.

Anything that should be applied to multiple agents (formatting rules, output
conventions) lives here so changes propagate consistently.
"""

# Single source of truth for how math expressions appear in *display* fields
# (problem statements, plan steps, solution_steps, explanation, answer values,
# practice_questions, etc.). Tool-facing fields (e.g. `verifiable.payload`)
# follow their own rules — see solving_verification.py.
#
# Frontends will render display fields with KaTeX/MathJax. Mixing ASCII
# (`x^2`), Unicode (`²`, `√`, `π`), and LaTeX in the same payload makes that
# rendering and any downstream search/normalization harder than it needs to be.
MATH_FORMAT_RULES = """\
Math formatting (applies to ALL human-readable string fields you output):
- Wrap every math expression in `$...$` for inline math, or `$$...$$` for
  display math used in solution steps that contain a multi-line derivation.
- Inside `$...$`, use LaTeX syntax:
    * Powers: `x^{2}`, `x^{n+1}` (always brace exponents with more than one char).
    * Roots: `\\sqrt{3}`, `\\sqrt[3]{x}`. Never write the bare `√` character.
    * Greek letters: `\\pi`, `\\alpha`, `\\theta`. Never write `π`, `α`, `θ`.
    * Fractions: `\\dfrac{a}{b}` for display, `\\frac{a}{b}` is also accepted.
    * Functions: `\\sin`, `\\cos`, `\\log`, `\\ln`, `\\tan` (with backslash).
    * Set/relation symbols: `\\in`, `\\leq`, `\\geq`, `\\neq`, `\\cdot`,
      `\\pm`, `\\to`, `\\Rightarrow`. Never write the bare Unicode glyphs.
    * Intervals/sets are math: write `$[-2, 2]$`, `$(0, \\pi)$`, `$\\triangle ABC$`.
- Outside `$...$`, the text is plain Chinese/English prose. Do NOT mix bare
  math symbols (`²`, `√`, `π`, `∈`, `≤`) into prose.
- Multi-character variables and function names that are NOT standard LaTeX
  commands should be wrapped in `\\operatorname{}` or `\\text{}` if needed,
  but for typical school math (single-letter variables) this is rarely needed.

Examples:
- GOOD: "求导得 $f'(x) = 3x^{2} - 3 = 3(x+1)(x-1)$。"
- BAD:  "求导得 f'(x)=3x^2-3=3(x+1)(x-1)。"           (no `$...$`, ASCII `^`)
- BAD:  "求导得 $f'(x) = 3x² - 3$。"                  (Unicode superscript)
- GOOD: "代入得 $\\cos B = -\\dfrac{\\sqrt{3}}{2}$，所以 $B = \\dfrac{5\\pi}{6}$。"
- BAD:  "代入得 cos B = -√3/2，所以 B = 5π/6。"        (bare Unicode + no `$...$`)
- GOOD: "在 $\\triangle ABC$ 中，$B \\in (0, \\pi)$。"
- BAD:  "在 △ABC 中，B∈(0, π)。"                     (bare Unicode + no `$...$`)

NOTE: These LaTeX rules apply to display fields (problem statements, plan
steps, solution_steps, explanation, AnswerItem.latex, practice_questions,
etc.). They DO NOT apply to tool-facing fields:
- `assertions[i].expr` and `assertions[i].expected` use raw SymPy syntax,
  no `$...$`, no `\\sqrt`/`\\pi`. See solving_verification prompt.
- `AnswerItem.sympy` uses raw SymPy syntax (e.g. "5*pi/6").
"""
