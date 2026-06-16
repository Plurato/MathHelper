"""Repair LLM-emitted JSON strings that contain LaTeX commands.

Models often emit `"$\\triangle ABC$"` as `"$\\triangle ABC$"` in the JSON
literal but write only ONE backslash, like `"$\triangle ABC$"`. JSON parses
`\\t` as TAB, so after `json.loads` the string actually contains a TAB
followed by `riangle` — which KaTeX cannot render.

The bug is recoverable post-parse: a TAB that appears inside a `$...$` math
span was almost certainly meant as `\\t` (the start of `\\triangle`,
`\\theta`, `\\tan`, `\\to`, `\\times`, `\\top`). Replacing it with the
literal two-character sequence backslash-`t` reconstructs the LaTeX command.

Same mechanism for `\\n` → `\\neq`/`\\nu`/`\\nabla`, `\\r` → `\\rho`/
`\\rightarrow`, `\\b` → `\\beta`/`\\binom`, `\\f` → `\\frac`/`\\forall`.

Trade-off: legitimate TAB/NEWLINE characters inside `$...$` get rewritten
too. In school-math contexts this is essentially never wrong (KaTeX ignores
literal whitespace anyway).
"""

from __future__ import annotations

import re
from typing import Any

# JSON control character → its literal LaTeX command prefix.
# Order doesn't matter for correctness; declared in spec order for readability.
_CTRL_TO_LATEX_LITERAL: dict[str, str] = {
    "\t": "\\t",   # \triangle, \theta, \tan, \to, \times, \top
    "\n": "\\n",   # \neq, \nu, \nabla, \ne
    "\r": "\\r",   # \rho, \rightarrow
    "\b": "\\b",   # \beta, \binom
    "\f": "\\f",   # \frac, \forall
}

# ! NOTE on `\u`: JSON's `\uXXXX` escape requires 4 hex digits. If the LLM
# writes `\Upsilon` or `\uparrow` with a single backslash, json.loads itself
# raises "Invalid \\uXXXX escape" BEFORE we get a chance to repair anything.
# Such inputs surface as parse errors and rely on BaseAgent.max_parse_retries
# to ask the model to fix them. We can't recover those here.

# Match $$...$$ display blocks (greedy/non-greedy with DOTALL) OR $...$ inline
# blocks. Alternation tries the longer pattern first so $$a$$ is one region,
# not two empty $$.
_LATEX_REGION = re.compile(
    r"\$\$.*?\$\$|\$[^$]*?\$",
    re.DOTALL,
)


def fix_latex_escapes(text: str) -> str:
    """Restore JSON control chars to literal escape form inside `$...$` blocks.

    No-op outside math regions. Idempotent — safe to call multiple times on
    the same string.
    """
    def _fix(match: re.Match[str]) -> str:
        body = match.group(0)
        for ctrl, literal in _CTRL_TO_LATEX_LITERAL.items():
            if ctrl in body:
                body = body.replace(ctrl, literal)
        return body

    return _LATEX_REGION.sub(_fix, text)


def repair_latex_in_payload(payload: Any) -> Any:
    """Apply `fix_latex_escapes` to every string in a parsed JSON tree.

    Walks dicts, lists, and tuples recursively. Non-string scalars are
    returned unchanged. Returns a new structure; the input is not mutated.
    """
    if isinstance(payload, str):
        return fix_latex_escapes(payload)
    if isinstance(payload, dict):
        return {k: repair_latex_in_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [repair_latex_in_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(repair_latex_in_payload(item) for item in payload)
    return payload
