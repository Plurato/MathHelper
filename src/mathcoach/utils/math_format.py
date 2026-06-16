"""Repair LLM-emitted JSON strings whose LaTeX commands collide with JSON escapes.

When a model writes `"$\triangle$"` (single backslash) inside a JSON string,
`json.loads` decodes `\t` as TAB. The resulting Python string contains a TAB
followed by `riangle` — KaTeX cannot render that. We rescue this post-parse
by mapping the control char back to its literal LaTeX form, but only inside
`$...$` math regions to avoid touching legitimate prose whitespace.
"""

from __future__ import annotations

import re
from typing import Any

# JSON control char → literal LaTeX command prefix the model meant to emit.
# Order doesn't matter for correctness; declared in spec order.
_CTRL_TO_LATEX_LITERAL: dict[str, str] = {
    "\t": "\\t",   # \triangle, \theta, \tan, \to, \times, \top
    "\n": "\\n",   # \neq, \nu, \nabla, \ne
    "\r": "\\r",   # \rho, \rightarrow
    "\b": "\\b",   # \beta, \binom
    "\f": "\\f",   # \frac, \forall
}

# ! NOTE on `\u`: JSON's \uXXXX needs 4 hex digits. `\Upsilon` / `\uparrow`
# trip json.loads BEFORE we ever run, so they bubble up as parse errors and
# rely on BaseAgent.max_parse_retries to fix.

_LATEX_REGION = re.compile(
    r"\$\$.*?\$\$|\$[^$]*?\$",
    re.DOTALL,
)


def fix_latex_escapes(text: str) -> str:
    def _fix(match: re.Match[str]) -> str:
        body = match.group(0)
        for ctrl, literal in _CTRL_TO_LATEX_LITERAL.items():
            if ctrl in body:
                body = body.replace(ctrl, literal)
        return body

    return _LATEX_REGION.sub(_fix, text)


def repair_latex_in_payload(payload: Any) -> Any:
    if isinstance(payload, str):
        return fix_latex_escapes(payload)
    if isinstance(payload, dict):
        return {k: repair_latex_in_payload(v) for k, v in payload.items()}
    if isinstance(payload, list):
        return [repair_latex_in_payload(item) for item in payload]
    if isinstance(payload, tuple):
        return tuple(repair_latex_in_payload(item) for item in payload)
    return payload
