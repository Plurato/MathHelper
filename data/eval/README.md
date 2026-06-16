# Eval datasets

Hand-curated problem sets for end-to-end evaluation of the MathCoach pipeline.

## Files

- `dev.json` — v01 dev set, 20 problems.

## Schema

Top-level: `{version, description, problems: list[Problem]}`.

Each `Problem`:

| field | type | notes |
|------|------|------|
| `id` | string | unique, format `<Group><Index>` (e.g. `A01`, `C05`) |
| `group` | `"A"` / `"B"` / `"C"` | verifier-expectation grouping |
| `type` | string | problem-type label |
| `knowledge_points` | list[string] | tags |
| `difficulty` | `"easy"` / `"medium"` / `"hard"` | |
| `expected_verifier` | `"should_pass"` / `"edge_case"` / `"unverifiable"` | analyst's prior expectation of verifier behavior; used to compute FP/FN |
| `question` | string | LaTeX in `$...$` |
| `truth.answer` | list[AnswerItem] | structure aligned with `mathcoach.schemas.AnswerItem` |
| `notes` | string, optional | reviewer notes (not used by grader) |

`AnswerItem`: `{label, latex, sympy, numeric, unit}`. `sympy=null` means the
problem has no machine-comparable answer (typical for proofs); the grader
returns `correct=None` for such items.

## Group semantics

| Group | Meaning | Verifier should... | Used to compute |
|-------|---------|-------------------|-----------------|
| **A** | Standard textbook problems with clean SymPy-friendly answers | …pass with high confidence | base correctness rate, FP rate (signal floor) |
| **B** | Edge cases (parameters, free variables, awkward forms, geometric edge) | …still pass but may surface false answers | grader robustness |
| **C** | Proofs, propositions, applied problems with significant text | …emit `assertions=[]` and trigger confidence cap=0.6 | confirms cap behavior |

## Adding problems

1. Pick the next free `id` in the appropriate group.
2. Write `question` and pre-compute `truth.answer`. Verify each `truth.answer[i].sympy` parses with the eval grader before committing (run `python -m mathcoach.eval.loader data/eval/dev.json --check`).
3. Choose `expected_verifier` based on whether the verifier is *supposed* to give a high-confidence pass on this problem. This is your prior, not a guess at LLM behavior.
4. Keep `notes` short — derivation hint is enough.
