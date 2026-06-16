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

The grouping is a *taxonomy of problem shape*, not a strict promise about
verifier behavior. The `expected_verifier` field on each problem is the
prior assertion against which FP/FN are computed; it gets updated when real
runs reveal it was wrong (this happened in v1 — see `docs/journal.md`).

| Group | Meaning |
|-------|---------|
| **A** | Standard textbook problems with clean SymPy-friendly answers |
| **B** | Edge cases (parameters, free variables, awkward answer shapes like Interval / Relational) |
| **C** | Proofs, propositions, applied problems with significant text — answer often still numeric/symbolic, but the path through the LLM is more verbal |

`expected_verifier` values:
- `should_pass` — verifier should give a high-confidence pass
- `edge_case` — verifier may pass but the answer form / corner case is non-trivial
- `unverifiable` — we expect verifier to bail (e.g. `assertions=[]`, `cap=0.6`); reserved for problems where the LLM truly cannot emit machine-checkable claims. Note: in the v01 dev set this expectation never matched reality — LLMs found ways to emit verifiable assertions even for proofs.

## Adding problems

1. Pick the next free `id` in the appropriate group.
2. Write `question` and pre-compute `truth.answer`. Verify each `truth.answer[i].sympy` parses with the eval grader before committing (run `python -m mathcoach.eval.loader data/eval/dev.json --check`).
3. Choose `expected_verifier` based on whether the verifier is *supposed* to give a high-confidence pass on this problem. This is your prior, not a guess at LLM behavior.
4. Keep `notes` short — derivation hint is enough.
