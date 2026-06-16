# Eval Report — 20260616-222652

- Model: `default`
- Agents: Understanding → Planning → Verification
- Concurrency: 1
- Problems: 20 (from `/Users/didi/Code/Rwei/MathHelper/data/eval/dev.json`)
- Git SHA: `8d98fe161f6e0364ed5e2297196ed1fc4a3eca48`

## Aggregate

- correct: **18/20 (90.0%)**
- incorrect: 1
- grader skipped (no_sympy / error): 1
- pipeline failed: 0
- FP rate (expected should_pass but answer wrong): **0.0%**
- FN rate (verifier failed but answer correct): **5.3%**

## By group

| Group | N | correct | incorrect | skipped | failed |
|------|---|---------|-----------|---------|--------|
| A | 10 | 10 | 0 | 0 | 0 |
| B | 5 | 4 | 1 | 0 | 0 |
| C | 5 | 4 | 0 | 1 | 0 |

## Failures and skips (2)

### B04 [B] — 圆的切线

- expected_verifier: `edge_case`
- pipeline_status: `ok`
- grader: `ok` (layer=exact)
- grader_reason: item '切线方程': pipeline='Eq(3*x + 4*y, 25)' != truth='3*x + 4*y - 25'
- verifier: status=`passed`, confidence=`0.98`
- pipeline_answer: `切线方程=Eq(3*x + 4*y, 25)`
- truth_answer: `切线方程=3*x + 4*y - 25`

### C02 [C] — 立体几何证明

- expected_verifier: `should_pass`
- pipeline_status: `ok`
- grader: `no_sympy` (layer=no_sympy)
- grader_reason: at least one truth item has sympy=null; cannot grade programmatically
- verifier: status=`passed`, confidence=`0.98`
- pipeline_answer: `结论=<no sympy>`
- truth_answer: `结论=<no sympy>`
