# BC-POT-R results

This is retrospective redevelopment evidence: earlier aggregate 2025 results were known.
The 2026 confirmation gate remains unopened.

## Primary result

- Q4-2024-selected method: `boundary_gated_ensemble`
- Standard argmax accuracy: 0.773511513
- Original schedule-baseline accuracy: 0.767492500
- Previous meta-stack accuracy: 0.773308359
- Absolute accuracy gain vs original schedule baseline: +0.006019013
- Date-cluster 95% interval for that gain: [+0.005054471, +0.007004311]
- Absolute accuracy gain vs previous meta-stack: +0.000203154
- Requested +0.05 gate passed: False
- Requested +0.10 gate passed: False
- Joint log loss: 0.538449266
- Multiclass Brier score: 0.326679482

Accuracy changes above are absolute proportions; multiply by 100 for percentage points.
No relative-percentage substitution is used. Boundary residuals are computational
contrasts, not causal effects, and capacity/queue fields are model states rather than
observed airport operations.
