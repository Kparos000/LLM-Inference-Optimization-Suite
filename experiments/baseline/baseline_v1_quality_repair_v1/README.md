# Baseline V1 Quality Repair V1

This folder preserves the Baseline_V1 quality-repair scorecard and targeted validation evidence. It does not overwrite `experiments/baseline_v1/` and does not run Main_Inference_V1.

- Baseline_V1 execution: complete.
- Frozen Baseline_V1 deployability: NOT_DEPLOYABLE_SLO_FAILURES.
- Targeted repair validation: passed on 1,600 selected requests.
- Final safety-risk repair: passed with zero safety findings.
- Main_Inference_V1 allowed: true.

## Final Safety Risk Repair

`final_safety_risk_repair/` archives the closeout for the monitored
SGLang MM4 c32 Healthcare Admin safety finding. The issue was isolated to
`healthcare_admin_scaleup_2000_0027`, where safe administrative boundary
wording repeated the prohibited phrase `medical advice`. The repair rewrites
only normalized final-answer boundary wording, preserves raw output for audit,
and does not weaken the evaluator, SLO targets, gold data, or MM4.

Targeted replay covered the exact row plus 39 neighboring Healthcare Admin
SGLang MM4 c32 rows and reached 100% JSON validity, 100% contract validity,
100% evidence match, 100% groundedness, and zero safety findings. Broader
deterministic replay covered 2,200 rows, including the selected 1,600 repair
rows plus vLLM comparison configs, and also reached zero safety findings.
