# Block Phase 2 Targeted Baseline Repairs Summary

Status: `TARGETED_REPAIR_GATE_PASSED`

The targeted baseline repair phase ran the selected eight-config before/after
validation plan without running the full final 10,000 experiment. It completed
1,600/1,600 requests with zero request failures.

Overall quality improved from 84.13% contract validity, 79.63% evidence match,
76.19% groundedness, and 14 safety findings to 100.00% contract validity,
100.00% evidence match, 100.00% groundedness, and 1 safety finding.

Research AI improved from 20.94% contract validity, 37.81% evidence match, and
20.94% groundedness to 100.00% across all three metrics. Healthcare Admin
safety findings dropped from 12 to 1.

The targeted gate passed and the final/main 10,000-request experiment is
allowed as a separate explicit run. The one remaining safety finding is isolated
to SGLang MM4 c32 Healthcare Admin and should be watched in the next full run.
