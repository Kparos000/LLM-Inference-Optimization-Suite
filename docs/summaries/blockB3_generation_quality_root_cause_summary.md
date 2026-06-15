# Block B3 Generation Quality Root-Cause Summary

Status: `AUDIT_COMPLETE_QUALITY_REMAINS_BLOCKED`

Phase B3 audited all 65 failed B1 rows offline. It used unchanged evaluator
rows, raw generations, citation aliases, rendered E1-E5 context, and frozen
runner metadata. No inference ran.

Main findings:

- 52 failures lacked at least one required gold evidence ID in E1-E5.
- 18 failures had available required evidence that was not cited.
- 27 were partial multi-evidence citations.
- 7 had invalid JSON, 8 had invalid contracts, and 6 truncated.
- The two safety violations were Airline literal prohibited-phrase matches.
- Finance had 19 failures: 18 lacked exact required SEC/XBRL evidence, while
  ticker and company metadata remained visible in every rendered context.
- Finance had no safety violation or investment/advice/projection wording
  match.

Finance is primarily a frozen workload/rendered-context alignment problem with
a secondary model citation-selection and truncation problem. This finding does
not change the promoted retrieval source of truth.

The exact next block is
`B3R1_FROZEN_WORKLOAD_CONTEXT_ALIGNMENT_REPAIR`: trace and re-export the same
100 prompts, require every gold ID to map to E1-E5, rerun the offline audit,
then use no more than five Finance prompts for a citation-selection replay.

Full report:
`docs/100_generation_quality_root_cause_audit.md`.
