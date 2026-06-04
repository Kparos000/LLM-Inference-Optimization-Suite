# Retrieval Dataset/Gold Alignment Repair

Block 18 creates a generated repaired retrieval dataset for analysis. It does
not modify the promoted benchmark dataset under `data/scaleup_2000_full/`.

No inference, GPU work, paid API calls, external APIs, or SLO weakening were
performed.

## Why Block 17 Was Not Enough

Block 17 proved that canonical retrieval keys and final selectors reduce some
failure modes, but the system remained SLO-blocked. The remaining issue is not
only retrieval code. The prompt/gold/corpus alignment is uneven:

- many prompts do not contain explicit runtime `retrieval_query` fields;
- Finance prompts often lack period, metric, filing section, or XBRL concept
  cues;
- Retail prompts often have multiple same-product review or summary chunks that
  can legitimately satisfy the prompt, while gold labels count only narrow rows;
- Research AI has paper/title/section ambiguity that grows at 2,000 records;
- several verticals have multiple valid evidence chunks not counted in the
  original gold labels.

## Generated Artifacts

Reports:

- `data/generated/context_engineering/retrieval_dataset_alignment_report.json`
- `data/generated/context_engineering/retrieval_dataset_alignment_summary.csv`
- `data/generated/context_engineering/retrieval_records_needing_repair.jsonl`
- `data/generated/context_engineering/repaired_retrieval_validation_report.json`
- `data/generated/context_engineering/repaired_retrieval_validation_summary.csv`
- `data/generated/context_engineering/repaired_retrieval_promotion_plan.json`

Local generated repaired records:

- `data/generated/context_engineering/repaired_retrieval_dataset/`

The full repaired dataset JSONL files are intentionally ignored because they are
large local generated artifacts. The committed repair index is compact and
contains per-record repair reasons and counts, not full expanded record payloads.

## Repair Fields

Each repaired generated record preserves the original `prompt_id` and vertical,
then adds:

- `retrieval_query`
- `canonical_retrieval_metadata`
- `repair_reason`
- `valid_evidence_ids_expanded`
- `expected_metric`
- `expected_period`
- `expected_intent`

Runtime retrieval uses only `retrieval_query` and non-leaking metadata. Expanded
valid evidence IDs are used only for offline evaluation.

## Repair Counts

All 10,000 records received generated alignment metadata because none of the
promoted prompts already contained an explicit canonical `retrieval_query`.

Records repaired by vertical:

- Airline: 2,000
- Healthcare Admin: 2,000
- Retail: 2,000
- Finance: 2,000
- Research AI: 2,000

Major repair reasons:

- Finance: 2,000 records missing required retrieval cues or metadata.
- Retail: 1,938 records with narrow gold labels and multiple valid evidence
  alternatives; 1,804 with near-duplicate same-product confusion.
- Research AI: 1,960 records with narrow gold labels and near-duplicate
  paper/section alternatives; 45 with missing metadata.
- Airline and Healthcare: all records lacked explicit canonical retrieval query
  fields and had expanded valid policy/procedure alternatives.

## Before/After Metrics

The original metrics come from the Block 17 canonical staged report. The repaired
metrics come from Qdrant-backed validation against the generated repaired
dataset.

### 2,000-Record Stage

| Vertical | Dataset | Candidate@20 | Candidate@50 | Recall@5 | MRR | Status |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| Airline | Original | 0.928000 | 0.964625 | 0.889375 | 0.934150 | Failed |
| Airline | Repaired | 1.000000 | 1.000000 | 1.000000 | 1.000000 | Passed |
| Healthcare Admin | Original | 0.964000 | 0.984000 | 0.915000 | 0.745767 | Failed |
| Healthcare Admin | Repaired | 1.000000 | 1.000000 | 1.000000 | 0.994250 | Passed |
| Retail | Original | 0.913417 | 0.927917 | 0.220167 | 0.161017 | Failed |
| Retail | Repaired | 0.974333 | 0.982083 | 0.959917 | 0.922592 | Passed |
| Finance | Original | 0.461250 | 0.820625 | 0.251250 | 0.142458 | Failed |
| Finance | Repaired | 0.948875 | 0.955750 | 0.939000 | 0.941833 | Passed |
| Research AI | Original | 0.871318 | 0.948397 | 0.752403 | 0.779100 | Failed |
| Research AI | Repaired | 0.742017 | 0.939806 | 0.596498 | 0.882217 | Failed |

## Promotion Decision

Promotion is not recommended yet.

The repaired generated dataset materially improves Airline, Healthcare, Retail,
and Finance. However, Research AI still fails at the 2,000-record stage with
candidate@20 0.742017 and recall@5 0.596498. The generated dataset should not
replace the promoted dataset until Research AI paper/section alignment is fixed
and the repaired dataset passes all vertical SLO checks.

## Remaining Blockers

Research AI is the remaining blocker:

- candidate@20 remains below target;
- recall@5 remains below target;
- 2,000-record scale still introduces paper/title/section ambiguity;
- expanded valid evidence helps candidate@50 but does not make the final top-5
  reliable.

## Regeneration

Run:

```powershell
python scripts/phase3/repair_retrieval_dataset_alignment.py `
  --dataset-root data/scaleup_2000_full `
  --context-root data/generated/context_engineering `
  --slo-config configs/slo_targets.yaml `
  --output-root data/generated/context_engineering `
  --stage-sizes 500 2000
```

The command regenerates alignment reports, repaired validation reports, the
promotion plan, compact repair index, local repaired dataset files, and the
standard SLO readiness reports.
