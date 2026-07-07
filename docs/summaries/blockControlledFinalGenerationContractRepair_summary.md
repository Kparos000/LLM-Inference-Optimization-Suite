# Controlled Final Generation-Contract Repair Summary

## Outcome

The controlled-final runner now builds requests through the B6/B7/A100
context-aligned generation-contract path instead of raw promoted prompts.

## Gate Results

- Contract preflight: PASS on 10,000/10,000 matrix rows.
- Repaired 25-row replay: completed 25/25 requests.
- JSON validity improved from 0.0% to 100.0%.
- Contract validity improved from 0.0% to 100.0%.
- Evidence match and groundedness improved from 3.97% to 72.0%.
- Safety violations fell from 12 to 0.
- Natural-language/no-JSON majority failure was eliminated.
- 25-row replay gate: PASS.
- MM4 safety targeted replay: completed 11/11 requests with zero safety
  violations and 100.0% JSON/contract/format validity.
- 500-row validation: ran 500/500 requests with 100.0% JSON/contract/format
  validity, 73.2% evidence match, 73.2% groundedness, and zero safety
  violations.

## Decision

Full 10,000 rerun is now allowed as a separate explicit run. It was not
executed in this repair block.
