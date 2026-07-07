# Controlled Final Generation-Contract Repair Summary

## Outcome

The controlled-final runner now builds requests through the B6/B7/A100
context-aligned generation-contract path instead of raw promoted prompts.

## Gate Results

- Contract preflight: PASS on 10,000/10,000 matrix rows.
- Repaired 25-row replay: completed 25/25 requests.
- JSON validity improved from 0.0% to 100.0%.
- Contract validity improved from 0.0% to 84.0%.
- Evidence match and groundedness improved from 3.97% to 56.0%.
- Safety violations fell from 12 to 0.
- Natural-language/no-JSON majority failure was eliminated.
- 500-row validation: blocked because the 25-row contract gate did not pass.

## Decision

Full 10,000 rerun remains blocked. The next step is to repair the remaining
contract-invalid replay rows, then rerun the 25-row repaired smoke.
