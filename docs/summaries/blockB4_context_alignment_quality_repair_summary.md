# Block B4 Context Alignment Quality Repair Summary

Status: `QUALITY_BLOCKED`

B4 executed the B3 context-alignment repair on the exact 100 B1 prompt IDs. It
did not modify gold data, evaluator semantics, or the promoted retrieval source
of truth.

Main results:

- B1 all-required-gold-present rate improved from 48/100 to 100/100.
- Finance context alignment improved from 2/20 to 20/20.
- B4 completed 100/100 vLLM requests on the remote RTX 3070.
- JSON validity improved from 93% to 97%.
- Contract validity improved from 92% to 97%.
- Evidence match improved from 35% to 76%.
- Groundedness improved from 35% to 76%.
- Truncation fell from 6% to 3%.
- Safety violations remained 2, so the gate is still blocked.

Per-vertical evidence match and groundedness:

- Airline: 65%, with 2 safety violations.
- Healthcare Admin: 75%.
- Retail: 95%.
- Finance: 70%.
- Research AI: 75%.

The post-B4 audit found 25 failed rows. Required gold evidence was absent from
zero failed rows. Evidence was present but not cited in 24 failed rows, and all
25 failed rows were classified as model instruction-following failures.

Finance is no longer a retrieval/context availability problem in B4. All six
failed Finance rows had required evidence present in E1-E5. The remaining
Finance failures are citation-selection and instruction-following failures,
with one truncation and no Finance safety/advice/projection wording.

The B2 SLO diagnosis engine reported nine failed SLOs across B4 vertical slices:
Airline failed evidence, groundedness, and safety; Healthcare Admin, Finance,
and Research AI failed evidence and groundedness; Retail had no failed selected
SLOs. The catalog primary recommendation is `use_stronger_model` for the
failed verticals, but B4's directly observed next repair is narrower.

Exact next block:

```text
B4R1_SAFETY_AND_CITATION_SELECTION_REPAIR
```

Keep the B4 100-prompt matrix frozen. Repair the safety retry prompt so unsafe
phrases are not repeated back to the model, add a lexical guard for prohibited
terms, and improve short-label evidence presentation for multi-evidence tasks.
Do not scale prompt count or concurrency until safety is zero and the quality
gate passes.

Full report:
`docs/101_context_alignment_and_generation_quality_repair.md`.
