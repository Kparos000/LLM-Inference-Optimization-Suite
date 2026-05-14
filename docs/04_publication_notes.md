# Publication Notes

## Purpose

Preserve material that may later support a short technical paper, LinkedIn post, and Twitter/X thread.

## Paper Notes

- Frame the project as a controlled benchmark of inference optimization trade-offs.
- Emphasize methodology, reproducibility, and staged validation before GPU spending.
- Preserve exact model, backend, workload, and optimization configurations for any reported result.
- Use the benchmark methodology document to guide the technical paper structure and experimental framing.

## LinkedIn Notes

- Focus on the engineering workflow: validate locally, measure carefully, then scale.
- Highlight the benchmark pipeline components once real inference results are available.
- Keep claims tied to measured results and documented limitations.

## Twitter/X Thread Notes

- Potential structure: problem, benchmark design, no-GPU harness, first model baseline, optimization comparison, key takeaway.
- Use concise charts only after representative results are available.
- Avoid overstating early smoke-test outcomes.

## Methodology Points To Preserve

- Configuration-driven experiments
- Stable workload definitions
- Separate latency, throughput, memory, and cost metrics
- Raw CSV outputs and report-ready plots
- Comparison CSVs for paper and social-summary preparation
- Clear distinction between mock, smoke, and performance benchmark runs

## Result Points To Preserve Later

- Baseline latency and throughput by model size
- Impact of serving backend changes
- Memory and cost trade-offs for each optimization
- Workload sensitivity across short, long-context, and shared-prefix prompts

## HF Baseline Findings To Preserve

- `long_context` increased TTFT materially compared with `short_chat`.
- TPOT remained relatively stable across workloads.
- The HF baseline establishes the reference point before vLLM.
- The `shared_prefix` workload will support prefix caching evaluation later.

## vLLM Baseline Learning Preserved

- The vLLM RunPod L40S baseline showed fast TPOT.
- Prompt-level traces revealed quality and truncation issues.
- Future public writeups should discuss speed and quality together.

## Controlled HF Baseline Artifacts

Controlled Hugging Face baseline results should preserve:

- `system_info.json`
- `hf_smoke_results.csv`
- `hf_smoke_generations.jsonl`
- `hf_structured_output_results.csv`
- `hf_structured_output_generations.jsonl`
- Structured JSON validity score
- Figures from both workloads

## Candidate Sample Artifacts For Final Publication Package

- System info sample
- HF baseline metrics sample
- Structured-output metrics sample
- Structured-output generation trace sample
- Expanded workload comparison CSV
- Workload-specific trace samples
- Latency and throughput plots

## Scaled Benchmark Evidence To Preserve

- `stress_plan.yaml`
- Concurrency results
- Workload-scale comparisons
- Tail latency summaries
- Throughput vs concurrency plots
- Memory/cost trade-off tables

## Limitations To Track

- Hardware differences across runs
- Backend version differences
- Tokenization differences across model families
- Small sample sizes in smoke workloads
- Quality evaluation scope

## Future Work Ideas

- Add Hugging Face baseline runner
- Add vLLM serving benchmark path
- Add quantization experiments
- Add prefix caching and shared-prefix workloads
- Add speculative decoding comparison
- Add selected reproducible result snapshots for publication artifacts
