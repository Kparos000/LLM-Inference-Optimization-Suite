# Scaled Workload Generation

## Purpose

This document describes how deterministic synthetic workload files are generated for larger benchmark runs. The goal is to move beyond 3 to 5 prompt calibration files while keeping the workload source reproducible, inspectable, and free of private data.

## Why Synthetic Scaled Workloads Are Used

Synthetic scaled workloads provide stable prompt sets for load testing, backend comparison, and repeatable latency analysis. They avoid external API calls, private data, and copyrighted passages while still exercising the benchmark harness across multiple prompt families.

The generated prompts are not a replacement for later real-world evaluation datasets. They are controlled benchmark inputs for measuring serving behavior, output traces, structured-response handling, and concurrency effects.

## Workload Families

The generator supports five workload families:

- `short_chat`: short professional writing, summarization, rewrite, confirmation, and explanation prompts.
- `code_helpdesk`: debugging, Git, CLI, Python, dependency, environment, and troubleshooting prompts.
- `long_context`: synthetic passages around platform operations, support processes, data pipelines, incidents, rollout notes, and documentation updates.
- `shared_prefix`: repeated internal IT support prefix with varied user requests.
- `structured_output`: prompts requesting valid JSON with `category`, `answer`, and `confidence` fields.

## Determinism and Reproducibility

The generator creates deterministic JSONL workloads from templates using:

- `count`: number of prompts per workload file.
- `seed`: deterministic generation seed.
- `workload selection`: the workload families to generate.
- `output directory`: where JSONL files are written.

For the same count, seed, workload list, and generator version, output files should be stable.

## Prompt Schema

Each generated JSONL row contains:

- `prompt_id`
- `workload_name`
- `prompt`
- `metadata`

Metadata includes:

- `scale_count`
- `template_family`
- `synthetic`
- `seed`

## Scaling Strategy

- 100 prompts per workload: validation and first concurrency check.
- 1,000 prompts per workload: first serious GPU benchmark.
- 5,000 prompts per workload: medium-scale benchmark.
- 10,000 prompts per workload: larger benchmark.

The 100-prompt files may be committed when intentionally generated and reviewed. Larger generated files are ignored by default and should usually be generated on RunPod or the execution environment when needed.

## Committed vs Generated Artifacts

The repository keeps the generator, configuration, tests, and documentation under version control. Large generated workload files should not be committed unless they are intentionally curated and reviewed for repository size and public-content suitability.

Generated 1,000-, 5,000-, and 10,000-prompt files are ignored by default. This keeps the public repository lightweight while preserving reproducibility through deterministic generation commands.

## Example Commands

```text
inference-bench generate-workloads --count 100 --output-dir data/prompts/scaled --seed 42
```

```text
inference-bench generate-workloads --count 1000 --output-dir data/prompts/scaled --seed 42
```

```text
inference-bench generate-workloads --count 5000 --output-dir data/prompts/scaled --seed 42
```

## Limitations

- Template-based prompts do not capture the full diversity of production traffic.
- Synthetic prompts may underrepresent ambiguous, adversarial, or domain-specific inputs.
- Quality evaluation still requires prompt-level trace review and scoring.
- Large-scale runs remain hardware- and environment-specific.

## Future Dataset Expansion

Future benchmark phases can add more template families, longer context variants, structured-output schemas, quality-scored subsets, and curated public datasets where licensing and privacy constraints are clear.
