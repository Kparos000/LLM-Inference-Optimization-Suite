# Decision Log

| Date | Decision | Rationale | Impact |
| --- | --- | --- | --- |
| 2026-05-12 | Use main branch only for solo development. | The project is currently maintained by one developer and does not need branch coordination overhead. | Keeps workflow simple while the scaffold is being built. |
| 2026-05-12 | Use CI/CD from the beginning. | Automated checks catch regressions before benchmark complexity increases. | Establishes a quality baseline for later runtime integrations. |
| 2026-05-12 | Do not use paid GPU until the harness is validated. | Experiment design should be proven before spending compute budget. | Reduces wasted GPU time and supports reproducible execution. |
| 2026-05-12 | Keep educational explanations in working conversations, not in the repo. | The repository should remain professional and focused on benchmark implementation. | Keeps docs concise and portfolio-ready. |
| 2026-05-12 | Use Qwen small models first because they are easier to access on Hugging Face. | Small accessible models are suitable for smoke tests and integration validation. | Provides a practical path from local validation to first model runs. |
| 2026-05-12 | Include larger model placeholders for future scale comparison. | Scale comparison is part of the project direction, but larger runs are not active yet. | Preserves future planning without adding runtime or cost. |
| 2026-05-12 | Start with vLLM before adding SGLang. | vLLM is the first serving benchmark target after the Hugging Face baseline is stable. | Limits backend scope and leaves SGLang as a later extension. |
| 2026-05-12 | Prepare vLLM baseline only after HF baseline and expanded workloads are stable. | Avoid wasting GPU resources and ensure fair baseline comparison. | vLLM is planned but not executed until readiness checklist is complete. |
