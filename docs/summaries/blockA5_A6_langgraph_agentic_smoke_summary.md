# Block A5/A6 LangGraph Agentic Smoke Summary

Status: complete on June 14, 2026.

Implemented an executable bounded LangGraph `mm4_bounded_agentic` mode with
validated state, eight nodes, seven approved tools, one optional repair, public
trace events, per-node timings, token accounting, and hard runtime limits.

The matched 50-prompt RTX 3070 smoke completed without request failures:

- 47 answers and 3 escalations;
- 94% contract validity;
- 44% evidence match;
- 42% deterministic groundedness;
- 4% safety violations;
- 6% repair rate and 6% escalation rate;
- 181.903 ms mean TTFT;
- 1,022.239 ms mean E2E latency.

Compared with mm2/mm3, mm4 improved groundedness by 14/16 percentage points and
contract validity by 22/28 points. It used more normalized tokens and increased
mean E2E latency by 141.743/251.612 ms. GPU cost remains unavailable because the
remote server has no registered hourly price.

Decision:

```text
KEEP_MM4_IN_CONTROLLED_BENCHMARK_MATRIX
DO_NOT_DEFAULT_TO_MM4
DO_NOT_SCALE_BEYOND_50_PROMPTS_YET
```

The graph used one retrieval round for every prompt. The maximum observed
counters stayed within the contract: two generation attempts, one repair, and
three action-tool calls.
