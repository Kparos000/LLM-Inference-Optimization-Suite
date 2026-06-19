# Repository Cleanup And CI Validation Hardening

Status: implemented on June 19, 2026

Phase 1D removes local-only test/cache folders from the repository workspace,
strengthens ignore rules, and aligns GitHub Actions with the local validation
order used before production benchmark work.

## Cleanup Scope

The cleanup targets only local tool and temporary run folders:

- `pytest_run_tmp_*`
- `pytest_tmp`
- `.pytest_cache`
- `.ruff_cache`
- `.mypy_cache`
- `.tmp`
- `.tmp_pytest`

The cleanup does not remove `results/`, `data/`, `docs/`, `configs/`, `src/`,
`tests/`, or promoted project artifacts.

## CI Validation Coverage

CI now runs the same ordered gates used locally:

1. targeted config validation tests;
2. repository hygiene tests;
3. CI workflow audit tests;
4. `mypy src tests`;
5. full `pytest`;
6. `ruff check .`;
7. `ruff format --check .`;
8. public-content and secret audit;
9. `inference-bench doctor`;
10. `inference-bench validate-config`.

`validate-config` now loads the model registry, runtime registry, load
profiles, optimization negative rules, SLO targets, SLO profiles, and a sample
unified result-track schema row.

## Guardrail Result

This block is a repository-readiness block. It does not run inference, touch
gold data, change promoted retrieval, or authorize larger benchmark runs.
