# Production Workload Profiles

Status: implemented June 19, 2026

Phase 1C adds production load metadata in `configs/load_profiles.yaml` and
`src/inference_bench/load_profiles.py`.

## Required Report Fields

Every production benchmark report must carry:

- input token distribution;
- output token distribution;
- traffic profile;
- concurrency;
- request arrival mode.

Result rows and generation traces reserve traffic profile, arrival mode,
concurrency, input sequence bucket, and output sequence bucket fields.

## ISL And OSL Buckets

Input sequence length buckets:

- `isl_0_512`
- `isl_512_1024`
- `isl_1024_2048`
- `isl_2048_4096`
- `isl_4096_8192`
- `isl_8192_plus`

Output sequence length buckets:

- `osl_0_64`
- `osl_64_128`
- `osl_128_256`
- `osl_256_512`
- `osl_512_1024`
- `osl_1024_plus`

## Traffic Profiles

| Profile | Default concurrency | Arrival mode | Purpose |
| --- | ---: | --- | --- |
| `online_low_latency` | 1 | `jittered_poisson` | Low-latency interactive traffic |
| `office_hours_bursty` | 4 | `bursty_jittered` | Short daytime bursts |
| `offline_throughput` | 8 | `closed_loop` | Batch throughput |
| `custom` | 1 | `custom` | Caller-defined profile |

Jittered arrivals are deterministic by seed. They are metadata and simulation
inputs, not evidence that a live server handled that traffic shape.
