# Pipeline Architecture

## Data flow

```text
model manifest ─┐
                ├─> validated run config ─> runner ─> immutable JSONL records
benchmark spec ─┘                          │
                                           ├─> metrics + confidence intervals
environment/provenance ───────────────────┘
```

## Core abstractions

- **Model manifest:** identity, loading, licence, checkpoints, and valid comparison groups.
- **Model adapter:** tokenisation, forward scoring, and generation. Chat formatting is explicit policy, never inferred merely from the presence of a template.
- **Benchmark adapter:** versioned items, prompt rendering, label isolation, and valid metrics.
- **Runner:** batching, stable item IDs, failure recording, resumability, and append-only raw records.
- **Metrics:** point estimates, paired comparisons, calibration, uncertainty, missingness, and multiplicity correction.
- **Provenance:** code, model, dataset, environment, hardware, prompt, configuration, seed, timing, and output hashes.

## Portability contract

The scientific record is portable JSON/JSONL. A local database may index these
files, but it is never the sole copy of an outcome.

Execution profiles describe evidence scope rather than pretending every
machine can run the same workload:

- `smoke_cpu` validates plumbing on a very small sample.
- `sampled_accelerator` produces a versioned, stratified estimate.
- `full_accelerator` runs every licensed item on suitable hardware.

Automatic recommendations are conservative and may be overridden explicitly.
The selected profile, detected hardware, precision, quantisation, item IDs, and
failures must accompany every run. Reports must label smoke, sampled, and full
runs distinctly.

`ModelAdapter` and `SafetyCheck` protocols define the extension boundary. A new
model backend or safety check implements those interfaces; it does not modify
the runner or reporting format.

## Scaling to a new release

If a Nemotron family with intermediate checkpoints is released:

1. Add `Nemotron/configs/models.json` with pinned checkpoint relationships.
2. Add a family adapter only if standard loading is insufficient.
3. Reference existing benchmark specifications.
4. Create a Nemotron-specific protocol defining valid comparisons.
5. Store its outputs under `Nemotron/experiments/...`.

No Puro source file should need to be copied or edited.
