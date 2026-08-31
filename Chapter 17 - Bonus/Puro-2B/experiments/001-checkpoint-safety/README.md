# Experiment 001: Checkpoint Safety

## Status

Pilot complete; confirmatory study not yet run.

## Source attribution

This is an independent safety evaluation of Puro-2B, created by the PACMAN Group at Tsinghua University. It is not an official Puro-2B experiment and is not affiliated with or endorsed by the model's creators. Full creator credit, primary sources, licence notes, and citations are maintained in [`../../ATTRIBUTION.md`](../../ATTRIBUTION.md).

## Research question

Do Puro-2B models trained with different Phase 2 curricula develop materially different safety profiles despite having similar capability?

## Scope of the pilot

Compare a small, deliberately selected set of Puro checkpoints using identical inference and evaluation settings. Begin with a technical smoke test, then expand only if the measurements are reliable.

The initial matched endpoint set is:

1. `Puro-2B-Uniform`
2. `Puro-2B-Curriculum-DecayFinal`
3. `Puro-2B-Base`

All three are reported at approximately 1.40T total training tokens and 22,514 GPU-hours. They provide the cleanest released comparison of uniform ordering, curriculum with final decay, and curriculum with late constant learning rate plus SMA6. This is not a perfect isolation of curriculum because late optimisation and model averaging also differ. See [`../../model-inventory.md`](../../model-inventory.md).

## Hypotheses

The draft hypotheses and analysis rules are defined in [`protocol.md`](protocol.md). They remain subject to benchmark/licence inspection and technical validation, but must be frozen before pilot outcomes are examined.

## Planned safety dimensions

- Bilingual safety knowledge and tendencies through likelihood-based multiple-choice scoring
- Calibration of safety-relevant answers
- Toxic-continuation propensity as a secondary generative measure
- Truthfulness as a secondary likelihood-based measure if resources permit

Refusal or harmful-instruction compliance is not a primary outcome because these are base models without an established assistant-safety contract.

## Required controls

- Exact model and dataset revisions
- Identical prompt templates and chat formatting
- Identical decoding parameters and output limits
- Recorded seeds and software versions
- Capability controls to distinguish safety differences from comprehension differences
- Human validation of a sample of automated scores

## Exit criterion

Proceed to a larger study only if observed differences are measurable, reproducible, and not readily explained by evaluation artefacts or capability differences.
# Experiment configuration

[`experiment.json`](experiment.json) is the machine-portable execution request
for this study. It identifies the three registered model aliases, SafetyBench,
the fixed seed, automatic execution-profile selection, and the append-only runs
directory. The shared pipeline resolves research runs. The older neutral smoke
script remains only as a technical-validation diagnostic.

Preview a small plan without downloading weights:

```bash
cd ../../../safety-evaluation-pipeline
PYTHONPATH=src python3 -m safety_eval plan \
  ../Puro-2B/experiments/001-checkpoint-safety/experiment.json \
  --profile smoke_cpu --max-items 3
```

## Frozen blinded smoke sample

`samples/blinded-smoke-70.json` freezes 70 item IDs before further outcomes are
generated: five items from each of the seven SafetyBench categories in English
and Chinese. Its item-ID hash is
`f9c8deb1bb43b5647fdbbb05bbf9904c7abf510eb3ca3831d6356f2f53f98746`.
This stage validates matched three-checkpoint execution and is not a reportable
safety outcome.

## Frozen pilot sample

[`pilot-preregistration.md`](pilot-preregistration.md) was written before pilot
outcomes were examined. `samples/pilot-700.json` freezes 700 item IDs: 50 from
each of the seven SafetyBench categories in English and Chinese. It excludes all
70 smoke-test items and has item-ID hash
`a543a785cf4f8a6c3d6b9afcc211a9254632d42dc03a603c8d134236ead2c519`.

All three checkpoints completed the same 700 items under matched MPS, float16,
unquantized inference. The sanitized preliminary results are stored under
[`../../results/001-checkpoint-safety/pilot-700/`](../../results/001-checkpoint-safety/pilot-700/).
This pilot may guide a separately preregistered confirmatory study, but it is not
final evidence.
