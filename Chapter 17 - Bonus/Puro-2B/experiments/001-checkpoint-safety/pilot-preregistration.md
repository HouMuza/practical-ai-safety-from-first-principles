# Puro-2B checkpoint-safety pilot preregistration

Frozen before pilot inference on 2026-08-31.

## Purpose

Estimate effect sizes, disagreement rates, uncertainty, runtime, and
missingness for a later confirmatory design. Pilot results are preliminary and
must not be described as confirmatory evidence.

## Models and inference policy

The three pinned Puro-2B endpoints—Uniform, Curriculum Decay, and Curriculum
SMA6—will be evaluated sequentially with Transformers on Apple MPS, float16,
no quantisation, identical prompt/scoring versions, and identical item order.

## Sample

- 700 SafetyBench items: 50 per each of 14 English/Chinese category strata.
- Seed: 17002.
- The frozen 70-item blinded-smoke sample is excluded.
- The ordered item-ID list and its SHA-256 are frozen before inference.

## Estimands

- Overall accuracy for each checkpoint with 95% Wilson intervals.
- Accuracy by language and category with 95% Wilson intervals.
- Pairwise accuracy differences on identical items with paired bootstrap 95%
  intervals.
- Pairwise discordance with exact McNemar tests and Holm adjustment across the
  three preregistered checkpoint comparisons.
- Missingness and runtime failures for every checkpoint.

## Decision rules

Paired comparisons are withheld unless every checkpoint completes the exact
frozen sample under matched backend, precision, quantisation, dataset, prompt,
and scoring versions. The pilot will inform confirmatory sample sizing and will
not itself be marked as a publishable final outcome.
