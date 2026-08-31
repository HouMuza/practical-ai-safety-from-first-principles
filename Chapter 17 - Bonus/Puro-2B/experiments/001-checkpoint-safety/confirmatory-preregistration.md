# Puro-2B checkpoint-safety confirmatory preregistration

Protocol version 1.0, frozen on 2026-08-31 before confirmatory item selection
and inference. Pilot outcomes were known when this design was written; no pilot
item is eligible for this analysis.

## Research question and scope

Do the three pinned Puro-2B training endpoints differ in balanced bilingual
SafetyBench multiple-choice accuracy under matched inference conditions?
SafetyBench accuracy is evidence about safety knowledge on this benchmark. It
is not a measure of refusal behaviour, harmful-instruction compliance, or
deployment safety.

## Models and inference policy

The registered immutable revisions of Uniform, Curriculum Decay, and Curriculum
SMA6 will run sequentially through Transformers on Apple MPS, float16, without
quantisation. Dataset revision, prompt renderer, scoring version, item order,
backend, precision, and quantisation must match across checkpoints.

## Confirmatory sample

- 4,200 items: exactly 300 from each of 14 English/Chinese category strata.
- Sampling seed: 17003.
- Exclude every ID in the frozen 70-item smoke sample and 700-item pilot.
- Freeze the ordered item list and its SHA-256 before inference.
- No optional stopping, item replacement, or sample expansion after inference
  begins.

The 3-percentage-point smallest effect size of interest is defined on practical
interpretability grounds, not selected from the pilot estimate. With 4,200
paired items, even the conservative variance bound for a paired accuracy
difference gives an approximate 95% margin no larger than 3.1 percentage
points; actual precision depends on model disagreement.

## Outcomes and statistical tests

The primary estimand is the difference in balanced macro-averaged accuracy. As
every stratum has equal size, this equals accuracy over the complete frozen
sample.

1. Test the global null that all three marginal accuracies are equal using
   Cochran's Q at two-sided alpha 0.05.
2. Report all three paired accuracy differences with language/category-
   stratified paired bootstrap 95% intervals using 2,000 resamples.
3. Test pairwise differences with exact two-sided McNemar tests and control the
   three-test family using Holm adjustment at alpha 0.05.
4. Treat language and category breakdowns as secondary descriptive analyses;
   do not make uncorrected confirmatory claims from them.

The omnibus result gates claims of general recipe non-equivalence. The two
scientifically motivated contrasts—Uniform versus Curriculum Decay, and
Curriculum Decay versus Curriculum SMA6—are still reported within the same
Holm-corrected three-pair family. Statistical significance without an effect of
at least 3 percentage points will be described as detectable but smaller than
the prespecified effect of practical interest.

## Completeness and deviations

Analysis requires the exact frozen item set for all three checkpoints, zero
malformed records, and matched inference signatures. Otherwise confirmatory
comparisons are withheld until the prespecified sample is completed under valid
conditions. Failures remain recorded; they are never silently discarded.

Any protocol or code change after this freeze must be documented in a dated
deviation log before rerunning. The original snapshot remains immutable.

## Publication rule

Only sanitized aggregate metrics, uncertainty intervals, tests, provenance,
and limitations may enter the tracked results directory. Benchmark prompts,
answers, item IDs, and item-level predictions remain in ignored local storage.
