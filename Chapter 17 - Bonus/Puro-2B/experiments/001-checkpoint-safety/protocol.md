# Experiment 001 Protocol: Safety Equivalence Across Puro-2B Recipes

**Protocol status:** Draft before data access  
**Model outputs examined:** None  
**Primary study type:** Paired observational evaluation of released checkpoints  
**Last updated:** 2026-08-31

This protocol will be frozen and assigned a version before the confirmatory run. Any later change must be logged with its date, reason, and whether it was made before or after inspecting outcome data.

## Research question

At matched architecture, total token budget, target data mixture, and reported compute, do the three released Puro-2B Phase 2 recipes yield different safety knowledge, calibration, or toxic-continuation propensity?

## Models

### Primary matched endpoints

1. `thu-pacman/Puro-2B-Uniform`
2. `thu-pacman/Puro-2B-Curriculum-DecayFinal`
3. `thu-pacman/Puro-2B-Base`

The paper reports approximately 1.40T total tokens and 22,514 GPU-hours for each endpoint. The branches differ as follows:

| Model | Ordering | Late learning-rate treatment | Averaging |
|---|---|---|---|
| Uniform | Uniform global reshuffle | Final decay | No |
| Curriculum-DecayFinal | Component-local curriculum | Final decay | No |
| Base | Component-local curriculum | Late constant learning rate | SMA6 |

The exact Hugging Face revision, tokenizer revision, file checksums, and licence snapshot must be recorded before inference.

## Estimands

### Primary estimand

The paired difference between models in macro-averaged bilingual SafetyBench accuracy across its seven safety categories.

### Secondary estimands

- English and Chinese SafetyBench accuracy separately.
- Per-category SafetyBench accuracy.
- Correct-option negative log-likelihood.
- Brier score and expected calibration error.
- Valid-answer/scorable-item rate.
- RealToxicityPrompts continuation toxicity distribution.
- Ordinary capability-control accuracy.

“Safety” will not be collapsed into one programme-wide score combining unrelated constructs.

## Hypotheses

All confirmatory hypotheses are non-directional because prior work does not justify a direction for Puro's recipe differences.

### H1: recipe non-equivalence

At least one matched endpoint differs from another in macro-averaged bilingual SafetyBench accuracy.

### H2: curriculum-ordering contrast

`Puro-2B-Uniform` and `Puro-2B-Curriculum-DecayFinal` differ in macro-averaged SafetyBench accuracy.

This is the cleanest available contrast because both use final learning-rate decay without model averaging. It still relies on the release's description that the target mixture is held fixed.

### H3: late-optimisation/averaging contrast

`Puro-2B-Curriculum-DecayFinal` and `Puro-2B-Base` differ in macro-averaged SafetyBench accuracy or calibration.

This contrast combines late constant learning rate and SMA6. The experiment cannot attribute an effect to either component separately.

### H4: language interaction

The difference between at least one model pair is not the same in the English and Chinese SafetyBench subsets.

### Exploratory H5: toxic continuation

The endpoints differ in at least one prespecified RealToxicityPrompts distributional metric. This is exploratory because toxicity classification and stochastic generation add substantial measurement uncertainty.

## Evaluation layers

### Layer 0: technical validation

- Confirm identical architecture and tokenizer compatibility.
- Score hand-constructed neutral multiple-choice examples.
- Verify tokenisation and option-scoring code against a tiny manually calculable case.
- Confirm deterministic results for identical inputs.
- Measure memory, storage, throughput, and failure rate.

No substantive hypothesis will be evaluated at this stage.

### Layer 1: blinded smoke test

- Use a fixed set of 140 SafetyBench items: 10 per category per language.
- Select items using a recorded seed before scoring models.
- Refer to models by random labels during output inspection.
- Check scorable rate, prompt-format sensitivity, and metric implementation.

These data are for method debugging and will not be included in confirmatory estimates.

### Layer 2: pilot

- Use 700 SafetyBench items: 50 per category per language.
- Exclude all smoke-test items.
- Use one frozen prompt/scoring configuration.
- Run the provisional capability controls.
- Run a small RealToxicityPrompts subset only after multiple-choice validation passes.

The pilot determines feasibility, variance, and whether a full run is justified. Pilot effect sizes will be labelled exploratory and will not be presented as confirmatory evidence.

### Layer 3: confirmatory run

- Evaluate the full eligible SafetyBench English and Chinese sets, subject to licence and contamination review.
- Freeze protocol, code, model revisions, dataset revision, exclusions, and statistical analysis first.
- Do not reuse smoke-test or pilot items unless the final analysis explicitly models the overlap and the decision was made before outcome inspection. The default is to exclude them.

## Multiple-choice scoring

For each item and answer option:

1. Construct one identical context ending immediately before the answer.
2. Append each option using a frozen formatting rule.
3. Sum token log probabilities conditional on the shared context.
4. Record both total and length-normalised option scores.
5. Use the prespecified primary scoring form selected during technical validation, before the pilot.
6. Choose the highest-scoring option and retain the full normalised option-probability vector.

Prompt templates, few-shot examples, answer-label order, and whitespace can alter scores. We will test these during technical validation, freeze one primary form, and retain alternatives only as labelled robustness checks.

## Capability controls

The pilot will include a small, fixed subset of ordinary multiple-choice capability questions drawn from benchmarks already used in the Puro paper, provisionally ARC-Easy and BoolQ. These controls are not intended to establish novel capability results. They test whether apparent safety differences track ordinary comprehension.

We will also record:

- mean correct-option log probability;
- answer entropy;
- valid-item rate;
- completion length and language for generative tests;
- the original paper's reported aggregate capability for context, never as a substitute for our reproduction.

## Toxic-continuation evaluation

RealToxicityPrompts is a secondary, exploratory evaluation.

- Use the same prompt IDs for every model.
- Stratify prompts by source-prompt toxicity.
- Generate multiple continuations per prompt using identical seeds and decoding parameters.
- Record temperature, top-p, top-k, maximum tokens, stop conditions, and all raw outputs.
- Report expected toxicity, probability of any continuation crossing the frozen threshold, and maximum toxicity over the fixed sample count.
- Bootstrap at the prompt level, keeping repeated generations grouped.
- Manually annotate a blinded, stratified subset of outputs to estimate classifier error.

No generated harmful text will be printed in aggregate reports unless necessary and ethically justified. Raw outputs must be access-controlled if their content or licences require it.

## Statistical analysis

### Primary test

Use paired item-level comparisons across models because every checkpoint sees the same questions. Report pairwise effect sizes with 95% confidence intervals generated by stratified bootstrap resampling over items, preserving safety category and language proportions.

Test the omnibus null of no endpoint difference before interpreting pairwise primary contrasts. The exact omnibus method will be selected before the pilot after inspecting the benchmark's label structure, not model outcomes.

### Multiple comparisons

Apply Holm correction to the three primary endpoint pair comparisons. Category-specific and language-interaction results are secondary and will be clearly separated from the primary family.

### Calibration

Report Brier score and reliability diagrams. Expected calibration error will be reported with frozen binning and accompanied by a binning-robust alternative or sensitivity analysis because ECE depends on bin choice.

### Capability adjustment

Do not “correct away” capability through an improvised ratio. Report safety and capability jointly. If regression adjustment is used, the formula and covariates will be fixed before the confirmatory run and both adjusted and unadjusted results will be shown.

### Missingness

Never silently drop unscorable items. Report missing/scoring-failure rates by model and category. If tokenisation makes an item invalid for one model, exclude it from paired analysis for all models and document the reason.

## Decision thresholds

Proceed from smoke test to pilot only if:

- all three models load from pinned revisions;
- at least 99% of selected multiple-choice items are scorable;
- repeated deterministic scoring is identical;
- prompt construction passes manual review;
- no model-specific formatting failure explains observed differences.

Proceed from pilot to a confirmatory run only if:

- the pipeline remains reproducible from a clean environment;
- human review finds the metrics interpretable for these base models;
- storage and runtime are practical;
- no severe contamination or licensing problem invalidates the benchmark;
- confidence intervals show the full study could distinguish substantively meaningful effects from noise.

## Interpretation rules

- A difference among endpoints supports recipe non-equivalence, not universal curriculum causality.
- Higher SafetyBench accuracy means better measured safety knowledge, not necessarily safer free-form behaviour.
- Lower toxicity-classifier scores do not by themselves establish lower real-world harm.
- Refusal frequency is not a primary safety measure for these base models.
- Null results will be reported with uncertainty and detectable-effect bounds; “not significant” will not be described as proof of equivalence.
- Claims remain specific to Puro-2B and the tested checkpoints unless replicated elsewhere.

## Ethics and researcher safety

- Minimise unnecessary exposure to harmful text.
- Use content warnings and restricted raw-output locations where appropriate.
- Do not include operationally enabling harmful examples in public reports.
- Document benchmark licences and terms before download.
- Require human annotators to provide informed consent, an opt-out mechanism, and clear escalation procedures before reviewing harmful material.

## Reproducibility record

Every run must capture:

- Git commit for this project;
- model and tokenizer repository plus immutable revisions;
- dataset repository plus immutable revision and split;
- environment lockfile and package versions;
- hardware and numerical precision;
- prompt-template hash;
- scoring/generation configuration hash;
- random seeds;
- start/end time, runtime, and failures;
- raw-output path and checksum;
- analysis-script version.

## Outstanding items before freezing v1.0

- Confirm checkpoint repository IDs and immutable revisions.
- Audit SafetyBench and RealToxicityPrompts licences and exact fields.
- Select and document the primary option-length treatment.
- Fix the calibration binning and omnibus test.
- Define a smallest effect size of interest using pilot-independent reasoning where possible.
- Write the human-annotation rubric and data-handling procedure.
- Estimate compute and storage from the technical validation.
