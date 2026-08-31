# Focused Literature Review: Safety Across Puro-2B Pretraining

**Status:** Design-oriented first pass  
**Search date:** 2026-08-31  
**Scope:** Work that changes how Experiment 001 should measure safety across base-model pretraining checkpoints

## Review question

What methods and evidence should guide a controlled comparison of safety-relevant behaviour across Puro-2B checkpoints trained with different Phase 2 recipes?

## Search approach

The first pass prioritised original papers and official benchmark repositories found through combinations of:

- pretraining checkpoint safety and trustworthiness dynamics;
- toxicity, privacy, fairness, and memorisation during pretraining;
- training trajectories and capability confounding;
- safety pretraining and pretraining-data interventions;
- base-model safety evaluation;
- bilingual safety benchmarks;
- automatic safety classifiers and LLM-judge reliability.

This is a targeted design review, not yet a systematic review. Citation chaining, formal inclusion/exclusion criteria, duplicate screening, and a reproducible search log remain future work if the pilot supports publication.

## Evidence table

| Work | Design and evidence | Result relevant to Puro | Limitation for our use | Design implication |
|---|---|---|---|---|
| Qian et al. (2024), *Towards Tracing Trustworthiness Dynamics* | Studies intermediate pretraining checkpoints across reliability, privacy, toxicity, fairness, and robustness using probes, steering vectors, and mutual-information analysis | Trustworthiness-related representations may become detectable early, and their trajectories need not be monotonic | Representation probes do not directly establish harmful generation or deployed risk | Behavioural evaluation can be paired later with representation analysis, but probes should not be our first outcome |
| Xia et al. (2022), *Training Trajectories of Language Models Across Scales* | Analyses intermediate OPT checkpoints from 125M to 175B on token prediction, generation, and tasks | Perplexity can predict behaviour better than parameter count or raw compute; undesirable-looking phenomena may change with training progress | Not a dedicated safety study and does not test curriculum ordering | Record validation loss or matched capability metrics and include them as covariates/stratification variables |
| Biderman et al. (2023), *Emergent and Predictable Memorization* | Measures memorisation across Pythia model sizes and intermediate checkpoints | Memorisation can be studied longitudinally and partially forecast from earlier checkpoints | Requires access to known training sequences; broad extraction tests can create privacy and redistribution concerns | Keep memorisation as a separate experiment unless Puro manifests permit a tightly governed canary/known-sequence design |
| Longpre et al. (2023), *A Pretrainer's Guide to Training Data* | Trains 28 matched 1.5B models with different data age, domain, quality, and toxicity filtering | Data filtering changes toxic-generation risk and can trade off with general capability; data effects are not reliably inferred from domain labels alone | Studies data composition/filtering, not the order of an otherwise matched mixture | Do not infer safety from Puro's data labels alone; measure outputs and control for capability |
| Prabhumoye et al. (2023), *Adding Instructions during Pretraining* | Adds toxicity metadata or instructions during pretraining and evaluates toxicity and utility | Pretraining interventions can reduce generated toxicity without necessarily reducing standard task performance | Deliberately changes training examples; Puro's curriculum was not designed as a safety intervention | Supports the broader premise that pretraining choices affect safety, but not our specific causal claim |
| Maini et al. (2025), *Safety Pretraining* | Tests filtering, synthetic safety data, refusal-style web data, and harmfulness tags before instruction tuning | Base-model safety can be measured, and safety-oriented pretraining can substantially change harmful-generation outcomes | A designed safety intervention differs greatly from observational comparison of Puro recipes | Use its base-model evaluation framing, while avoiding claims that ordinary curriculum equals safety pretraining |
| Gehman et al. (2020), *RealToxicityPrompts* | Prompted continuation benchmark for toxic degeneration with distributional generation metrics | Toxicity is naturally measurable for a base autoregressive model using continuations rather than chat refusals | Classifier scores can encode demographic/dialect bias; repeated stochastic generation is relatively expensive | Suitable secondary generation measure if classifier limitations and bootstrap uncertainty are reported |
| Zhang et al. (2024), *SafetyBench* | 11,435 English and Chinese multiple-choice safety questions across seven categories | Offers low-cost bilingual safety-understanding measurement and reports correlation with safety generation | Safety knowledge is not identical to safe behaviour; possible pretraining contamination must be considered | Strong candidate for the primary pilot because likelihood-based MCQ evaluation suits base models and Puro is English/Chinese |
| Wang et al. (2024), *Do-Not-Answer* | Harmful instructions with automatic response classification; compares smaller classifiers with GPT-4 evaluation | Provides open harmful prompts and lower-cost response scoring | Primarily evaluates safeguards/refusal in assistant-like models; raw base models lack that behavioural contract | Use only as secondary exploratory generation, not the primary endpoint; manually validate a stratified sample |
| Lin et al. (2021), *TruthfulQA* | Tests generated and multiple-choice answers to misconception-sensitive questions | More training or larger scale does not guarantee greater truthfulness | Covers truthfulness rather than harmful compliance; some questions are temporally/culturally sensitive | Multiple-choice form is suitable as a secondary likelihood-based outcome with versioned data and contamination caveats |
| Chen et al. (2022), *A Close Look into Calibration of Pre-trained Language Models* | Controlled study of confidence and calibration over training | Confidence may rise during training even when predictions are wrong | Experimental setting differs from autoregressive checkpoint safety evaluation | Report proper scoring/calibration measures, not accuracy alone, for multiple-choice outcomes |
| Chen et al. (2024), *Humans or LLMs as the Judge?* | Tests human and LLM evaluator sensitivity to several biases and perturbations | Even strong LLM judges can be systematically biased and attacked | Not specific to safety classification or Puro | Do not treat one LLM judge as ground truth; validate against humans and rule/classifier alternatives |

## Synthesis

### 1. The research question is supported, but the causal claim must stay narrow

Prior work establishes that trustworthiness signals change across pretraining and that interventions to data filtering or presentation can change toxicity and harmful generation. It does not establish that Puro's curriculum ordering produces a particular safety effect. The released Puro endpoints therefore offer a useful new controlled case.

However, `Puro-2B-Base` changes curriculum ordering, late learning-rate treatment, and checkpoint averaging together. Experiment 001 can test whether the **released recipes are safety-equivalent**. It cannot initially identify curriculum ordering as the sole cause.

### 2. Base models require different evaluation semantics from chat models

The Puro checkpoints are base autoregressive models. A refusal-oriented benchmark assumes that a model has been trained to interpret a user instruction and obey a safety policy. That assumption is weak here.

Accordingly:

- failure to refuse must not automatically be called unsafe;
- chat templates and system safety prompts should not be imposed unless the official model format supports them;
- primary measures should rely on conditional likelihood, multiple choice, or natural text continuation;
- open-ended instruction following should be labelled exploratory;
- the study should distinguish **safety knowledge**, **toxic continuation propensity**, and **assistant safeguard behaviour**.

### 3. Capability is a central confounder

An earlier checkpoint may appear safer because it cannot understand a harmful request, cannot sustain a coherent response, or produces shorter text. Conversely, a later checkpoint may score better on safety knowledge simply because it is more capable.

For every safety outcome, we should record at least one matched capability measure and response-quality indicators such as valid-answer rate, completion length, and language. Analysis should compare endpoints at the same token budget first and model safety jointly with capability.

### 4. Closed-form and open-ended measurements answer different questions

Multiple-choice scoring is cheap, deterministic, and appropriate for base models, but primarily measures recognition or stated tendency. Free-form continuation better reflects generative risk, but introduces sampling variance and subjective scoring.

The defensible pilot therefore uses two layers:

1. **Primary:** bilingual SafetyBench likelihood-based scoring, including confidence and calibration where labels permit.
2. **Secondary:** a small RealToxicityPrompts continuation sample with repeated generations and distributional toxicity statistics.

TruthfulQA multiple choice is a useful secondary outcome. Do-Not-Answer should wait until we have verified that its prompt-response setup yields interpretable outputs from Puro base checkpoints.

### 5. Automatic judges need independent validation

No LLM judge or toxicity classifier should be treated as an oracle. The pilot should prefer exact multiple-choice scoring where possible. Any generated-response scoring should include:

- a fixed, versioned evaluator;
- a written rubric;
- blinded human annotation of a stratified sample;
- inter-annotator agreement;
- sensitivity to alternative thresholds or evaluators;
- examples of false positives and false negatives;
- uncertainty intervals that include generation variability.

## Gap and proposed contribution

The closest checkpoint-dynamics work studies broad trustworthiness representations or ordinary training progression. The closest data-centric safety work intentionally changes filtering or injects safety data. The Puro release enables a different comparison:

> At a matched model architecture, token budget, data-mixture target, and reported compute budget, do released Phase 2 ordering and late-optimisation recipes yield different bilingual safety knowledge, calibration, or toxic-continuation propensity?

The initial contribution is therefore a controlled behavioural comparison of open pretraining recipe endpoints, not a claim that we have discovered a universal law of curriculum safety.

## Decisions for Experiment 001

### Include in the pilot

- Matched 1.4T-token endpoints as the primary model comparison.
- SafetyBench English and Chinese subsets as the primary outcome.
- Answer log probabilities, accuracy, valid-answer rate, negative log-likelihood, Brier score, and expected calibration error where technically appropriate.
- A small ordinary capability control using the same evaluation mechanism.
- A small RealToxicityPrompts generation study as a secondary outcome.
- Manual validation for generated-response classifications.

### Defer

- Memorisation/privacy extraction, because it requires separate governance and known training examples.
- Cybersecurity capability, because safe handling and specialised evaluation infrastructure make it a separate study.
- Deception and sycophancy, because constructs and prompting assumptions are difficult to validate in a 2B base model.
- Post-training persistence, until base-checkpoint differences are established.
- Claims about data-component causality, until manifests and ordering are audited.

## Remaining literature tasks before preregistration

- Inspect SafetyBench's precise scoring and licence terms.
- Inspect RealToxicityPrompts redistribution and Perspective API dependencies or validated open alternatives.
- Identify a capability-control subset with minimal overlap with Puro's reported benchmark suite.
- Search specifically for model-averaging effects on safety and calibration.
- Search for data-order/curriculum effects under fixed data mixtures, including negative results.
- Define human annotation guidance and ethical handling for harmful generated text.

## Provisional conclusion

Experiment 001 is feasible and appears differentiated, but it should be narrower than the original programme. Its primary claim should concern safety equivalence among released Puro recipes. The initial pilot should prioritise bilingual, likelihood-based safety understanding and calibration, with toxicity generation as a secondary behavioural measure. This design is better matched to raw base models and reduces dependence on unreliable refusal semantics and LLM judges.
