# Puro-2B Artefact Inventory

**Status:** Initial official-source inventory  
**Inventory date:** 2026-08-31  
**Primary paper version:** arXiv:2608.27370v1 (2026-08-27)

This inventory records what exists before Experiment 001 is finalised. Repository revisions and file checksums will be added when artefacts are downloaded. Facts below are taken from the original paper and official PACMAN Group releases; interpretations for our study are explicitly labelled.

## Official release surfaces

| Artefact | Official location | Reported release terms | Relevance |
|---|---|---|---|
| Paper | <https://arxiv.org/abs/2608.27370> | Paper page identifies CC BY-SA 4.0 | Defines the recipe, comparisons, limitations, and reported results |
| Model collection | <https://huggingface.co/collections/thu-pacman/puro-2b> | Authors report Apache-2.0 unless separately noted | Released endpoints and intermediate checkpoints |
| Canonical base model | <https://huggingface.co/thu-pacman/Puro-2B-Base> | Model card identifies Apache-2.0 | Curriculum/CMA endpoint |
| Dataset and manifests | <https://huggingface.co/datasets/thu-pacman/Puro-2B> | Component-specific upstream terms may differ | Data provenance, ordering, and contamination research |
| Training code | <https://github.com/thu-pacman/Puro-Megatron> | Licence must be captured at the exact commit used | Reproduction and continued-pretraining studies |
| Data-processing code | <https://github.com/thu-pacman/Kaiyuan-Spark> | Licence must be captured at the exact commit used | Reconstructing preprocessing and data shards |

The paper states that model weights, intermediate checkpoints, configurations, data manifests, and implementation artefacts are released. It also warns that different licences can apply to individual dataset components, including non-permissive terms.

## Training structure

- Architecture: shared dense 2B-parameter architecture across the reported Puro checkpoints.
- Phase 1: 438.84B tokens, shared by the compared Phase 2 branches.
- Phase 2: up to 959.99B tokens.
- Full schedule: approximately 1.40T tokens.
- Precision: blockwise FP8 training.
- Optimisation: Muon with Hyperball (MuonH) as part of the recipe.
- Hardware reported for production runs: 24 RTX 5090 GPUs in Phase 1 and 96 in Phase 2.

## Released checkpoint ledger

The following values reproduce Table 15 of the original paper. “Cost” is the authors' reproduction-cost accounting and is not a price we independently verified.

| Checkpoint ID | Phase 2 tokens | Total tokens | Phase 2 ordering | Averaging | Reported GPU-h | Reported cost |
|---|---:|---:|---|---|---:|---:|
| `Puro-2B-Base-Phase1` | 0 | 438.84B | Not applicable | No | Not separately listed in Table 15 | Not separately listed |
| `Puro-2B-Uniform-Phase2-1of16` | 60.00B | 498.83B | Uniform global reshuffle | No | 7,041 | $2.16K |
| `Puro-2B-Uniform-Phase2-1of8` | 120.00B | 558.83B | Uniform global reshuffle | No | 8,073 | $2.47K |
| `Puro-2B-Uniform-Phase2-1of4` | 240.00B | 678.83B | Uniform global reshuffle | No | 10,136 | $3.10K |
| `Puro-2B-Uniform-Phase2-1of2` | 480.00B | 918.83B | Uniform global reshuffle | No | 14,262 | $4.37K |
| `Puro-2B-Uniform` | 959.99B | 1.40T | Uniform global reshuffle | No | 22,514 | $6.89K |
| `Puro-2B-Curriculum-DecayFinal` | 959.99B | 1.40T | Component-local curriculum; final decay | No | 22,514 | $6.89K |
| `Puro-2B-Base` | 959.99B | 1.40T | Curriculum plus late constant learning rate | SMA6 | 22,514 | $6.89K |

The official collection also exposes `Puro-2B-Curriculum-SMA6-Inputs`, an auxiliary artefact associated with the six-checkpoint average. Its exact contents and whether every constituent is directly loadable must be verified before it is included in an experiment.

## What the original evaluation covers

The paper reports capability rather than a substantive safety evaluation. Its main evaluation spans 15 mathematics, code, reasoning, and knowledge benchmarks:

- GSM8K
- MATH
- sanitized-MBPP
- HumanEval
- MMLU
- MMLU-Pro
- ARC-Challenge
- ARC-Easy
- BoolQ
- CommonsenseQA
- HellaSwag
- PIQA
- Social IQa
- WinoGrande
- BIG-Bench Hard

The authors also report matched post-training comparisons intended to test whether curriculum-related capability differences persist after supervised adaptation. This makes safety evaluation across the corresponding base checkpoints a direct extension rather than a repetition of their reported work.

## Authors' stated limitations relevant to us

- Corpus-wide exact and near-duplicate contamination was not fully audited.
- Processed-data provenance and component-specific licensing require care.
- The architecture and scale limit generalisation beyond this model family and regime.
- The model is heavily trained relative to its parameter count, so conclusions may be specific to an overtrained dense regime.

These limitations motivate later studies but should not all be folded into Experiment 001.

## Proposed Experiment 001 model sets

### Pilot A: matched endpoint comparison

Use these three endpoints first:

1. `Puro-2B-Uniform`
2. `Puro-2B-Curriculum-DecayFinal`
3. `Puro-2B-Base`

**Reasoning:** all three report the same total token budget and GPU-hours. They differ in data ordering and late-stage optimisation/averaging. This is the cleanest available safety comparison, although curriculum, learning-rate treatment, and averaging are not fully isolated from one another.

### Pilot B: training-progress control

If storage and inference cost permit, add:

1. `Puro-2B-Base-Phase1`
2. `Puro-2B-Uniform-Phase2-1of16`
3. `Puro-2B-Uniform-Phase2-1of8`
4. `Puro-2B-Uniform-Phase2-1of4`
5. `Puro-2B-Uniform-Phase2-1of2`
6. `Puro-2B-Uniform`

**Reasoning:** this provides a monotonic token-budget trajectory under uniform Phase 2 ordering. It can show whether a safety metric changes with general training progress, which helps interpret endpoint differences.

## Threats to causal interpretation

- `Puro-2B-Base` combines curriculum ordering, a late constant learning rate, and SMA6; it does not isolate curriculum alone.
- Base-model responses may be incomplete or unstable under instruction-style safety prompts because the checkpoints are not necessarily instruction tuned.
- Apparent safety gains may actually reflect weaker comprehension, shorter outputs, or lower capability.
- Automated judges may favour certain response styles or fail on Chinese-language outputs.
- A single deterministic generation per prompt estimates one decoding policy, not the model's full behavioural distribution.
- Exact checkpoint revisions may change unless immutable Hugging Face commits are pinned.

## Verification tasks before inference

- Record the immutable revision, file list, weight format, parameter count, tokenizer revision, and download size for each pilot model.
- Capture the licence and README at each pinned revision.
- Inspect whether all checkpoints use identical architecture and tokenizer files.
- Establish whether a chat template exists and whether using it is appropriate for a base model.
- Estimate local storage, memory, and inference time before downloading weights.
- Determine whether the six SMA input checkpoints are independently usable.
- Reproduce at least one small capability measurement before interpreting safety results.

## Sources

- Luo et al., “Puro-2B: Poor Lab's Qwen2-1.5B Trained on RTX 5090 within $5090,” arXiv:2608.27370v1, especially Sections 2–4 and Appendices A, G, H, and I.
- Official PACMAN Group Puro-2B Hugging Face collection and model cards.
- Official `thu-pacman/Puro-Megatron` repository.
