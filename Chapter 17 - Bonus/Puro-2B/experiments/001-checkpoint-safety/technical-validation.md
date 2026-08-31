# Technical Validation Record

**Status:** In progress  
**Validation date:** 2026-08-31

## Pinned models

Exact revisions, weight hashes, and sizes are stored in [`../../configs/models.json`](../../configs/models.json).

Initial metadata inspection found:

- all three primary checkpoints use `Qwen3ForCausalLM`;
- configuration, tokenizer, and tokenizer-configuration SHA-256 hashes are identical across checkpoints;
- each repository contains one BF16 `model.safetensors` file of 4,063,515,640 bytes;
- the three weights total 12,190,546,920 bytes (approximately 11.35 GiB), excluding cache overhead;
- the architecture has 28 layers, hidden size 2,048, 16 attention heads, 8 key/value heads, and vocabulary size 151,936;
- the model configuration declares a 4,096-token maximum position embedding length;
- the tokenizer metadata includes a generic assistant chat template, but the primary likelihood evaluation will not use it because these are base checkpoints.

## Pinned SafetyBench sources

Exact benchmark revisions are stored in the shared [`SafetyBench registry entry`](../../../safety-evaluation-pipeline/configs/benchmarks/safetybench.json).

- Hugging Face dataset licence: MIT.
- Official GitHub repository licence: MIT.
- Languages: English and Chinese, plus a filtered Chinese subset.
- Test questions are available from Hugging Face and GitHub.
- Test labels are omitted from the Hugging Face dataset but published as separate answer JSON files in the official GitHub repository.
- Development data contains five labelled examples per category per language.

Labels must not be included in model prompts or model-facing input records. Evaluation code should join predictions to labels only after scoring, by language and stable item ID.

## Scoring implementation

The shared [`mcq.py`](../../../safety-evaluation-pipeline/src/safety_eval/mcq.py) implements deterministic conditional-likelihood scoring without a chat template. Context and options are tokenized separately, concatenated, and only option-token log probabilities are accumulated. Its SafetyBench renderer matches the official base-model CLP convention: English `Question:/Options:/Answer:(` or Chinese `问题：/选项：/答案：(` followed by comparison of the next `A`–`D` token.

The scorer reports both total and mean option log probability. The primary length treatment remains to be frozen after technical tests and before pilot outcomes are inspected.

Six unit tests pass in the project's existing virtual environment. They verify that only option tokens are scored, invalid input is rejected, the specialised single-token path matches the generic scorer, and the English and Chinese benchmark prompt renderers are stable.

The first neutral smoke run loaded the pinned Uniform checkpoint successfully in BF16 on CPU. The official-format scorer selected the expected answer for two of three deliberately simple questions; it selected `12` rather than `10` for a sequence question. This is not a research result and is retained only as a pipeline sanity check. The initial unoptimised implementation took 20.64 seconds for nine forward passes across the three items; a one-forward-pass single-token path was added in response.

The optimised single-token run took 19.92 seconds for three item-level forward passes, or approximately 6.64 seconds per item on this CPU-only environment. The reduction in forward-pass count did not materially reduce wall time, indicating that item-level batching or an accelerated backend is required before a 700-item, three-model pilot. A naive unbatched projection is roughly 3.9 hours for the 700-item pilot and 63 hours for the full 11,435-item benchmark across three models, excluding model swaps and analysis.

## Local feasibility

- Host: Apple M4 MacBook Air, 10 CPU cores, 16 GB unified memory.
- Free disk space observed during validation: approximately 31 GiB.
- Existing environment: Python 3.13.2, PyTorch 2.13.0, Transformers 4.57.6.
- Transformers successfully resolves the pinned local metadata as `Qwen3Config` with the expected dimensions.
- The installed PyTorch build reports neither MPS nor CUDA availability, so it would currently run inference on CPU.

One 3.78-GiB BF16 checkpoint should fit in the 16-GB memory envelope for inference, but CPU throughput must be measured. Keeping all three downloaded weights would consume roughly 11.35 GiB before Hugging Face cache metadata, datasets, results, or temporary files. With only about 31 GiB free, checkpoints should be downloaded and validated one at a time unless storage conditions change.

The absence of MPS support is an environment issue, not a Puro incompatibility. We should first obtain a CPU timing baseline, then decide whether installing an Apple-accelerated PyTorch build or using an MLX conversion is justified. Any converted or quantised model would be a separate deployment condition and cannot replace the BF16 primary comparison without validation.

## Remaining validation

- Test the scorer against a tiny public causal LM or a locally available compatible model.
- Measure CPU, Apple accelerator, or GPU memory and throughput on the available hardware.
- Implement padded item batching and benchmark several batch sizes.
- Determine why the installed PyTorch build reports no MPS support on Apple M4.
- Download one checkpoint only after confirming at least 5 GiB of safe cache headroom.
- Compare total versus length-normalised option scoring on development examples without using test outcomes.
- Freeze the prompt format and option-scoring convention.
# Shared-pipeline smoke test — 2026-08-31

- Run ID: `puro_2b_001_checkpoint_safety-45cbc98e8947`
- Model: Puro-2B Uniform at revision
  `46d9da353060356169438bcb3730c41ebd116216`
- Runtime: Hugging Face Transformers, Apple MPS, float32, no quantisation
- SafetyBench source: official GitHub revision
  `960833692a1ec82b191ff3e902daca7993c35cc1`
- Prepared bilingual dataset SHA-256:
  `e4247b102971bb4515e49ebcd5db5c33420b14ccb518baea0f4ae8e7ab9f0e2e`
- Scope: two deterministically sampled items, one English and one Chinese
- Outcome: two completed records, zero runtime failures
- Resume check: second invocation completed zero items and resumed two

This validates model loading, bilingual rendering, conditional-likelihood
scoring, provenance capture, append-only output, and restart behaviour. Two
items cannot estimate safety performance; their answers must not be reported as
a model score or used to compare checkpoints.

## Matched blinded smoke — 2026-08-31

- Run ID: `puro_2b_001_checkpoint_safety-a3ff95b2c916`
- Frozen sample: 70 items, five per language/category stratum
- Models: Uniform, Curriculum Decay, Curriculum SMA6
- Runtime: Transformers, Apple MPS, float16, no quantisation
- Coverage: 70/70 items for every checkpoint; 210 unique model/item pairs
- Failures: zero
- Inference-condition match: passed
- Publication status: technical validation; outcomes withheld

An initial float32 feasibility attempt was stopped after three recorded items
because its projected runtime was excessive. Its distinct run fingerprint
prevents those records from entering the matched float16 analysis.
