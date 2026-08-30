# Chapter 16: From Safety Experiment to Research Programme

Every earlier chapter isolated one problem so its mechanics were visible: a classifier, a threshold, a jailbreak, a reward model, a fine-tune's side effects, a guardrail stack, an agent's trajectory, a capability evaluation, an unlearning intervention. This final, capstone chapter is about the infrastructure around good experiments: hypotheses written before results are known, versioned artefacts, per-example evidence, paired statistics, a genuinely held-out challenge set, a regression suite, and a report that states what the evidence does and does not support.

Rather than teach that infrastructure in the abstract, the notebook [`chapter_16_from_safety_experiment_to_research_programme.ipynb`](chapter_16_from_safety_experiment_to_research_programme.ipynb) applies it to one concrete intervention the book already built: **Chapter 10's DPO harmlessness adapter**, evaluated for whether it reduces harmful compliance without materially increasing benign refusal, the running example the book itself uses throughout this chapter. It reuses Chapter 10's own training and evaluation code (regenerating the adapter locally if Chapter 10 has not been run first, the same cross-chapter convention every prior chapter follows) and adds the research-infrastructure layer this chapter is actually about.

Running it top to bottom will:

- write a pre-result design table (baseline, intervention, primary/secondary outcomes, development vs. challenge sets, ablation, generation seeds, uncertainty method, multiple-comparison handling, adoption constraint, stopping rule) *before* any evaluation code runs, and state three versions of the claim in advance: a narrow empirical statement, an operational interpretation, and the broad claim ("the model is now safe against jailbreaks") that nothing in the notebook licenses;
- build stable content-derived example IDs and file/directory content hashes, define one common `EvalRecord` schema shared across every benchmark, and load or regenerate Chapter 10's DPO adapter (verifying its content hash) before writing a machine-readable experiment registry entry (`registry/ch16_dpo_capstone_001.yaml`) that records the exact model, adapter hash, judge, data sources, seeds and adoption constraint used;
- split JailbreakBench's harmful and benign categories into a development set (inspected while building the evaluation) and a disjoint challenge set (never inspected until the primary analysis is frozen), then evaluate the reference model and the DPO adapter on both, writing every result as an `EvalRecord` to a per-example parquet file;
- run a matched-control ablation, the untrained reference model plus a safety-emphasising system prompt only, to test whether a much cheaper intervention explains most of the adapter's apparent effect, and build a paired transition analysis (persistent failure / fixed failure / new failure / persistent success) that shows exactly which examples changed rather than only an aggregate rate;
- score a small TruthfulQA (mc1) subset as a secondary utility outcome, assemble a multi-dimensional scorecard that keeps harmful compliance, benign refusal, challenge-set generalisation and truthfulness visible as separate rows rather than one collapsed number, and compute paired bootstrap confidence intervals with a Holm correction across the primary+secondary metric family; and
- turn the scorecard into regression gates with explicit tolerances, write a manifest and a technical safety card (including an explicit "what was not tested" list), generate a primary-effect table and a final evidence-based conclusion paragraph from that run's own numbers, and close with a research-question backlog built from the capstone's own residual failures plus an illustrative (not yet multi-beta) Pareto-frontier demonstration.

*A note on scope*: another pass is in progress to rename one section header and flesh out part of this notebook's text; the experiment structure and outputs described above reflect its content as read for this README and should be stable, though exact section numbering may shift slightly.

## What the results actually looked like

This notebook has not been executed in this checkout, its cells carry no cached outputs, so there are no run numbers to report here yet. Running it end to end is designed to produce, among other artefacts:

- a written experiment registry entry and a technical safety card, both populated with this run's real content hashes rather than placeholders;
- a per-example parquet file (`results/chapter16/per_example/ch16_dpo_capstone_001.parquet`) with one `EvalRecord` row per example, per condition (reference, dpo, and the system-prompt ablation), per benchmark (harmful/benign development and challenge sets, TruthfulQA);
- an ablation table comparing harmful-compliance and benign-refusal rates for the reference model, the reference model with a safety system prompt, and the DPO adapter, letting the notebook check whether the adapter's effect exceeds what a prompt alone achieves;
- paired transition tables for both the harmful-dev and benign-dev sets, showing counts of persistent failure, fixed failure, new failure and persistent success rather than one before/after percentage;
- a multi-dimensional scorecard (harmful compliance and benign refusal on both development and challenge sets, TruthfulQA accuracy, mean response length) and an uncertainty table with paired bootstrap 95% intervals and Holm-adjusted p-values for each primary/secondary metric;
- a regression-gate table (pass/fail against predeclared tolerances) and a final conclusion paragraph, generated from that specific run's numbers, stating the adapter's measured effect on development and challenge categories, the matched-control comparison, and an explicit statement of what the study does and does not establish (no claim about adaptive attacks, other languages, other model sizes, or deployment conditions it did not test).

Run the notebook (`jupyter notebook chapter_16_from_safety_experiment_to_research_programme.ipynb`, executing all cells) to generate the actual registry entry, tables and conclusion paragraph described above.

## Data and dependencies

JailbreakBench's harmful/benign behaviours are loaded from `dedeswim/JBB-Behaviors` directly (same approach and Python 3.13 rationale as Chapters 6 and 10), and split by category into disjoint development and challenge sets, recorded in `registry/challenge_sets.yaml`. TruthfulQA (`truthful_qa`, `multiple_choice`) is downloaded via `datasets`. The Chapter 3 BeaverTails classifier is reused as the harmful-response judge (retrained automatically if not found locally), and Chapter 10's DPO adapter is loaded from `../Chapter 10/models/qwen3_0.6b_dpo_harmlessness_lora` if present, or regenerated locally using Chapter 10's exact procedure otherwise. New dependency: `statsmodels` (for the Holm multiple-comparison correction via `multipletests`), plus `pyyaml` for the registry files.

This chapter's folder also contains `models/`, `registry/` and `results/` subfolders (all gitignored, populated by running the notebook): `models/` holds the DPO adapter if regenerated here, `registry/` holds the experiment and challenge-set YAML files, and `results/` holds the per-example parquet file plus aggregate CSVs (scorecard, regression gates), the manifest and the technical safety card.

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 16/chapter_16_from_safety_experiment_to_research_programme.ipynb"
```

Sample sizes match Chapter 10's own evaluation scale so the two are comparable: `N_TRAIN = 200`, `N_VAL = 50` (only used if the adapter needs regenerating), `N_EVAL = 15` per development/challenge harmful and benign set, `N_TRUTHFULQA = 15`. If Chapter 10 has already been run, this notebook reuses its saved adapter directly and skips retraining; if not, expect the added DPO training time documented in Chapter 10's own README. At `N_EVAL = 15`, expect wide bootstrap intervals, this is an honest reflection of sample size rather than a reason to treat any single run's point estimates as settled; raise `N_EVAL` (and the JailbreakBench category split sizes) toward the book's fuller scale for a narrower, more defensible answer.
