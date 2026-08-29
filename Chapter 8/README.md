# Chapter 8: Bias and Fairness as Measurable Safety Properties

Chapter 7 separated "hallucination" into truthfulness, faithfulness and groundedness. This chapter applies the same discipline to "bias": a disparity between groups is an observation, whether it counts as unfair depends on which fairness definition applies, and several reasonable definitions can conflict with each other whenever group base rates differ. The chapter moves from classical group-fairness metrics (allocative harms) to controlled multiple-choice bias evaluation on a real benchmark (BBQ) to a small open-ended generation comparison (BOLD), using the same Qwen/Qwen3-0.6B target model as Chapters 6 and 7.

The notebook [`chapter_8_bias_and_fairness.ipynb`](chapter_8_bias_and_fairness.ipynb) reproduces the chapter's experiments end to end. Running it top to bottom will:

- build a reusable `subgroup_metrics` function and demonstrate, with actual numbers on a synthetic two-group prediction table, that equalizing one fairness criterion (equal opportunity / true-positive rate) can *worsen* another (predictive parity / precision) whenever the groups' base rates genuinely differ, the mathematical content of the fairness-impossibility results, not an abstract claim;
- clone the official [BBQ](https://github.com/nyu-mll/BBQ) repository, load and merge its metadata, and score every answer option directly from the model's token likelihoods (the same approach as Chapter 7's TruthfulQA scorer, adapted to three options);
- compute accuracy separately for ambiguous and disambiguated contexts, reproduce BBQ's own directional bias score and accuracy-adjusted ambiguous bias score, and measure the evidence-alignment accuracy gap (is the model more accurate when the evidence-backed answer happens to match a stereotype than when it contradicts one?);
- break every result down by BBQ's 11 social-dimension categories;
- test answer-position sensitivity with a genuinely separate letter-choice prompting method (the direct-likelihood scorer is immune to position bias by construction, so testing it against itself would be circular);
- build a real, automated counterfactual experiment: swap each example's two demographic surface forms (e.g. "grandfather" / "grandson") everywhere they appear in the context and question, and measure both the prediction flip rate and the movement in the stereotype-target probability;
- bootstrap confidence intervals for a category gap, and apply a Holm multiple-comparisons correction across all 11 categories rather than treating every one as an independent, pre-registered test;
- run a small, real BOLD extension (one domain, a handful of prompts per subgroup, frozen decoding settings across groups) and score the generations with a sentiment classifier, explicitly flagged as needing its own validation rather than being trusted as neutral; and
- test an evidence-first mitigation prompt against the full set of BBQ metrics it was meant to move, not just the one easiest to improve.

## What the results actually looked like

Run with this notebook's default (small, stratified) BBQ sample on Qwen3-0.6B:

- The synthetic classical-fairness demo showed the textbook tension directly: adjusting group B's threshold to match group A's true-positive rate (0.924) left the precision gap slightly *wider*, not narrower, purely because the two groups' base rates differ (0.201 vs 0.342).
- BBQ accuracy came out modest in both conditions (ambiguous ~0.39, disambiguated ~0.35, a 3-option benchmark where chance is ~0.33), with real, uneven category-level bias scores, e.g. `Race_x_SES` reached ~0.91 ambiguous accuracy (and a bias score near zero, correctly, since there is little room left for stereotype-driven errors), while several other categories showed meaningfully negative or positive bias scores at the same sample size.
- The letter-choice position-sensitivity check came out at 0.800, a real, informative number once it was tested with a method that could actually show sensitivity (the direct-likelihood method, tested against itself, is tautologically 1.000, which is why the notebook uses a separate prompting method for this specific check).
- The automated counterfactual swap found a usable literal substitution in about a quarter of ambiguous examples, with a genuine ~21-31% prediction-flip rate across runs and real spread in stereotype-target-probability movement, including some large swings that never actually flipped the final answer.
- `Race_x_SES` was the only category to survive Holm correction across the 11-category comparison at this sample size, exactly the expected outcome of that discipline: most category-level "findings" at a small sample are not yet statistically distinguishable from the overall rate, and the correction is what keeps that honest.

## Data and dependencies

BBQ (58,492 examples, `git clone`d automatically into `data/external/BBQ`) and BOLD (`git clone`d into `data/external/BOLD`) are both cloned directly from their official repositories rather than redistributed here; the resulting ~160MB of external data is gitignored. `statsmodels` is required for the Holm correction and `sentence-transformers`/`transformers` pins are shared with earlier chapters, see the root `requirements.txt` for the version constraints that keep Chapters 6-8 mutually compatible.

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 8/chapter_8_bias_and_fairness.ipynb"
```

BBQ scoring is a few forward passes per example, roughly 1.5 seconds on Apple Silicon MPS, so the default `N_BBQ = 300` stratified sample finishes in well under 20 minutes total, including the mitigation re-scoring pass and the small BOLD extension. Raise `N_BBQ` toward the full 58,492 examples to reproduce the book's full-scale experiment.
