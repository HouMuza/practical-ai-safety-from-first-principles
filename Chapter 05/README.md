# Chapter 5: Distribution Shift and Adversarial Robustness

Chapter 4 left us with a full safety operating point: a fitted classifier, a Platt calibrator, and a threshold selected on held-out data. Everything measured there assumed the future looks like BeaverTails. This chapter freezes that model and threshold and asks what happens when the input distribution moves, first through natural, non-adversarial shift, then through a threat-modelled adversarial lens, and finally through unlabeled production drift signals.

The notebook [`chapter_5_distribution_shift_and_robustness.ipynb`](chapter_5_distribution_shift_and_robustness.ipynb) reproduces the chapter's experiments end to end. Running it top to bottom will:

- load the frozen Chapter 4 model, Platt calibrator and safety threshold, regenerating any missing upstream artifact (Chapter 2's data split, Chapter 4's model/calibrator/policy) using the exact same procedure as those chapters;
- re-establish the clean test-set baseline as the fixed point everything else is measured against;
- build reproducible, label-preserving shifted test sets: lowercasing and punctuation removal (negative controls), and a seeded 3% adjacent-character-swap typo transform;
- measure absolute degradation in precision, recall and F1 under each shift, and calculate a paired evasion success rate among unsafe examples the clean model originally caught;
- inspect score erosion before the hard decision flips, including a clean-vs-shifted score scatterplot;
- show numerically how a changing production base rate shifts precision even when the classifier itself has not changed;
- define an explicit black-box threat model for adversarial shift;
- train character n-gram and combined word-plus-character TF-IDF classifiers (thresholds selected on a held-out validation split, never on the shifted test data), and compare all three model families across all shift conditions in a robustness matrix with worst-case and average shifted recall;
- break down robustness by BeaverTails harm category;
- run a typo-augmentation experiment and check whether the improvement generalizes to a transformation (punctuation removal) the augmentation never trained on, rather than just the exact attack it saw;
- check whether calibration itself degrades under shift, without refitting the calibrator;
- compute Population Stability Index and Jensen-Shannon divergence on the score distribution as label-free drift signals;
- build a feature-space familiarity signal (cosine similarity to a training centroid) calibrated against a clean reference distribution, and extend Chapter 4's allow/review/block policy with it; and
- save a versioned shift registry and robustness reports under `reports/` and `config/`.

## Data and evaluation scope

The core experiment uses the BeaverTails `30k_train` split, released under CC BY-NC 4.0. The dataset is not redistributed, the notebook downloads it from Hugging Face only if the processed Chapter 2 split is not already on disk.

The automated perturbations here are reproducible stress tests, not proof of general adversarial robustness. As the chapter stresses, a transformation only counts as a valid robustness test if it actually preserves the original safety label, the mild shifts used here (casing, punctuation, light typos) are chosen specifically because that assumption is easy to defend; a more aggressive challenge set (paraphrase, synonym substitution, translation) needs sampled human review before its results can be trusted.

WildGuardMix is discussed in the notebook as an optional external challenge set but is not downloaded automatically, access requires accepting AI2's Responsible Use Guidelines. Any external evaluation would need to document explicitly how its labels map onto the binary target used here.

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 5/chapter_5_distribution_shift_and_robustness.ipynb"
```

No GPU is required. The character and combined TF-IDF models use more memory and take longer to fit than the word-only baseline, but everything still runs comfortably on a laptop CPU. Generated data, models, configuration and reports are gitignored since the notebook recreates them on each run.
