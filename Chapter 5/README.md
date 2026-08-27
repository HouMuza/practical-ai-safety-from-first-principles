# Chapter 5: Distribution Shift and Adversarial Robustness

Chapter 4 produced a calibrated safety classifier and a documented operating threshold. This chapter freezes that package and tests whether its evidence survives when the input distribution changes. It separates clean generalisation, natural robustness, adversarial robustness and external generalisation rather than compressing them into one headline score.

The notebook [`chapter_5_distribution_shift_and_robustness.ipynb`](chapter_5_distribution_shift_and_robustness.ipynb) reproduces the chapter's experiments end to end. Running it top to bottom will:

- load the Chapter 4 model, Platt calibrator and safety policy, rebuilding missing upstream artifacts automatically;
- create traceable, deterministic test shifts for casing, punctuation, a typographical-noise severity sweep and readable word fragmentation;
- compare clean and shifted precision, recall, F1, false-positive rate, false-negative rate and calibration;
- calculate paired evasion success among unsafe examples detected in their clean form;
- inspect score erosion before examples cross the decision threshold;
- compare the frozen recall-first policy with a validation-selected diagnostic policy without tuning either on shifted test data;
- demonstrate how label prevalence changes expected precision without changing the classifier;
- train character TF-IDF and combined word-plus-character baselines using validation-selected operating points;
- compare models with a robustness matrix, worst-case recall and average shifted recall;
- evaluate typographical data augmentation without contaminating the shifted test suite;
- calculate Population Stability Index, Jensen-Shannon distance and feature-space familiarity signals;
- extend allow/review/block routing with an unfamiliar-input review path; and
- save versioned shift definitions and detailed robustness reports under `reports/`.

## Data and evaluation scope

The core experiment uses the BeaverTails `30k_train` split, released under CC BY-NC 4.0. The dataset is not redistributed. The notebook downloads it from Hugging Face only when the processed Chapter 2 split is unavailable.

The automated perturbations are reproducible stress tests, not proof of general adversarial robustness. The stronger typo and word-fragmentation conditions require sampled semantic review before their labels are treated as validated challenge data. Aggressive paraphrases, translations or synonym substitutions require even more careful semantic validation before the original safety label can be reused.

WildGuardMix is discussed as an optional external challenge set but is not downloaded automatically because access requires accepting AI2's Responsible Use Guidelines. Any external evaluation must document how its labels map to the binary target used by the BeaverTails classifier.

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 5/chapter_5_distribution_shift_and_robustness.ipynb"
```

No GPU is required. Character and combined TF-IDF models use more memory and take longer to fit than the word-only baseline, but the experiment remains designed for a laptop CPU. Generated data, models, configuration and reports are gitignored because the notebook recreates them.
