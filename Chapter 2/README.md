# Chapter 2: Turning AI Safety into Data

This chapter takes a broad safety concept — "unsafe" — and turns it into something a machine-learning pipeline can actually work with: an operational label, a dataset, and a set of metrics chosen to match a deployment context rather than habit.

The notebook [`chapter_2_beavertails_audit.ipynb`](chapter_2_beavertails_audit.ipynb) reproduces the chapter's walkthrough end to end using the [BeaverTails](https://huggingface.co/datasets/PKU-Alignment/BeaverTails) dataset (PKU-Alignment). Running it top to bottom will:

- load the `30k_train` split of BeaverTails from Hugging Face and inspect its schema;
- check for missing/empty prompts and responses, and inspect text length distributions;
- compute the base rate of the `unsafe` class and show why accuracy alone is misleading under class imbalance;
- walk through the confusion matrix, precision, recall, specificity, F1/Fβ, and how they move with the decision threshold;
- use Bayes' theorem to show why precision collapses when a classifier trained on a balanced benchmark is deployed into a low-prevalence production environment;
- expand the multilabel harm-category taxonomy and plot category prevalence;
- check for exact-duplicate leakage between rows;
- run a reusable dataset-audit function; and
- build the `[PROMPT]/[RESPONSE]` text field and target label, then produce a stratified train/test split saved to `data/processed/` for Chapter 3.

## Data

BeaverTails is released under **CC BY-NC 4.0** — non-commercial use only. The dataset is not redistributed in this repository; the notebook downloads it directly from Hugging Face via the `datasets` library. It intentionally contains harmful and distressing content because it was built for safety research; the notebook works with metadata, aggregate statistics, and truncated text rather than printing raw examples in full.

Reference: Ji, J., Liu, M., Dai, J., Pan, X., Zhang, C., Bian, C., Zhang, C., Sun, R., Wang, Y., & Yang, Y. (2023). *BeaverTails: Towards Improved Safety Alignment of LLM via a Human-Preference Dataset.* NeurIPS Datasets and Benchmarks Track. https://arxiv.org/abs/2307.04657

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 2/chapter_2_beavertails_audit.ipynb"
```

The first run downloads the `30k_train` split (~30k rows) from Hugging Face, which requires an internet connection. No GPU is required — everything in this chapter runs comfortably on a laptop CPU.
