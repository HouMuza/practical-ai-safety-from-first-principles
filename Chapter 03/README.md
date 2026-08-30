# Chapter 3: Building a Safety Classifier from First Principles

This chapter builds the first real model in the book: TF-IDF features combined with logistic regression, trained on the BeaverTails split prepared in Chapter 2. The point of this deliberately simple architecture is transparency. Every step from raw text to a decision score is inspectable, so before reaching for a neural classifier, we understand exactly what a classical baseline can and cannot do.

The notebook [`chapter_3_baseline_classifier.ipynb`](chapter_3_baseline_classifier.ipynb) reproduces the chapter's walkthrough end to end. Running it top to bottom will:

- load the Chapter 2 train/test split (and regenerate it automatically from BeaverTails if it is not already on disk, so this notebook runs on its own);
- train and evaluate a majority-class dummy baseline for comparison;
- walk through TF-IDF term weighting with a small worked example, then build the real vectorizer;
- explain logistic regression in terms of log-odds, cross-entropy loss and L2 regularisation, and fit a TF-IDF + logistic regression pipeline;
- evaluate the classifier with a confusion matrix, precision, recall and F1;
- inspect the learned coefficients to see which terms push predictions towards "safe" and "unsafe";
- run a prompt-only vs. response-only vs. combined ablation;
- review false positives and false negatives as data, separating boundary mistakes from confident ones;
- compare logistic regression against Multinomial Naive Bayes and a linear SVM;
- run a 5-fold stratified cross-validation check for split stability;
- record training time and vocabulary size; and
- save the fitted pipeline to `models/` for Chapter 4's threshold and calibration work.

## Data and dependencies

This notebook depends on the processed split from [Chapter 2](../Chapter%202). If `data/processed/beavertails_train.parquet` and `beavertails_test.parquet` already exist (because you ran the Chapter 2 notebook first), it uses them directly. If they do not exist, it regenerates them itself using the same procedure, so cloning the repo and running only this notebook still works.

As in Chapter 2, BeaverTails is CC BY-NC 4.0 (non-commercial) and is not redistributed in this repository; it is downloaded directly from Hugging Face.

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 3/chapter_3_baseline_classifier.ipynb"
```

No GPU is required. Training a TF-IDF + logistic regression model on the 30k split runs comfortably on a laptop CPU, and the notebook records the actual training time and vocabulary size it observes on your machine.
