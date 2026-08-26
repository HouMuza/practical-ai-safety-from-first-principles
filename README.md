# Practical AI Safety from First Principles

Companion code repository for **Practical AI Safety from First Principles**, a book about technical AI safety taught from the ground up: real datasets, real classifiers, real experiments, and the reasoning behind each one, before reaching for a library that hides the mechanism.

The book treats AI safety as an empirical discipline. Each chapter turns a broad safety concept, such as hallucination, harmful compliance, reward hacking, jailbreak robustness, distribution shift, unlearning, or scalable oversight, into something that can be measured, modelled, and tested. This repository holds the notebooks that go with that work, organized one folder per chapter, so a reader can clone the repo and run the same experiments described in the book.

## Repository structure

Each chapter gets its own folder containing a `README.md` and, where the chapter includes hands-on work, a Jupyter notebook that reproduces it end to end. Chapters that are purely conceptual (no code) say so plainly in their `README.md` rather than including an empty notebook.

```
Chapter 1/   AI Safety Overview                          (no code, conceptual chapter)
Chapter 2/   Turning AI Safety into Data                  (notebook: BeaverTails dataset audit)
Chapter 3/   Building a Safety Classifier from First Principles  (notebook: TF-IDF + logistic regression baseline)
```

More chapters will be added here as they are written.

## Getting started

```bash
git clone https://github.com/HouMuza/practical-ai-safety-from-first-principles.git
cd practical-ai-safety-from-first-principles
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
jupyter notebook
```

Then open the notebook inside the chapter folder you're interested in. Each notebook is self-contained: it downloads any data it needs (e.g. from Hugging Face) and runs top to bottom without requiring outputs from another chapter, unless a chapter explicitly builds on artifacts saved by a previous one.

## About the book

*Practical AI Safety from First Principles* is currently being written. This README will be updated with links to get the book once it is available.

## License

Code in this repository is released under the MIT License (see `LICENSE`). Datasets used by the notebooks retain their own licenses. See each chapter's README for details.
