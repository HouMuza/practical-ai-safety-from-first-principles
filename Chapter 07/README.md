# Chapter 7: Truthfulness, Hallucination and Model Uncertainty

Chapter 6 red-teamed Qwen/Qwen3-0.6B for harmful compliance. This chapter keeps the same target model but studies a different failure mode: how often it is simply wrong when nobody is attacking it, and whether its own confidence tells us anything useful about that. Factuality is treated as three separable regimes rather than one "hallucination score": truthfulness under forced choice (TruthfulQA), free-form factual accuracy with abstention (SimpleQA), and document-grounded factuality (a small claim-verification pipeline).

The notebook [`chapter_7_truthfulness_and_hallucination.ipynb`](chapter_7_truthfulness_and_hallucination.ipynb) reproduces the chapter's harness end to end. Running it top to bottom will:

- reconstruct TruthfulQA's binary (truthful-vs-false) format from the maintained `truthfulqa/truthful_qa` dataset, since the original `EleutherAI/truthful_qa_binary` release uses a legacy loading script current `datasets` versions no longer execute;
- score both answer candidates directly from the model's token log-probabilities (a single forward pass per candidate, not generation), verify the token-offset arithmetic on a worked example before trusting it across the sample, and check answer-position sensitivity with a separate letter-choice prompting method;
- compute accuracy (overall and by category), predictive entropy, a reliability diagram, Brier score, log loss, and error-detection AUROC, reusing the calibration machinery from Chapter 4;
- build a risk-coverage curve showing how selective accuracy changes as the model is allowed to abstain from lower-confidence questions;
- download SimpleQA (4,326 short fact-seeking questions) and generate short, greedy answers under a baseline prompt and a separate abstention-aware prompt, grading both with a deterministic CORRECT / INCORRECT / NOT_ATTEMPTED matcher;
- report overall accuracy, attempt rate, attempted accuracy and incorrect-attempt rate for both prompts side by side, so the abstention trade-off is visible rather than implied; and
- run a small, fully self-contained grounded-factuality pipeline: generate an answer from a supplied document, split it into claims, retrieve the most relevant evidence sentences with a sentence-transformer, and judge entailment/contradiction/neutral with a small NLI model, reporting grounded precision, unsupported rate and contradiction rate.

## What the results actually looked like

Run with this notebook's default (small) sample sizes on Qwen3-0.6B, the results are a genuinely sharp illustration of the chapter's point, not a smoothed-over demo:

- **TruthfulQA accuracy came out at 43%, below chance**, and error-detection AUROC came out at 0.36, *worse* than random, with mean entropy on correct answers actually *higher* than on incorrect ones. That is not a bug: it is TruthfulQA doing exactly what it was built to do. A small model is more likely to reproduce a common misconception fluently and confidently than a large one, so it can look simultaneously wrong *and* sure of itself, precisely the dangerous combination section 7.4 warns about.
- **SimpleQA baseline accuracy came out at 0%** on this sample of deliberately hard questions, with confident, fluent, specific wrong answers (a wrong dealership name, a wrong forest-cover figure, a wrong district) rather than visible confusion. Under the abstention-aware prompt, the same model **attempted 0% of questions**, an extreme but real illustration of section 7.6's warning that an abstention instruction can suppress far more than the genuinely unanswerable questions.
- **Grounded precision came out at 100%** on the small toy document set, once evidence retrieval correctly used the top three matching sentences rather than only the single closest one (a compound generated claim spanning two facts needs more than one evidence sentence to be recognized as supported, an easy way to under-count grounding if you're not careful).

## Data and dependencies

TruthfulQA and SimpleQA are downloaded automatically (SimpleQA via `requests`, not `pandas.read_csv(url)` directly, to sidestep an SSL certificate issue some Python installs hit with the raw method). The grounded-factuality section uses a small, original set of synthetic documents defined in the notebook itself, not a redistributed dataset, precisely because a real benchmark like FACTS Grounding needs infrastructure well beyond a companion notebook (1,719 examples, multiple judge models, a private held-out split). The practical exercise at the end of the notebook is where a reader would substitute a larger public document set.

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 7/chapter_7_truthfulness_and_hallucination.ipynb"
```

Log-likelihood scoring (TruthfulQA) is much faster than free-form generation, roughly a second per question on Apple Silicon MPS, so the default sample sizes (`N_TRUTHFULQA = 150`, `N_SIMPLEQA = 50`) finish in well under 20 minutes total. Raise them toward the full benchmarks (817 and 4,326 respectively) to reproduce the book's full-scale experiment. No GPU is required, but this chapter, like Chapter 6, benefits from one; `torch`'s MPS backend is used automatically on Apple Silicon.
