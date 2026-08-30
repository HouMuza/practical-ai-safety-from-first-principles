# Chapter 12: Guardrails and System-Level Safety

Chapters 1-11 asked questions about a model or a component sitting very close to it: a classifier, its threshold, its calibration, its robustness under shift, a language model under red-teaming, its truthfulness, its bias, a reward model, and finally a fine-tuned model whose behaviour moved in unexpected directions. This chapter changes the unit of analysis from the model to the system around it. The same model producing the same mistaken output can be harmless in a notebook and consequential in a deployed pipeline, purely because of the architecture surrounding it.

A guardrail is any control that changes what a system is allowed to receive, retrieve, generate, execute or return. Calling something a guardrail does not tell us whether it works. The notebook [`chapter_12_guardrails_and_system_level_safety.ipynb`](chapter_12_guardrails_and_system_level_safety.ipynb) treats guardrails as experimental interventions: it builds a small layered safety pipeline out of real, fitted classifiers and a real language model, measures each control in isolation, measures the same controls when they interact, and runs a safe, harmless indirect prompt-injection experiment that preserves the structure of the real trust-boundary problem without touching anything dangerous.

Running it top to bottom will:

- fit two fresh, deliberately narrower classifiers as ablations of Chapter 3's combined prompt+response model, `input_clf` (prompt-only) and `output_clf` (response-only), plus regenerate Chapter 3's own combined pipeline to play an independent judge role, and wire them and a real Qwen3-0.6B generator into a transparent `run_system` pipeline that returns an explicit decision object at every stage;
- run a system-level failure funnel on real BeaverTails unsafe and safe test prompts across four configurations (no guardrails, input-only, output-only, both), walk the funnel stage by stage with denominators attached, and test whether the input and output guardrails fail on the *same* examples more than independence would predict (an overlap ratio and Jaccard similarity over paired failure sets);
- run a four/five-condition indirect prompt-injection experiment on a fictional research-assistant summarisation task (naive concatenation, explicit trust boundaries, an injection filter falling back from Llama Prompt Guard 2 to a locally trained TF-IDF classifier, structured extraction, and a layered combination), measuring attack success and legitimate task success on the same paired clean/injected documents in every condition, plus a held-out challenge set with different phrasing never used while building the defences;
- demonstrate what a JSON schema and an execution-layer allow-list can and cannot guarantee (form validity, not truth or policy correctness); and
- assemble a safety-utility trade-off table (safety gain, utility loss, latency), apply the rule-of-three correction to small-sample zero-attack-success observations, compute paired bootstrap intervals resampled by case, sweep the injection filter's threshold to trace a Pareto frontier between attack resistance and clean-document pass rate, and generate a system-level conclusion paragraph from that run's own numbers.

## What the results actually looked like

This notebook has not been executed in this checkout, its cells carry no cached outputs, so there are no run numbers to report here yet. The notebook is built so that running it produces a specific set of checkable artefacts rather than one headline number:

- a funnel table showing how many of the entering unsafe prompts pass the input guardrail, go on to produce a model response judged unsafe by the independent `pair_judge_clf`, and then escape the output guardrail, with a denominator attached at every stage;
- an overlap ratio (observed joint failure rate divided by the rate independence would predict) and Jaccard similarity between the input and output guardrails' failure sets on the same `D_input_and_output` run, the notebook's direct test of whether "defence in depth" is buying independent coverage or catching the same examples twice;
- an ablation table comparing all four pipeline configurations on both unsafe-escape rate and safe-false-block rate, so a configuration that "improves" safety by refusing everything cannot hide behind one number;
- a five-condition table of attack success rate, clean- and attacked-task success, false-block rate on clean documents, and median latency, plus paired bootstrap intervals for the trust-boundary-vs-naive and structured-extraction-vs-naive deltas, and a rule-of-three upper bound wherever a condition observes zero attack successes at this notebook's small sample size;
- a scatter plot and printed table of the injection filter's safety-utility Pareto frontier across nine threshold values; and
- a final conclusion paragraph, generated from that specific run's numbers, stating which configuration achieved the lowest observed attack success rate and what it did and did not establish about general prompt-injection resistance.

Run the notebook (`jupyter notebook chapter_12_guardrails_and_system_level_safety.ipynb`, executing all cells) to generate the actual tables, plot and conclusion paragraph described above.

## Data and dependencies

BeaverTails is downloaded via `datasets` and reuses (or regenerates) the Chapter 2 train/test split. The two new ablation classifiers and the regenerated combined judge are saved to `models/` (gitignored); summary tables (never the raw BeaverTails prompts or responses that produced them) are saved to `results/chapter12/` (gitignored). The injection filter first tries to load `meta-llama/Llama-Prompt-Guard-2-22M` (a gated Hugging Face model) and falls back to a small, locally trained TF-IDF + logistic regression classifier if that model is unavailable, following the same fallback precedent Chapter 6 used for WildGuard. The fictional station reports, injection strings and structured-extraction schema are all generated inside the notebook; nothing dangerous is executed and no private data is touched. No new dependencies beyond earlier chapters (`transformers`, `scikit-learn`, `pydantic`).

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 12/chapter_12_guardrails_and_system_level_safety.ipynb"
```

Sample sizes are kept small so the notebook completes in a reasonable time on a laptop: `N_FUNNEL_UNSAFE = 15`, `N_FUNNEL_SAFE = 15` BeaverTails prompts run through the pipeline, and `N_INJECTION_PAIRS = 10` clean/injected document pairs (plus a small held-out challenge set). The expensive steps are fitting the two ablation classifiers once and running Qwen3-0.6B generation across the funnel and injection experiments; raise the sample-size constants toward the book's suggested scale (at least 200 paired documents for the injection experiment) for narrower confidence intervals and a more defensible overlap-ratio estimate.
