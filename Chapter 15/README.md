# Chapter 15: Machine Unlearning and Capability Reduction

Chapter 14 measured whether a model possesses a capability. This chapter asks a more interventionist question: if a model has learned something we no longer want it to retain, can we remove it without retraining from scratch and without destroying everything else the model can do? "The model forgot 90% of the data" is a much weaker claim than it sounds, since low accuracy on a forget set is compatible with genuine forgetting, learned refusal, broad capability damage, or information that is still there but no longer surfaces under the exact evaluation format used.

The notebook [`chapter_15_machine_unlearning_and_capability_reduction.ipynb`](chapter_15_machine_unlearning_and_capability_reduction.ipynb) uses TOFU (a synthetic dataset of fictitious-author facts) as the main laboratory, because it lets us fine-tune a real model on a real forget/retain split, inspect every failure, and deliberately try to recover forgotten information, all without touching sensitive content. It trains two LoRA branches from the same base checkpoint (a "full" model that sees the forget data, and a retain-only "oracle" that never does), then builds two unlearning baselines on top of the full model: gradient ascent and a retain-regularised objective swept across several forget/retain weightings. Every claim is checked against a hierarchy of evidence, exact accuracy, paraphrase transfer, oracle comparison, membership-style probes, and relearning speed, before anything is called "forgotten." A short, deliberately abstract section connects the same logic to WMDP and RMU-style representation unlearning, and a final section applies MUSE's six-way framing to a small sequential-unlearning simulation.

Running it top to bottom will:

- load TOFU's `full`, `forget01` (roughly two fictitious authors, ~1% of the data) and `retain99` splits, carve out disjoint retain slices so no example ever appears in more than one role, and train `M_full` (forget + retain), `M_oracle` (retain only, same retain examples) and evaluate untouched `M_base` as three reference points, using a token-F1 free-form grader and an answer-only log-probability scorer built for this comparison;
- run gradient-ascent unlearning on `M_full` (maximising forget-set loss directly), tracking forget accuracy, retain accuracy, target log-probability, gradient norm and first-answer-token entropy every few steps rather than only at the end, since a sudden entropy spike or retain-accuracy collapse is often the real explanation for apparent "extra forgetting";
- run a retain-regularised unlearning objective (`-lambda_forget * forget_loss + lambda_retain * retain_loss`) swept across four `(lambda_forget, lambda_retain)` configurations, producing a full training curve per configuration;
- build a forgetting-utility Pareto frontier across all gradient-ascent and retain-regularised checkpoints, a paired bootstrap comparison of the best frontier configuration against gradient ascent, a local paraphrase-transfer probe (does forgetting survive different wording of the same question), a relearning-speed comparison (how many fine-tuning steps to recover forgotten answers, versus the oracle learning the same facts for the first time), a membership-style ROC-AUC probe (can forget examples still be distinguished from held-out examples by loss alone), and a Jensen-Shannon divergence comparison of each model's answer distribution against the oracle's;
- discuss WMDP and RMU as the safety-motivated version of the same measurement structure (kept deliberately abstract and content-safe, following Chapter 14's discipline), with a small illustrative base-model WMDP-Bio accuracy check; and
- assemble an evidence ladder (exact-forgotten / paraphrase-forgotten / membership-indistinguishable) across the unlearning methods, generate a claim paragraph from the run's own numbers, and run a MUSE-style six-property proxy report including a genuine sequential-unlearning simulation (issuing a second forget request on top of the first and checking whether the first request's forgetting survives it).

## What the results actually looked like

This notebook has not been executed in this checkout, its cells carry no cached outputs, so there are no run numbers to report here yet. Running it end to end is designed to produce:

- a baseline table where `M_full` should be clearly better than `M_base` on exact forget accuracy (evidence the facts were actually learned) and `M_oracle` should sit near `M_base` on forget accuracy while matching or beating `M_full` on retain accuracy, the notebook explicitly flags this pattern as something to check before trusting anything downstream;
- training traces for gradient ascent and each retain-regularised configuration, each row carrying forget accuracy, retain accuracy, target log-probability and forget-answer entropy at every evaluation checkpoint;
- a Pareto frontier plot of forgetting amount against retained accuracy, a paired bootstrap interval for the best frontier configuration versus gradient ascent, a paraphrase table (exact vs. paraphrased forget accuracy per method, a large gap between the two is the signature of surface-level suppression rather than robust forgetting), a relearning-speed comparison (steps to recover 80% of the original forget accuracy, unlearned model vs. oracle), and a membership-AUC table (values near 0.5 mean forget examples are no longer distinguishable from held-out examples by loss);
- an illustrative base-model WMDP-Bio accuracy printout (no unlearning applied, a lightweight check only) if WMDP is reachable in the run environment;
- an evidence-ladder table combining exact-forgotten, paraphrase-forgotten and membership-indistinguishable flags per method, a claim paragraph generated from the run's own numbers stating exactly what forget/retain accuracy each method achieved, and a sustainability table from the sequential-unlearning demo showing whether the first forget request's accuracy regressed once a second, unrelated request was applied on top of it.

Run the notebook (`jupyter notebook chapter_15_machine_unlearning_and_capability_reduction.ipynb`, executing all cells) to generate the actual tables, plot and claim paragraph described above.

## Data and dependencies

TOFU (`locuslab/TOFU`, configs `full`, `forget01`, `retain99`) is downloaded via `datasets`; since it is entirely synthetic, the notebook prints examples freely, unlike WMDP. Dataset configuration names are an active research artefact and can change; the notebook notes that the structure (full knowledge, a designated forget subset, its retained complement) is what matters if `locuslab/TOFU`'s exact config names differ from what is used here. WMDP-Bio (`cais/wmdp`) is used only for the small illustrative check in section 15.6, with the same no-raw-content discipline Chapter 14 established. LoRA adapters (`m_full`, `m_oracle`, plus the retain-regularised and gradient-ascent branches) are saved to `models/` (gitignored); result tables and the MUSE-style JSON report are saved to `results/chapter15/` (gitignored). Reuses the manual LoRA training loop pattern from Chapter 11. No new dependencies beyond earlier chapters (`peft`, `scikit-learn` for the ROC-AUC utility).

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 15/chapter_15_machine_unlearning_and_capability_reduction.ipynb"
```

Sample sizes and step counts are kept small so the notebook completes in a reasonable time on a laptop: `N_RETAIN_FINETUNE = 60`, `N_RETAIN_EVAL = 20`, `SFT_EPOCHS = 3`, `GA_STEPS = 40`, `RR_STEPS = 40` across four `(lambda_forget, lambda_retain)` configurations, `RELEARN_STEPS = 15`. The forget set itself (`forget01`) is used in full throughout, since it is already small. This notebook trains several LoRA branches in sequence (full, oracle, gradient-ascent, four retain-regularised configurations, plus a sequential-unlearning continuation), so total runtime is the largest of the six new chapters; raise the sample-size and step-count constants toward a larger-scale run for a more defensible forgetting-utility frontier.
