# Chapter 14: Dangerous Capabilities and Scalable Oversight

Every earlier chapter measured a failure close to the behaviour we care about: an unsafe response, a missed classification, a fine-tune's side effect, an agent crossing a permission boundary. This chapter measures something more awkward: whether a model has acquired a capability that would matter under different circumstances even if it normally refuses to use it, whether an agent can work autonomously across increasingly long tasks, and whether a human supervisor can still judge a system's work once that system becomes better at the task than the person checking it.

The notebook [`chapter_14_dangerous_capabilities_and_scalable_oversight.ipynb`](chapter_14_dangerous_capabilities_and_scalable_oversight.ipynb) currently covers five parts, matching the book's own scope but scaled down to run on a laptop: a hazardous-knowledge capability evaluation on WMDP (content handled as sensitive research material throughout), a small capability-scaling ladder across three Qwen3 sizes, a human-calibrated autonomy time-horizon proxy built on a local toy task suite, a synthetic control-evaluation (sabotage) simulation, and a weak-to-strong generalisation plus assisted-oversight (consultancy/debate) experiment on ARC-Challenge. Every experiment ends with an explicit statement of what it does, and does not, license us to conclude.

Running it top to bottom will:

- load WMDP's bio/chem/cyber splits and ARC-Challenge, score every multiple-choice question by comparing option log-probabilities rather than parsing free-form text (after verifying each answer label tokenises to a single token), and evaluate Qwen3-0.6B on each WMDP domain with bootstrap confidence intervals and a chance-adjusted accuracy, alongside an ARC-Challenge benign-utility control kept in a separate table from the hazardous score;
- repeat that same evaluation (identical scoring scaffold) across a three-model Qwen3 size ladder (0.6B / 1.7B / 4B), fit a descriptive log-odds-vs-log-parameters scaling curve, and immediately check its stability with a leave-one-model-out forecast rather than trusting the in-sample fit;
- build a small local toy task suite (13 short, objectively gradable tasks with author-assigned, illustrative human-time estimates, not measured baselines) to reproduce the mechanics of METR's human-calibrated time-horizon framework: fit a logistic regression of success against log(human minutes), extract 50%/80% horizon estimates, and bootstrap a task-level confidence interval (resampling whole tasks, not individual trials);
- run a fully synthetic sabotage/control-evaluation simulation for two agents with different hidden side-task capabilities, sweep a monitor's suspicion threshold to trace an ROC-style detection curve, and separate "low undetected-sabotage rate because the agent is weak" from "low rate because the monitor is strong";
- build a weak-to-strong generalisation experiment from first principles: Qwen3-0.6B as a noisy-label weak supervisor, frozen Qwen3-1.7B hidden-state features as the strong representation, two linear heads (one trained on weak labels, one on gold labels as an oracle) evaluated on held-out ARC-Challenge, a recovery-rate statistic (rho), a disagreement-subset analysis specifically on the weak supervisor's errors, and a comparison of confidence-filtering policies for the weak labels; and
- compare direct judging, consultancy (a stronger model argues for its own top answer) and debate (arguing for its own top-2 candidates) as ways to help the weak judge on the items it found hardest, including a deliberately misleading-consultant condition, with paired bootstrap intervals and a persuasion-rate statistic for how often an honest answer gets flipped by a dishonest argument.

**Scope note.** As of this notebook, section 14.6's outcome-vs-process scalable-oversight synthetic experiment described in the book is not yet implemented here; the weak-to-strong and assisted-oversight material above corresponds to the book's sections 14.6-14.8 material on generalisation and assisted judging. This README describes the notebook's current content as read; it may be extended later.

## What the results actually looked like

This notebook has not been executed in this checkout, its cells carry no cached outputs, so there are no run numbers to report here yet. Each part is built to produce a specific artefact when run:

- per-domain WMDP accuracy with bootstrap 95% intervals and chance-adjusted scores, printed alongside an ARC-Challenge benign-control accuracy, with an explicit reminder that WMDP asks whether the model *can* answer hazardous-knowledge questions while a refusal test asks whether the deployed assistant *will*, and that the two should never be collapsed into one number;
- a three-model capability profile table (WMDP domains plus benign control, each with bootstrap intervals) and a leave-one-out forecast-error printout per metric, where a large forecast error is itself a legitimate result about how well a three-point, parameter-count-only scaling law describes this capability under this elicitation procedure;
- either a fitted 50%/80% time horizon in minutes with a task-level bootstrap interval, or an explicit "degenerate outcome" message if every trial at this toy scale succeeded or failed uniformly (a real possibility the notebook checks for rather than assuming);
- an ROC-style monitor-detection plot and a threshold-0.5 summary table (main-task success, side-task success with no monitor, monitor TPR/FPR, undetected-side-task rate) for the low- and high-side-task-capability agents;
- printed weak-supervisor, weakly-supervised-student and oracle-student accuracies plus the recovery ratio rho, a disagreement-subset breakdown (recovery on the weak supervisor's errors vs. regression on its correct predictions), and a three-way comparison of confidence-filtering policies for the weak labels;
- a table of direct/consultancy/debate accuracy on the hardest ARC-Challenge items, paired bootstrap intervals for consultancy-vs-direct and debate-vs-direct, and a wrong-answer persuasion rate for the deliberately misleading-consultant condition; and
- a final claim paragraph, generated from that specific run's numbers, connecting the scaling ladder's hazardous-vs-benign trend to the weak-to-strong recovery fraction, with explicit caveats about what none of it establishes (real-world dangerousness, a validated job-automation percentage, or a solution to superhuman oversight).

Run the notebook (`jupyter notebook chapter_14_dangerous_capabilities_and_scalable_oversight.ipynb`, executing all cells) to generate the actual tables, plot and claim paragraph described above.

## Data and dependencies

WMDP (`cais/wmdp`, all three configs) and ARC-Challenge (`allenai/ai2_arc`) are downloaded via `datasets`. Consistent with the book's content-handling discipline, no raw WMDP question or choice text is ever printed or saved by this notebook, only example ids, predicted/target choice indices, correctness and log-probability margins. Results (capability profile, toy time-horizon attempts, control-evaluation summary, weak-label filtering comparison, assisted-oversight protocol results) are saved to `results/chapter14/` (gitignored). Loading the 4B checkpoint is meaningfully heavier than 0.6B or 1.7B; the notebook notes that `MODELS` can be trimmed to two entries on memory-constrained machines (the leave-one-out analysis still runs with less to hold out, though three points is what makes it meaningful). No new dependencies beyond earlier chapters.

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 14/chapter_14_dangerous_capabilities_and_scalable_oversight.ipynb"
```

Because every WMDP/ARC score comes from a single forward pass (option log-probabilities, not autoregressive generation) rather than free-form generation, most of this notebook is cheap even at larger sample sizes; the generation-heavy parts (the toy time-horizon suite and the consultancy/debate arguments) are kept smaller by default: `N_WMDP_PER_DOMAIN = 30`, `N_BENIGN_CONTROL = 40`, `N_ARC_TRAIN = 200`, `N_ARC_TEST = 80`, `N_TRIALS_PER_TOY_TASK = 3`, `N_DIFFICULT_OVERSIGHT_ITEMS = 12`. Two models (0.6B and 1.7B) stay loaded throughout for reuse across sections; raise the sample-size constants toward a larger-scale run for narrower confidence intervals.
