# Chapter 10: RLHF, Preference Optimisation and Reward Hacking

Chapter 9 built a reward model that could score two responses to the same prompt. On its own, that score is descriptive, the language model keeps generating exactly as before. This chapter takes the next step: using preference signal to actually *change the policy*, and then deliberately breaking that process to see what reward hacking looks like when every quantity involved is visible. We derive the PPO-style policy-gradient machinery and Direct Preference Optimisation from the same KL-regularised objective, run a real (small-scale) DPO fine-tune of Qwen3-0.6B with a LoRA adapter on the PKU-SafeRLHF harmlessness preferences audited in Chapter 9, evaluate the tuned policy against the reference model with independent behavioural checks, and then run a controlled best-of-N reward-hacking experiment where we know exactly why the proxy is flawed.

The notebook [`chapter_10_rlhf_and_reward_hacking.ipynb`](chapter_10_rlhf_and_reward_hacking.ipynb) reproduces the chapter's experiments end to end. Running it top to bottom will:

- work through a toy four-action policy-gradient problem where a hidden true utility and a deliberately mis-specified proxy reward are both visible, and watch REINFORCE push expected proxy reward up while expected true utility falls;
- sweep a KL penalty (`beta`) against a reference policy on the same toy problem and show the trade-off between preserved utility and permitted reward-seeking directly, then illustrate PPO's clipped surrogate objective numerically;
- derive the DPO loss from the same KL-regularised objective (policy-vs-reference log-probability margins, no explicit reward model or rollout loop), and unit-test the response-only sequence log-probability function on a worked example before trusting it across a dataset;
- fine-tune Qwen3-0.6B with a LoRA adapter using a from-scratch DPO training loop on PKU-SafeRLHF's harmlessness preferences (audited in Chapter 9), with cached reference log-probabilities and a full training log (loss, relative margin, gradient norm per step);
- evaluate the tuned policy against the untouched reference model: held-out preference accuracy first, then paired generation under matched decoding settings, reusing Chapter 6's JailbreakBench harness and BeaverTails-classifier judge to measure harmful-compliance and over-refusal, then paired bootstrap confidence intervals on both changes; and
- run a controlled reward-hacking experiment on a harmless QA task: a known true utility (correct, concise, not overconfident) versus a deliberately mis-specified proxy (rewards length, confidence markers and heading-like structure almost as much as correctness), a best-of-N optimisation-pressure sweep from N=1 to N=32, a feature regression that names the exploited shortcut, and a causal intervention (removing the shortcut from the proxy) that tests whether it was actually responsible for the gap.

## What the results actually looked like

This notebook has not been executed in this checkout, its cells carry no cached outputs, so there are no run numbers to report here yet. Every experiment above is built to produce a specific, checkable signature rather than a single headline figure, and running the notebook top to bottom is what produces them:

- the toy REINFORCE experiment should show expected proxy reward climbing steadily while expected true utility falls, and the `beta` sweep should show larger `beta` preserving more of the original utility at the cost of smaller proxy-reward gains, printed as a small summary table (`final_proxy`, `final_utility`, `final_kl` per `beta`);
- the DPO training log should show `relative_margin` (how much more strongly the policy prefers the winner than the reference already did) rising over the run, not just loss falling, this is the notebook's own diagnostic for "DPO is doing something" versus a masking or batching bug;
- the reference-vs-DPO evaluation table should be read across all four rows together (harmful-compliance rate, benign-refusal rate, and mean response length on both harmful and benign prompts) rather than the harmful-compliance row alone, since a harmful-compliance improvement paired with a large over-refusal increase and collapsed response length would indicate blanket refusal rather than a targeted safety change; the paired bootstrap intervals in this section will be wide at the notebook's default `N_EVAL = 15`, which is expected, not a bug;
- the best-of-N sweep should show proxy reward continuing to climb with N while true utility plateaus or falls, and the selected responses' length and confidence-marker counts rising with N even though neither is part of the true utility; the standardized proxy gap and the top-decile tail utility should both move in the direction that says the proxy is least trustworthy exactly where optimisation pressure is strongest; and
- the causal intervention (correcting the proxy to reward only correctness) should shrink or remove that gap, which is the notebook's evidence that the length/confidence shortcut, not something else, was actually driving the divergence.

Run the notebook (`jupyter notebook chapter_10_rlhf_and_reward_hacking.ipynb`, executing all cells) to generate the actual figures, tables and printed diagnostics described above.

## Data and dependencies

PKU-SafeRLHF is downloaded automatically via the `datasets` library, reusing the same prompt-disjoint train/validation split logic Chapter 9 built (`GroupShuffleSplit`). The harmful/benign red-team prompts for the behavioural evaluation come from JailbreakBench's underlying Hugging Face dataset, `dedeswim/JBB-Behaviors`, loaded directly through `datasets` rather than the `jailbreakbench` PyPI package, for the same Python 3.13 compatibility reasons Chapter 6 documents. The Chapter 3 BeaverTails classifier (`../Chapter 3/models/beavertails_tfidf_logreg.joblib`) is reused as the harmful-response judge if present, and retrained from BeaverTails automatically if not. The trained LoRA adapter and its run config are written to `models/` (gitignored, regenerated by running the notebook); this adapter is also the one Chapter 16's capstone loads (or regenerates, if this notebook has not been run first).

New dependency: `peft` (LoRA), already added to `requirements.txt` by Chapter 11. Training is a plain PyTorch loop implementing the DPO loss directly, not a trainer library.

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 10/chapter_10_rlhf_and_reward_hacking.ipynb"
```

This is the first chapter that actually trains the language model (a LoRA adapter, not the full weights) rather than only running inference over it. Measured on Apple Silicon MPS, one batched DPO training step (batch size 4, chosen and rejected together) took about 3.2 seconds during authoring. The defaults (`N_TRAIN = 200`, `N_VAL = 50`, `N_EVAL = 15` harmful/benign prompts per condition) keep the whole notebook, DPO training plus every evaluation plus the reward-hacking study, to well under 30 minutes; the book's own suggested scale (thousands of preference pairs) is a multi-hour undertaking and is left as a scaling exercise, not a debugging default.
