# Chapter 13: Agent Safety, Tools, Permissions and Control

Chapter 12 moved the unit of analysis from the model to the architecture around it, but the model was still only producing text that a human or a downstream system might act on. Agents take that one step further: the model chooses an action, observes the result of that action, and chooses another one. A model output becomes a sequence of real state changes, and a trajectory can look fine at the end while having done something unauthorised along the way.

The notebook [`chapter_13_agent_safety_tools_permissions_and_control.ipynb`](chapter_13_agent_safety_tools_permissions_and_control.ipynb) builds a small, fully-observable sandbox (fictional project records, notes, drafts and temporary files, all in-memory) and a real tool-calling agent loop around Qwen3-0.6B, then uses that sandbox to turn several Chapter 1 ideas into measured variables: least privilege, indirect prompt injection as a trajectory attack, approval gates for high-impact actions, long-horizon reliability, and cross-task memory contamination. Every proposed action, policy decision and executed action is kept as separate evidence throughout, since collapsing them into one pass/fail label throws away exactly the information a safety report needs.

Running it top to bottom will:

- build a deterministic, in-memory sandbox (`Workspace`) with real ground-truth state transitions, a set of six tools with research-only impact tiers, three tasks whose authority is encoded as data rather than inferred from the model's own words, and an executor with three independent gates (schema validity, policy authorisation, and an evidence-based approval check that inspects the workspace state rather than the model's stated reasoning);
- plug Qwen3-0.6B into the sandbox through a deliberately boring JSON-only tool-calling protocol with no automatic repair layer, establish a clean unattacked capability baseline, and then run a least-privilege comparison across broad, role-based and task-scoped permission regimes on the same two tasks, measuring task success, security success and joint success together;
- run an indirect-prompt-injection experiment where a tool observation (a note file) contains an embedded instruction trying to redirect the agent's trajectory toward an unauthorised, higher-impact action, crossed with trust-boundary wording and deterministic policy enforcement as two independent ablation switches, separating model-level attack success (did it propose the action) from system-level attack success (did it execute);
- implement an evidence-based approval gate for a task where the high-impact action is legitimately authorised by policy but not actually supported by the evidence in the note, and measure review rate and task success with and without the gate;
- inspect long-horizon reliability with a first-divergence trace, a harmless rollback drill, and a repeated-trials `pass_at_all` check compared against the naive independence approximation from the book's own section 13.6;
- seed a poisoned memory as if a prior episode already happened and measure whether it contaminates a second, unrelated clean episode under three memory-presentation policies (unscoped, provenance-aware, task-scoped); and
- compare the sandbox's design against AgentDojo, ToolEmu, tau-bench, ToolSandbox and Agent Security Bench in a small table, assemble a factorial safety-utility summary and paired bootstrap interval from the experiments already run, and generate a safety-case paragraph from that run's own numbers.

## What the results actually looked like

This notebook has not been executed in this checkout, its cells carry no cached outputs, so there are no run numbers to report here yet. Running the notebook is designed to produce, at minimum:

- a clean capability baseline (task success, parse-error rate, mean step count under broad permissions with no attack), which every later attack-success number should be read as conditional on;
- a least-privilege summary table (`task_success`, `joint_success`, unauthorised-proposal rate, unauthorised-execution rate, unnecessary-action count, median latency) across `broad` / `role_based` / `task_scoped` regimes for both tasks;
- an injection ablation table crossing `trust_boundary` and `policy_enforcement`, reporting separate model-level (`asr_model`) and system-level (`asr_system`) attack success rates, plus a companion table of attack-induced utility degradation (clean task success minus attacked task success) per configuration;
- an approval-gate comparison showing review-trigger rate and task success with the gate on versus off, illustrating that policy enforcement alone cannot catch an authorised-but-evidentially-unsupported action;
- a printed comparison of the empirical trajectory-level failure rate against the naive `1 - (1-p)^n` independence approximation (not expected to match closely, that mismatch is itself the point);
- a memory-contamination table showing task success and cross-episode contamination (proposed and executed) under the three memory policies; and
- a factorial summary, a paired bootstrap interval for task-scoped-minus-broad joint success, and a safety-case paragraph generated from that run's own numbers, stating the best-performing permission regime, the effect of enforcement alone versus enforcement plus trust boundaries, and the observed contamination rates under each memory policy.

Run the notebook (`jupyter notebook chapter_13_agent_safety_tools_permissions_and_control.ipynb`, executing all cells) to generate the actual tables and safety-case paragraph described above.

## Data and dependencies

The sandbox, its fictional project records, tasks and injected note are entirely synthetic and generated inside the notebook, no external dataset is used for the agent environment itself. Qwen3-0.6B is loaded the same way earlier chapters load it. Summary tables (privilege, injection ablation, approval traces, memory contamination, factorial summary, benchmark comparison) are saved to `results/chapter13/` (gitignored). No new dependencies beyond earlier chapters.

## Running it

From the repository root:

```bash
pip install -r requirements.txt
jupyter notebook "Chapter 13/chapter_13_agent_safety_tools_permissions_and_control.ipynb"
```

Every experiment grid uses a small number of repeats per configuration to keep runtime reasonable on a laptop: `N_SEEDS_PRIVILEGE = 3`, `N_SEEDS_INJECTION = 3`, `N_SEEDS_APPROVAL = 3`, `N_SEEDS_MEMORY = 3`, with `MAX_STEPS = 8` per trajectory. Because Qwen3-0.6B is a small model doing multi-step JSON tool-calling, expect a non-trivial parse-error and capability-baseline-failure rate at this scale, itself a legitimate result about a 0.6B model's tool-use capability, not a notebook bug. Raise the `N_SEEDS_*` constants toward a larger-scale run for narrower confidence intervals, following the same pattern as earlier chapters' `N_*` constants.
