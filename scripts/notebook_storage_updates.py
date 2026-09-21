"""Storage-only additions applied by normalize_notebook_artifact_paths.py."""
import re


def text(cell):
    return "".join(cell.get("source", []))


def set_text(cell, value):
    cell["source"] = value.splitlines(keepends=True)


def add_after(nb, needle, name, source):
    identity = "artefact-" + name
    existing = next((c for c in nb["cells"] if c.get("id") == identity), None)
    if existing:
        set_text(existing, source)
        return
    matches = [i for i, c in enumerate(nb["cells"])
               if c["cell_type"] == "code" and needle in text(c)
               and not c.get("id", "").startswith("artefact-")]
    if len(matches) != 1:
        raise ValueError(f"Expected one anchor for {identity}: {needle!r}, got {matches}")
    nb["cells"].insert(matches[0] + 1, {
        "cell_type": "code", "id": identity, "execution_count": None,
        "outputs": [], "metadata": {}, "source": source.splitlines(keepends=True),
    })


def enhance_notebook(nb, chapter):
    for cell in nb["cells"]:
        if cell["cell_type"] != "code" or cell.get("id") == "persistent-artifact-setup":
            continue
        s = text(cell)
        # Keep figures before notebook backends close them at show().
        if "save_figures()" not in s:
            s = re.sub(r"(?m)^( *)plt\.show\(\)", r"\1save_figures()\n\1plt.show()", s)
        # Persist tokenizer and a dated copy alongside each existing weight save.
        s = re.sub(r"(?m)^( *)(\w+)\.save_pretrained\(([^\n]+)\)$",
                   r"\1save_model_artifact(\2, \3, tokenizer)", s)
        # Persist explicit fallback classifiers if a reader opts into rebuilding.
        anchor = '    beavertails_clf.fit(bt_df["text"], bt_df["target"])'
        if anchor in s and "joblib.dump(beavertails_clf, CH3_MODEL_PATH)" not in s:
            s = s.replace(anchor, anchor + '\n    CH3_MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)\n    joblib.dump(beavertails_clf, CH3_MODEL_PATH)')
        if chapter == "Chapter 04":
            s = s.replace('    joblib.dump(safety_clf, MODEL_PATH)',
                          '    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)\n    joblib.dump(safety_clf, MODEL_PATH)') if '    MODEL_PATH.parent.mkdir' not in s else s
        if chapter == "Chapter 05":
            # Chapter 5 owns its shift registry; Chapter 4's policy is read-only input.
            s = s.replace('with open(CONFIG_DIR / "ch5_shift_registry.yaml", "w") as f:',
                          'CH5_CONFIG_DIR = CHAPTER_DIR / "config"\nCH5_CONFIG_DIR.mkdir(parents=True, exist_ok=True)\nwith open(CH5_CONFIG_DIR / "ch5_shift_registry.yaml", "w") as f:')
            s = s.replace('print((CONFIG_DIR / "ch5_shift_registry.yaml").read_text())',
                          'print((CH5_CONFIG_DIR / "ch5_shift_registry.yaml").read_text())')
        if chapter == "Chapter 10":
            s = s.replace('print("Harmful-compliance change 95% CI :", paired_bootstrap_difference(reference_harmful["jailbreak_success"], dpo_harmful["jailbreak_success"]))',
                          'harmful_change_ci = paired_bootstrap_difference(reference_harmful["jailbreak_success"], dpo_harmful["jailbreak_success"])\nprint("Harmful-compliance change 95% CI :", harmful_change_ci)')
            s = s.replace('print("Benign-refusal change 95% CI     :", paired_bootstrap_difference(reference_benign["refused"].astype(float), dpo_benign["refused"].astype(float)))',
                          'benign_change_ci = paired_bootstrap_difference(reference_benign["refused"].astype(float), dpo_benign["refused"].astype(float))\nprint("Benign-refusal change 95% CI     :", benign_change_ci)')
            s = s.replace('"prompt_hash": hash(row["goal"]) & 0xFFFFFFFF, "refused": refused,',
                          '"prompt_hash": __import__("hashlib").sha256(row["goal"].encode()).hexdigest(),\n            "prompt": row["goal"], "response": response, "policy_condition": label, "refused": refused,')
            if 'def evaluate_policy_on(' in s and 'save_tables("behaviour"' not in s:
                s = s.replace('    return pd.DataFrame(rows)',
                              '        save_tables("behaviour", **{label.replace("/", "_"): pd.DataFrame(rows)})\n    return pd.DataFrame(rows)')
        set_text(cell, s)

    if chapter == "Chapter 09":
        for name, anchor in [("help_lr", "w_help = help_lr.coef_[0]"),
                             ("safe_lr", "w_safe = safe_lr.coef_[0]")]:
            add_after(nb, anchor, name, f'''import joblib
_reward_dir = CHAPTER_DIR / "models/reward_heads"
_reward_dir.mkdir(parents=True, exist_ok=True)
_reward_archive = RUN_DIR / "checkpoints/reward_heads"
_reward_archive.mkdir(parents=True, exist_ok=True)
joblib.dump({name}, _reward_archive / "{name}.joblib")
__import__("shutil").copyfile(_reward_archive / "{name}.joblib", _reward_dir / "{name}.joblib")
save_artifact_json(_reward_dir / "encoder_config.json", {{
    "model": MODEL_NAME, "pooling": "attention-masked mean", "max_length": 256,
    "format": "[PROMPT]\\n{{prompt}}\\n\\n[RESPONSE]\\n{{response}}", "run_id": RUN_ID,
}})
''')
        add_after(nb, 'np.save(CACHE_DIR / "h1_val.npy", h1_val)', "embedding-rows", '''# These exact row orders are necessary to interpret the cached embedding arrays.
fit_sample.to_json(CACHE_DIR / "fit_rows.json", orient="split", index=True)
val_sample.to_json(CACHE_DIR / "validation_rows.json", orient="split", index=True)
save_artifact_json(CACHE_DIR / "metadata.json", {
    "model": MODEL_NAME, "run_id": RUN_ID, "fit_shape": list(h0_fit.shape),
    "validation_shape": list(h0_val.shape), "source": "PKU-Alignment/PKU-SafeRLHF",
})
''')
        for name, anchor in [("help_head", 'help_head = train_reward_head('),
                             ("safe_head", 'safe_head = train_reward_head('),
                             ("help_mlp", 'help_mlp = train_reward_head(')]:
            add_after(nb, anchor, name, f'''# Save the fitted head without pickling a notebook-defined Python class.
_head_dir = CHAPTER_DIR / "models/reward_heads"
_head_dir.mkdir(parents=True, exist_ok=True)
_head_archive = RUN_DIR / "checkpoints/reward_heads"
_head_archive.mkdir(parents=True, exist_ok=True)
_head_payload = {{"state_dict": {{k: v.detach().cpu() for k, v in {name}.state_dict().items()}},
                 "class_name": type({name}).__name__, "hidden_size": hidden_size,
                 "encoder_model": MODEL_NAME, "run_id": RUN_ID,
                 "pooling": "attention-masked mean", "max_length": 256,
                 "mlp_width": 256 if "{name}" == "help_mlp" else None,
                 "mlp_dropout": 0.1 if "{name}" == "help_mlp" else None}}
torch.save(_head_payload, _head_archive / "{name}.pt")
__import__("shutil").copyfile(_head_archive / "{name}.pt", _head_dir / "{name}.pt")
''')

    if chapter == "Chapter 10":
        add_after(nb, 'print(run_config)', "dpo-config", '''save_artifact_json(RUN_DIR / "dpo_run_config.json", run_config)
save_tables("preference_pairs", train=train_small, validation=val_small)
''')
        add_after(nb, 'train_log_df = pd.DataFrame(train_log)', "dpo-training", '''save_tables("training", training_log=train_log_df)
''')
        add_after(nb, 'print(pref_results[["reference_correct", "dpo_correct"]].mean())', "dpo-preferences", '''save_tables("preferences", validation_preferences=pref_results)
''')
        add_after(nb, 'result_table["change"]', "dpo-behaviour", '''save_tables("behaviour_summary", summary=result_table)
''')
        add_after(nb, 'candidates = pd.DataFrame(candidate_rows)', "reward-candidates", '''save_tables("reward_hacking", candidates=candidates)
''')
        add_after(nb, 'selected_df = pd.concat(selected_runs', "reward-selection", '''save_tables("reward_hacking", selected=selected_df, summary=summary)
''')
        add_after(nb, 'print("Harmful-compliance change 95% CI', "bootstrap", '''# Retain the intervals as machine-readable results as well as printed output.
save_artifact_json(RUN_DIR / "behaviour_intervals.json", {
    "harmful_compliance_change": harmful_change_ci.tolist(),
    "benign_refusal_change": benign_change_ci.tolist(),
})
''')

    if chapter == "Chapter 15":
        add_after(nb, 'ga_model, ga_trace = run_gradient_ascent', "unlearning-ga", '''save_model_artifact(ga_model, MODEL_DIR / "gradient_ascent", tokenizer)
save_tables("unlearning", gradient_ascent_trace=ga_trace)
''')
        # Save each method before subsequent experiments mutate the chosen model in place.
        for cell in nb["cells"]:
            s = text(cell)
            anchor = '    rr_models[label] = model'
            if anchor in s and 'save_model_artifact(model, MODEL_DIR / label' not in s:
                set_text(cell, s.replace(anchor, anchor + '\n    save_model_artifact(model, MODEL_DIR / label, tokenizer)\n    save_tables("unlearning", **{label: trace})'))
        add_after(nb, 'seq_model, seq_trace = run_retain_regularized_on', "unlearning-sequential", '''if best_rr_model is not None:
    save_model_artifact(seq_model, MODEL_DIR / "sequential_unlearning", tokenizer)
    save_tables("unlearning", sequential_trace=seq_trace)
''')

    if chapter == "Chapter 16":
        # Resolve only complete adapters; an empty directory is not a checkpoint.
        for cell in nb["cells"]:
            s = text(cell)
            start = s.find('if not CH10_ADAPTER_PATH.exists() and not REBUILD_MISSING_ARTIFACTS:')
            end = s.find('    policy_base =', start)
            if start >= 0 and end >= 0:
                s = s[:start] + '''def adapter_is_complete(path):
    return ((path / "adapter_config.json").is_file()
            and any((path / filename).is_file() and (path / filename).stat().st_size > 0
                    for filename in ("adapter_model.safetensors", "adapter_model.bin")))

ADAPTER_PATH = next((p for p in (CH10_ADAPTER_PATH, LOCAL_ADAPTER_PATH)
                     if adapter_is_complete(p)), None)
if ADAPTER_PATH is None and not REBUILD_MISSING_ARTIFACTS:
    raise FileNotFoundError(f"Run Chapter 10 first. No complete DPO adapter at {CH10_ADAPTER_PATH}. "
                            "A checkpoint needs adapter_config.json and adapter weights. "
                            "Explicitly set REBUILD_MISSING_ARTIFACTS=True only to retrain here.")

if ADAPTER_PATH is not None:
    print("Loading existing DPO adapter:", ADAPTER_PATH)
''' + s[end:]
                s = s.replace('(CH10_ADAPTER_PATH.parent /', '(ADAPTER_PATH.parent /')
                set_text(cell, s)

    identity = "artefact-final-export"
    final = next((c for c in nb["cells"] if c.get("id") == identity), None)
    if final is None:
        nb["cells"].extend([
            {"cell_type": "markdown", "metadata": {}, "id": "artefact-export-guide",
             "source": ["## Save remaining results\n", "Run this cell after the experiments (or before stopping early). It exports available tables, configuration and a file manifest into this run's Drive folder. Model weights are saved in their training cells; this is not an optimiser-state checkpoint.\n"]},
            {"cell_type": "code", "metadata": {}, "id": identity,
             "execution_count": None, "outputs": [],
             "source": ["save_figures()\n", "export_notebook_tables(globals())\n"]},
        ])
