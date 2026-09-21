"""Make chapter notebook artifact paths independent of Jupyter's launch directory.

Run from the repository root with python scripts/normalize_notebook_artifact_paths.py. This is intentionally a mechanical notebook migration;
the notebooks remain the source of truth after it has run.
"""

from __future__ import annotations

import json
from pathlib import Path
from notebook_storage_updates import enhance_notebook


ROOT = Path(__file__).resolve().parents[1]


def bootstrap(chapter: str) -> str:
    template = (ROOT / "scripts" / "notebook_storage.py").read_text()
    return template.replace('CHAPTER_NAME = "__CHAPTER__"', f'CHAPTER_NAME = "{chapter}"')


def source(cell: dict) -> str:
    return "".join(cell.get("source", []))


def set_source(cell: dict, value: str) -> None:
    cell["source"] = value.splitlines(keepends=True)


def replace_all(nb: dict, replacements: dict[str, str]) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = source(cell)
        for old, new in replacements.items():
            text = text.replace(old, new)
        set_source(cell, text)


def replace_all_once(nb: dict, replacements: dict[str, str]) -> None:
    for cell in nb["cells"]:
        if cell.get("cell_type") != "code":
            continue
        text = source(cell)
        for old, new in replacements.items():
            if new not in text:
                text = text.replace(old, new)
        set_source(cell, text)


notebooks = sorted(ROOT.glob("Chapter */*.ipynb"))
for path in notebooks:
    chapter = path.parent.name
    nb = json.loads(path.read_text())
    setup_id = "persistent-artifact-setup"
    setup = next((c for c in nb["cells"] if c.get("id") == setup_id), None)
    if setup is None:
        setup = {"cell_type": "code", "execution_count": None, "metadata": {},
                 "outputs": [], "id": setup_id, "source": []}
        nb["cells"].insert(1, setup)
        nb["cells"].insert(1, {
            "cell_type": "markdown", "id": "persistent-artifact-guide", "metadata": {},
            "source": [
                "## Persistent artefacts (run this setup first)\n",
                "On Colab, authorise Google Drive when prompted. All chapters share "
                "`MyDrive/AI Safety from First Principles/artifacts/codes/`.\n",
                "Models and cross-chapter inputs use stable chapter paths; each execution "
                "gets a dated `runs/` folder for exported tables and figures. "
                "Run the final export cell too. Existing notebook outputs are historical, "
                "not evidence that this execution has finished. See `COLAB.md` for details.\n"
            ]
        })
    set_source(setup, bootstrap(chapter))
    for cell in nb["cells"]:
        if cell.get("id") == "persistent-artifact-guide":
            set_source(cell, source(cell).replace("\\`", "`"))

    # Every chapter owns its generated outputs under its own directory.
    replace_all_once(nb, {
        'Path("data/processed")': 'CHAPTER_DIR / "data/processed"',
        'Path("data/external/BBQ")': 'CHAPTER_DIR / "data/external/BBQ"',
        'Path("data/external/BOLD")': 'CHAPTER_DIR / "data/external/BOLD"',
        'Path("data/processed/reward_embeddings")': 'CHAPTER_DIR / "data/processed/reward_embeddings"',
        'Path("models")': 'CHAPTER_DIR / "models"',
        'Path("config")': 'CHAPTER_DIR / "config"',
        'Path("reports")': 'CHAPTER_DIR / "reports"',
        'Path("registry")': 'CHAPTER_DIR / "registry"',
        'Path("results/chapter7")': 'CHAPTER_DIR / "results/chapter7"',
        'Path("results/chapter11")': 'CHAPTER_DIR / "results/chapter11"',
        'Path("results/chapter12")': 'CHAPTER_DIR / "results/chapter12"',
        'Path("results/chapter13")': 'CHAPTER_DIR / "results/chapter13"',
        'Path("results/chapter14")': 'CHAPTER_DIR / "results/chapter14"',
        'Path("results/chapter15")': 'CHAPTER_DIR / "results/chapter15"',
        'Path("results/chapter16")': 'CHAPTER_DIR / "results/chapter16"',
        'Path(RESULTS_DIR_NAME)': 'CHAPTER_DIR / RESULTS_DIR_NAME',
        'Path("../Chapter 3/models/beavertails_tfidf_logreg.joblib")': 'CODES_DIR / "Chapter 03" / "models/beavertails_tfidf_logreg.joblib"',
        'Path("../Chapter 10/models/qwen3_0.6b_dpo_harmlessness_lora")': 'CODES_DIR / "Chapter 10" / "models/qwen3_0.6b_dpo_harmlessness_lora"',
    })

    # Chapter 2 writes its split; every consumer reads that same canonical split.
    if chapter in {"Chapter 03", "Chapter 04", "Chapter 05", "Chapter 12"}:
        replace_all(nb, {
            'DATA_DIR = CHAPTER_DIR / "data/processed"': 'DATA_DIR = CODES_DIR / "Chapter 02" / "data/processed"',
            'df[["prompt", "response", "text", "target"]]': 'df[["prompt", "response", "text", "category", "target"]]',
        })

    if chapter == "Chapter 02":
        replace_all(nb, {
            'os.makedirs("data/processed", exist_ok=True)\n': '',
            'train_df.to_parquet("data/processed/beavertails_train.parquet", index=False)': 'DATA_DIR = CHAPTER_DIR / "data/processed"\nDATA_DIR.mkdir(parents=True, exist_ok=True)\ntrain_df.to_parquet(DATA_DIR / "beavertails_train.parquet", index=False)',
            'test_df.to_parquet("data/processed/beavertails_test.parquet", index=False)': 'test_df.to_parquet(DATA_DIR / "beavertails_test.parquet", index=False)',
            'print(" data/processed/beavertails_train.parquet", train_df.shape)': 'print(" ", DATA_DIR / "beavertails_train.parquet", train_df.shape)',
            'print(" data/processed/beavertails_test.parquet", test_df.shape)': 'print(" ", DATA_DIR / "beavertails_test.parquet", test_df.shape)',
            'df[["prompt", "response", "text", "target"]]': 'df[["prompt", "response", "text", "category", "target"]]',
        })

    # Chapter 4 reads Chapter 3's model but writes its own calibrated artifacts.
    if chapter == "Chapter 04":
        replace_all(nb, {
            'MODEL_PATH = MODEL_DIR / "beavertails_tfidf_logreg.joblib"': 'MODEL_PATH = CODES_DIR / "Chapter 03" / "models/beavertails_tfidf_logreg.joblib"',
        })
        replace_all_once(nb, {
            'MODEL_DIR = CHAPTER_DIR / "models"\nTRAIN_PATH': 'MODEL_DIR = CHAPTER_DIR / "models"\nMODEL_DIR.mkdir(parents=True, exist_ok=True)\nTRAIN_PATH',
            '# matching the reproducibility checklist in section 4.39.\njoblib.dump': '# matching the reproducibility checklist in section 4.39.\nMODEL_DIR.mkdir(parents=True, exist_ok=True)\njoblib.dump',
        })

    # Chapter 5 consumes Chapter 4's frozen artifacts; only its reports stay local.
    if chapter == "Chapter 05":
        replace_all(nb, {
            'MODEL_DIR = CHAPTER_DIR / "models"': 'MODEL_DIR = CODES_DIR / "Chapter 04" / "models"',
            'CONFIG_DIR = CHAPTER_DIR / "config"': 'CONFIG_DIR = CODES_DIR / "Chapter 04" / "config"',
        })
        replace_all_once(nb, {
            '    train_df = pd.read_parquet(TRAIN_PATH)\n    test_df = pd.read_parquet(TEST_PATH)\nelse:\n    print("Processed split not found, regenerating it from BeaverTails (same steps as Chapter 2)...")': '''    train_df = pd.read_parquet(TRAIN_PATH)
    test_df = pd.read_parquet(TEST_PATH)

    # Early Chapter 2 artifacts omitted `category`. Repair that legacy schema once;
    # this only rebuilds the deterministic data split and never retrains a model.
    if "category" not in train_df.columns or "category" not in test_df.columns:
        print("Upgrading the Chapter 2 split to include category metadata...")
        from datasets import load_dataset
        from sklearn.model_selection import train_test_split

        dataset = load_dataset("PKU-Alignment/BeaverTails", split="30k_train")
        df = dataset.to_pandas()
        df["text"] = (
            "[PROMPT]\\n" + df["prompt"].fillna("")
            + "\\n\\n[RESPONSE]\\n" + df["response"].fillna("")
        )
        df["target"] = (~df["is_safe"]).astype(int)
        train_df, test_df = train_test_split(
            df[["prompt", "response", "text", "category", "target"]],
            test_size=0.20, stratify=df["target"], random_state=42,
        )
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        train_df.to_parquet(TRAIN_PATH, index=False)
        test_df.to_parquet(TEST_PATH, index=False)
        print("Upgraded Chapter 2 split:", TRAIN_PATH, TEST_PATH)
else:
    print("Processed split not found, regenerating it from BeaverTails (same steps as Chapter 2)...")''',
            'category_df = pd.json_normalize(test_df["category"]).astype(bool)': '''from sklearn.metrics import recall_score

if "category" not in test_df.columns:
    raise RuntimeError(
        "This is a legacy Chapter 2 split without category metadata. Rerun this notebook "
        "from its setup cell once; it will upgrade the saved split without retraining a model."
    )

category_df = pd.json_normalize(test_df["category"]).astype(bool)''',
            '\nfrom sklearn.metrics import recall_score\ncategory_robustness = pd.DataFrame(category_rows).sort_values("delta")': '''
category_robustness = pd.DataFrame(category_rows)
if not category_robustness.empty:
    category_robustness = category_robustness.sort_values("delta")''',
        })

    if chapter == "Chapter 16":
        replace_all_once(nb, {
            'if not CH10_ADAPTER_PATH.exists() and not REBUILD_MISSING_ARTIFACTS:\n    raise FileNotFoundError(\n        f"Required Chapter 10 adapter is missing: {CH10_ADAPTER_PATH}. Run Chapter 10 first, "\n        "or explicitly set REBUILD_MISSING_ARTIFACTS=True (this retraining can take over an hour)."\n    )\n\nif CH10_ADAPTER_PATH.exists():\n    print(f"Loading the DPO adapter Chapter 10 already trained, from {CH10_ADAPTER_PATH}")\n    ADAPTER_PATH = CH10_ADAPTER_PATH': '''def adapter_is_complete(path):
    return (
        (path / "adapter_config.json").is_file()
        and any((path / name).is_file() for name in ("adapter_model.safetensors", "adapter_model.bin"))
    )


if adapter_is_complete(CH10_ADAPTER_PATH):
    ADAPTER_PATH = CH10_ADAPTER_PATH
elif adapter_is_complete(LOCAL_ADAPTER_PATH):
    # Reuse a prior explicit fallback rebuild instead of training it again.
    ADAPTER_PATH = LOCAL_ADAPTER_PATH
else:
    ADAPTER_PATH = None

if ADAPTER_PATH is None and not REBUILD_MISSING_ARTIFACTS:
    raise FileNotFoundError(
        "No complete Chapter 10 DPO adapter was found. Expected adapter_config.json and "
        f"adapter weights under {CH10_ADAPTER_PATH}. A valid prior Chapter 16 fallback at "
        f"{LOCAL_ADAPTER_PATH} is also accepted. Run Chapter 10 first, or explicitly set "
        "REBUILD_MISSING_ARTIFACTS=True (this retraining can take over an hour)."
    )

if ADAPTER_PATH is not None:
    print(f"Loading the existing DPO adapter from {ADAPTER_PATH}")''',
            'run_config = json.loads((CH10_ADAPTER_PATH.parent / "qwen3_0.6b_dpo_harmlessness_run_config.json").read_text()) \\\n        if (CH10_ADAPTER_PATH.parent / "qwen3_0.6b_dpo_harmlessness_run_config.json").exists() else \\\n        {"model": MODEL_NAME, "preference_dimension": "harmlessness", "beta": BETA, "source": "chapter_10"}': 'run_config_path = ADAPTER_PATH.parent / "qwen3_0.6b_dpo_harmlessness_run_config.json"\n    run_config = json.loads(run_config_path.read_text()) if run_config_path.exists() else \\\n        {"model": MODEL_NAME, "preference_dimension": "harmlessness", "beta": BETA, "source": "existing_adapter"}',
        })

    # Cross-chapter inputs are contracts. Missing inputs stop with an actionable error by
    # default; setting REBUILD_MISSING_ARTIFACTS=True restores the documented fallback.
    replace_all_once(nb, {
        'if TRAIN_PATH.exists() and TEST_PATH.exists():': 'if not (TRAIN_PATH.exists() and TEST_PATH.exists()) and not REBUILD_MISSING_ARTIFACTS:\n    raise FileNotFoundError(\n        f"Required Chapter 2 split is missing: {TRAIN_PATH} / {TEST_PATH}. "\n        "Run Chapter 2 first, or explicitly set REBUILD_MISSING_ARTIFACTS=True."\n    )\n\nif TRAIN_PATH.exists() and TEST_PATH.exists():',
        'if MODEL_PATH.exists():\n    safety_clf = joblib.load(MODEL_PATH)': 'if not MODEL_PATH.exists() and not REBUILD_MISSING_ARTIFACTS:\n    raise FileNotFoundError(\n        f"Required Chapter 3 model is missing: {MODEL_PATH}. Run Chapter 3 first, "\n        "or explicitly set REBUILD_MISSING_ARTIFACTS=True."\n    )\n\nif MODEL_PATH.exists():\n    safety_clf = joblib.load(MODEL_PATH)',
        'if BASE_MODEL_PATH.exists() and CALIBRATOR_PATH.exists() and POLICY_PATH.exists():': 'required_ch4_artifacts = (BASE_MODEL_PATH, CALIBRATOR_PATH, POLICY_PATH)\nmissing_ch4_artifacts = [p for p in required_ch4_artifacts if not p.exists()]\nif missing_ch4_artifacts and not REBUILD_MISSING_ARTIFACTS:\n    raise FileNotFoundError(\n        "Required Chapter 4 artifacts are missing: " + ", ".join(map(str, missing_ch4_artifacts))\n        + ". Run Chapter 4 first, or explicitly set REBUILD_MISSING_ARTIFACTS=True."\n    )\n\nif BASE_MODEL_PATH.exists() and CALIBRATOR_PATH.exists() and POLICY_PATH.exists():',
        'if CH3_MODEL_PATH.exists():\n    beavertails_clf = joblib.load(CH3_MODEL_PATH)': 'if not CH3_MODEL_PATH.exists() and not REBUILD_MISSING_ARTIFACTS:\n    raise FileNotFoundError(\n        f"Required Chapter 3 judge is missing: {CH3_MODEL_PATH}. Run Chapter 3 first, "\n        "or explicitly set REBUILD_MISSING_ARTIFACTS=True."\n    )\n\nif CH3_MODEL_PATH.exists():\n    beavertails_clf = joblib.load(CH3_MODEL_PATH)',
        'if CH10_ADAPTER_PATH.exists():\n    print(f"Loading the DPO adapter Chapter 10 already trained, from {CH10_ADAPTER_PATH}")': 'if not CH10_ADAPTER_PATH.exists() and not REBUILD_MISSING_ARTIFACTS:\n    raise FileNotFoundError(\n        f"Required Chapter 10 adapter is missing: {CH10_ADAPTER_PATH}. Run Chapter 10 first, "\n        "or explicitly set REBUILD_MISSING_ARTIFACTS=True (this retraining can take over an hour)."\n    )\n\nif CH10_ADAPTER_PATH.exists():\n    print(f"Loading the DPO adapter Chapter 10 already trained, from {CH10_ADAPTER_PATH}")',
    })

    enhance_notebook(nb, chapter)
    path.write_text(json.dumps(nb, indent=2 if chapter == "Chapter 11" else 1, ensure_ascii=False) + "\n")

print(f"Updated {len(notebooks)} notebooks")
