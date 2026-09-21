"""Embedded into every notebook: no repository import is needed in Colab."""
import os
import json as _artifact_json
import platform as _artifact_platform
from pathlib import Path
from datetime import datetime, timezone
from uuid import uuid4
from importlib.metadata import version, PackageNotFoundError

CHAPTER_NAME = "__CHAPTER__"
REBUILD_MISSING_ARTIFACTS = False  # Explicit opt-in to legacy fallback training.


def _find_codes_dir():
    configured = os.environ.get("AI_SAFETY_CODES_DIR")
    try:
        from google.colab import drive
    except ImportError:
        drive = None
    if drive is not None:
        mount = Path("/content/drive")
        if not (mount / "MyDrive").is_dir():
            drive.mount(str(mount))
        if not (mount / "MyDrive").is_dir():
            raise RuntimeError("Google Drive is not mounted. Authorise Drive and rerun setup.")
        root = (Path(configured).expanduser() if configured else
                mount / "MyDrive/AI Safety from First Principles/artifacts/codes")
        # A typo must not silently redirect expensive outputs into disposable /content.
        root = root.resolve()
        if not root.is_relative_to(mount.resolve()):
            raise ValueError("In Colab, AI_SAFETY_CODES_DIR must be inside /content/drive. "
                             "Use a folder on the mounted Google Drive.")
    elif configured:
        root = Path(configured).expanduser().resolve()
    else:
        cwd = Path.cwd().resolve()
        root = next((candidate for parent in (cwd, *cwd.parents)
                     for candidate in (parent, parent / "codes")
                     if (candidate / CHAPTER_NAME).is_dir()),
                    cwd / "ai_safety_artifacts/codes")
    root.mkdir(parents=True, exist_ok=True)
    return root


CODES_DIR = _find_codes_dir()
CHAPTER_DIR = CODES_DIR / CHAPTER_NAME
CHAPTER_DIR.mkdir(parents=True, exist_ok=True)
RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ") + "-" + uuid4().hex[:8]
RUN_DIR = CHAPTER_DIR / "runs" / RUN_ID
RUN_DIR.mkdir(parents=True, exist_ok=False)


def save_artifact_json(path, payload):
    """Replace only after a full JSON write; surface storage errors immediately."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp-" + uuid4().hex)
    temporary.write_text(_artifact_json.dumps(payload, indent=2, default=str), encoding="utf-8")
    temporary.replace(path)


_artifact_versions = {}
_artifact_packages = ("torch", "transformers", "peft", "datasets", "numpy", "pandas", "scikit-learn")
for _package in _artifact_packages:
    try:
        _artifact_versions[_package] = version(_package)
    except PackageNotFoundError:
        pass
save_artifact_json(RUN_DIR / "environment.json", {
    "chapter": CHAPTER_NAME, "run_id": RUN_ID, "artifact_root": str(CODES_DIR),
    "python": _artifact_platform.python_version(), "packages_at_setup": _artifact_versions,
    "status": "started",
})


def save_tables(stage, **tables):
    """Save DataFrames with indexes and nested values in pandas split JSON."""
    directory = RUN_DIR / "tables" / stage
    directory.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        destination = directory / (name + ".json")
        temporary = destination.with_name(destination.name + ".tmp-" + uuid4().hex)
        table.to_json(temporary, orient="split", index=True, indent=2, force_ascii=False, double_precision=15)
        temporary.replace(destination)


def save_model_artifact(model, destination, tokenizer=None):
    """Archive a loadable checkpoint for this run, then publish its canonical copy.

    This saves model/adapter weights, not optimiser state for exact training resumption.
    """
    import shutil
    destination = Path(destination)
    relative = destination.relative_to(CHAPTER_DIR)
    archive = RUN_DIR / "checkpoints" / relative
    model.save_pretrained(archive)
    if tokenizer is not None:
        tokenizer.save_pretrained(archive)
    save_artifact_json(archive / "artifact_provenance.json", {
        "chapter": CHAPTER_NAME, "run_id": RUN_ID,
        "base_model": getattr(getattr(model, "config", None), "_name_or_path", None),
        "canonical_path": str(destination),
    })
    destination.mkdir(parents=True, exist_ok=True)
    for path in archive.rglob("*"):
        if path.is_file():
            target = destination / path.relative_to(archive)
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".tmp-" + uuid4().hex)
            shutil.copyfile(path, temporary)
            temporary.replace(target)
    print("Saved checkpoint:", destination, "| Run copy:", archive)


def save_figures():
    """Call immediately before plt.show(), which may close figures in notebooks."""
    import matplotlib.pyplot as plt
    directory = RUN_DIR / "figures"
    directory.mkdir(parents=True, exist_ok=True)
    for number in plt.get_fignums():
        figure = plt.figure(number)
        name = datetime.now(timezone.utc).strftime("%H%M%S%f") + f"-figure-{number}.png"
        figure.savefig(directory / name, dpi=150, bbox_inches="tight")


def export_notebook_tables(namespace):
    """Export remaining top-level tables and config; never pickle the live kernel."""
    import pandas as pd
    source_names = {"df", "train_df", "test_df", "safety_pairs", "dpo_train", "dpo_val",
                    "simpleqa_full", "simpleqa", "bbq_df", "bold_df"}
    skipped_sources = {name: len(value) for name, value in list(namespace.items())
                       if name in source_names and isinstance(value, pd.DataFrame) and len(value) > 5000}
    tables = {name: value for name, value in list(namespace.items())
              if not name.startswith("_") and isinstance(value, pd.DataFrame)}
    tables = {name: value for name, value in tables.items() if name not in skipped_sources}
    save_tables("final", **tables)
    config_names = {"MODEL_ID", "MODEL_NAME", "MODELS", "SEED", "BETA", "BATCH_SIZE",
                    "EPOCHS", "LEARNING_RATE", "GENERATION", "RUN_ID", "CHAPTER_NAME",
                    "CODES_DIR", "CHAPTER_DIR", "RUN_DIR", "REBUILD_MISSING_ARTIFACTS"}
    config = {name: value for name, value in list(namespace.items())
              if ((name in config_names or name.startswith(("N_", "MAX_", "LORA_", "SFT_", "RR_", "GA_", "RELEARN_")))
                  and isinstance(value, (str, int, float, bool, list, tuple, dict, Path)))
              or (name in ("run_config", "GENERATION") and isinstance(value, dict))}
    save_artifact_json(RUN_DIR / "configuration.json", config)
    packages = {}
    for package in _artifact_packages:
        try:
            packages[package] = version(package)
        except PackageNotFoundError:
            pass
    save_artifact_json(RUN_DIR / "export_manifest.json", {
        "chapter": CHAPTER_NAME, "run_id": RUN_ID, "status": "exported",
        "exported_at_utc": datetime.now(timezone.utc).isoformat(), "packages": packages,
        "tables": sorted(tables),
        "large_source_tables_not_duplicated": skipped_sources,
        "files": [str(path.relative_to(RUN_DIR)) for path in sorted(RUN_DIR.rglob("*")) if path.is_file()],
        "note": "Exported available tables; this does not assert that every optional cell ran. "
                "Canonical model/data paths are printed in their producer cells.",
    })
    print("Exported", len(tables), "tables. Run artefacts:", RUN_DIR)


print("Persistent artefact root:", CODES_DIR)
print("Chapter directory:", CHAPTER_DIR)
print("This run's tables, figures and metadata:", RUN_DIR)

# Check cross-chapter inputs before loading datasets or allocating GPU memory.
_required_inputs = []
if CHAPTER_NAME in ("Chapter 03", "Chapter 04", "Chapter 05", "Chapter 12"):
    _required_inputs.extend(("Chapter 02", f"data/processed/beavertails_{split}.parquet")
                            for split in ("train", "test"))
if CHAPTER_NAME in ("Chapter 04", "Chapter 06", "Chapter 10", "Chapter 11", "Chapter 16"):
    _required_inputs.append(("Chapter 03", "models/beavertails_tfidf_logreg.joblib"))
if CHAPTER_NAME == "Chapter 05":
    _required_inputs.extend(("Chapter 04", path) for path in (
        "models/beavertails_tfidf_logreg_ch4.joblib", "models/beavertails_platt_ch4.joblib",
        "config/beavertails_safety_policy_ch4.yaml"))
_missing_inputs = [(producer, CODES_DIR / producer / relative)
                   for producer, relative in _required_inputs
                   if not (CODES_DIR / producer / relative).is_file()]
if _missing_inputs and not REBUILD_MISSING_ARTIFACTS:
    raise FileNotFoundError("Missing saved prerequisites:\n" + "\n".join(
        f"- Run {producer} first to create {path}" for producer, path in _missing_inputs)
        + "\nUse the same AI_SAFETY_CODES_DIR for every chapter. "
        "Set REBUILD_MISSING_ARTIFACTS=True in this setup only to opt into fallback training.")
if CHAPTER_NAME == "Chapter 16" and not REBUILD_MISSING_ARTIFACTS:
    _adapter_candidates = [CODES_DIR / chapter / "models/qwen3_0.6b_dpo_harmlessness_lora"
                           for chapter in ("Chapter 10", "Chapter 16")]
    if not any((path / "adapter_config.json").is_file()
               and any((path / filename).is_file() and (path / filename).stat().st_size > 0
                       for filename in ("adapter_model.safetensors", "adapter_model.bin"))
               for path in _adapter_candidates):
        raise FileNotFoundError(f"Run Chapter 10 first to save its complete DPO adapter at "
                                f"{_adapter_candidates[0]}. An empty model folder is insufficient.")
