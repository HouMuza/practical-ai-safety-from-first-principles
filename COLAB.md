# Running the notebooks with persistent artefacts

Open or upload the updated notebook in Colab. Run the **Persistent artefacts** setup cell first and authorise Google Drive. The storage setup is embedded in each notebook; you do not need to upload the Python scripts with it. Install the repository's `requirements.txt` in your runtime before running the experiment cells. GPU chapters need a GPU runtime.

Every chapter defaults to this shared folder:

```text
MyDrive/AI Safety from First Principles/artifacts/codes/
  Chapter 02/data/processed/       # train/test splits
  Chapter 03/models/              # classifier
  Chapter 04/models/              # calibrated model
  Chapter 04/config/              # operating policy
  Chapter 10/models/              # DPO adapter, tokenizer and configuration
  Chapter XX/runs/<UTC-run-id>/    # tables, figures, metadata and model archives
```

Colab's `/content` files disappear with its runtime. A notebook saved in Drive does not itself preserve the runtime's models or generated files. The notebooks therefore write artefacts to the mounted Drive directly; setup stops if Drive cannot mount. See the [Colab FAQ](https://research.google.com/colaboratory/faq.html).

To choose a different Drive folder, add this cell **before** the storage setup in every notebook:

```python
import os
os.environ["AI_SAFETY_CODES_DIR"] = "/content/drive/MyDrive/My Safety Experiments/artifacts/codes"
```

The path is the shared parent of the chapter folders, not a chapter folder or the Git checkout. Drive mounts before this override is used. In hosted Colab the override must stay inside `/content/drive`.

## Chapter prerequisites

| Chapter | Saved inputs to produce first |
|---|---|
| 3 | Chapter 2 splits |
| 4 | Chapter 2 splits and Chapter 3 classifier |
| 5 | Chapter 2 splits and Chapter 4 models/policy |
| 6, 10, 11 | Chapter 3 classifier |
| 12 | Chapter 2 splits |
| 16 | Chapter 3 classifier and Chapter 10 DPO adapter |

Missing inputs stop early with their full path and producer chapter. Complete the producer notebook using the same Drive root, then retry. A missing model does not trigger automatic retraining. Setting `REBUILD_MISSING_ARTIFACTS = True` in the setup explicitly enables the existing fallback routines.

Chapter 2 now includes category metadata in its saved splits. Chapter 5 can upgrade an older split missing that column using the same deterministic split procedure; this requires a dataset download but does not retrain its models.

## What is saved

- Existing dataset, report, configuration and model saves now use persistent chapter paths.
- Each setup execution creates a new dated run directory. Hugging Face model/adapter saves include the tokenizer and a run archive before publishing to the stable chapter path. Stable paths hold the latest published checkpoint; previous run archives remain available. Classical Chapter 3/4 classifier files use stable paths and are replaced on rerun. Avoid concurrent writes to the same chapter's canonical paths.
- Figures are saved before explicit `plt.show()` calls. The final export cell also saves figures still open.
- The final **Save remaining results** cell exports top-level DataFrames as JSON, experiment configuration, package versions and a manifest. Large source dataset tables are excluded from this duplicate export (listed in the manifest); sampled training/evaluation tables and the canonical Chapter 2 splits are retained. It can also be run before stopping early; its manifest says which tables exist, not that all optional experiments succeeded.
- Chapter 9 additionally saves reward heads and the row ordering/configuration for its embedding arrays.
- Chapter 10 saves its training log, sampled preference pairs, validation preferences, behavioural responses and scores, bootstrap intervals, and reward-hacking candidate/selection tables. Behavioural rows are saved after each completed response.
- Chapter 15 additionally saves the gradient-ascent and retain-regularised unlearning adapters before subsequent experiments can modify them, plus the sequential-unlearning adapter.

Read an ordinary exported table with:

```python
import pandas as pd
table = pd.read_json(RUN_DIR / "tables/final/result_table.json", orient="split")
```

JSON exports retain nested response metadata and index values. For tables with multi-level columns/indexes, inspect the JSON's `columns`, `index` and `data` fields when reconstructing the MultiIndex.

Model checkpoints are loadable weights, **not exact training-resume checkpoints**: optimiser state, scheduler state and random-number-generator state are not saved. Only completed saves survive a disconnect; an interrupted training cell may need to be rerun. The final export does not save the notebook itself—use Colab's Save a copy in Drive / Download notebook for the executed `.ipynb` and its displayed outputs.

No model training or full Colab run is implied by the repository's saved historical notebook outputs. Run the updated notebook from the beginning to produce a fresh set of artefacts.

## Maintaining the embedded setup

The notebooks work independently. Their common storage cell is generated from `scripts/notebook_storage.py`:

```bash
python scripts/normalize_notebook_artifact_paths.py
python scripts/test_notebook_storage.py
```

The tests exercise storage using temporary directories and a simulated Colab mount. They do not download models or run training.
