"""Offline storage and notebook-contract tests; no model/dataset downloads."""
import ast
import contextlib
import io
import json
import os
from pathlib import Path
import tempfile
import types
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
TEMPLATE = (ROOT / "scripts/notebook_storage.py").read_text()


def setup(root, chapter="Chapter 02", source=TEMPLATE):
    namespace = {}
    with patch.dict(os.environ, {"AI_SAFETY_CODES_DIR": str(root)}), contextlib.redirect_stdout(io.StringIO()):
        exec(source.replace('CHAPTER_NAME = "__CHAPTER__"', f'CHAPTER_NAME = "{chapter}"'), namespace)
    return namespace


class StorageTests(unittest.TestCase):
    def test_actual_chapter_2_to_5_save_load_cells(self):
        import pandas as pd
        import joblib
        from sklearn.model_selection import train_test_split

        def cell_source(chapter, needle):
            path = next((ROOT / chapter).glob("*.ipynb"))
            matches = ["".join(c["source"]) for c in json.loads(path.read_text())["cells"]
                       if c["cell_type"] == "code" and needle in "".join(c["source"])
                       and c.get("id") != "persistent-artifact-setup"]
            self.assertEqual(len(matches), 1)
            return matches[0]

        with tempfile.TemporaryDirectory() as directory, contextlib.redirect_stdout(io.StringIO()):
            root = Path(directory)
            second = setup(root)
            second.update({"pd": pd, "train_test_split": train_test_split,
                           "df": pd.DataFrame({"prompt": ["prompt"] * 20, "response": ["answer"] * 20,
                                               "text": ["text"] * 20, "category": [{"fixture": True}] * 20,
                                               "target": [0, 1] * 10})})
            exec(cell_source("Chapter 02", 'train_df.to_parquet('), second)
            third = setup(root, "Chapter 03")
            exec(cell_source("Chapter 03", 'TRAIN_PATH ='), third)
            self.assertIn("category", third["train_df"])
            third["safety_clf"] = {"test_classifier": True}
            exec(cell_source("Chapter 03", 'joblib.dump(safety_clf,'), third)
            fourth = setup(root, "Chapter 04")
            exec(cell_source("Chapter 04", 'MODEL_PATH ='), fourth)
            self.assertEqual(fourth["safety_clf"], third["safety_clf"])
            fourth.update({"base_clf": {"test_model": True}, "platt": {"test_calibrator": True},
                           "threshold_table": pd.DataFrame({"threshold": [0.5]})})
            exec(cell_source("Chapter 04", 'joblib.dump(base_clf,'), fourth)
            policy_path = fourth["CHAPTER_DIR"] / "config/beavertails_safety_policy_ch4.yaml"
            policy_path.parent.mkdir(parents=True)
            policy_path.write_text("policy:\n  block_at_or_above: 0.5\n")
            fifth = setup(root, "Chapter 05")
            exec(cell_source("Chapter 05", 'DATA_DIR ='), fifth)
            self.assertEqual(joblib.load(fifth["BASE_MODEL_PATH"]), fourth["base_clf"])
            self.assertEqual(joblib.load(fifth["CALIBRATOR_PATH"]), fourth["platt"])
            self.assertEqual(fifth["POLICY_PATH"], policy_path)

    def test_standalone_and_second_runtime_share_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            first = setup(root)
            split = first["CHAPTER_DIR"] / "data/processed"
            split.mkdir(parents=True)
            for name in ("train", "test"):
                (split / f"beavertails_{name}.parquet").write_bytes(b"test fixture")
            second = setup(root, "Chapter 03")
            self.assertEqual(first["CODES_DIR"], second["CODES_DIR"])
            self.assertNotEqual(first["RUN_DIR"], setup(root)["RUN_DIR"])

    def test_missing_producer_is_actionable(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(FileNotFoundError, "Run Chapter 02 first"):
                setup(Path(directory), "Chapter 03")

    def test_colab_mounts_before_custom_root(self):
        with tempfile.TemporaryDirectory() as directory:
            mount = Path(directory) / "drive"
            drive = types.SimpleNamespace(mount=lambda path: (Path(path) / "MyDrive").mkdir(parents=True))
            fake_colab = types.ModuleType("google.colab")
            fake_colab.drive = drive
            source = TEMPLATE.replace('/content/drive', str(mount))
            with patch.dict("sys.modules", {"google.colab": fake_colab}):
                namespace = setup(mount / "MyDrive/custom", source=source)
                self.assertTrue((mount / "MyDrive").is_dir())
                self.assertEqual(namespace["CODES_DIR"], (mount / "MyDrive/custom").resolve())
                with self.assertRaisesRegex(ValueError, "must be inside"):
                    setup(Path(directory) / "ephemeral", source=source)

    def test_colab_default_and_mount_failure(self):
        with tempfile.TemporaryDirectory() as directory:
            mount = Path(directory) / "drive"
            fake_colab = types.ModuleType("google.colab")
            fake_colab.drive = types.SimpleNamespace(mount=lambda path: None)
            source = TEMPLATE.replace('/content/drive', str(mount))
            with patch.dict("sys.modules", {"google.colab": fake_colab}):
                with self.assertRaisesRegex(RuntimeError, "not mounted"):
                    setup(mount / "MyDrive/custom", source=source)
                fake_colab.drive.mount = lambda path: (Path(path) / "MyDrive").mkdir(parents=True)
                with patch.dict(os.environ, {"AI_SAFETY_CODES_DIR": ""}):
                    namespace = {}
                    exec(source.replace('__CHAPTER__', 'Chapter 02'), namespace)
                self.assertEqual(namespace["CODES_DIR"], (mount / "MyDrive/AI Safety from First Principles/artifacts/codes").resolve())

    def test_model_archive_survives_later_run(self):
        class Model:
            config = types.SimpleNamespace(_name_or_path="fixture/model")
            def __init__(self, weights):
                self.weights = weights
            def save_pretrained(self, path):
                Path(path).mkdir(parents=True, exist_ok=True)
                (Path(path) / "adapter_config.json").write_text("{}")
                (Path(path) / "adapter_model.safetensors").write_bytes(self.weights)

        class Tokenizer:
            def save_pretrained(self, path):
                (Path(path) / "tokenizer_config.json").write_text("{}")

        with tempfile.TemporaryDirectory() as directory:
            first = setup(Path(directory))
            canonical = first["CHAPTER_DIR"] / "models/example"
            first["save_model_artifact"](Model(b"old"), canonical, Tokenizer())
            second = setup(Path(directory))
            second["save_model_artifact"](Model(b"new"), canonical, Tokenizer())
            self.assertEqual((canonical / "adapter_model.safetensors").read_bytes(), b"new")
            self.assertEqual((first["RUN_DIR"] / "checkpoints/models/example/adapter_model.safetensors").read_bytes(), b"old")
            self.assertTrue((canonical / "tokenizer_config.json").is_file())

    def test_table_figure_and_manifest(self):
        import pandas as pd
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        with tempfile.TemporaryDirectory() as directory:
            namespace = setup(Path(directory))
            table = pd.DataFrame({"response": ["test"], "nested": [{"safe": True}]}, index=[9])
            multi = table.copy()
            multi.columns = pd.MultiIndex.from_tuples([("a", "response"), ("a", "nested")])
            namespace["save_tables"]("test", results=table, multi=multi)
            saved = pd.read_json(namespace["RUN_DIR"] / "tables/test/results.json", orient="split")
            pd.testing.assert_frame_equal(saved, table)
            plt.plot([0, 1], [0, 1])
            namespace["save_figures"]()
            plt.close("all")
            namespace["export_notebook_tables"]({"results": table, "run_config": {"seed": 42}, "HF_TOKEN": "not-a-real-token"})
            manifest = json.loads((namespace["RUN_DIR"] / "export_manifest.json").read_text())
            self.assertNotIn("HF_TOKEN", (namespace["RUN_DIR"] / "configuration.json").read_text())
            self.assertEqual(manifest["tables"], ["results"])
            self.assertTrue(list((namespace["RUN_DIR"] / "figures").glob("*.png")))
            self.assertFalse(list(namespace["RUN_DIR"].rglob("*.tmp-*")))

    def test_empty_adapter_fails_before_model_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            judge = root / "Chapter 03/models/beavertails_tfidf_logreg.joblib"
            judge.parent.mkdir(parents=True)
            judge.write_bytes(b"fixture")
            adapter = root / "Chapter 10/models/qwen3_0.6b_dpo_harmlessness_lora"
            adapter.mkdir(parents=True)
            with self.assertRaisesRegex(FileNotFoundError, "Run Chapter 10 first"):
                setup(root, "Chapter 16")
            (adapter / "adapter_config.json").write_text("{}")
            (adapter / "adapter_model.safetensors").write_bytes(b"fixture")
            setup(root, "Chapter 16")

    def test_all_notebook_code_and_storage_setup(self):
        from IPython.core.inputtransformer2 import TransformerManager
        transform = TransformerManager()
        notebooks = list(ROOT.glob("Chapter */*.ipynb"))
        self.assertEqual(len(notebooks), 15)
        for path in notebooks:
            notebook = json.loads(path.read_text())
            codes = [cell for cell in notebook["cells"] if cell["cell_type"] == "code"]
            self.assertEqual(codes[0]["id"], "persistent-artifact-setup")
            self.assertEqual("".join(codes[0]["source"]), TEMPLATE.replace(
                'CHAPTER_NAME = "__CHAPTER__"', f'CHAPTER_NAME = "{path.parent.name}"'))
            self.assertEqual(codes[-1]["id"], "artefact-final-export")
            for cell in codes:
                source = "".join(cell["source"])
                ast.parse(transform.transform_cell(source))
                if cell.get("id") != "persistent-artifact-setup":
                    self.assertNotIn('Path("../Chapter', source)
                    self.assertNotIn('Path("models")', source)
                    self.assertNotIn('Path("data/processed")', source)


if __name__ == "__main__":
    unittest.main()
