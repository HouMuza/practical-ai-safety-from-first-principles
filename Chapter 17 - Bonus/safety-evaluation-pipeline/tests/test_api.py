from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from safety_eval.api import discover_runs, discover_studies


class RunDiscoveryTests(unittest.TestCase):
    def test_discovers_summaries_without_exposing_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "Family" / "experiments" / "001" / "runs" / "run-one" / "check.jsonl"
            output.parent.mkdir(parents=True)
            record = {
                "run_id": "run-one",
                "experiment_id": "experiment-one",
                "item_id": "private-item",
                "model": {"family_id": "family", "model_alias": "model", "revision": "a" * 40, "backend": "fixture", "device": "cpu", "precision": "float32", "quantization": "none"},
                "check": {"check_id": "check", "dataset_revision": "rev", "dataset_sha256": "b" * 64},
                "status": "completed",
                "completed_at": "2026-08-31T00:00:00+00:00",
                "input": {"prompt": "must not escape"},
                "output": {"score": {"correct": True, "language": "en", "category": "ethics"}},
            }
            output.write_text(json.dumps(record) + "\n", encoding="utf-8")
            summary = discover_runs(root)
            serialised = json.dumps(summary)
            self.assertEqual(summary["totals"]["completed_records"], 1)
            self.assertFalse(summary["runs"][0]["publishable_outcome"])
            self.assertNotIn("must not escape", serialised)
            self.assertNotIn("private-item", serialised)

    def test_groups_shared_run_files_by_model_and_sanitizes_studies(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            output = root / "Family" / "experiments" / "001" / "runs" / "run-one" / "check.jsonl"
            output.parent.mkdir(parents=True)
            rows = []
            for alias in ("a", "b"):
                rows.append({"run_id": "run-one", "experiment_id": "confirmatory-study", "item_id": f"private-{alias}", "model": {"family_id": "family", "model_alias": alias}, "check": {"check_id": "check"}, "status": "completed", "output": {"score": {"correct": True}}})
            output.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            experiment = output.parents[2] / "confirmatory-experiment.json"
            experiment.write_text(json.dumps({"experiment_id": "confirmatory-study", "title": "Study", "models": [{"family_id": "family", "model_alias": "a"}, {"family_id": "family", "model_alias": "b"}], "checks": ["check"], "profile": "cpu", "seed": 1}), encoding="utf-8")
            samples = experiment.parent / "samples"
            samples.mkdir()
            (samples / "confirmatory.json").write_text(json.dumps({"experiment_id": "confirmatory-study", "evidence_class": "confirmatory", "item_count": 2, "item_ids": ["private-a", "private-b"], "item_ids_sha256": "c" * 64}), encoding="utf-8")
            summary = discover_runs(root)
            studies = discover_studies(root)
            self.assertEqual(len(summary["runs"]), 2)
            self.assertEqual(summary["totals"]["completed_records"], 2)
            self.assertEqual(len(studies), 1)
            self.assertNotIn("item_ids", studies[0]["samples"][0])


if __name__ == "__main__":
    unittest.main()
