from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from safety_eval.analysis import analyze_records
from safety_eval.sampling import item_ids_sha256


class AnalysisTests(unittest.TestCase):
    def test_writes_sanitized_paired_pilot_outputs(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            item_ids = ["one", "two", "three", "four"]
            sample = {
                "schema_version": "1.0",
                "experiment_id": "experiment",
                "check": {"check_id": "fixture", "dataset_sha256": "d" * 64},
                "evidence_class": "pilot",
                "seed": 17,
                "sampling_method": "fixture",
                "item_count": 4,
                "item_ids": item_ids,
                "item_ids_sha256": item_ids_sha256(item_ids),
                "strata": {"en/a": 4},
                "frozen_at": "2026-08-31T00:00:00+00:00",
            }
            experiment = {"experiment_id": "experiment", "seed": 17, "models": [{"model_alias": "a"}, {"model_alias": "b"}]}
            sample_path = root / "sample.json"
            experiment_path = root / "experiment.json"
            records_path = root / "records.jsonl"
            sample_path.write_text(json.dumps(sample), encoding="utf-8")
            experiment_path.write_text(json.dumps(experiment), encoding="utf-8")
            rows = []
            outcomes = {"a": [True, True, False, False], "b": [True, False, True, False]}
            for alias, values in outcomes.items():
                for item_id, correct in zip(item_ids, values):
                    rows.append({"model": {"model_alias": alias, "revision": alias * 40, "backend": "fixture", "precision": "float32", "quantization": "none"}, "check": {"dataset_revision": "fixture", "dataset_sha256": "d" * 64, "prompt_version": "1", "scoring_version": "1"}, "item_id": item_id, "status": "completed", "environment": {"machine": {}, "software": {}}, "input": {"prompt": "private benchmark text"}, "output": {"score": {"correct": correct, "language": "en", "category": "a"}}})
            records_path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            output = root / "publication"
            result = analyze_records(records_path, experiment_path=experiment_path, sample_path=sample_path, output_directory=output, bootstrap_iterations=100)
            self.assertEqual(result["reporting_status"], "preliminary")
            self.assertFalse(result["publishable_outcome"])
            self.assertTrue(result["complete_matched_coverage"])
            comparison = json.loads((output / "paired-comparisons.json").read_text())["comparisons"][0]
            self.assertIn("mcnemar_p_holm", comparison)
            paired = json.loads((output / "paired-comparisons.json").read_text())
            self.assertEqual(paired["omnibus"]["test"], "Cochran Q")
            self.assertEqual(paired["omnibus"]["degrees_of_freedom"], 1)
            self.assertIn("stratified", paired["method"]["interval"])
            published = "\n".join(path.read_text() for path in output.iterdir() if path.is_file())
            self.assertNotIn("private benchmark text", published)
            self.assertNotIn('"item_ids"', published)


if __name__ == "__main__":
    unittest.main()
