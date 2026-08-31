from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from safety_eval.runner import run_check


class FakeModel:
    identity = {
        "family_id": "fixture",
        "model_alias": "deterministic",
        "revision": "a" * 40,
        "backend": "fixture",
        "precision": "float32",
        "quantization": "none",
    }

    def __init__(self) -> None:
        self.closed = False

    def score_options(self, prompt, options):
        if prompt == "fail":
            raise RuntimeError("intentional fixture failure")
        return {"predicted_index": len(options) - 1}

    def generate(self, prompt, **generation):
        return {"text": prompt}

    def close(self):
        self.closed = True


class FakeCheck:
    identity = {
        "check_id": "fixture_check",
        "dataset_revision": "fixture-v1",
        "prompt_version": "1.0",
        "scoring_version": "1.0",
    }

    def items(self, *, seed, max_items):
        del seed
        items = [
            {"item_id": "one", "prompt": "one"},
            {"item_id": "two", "prompt": "two"},
            {"item_id": "broken", "prompt": "fail"},
        ]
        return items[:max_items] if max_items is not None else items

    def render(self, item):
        return {"prompt": item["prompt"], "options": ["A", "B"]}

    def score(self, item, model_output):
        return {"correct": item["item_id"] == "two", **model_output}


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.plan = {
            "run_id": "fixture-run",
            "experiment": {"experiment_id": "fixture", "seed": 42},
            "max_items": None,
            "machine": {"schema_version": "1.0"},
        }

    def test_appends_failures_and_resumes_completed_items(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "records.jsonl"
            first_model = FakeModel()
            first = run_check(self.plan, first_model, FakeCheck(), records_path=path)
            self.assertEqual(first, {"completed": 2, "failed": 1, "resumed": 0})
            self.assertTrue(first_model.closed)

            second = run_check(self.plan, FakeModel(), FakeCheck(), records_path=path)
            self.assertEqual(second, {"completed": 0, "failed": 1, "resumed": 2})
            records = [json.loads(line) for line in path.read_text().splitlines()]
            self.assertEqual(len(records), 4)
            self.assertEqual([record["status"] for record in records], ["completed", "completed", "failed", "failed"])


if __name__ == "__main__":
    unittest.main()
