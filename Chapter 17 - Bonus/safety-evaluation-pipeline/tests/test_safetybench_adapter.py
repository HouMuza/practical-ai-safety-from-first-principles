from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path


sys.path.insert(0, str(Path(__file__).parents[1] / "src"))

from safety_eval.adapters.safetybench import SafetyBenchCheck


MANIFEST = {
    "check_id": "safetybench",
    "dataset": {"revision": "fixture-v1"},
    "scoring": {"prompt_renderer": "fixture-prompt-v1", "scoring_version": "1.0"},
}


class SafetyBenchAdapterTests(unittest.TestCase):
    def make_check(self, directory: str) -> SafetyBenchCheck:
        items = [
            {"id": "en-a-1", "question": "Q1?", "options": ["x", "y"], "answer": "A", "language": "en", "category": "a"},
            {"id": "en-a-2", "question": "Q2?", "options": ["x", "y"], "answer": 1, "language": "en", "category": "a"},
            {"id": "zh-b-1", "question": "问题？", "options": ["甲", "乙"], "answer": "B", "language": "zh", "category": "b"},
            {"id": "en-c-1", "question": "Q3?", "options": ["x", "y"], "answer": 0, "language": "en", "category": "c"},
        ]
        path = Path(directory) / "items.json"
        path.write_text(json.dumps(items), encoding="utf-8")
        return SafetyBenchCheck(MANIFEST, path)

    def test_sampling_is_deterministic_and_stratified(self):
        with tempfile.TemporaryDirectory() as directory:
            check = self.make_check(directory)
            first = list(check.items(seed=7, max_items=3))
            second = list(check.items(seed=7, max_items=3))
            self.assertEqual([item["item_id"] for item in first], [item["item_id"] for item in second])
            self.assertEqual(len({(item["language"], item["category"]) for item in first}), 3)

    def test_render_and_score(self):
        with tempfile.TemporaryDirectory() as directory:
            check = self.make_check(directory)
            item = next(item for item in check.items(seed=1, max_items=None) if item["item_id"] == "zh-b-1")
            rendered = check.render(item)
            self.assertTrue(rendered["prompt"].startswith("问题："))
            score = check.score(item, {"predicted_index_total": 1})
            self.assertTrue(score["correct"])

    def test_frozen_ids_override_sampling_and_preserve_order(self):
        with tempfile.TemporaryDirectory() as directory:
            original = self.make_check(directory)
            ids = [item["item_id"] for item in original.items(seed=2, max_items=2)]
            frozen = SafetyBenchCheck(MANIFEST, original.dataset_path, frozen_item_ids=list(reversed(ids)))
            selected = list(frozen.items(seed=999, max_items=1))
            self.assertEqual([item["item_id"] for item in selected], list(reversed(ids)))


if __name__ == "__main__":
    unittest.main()
