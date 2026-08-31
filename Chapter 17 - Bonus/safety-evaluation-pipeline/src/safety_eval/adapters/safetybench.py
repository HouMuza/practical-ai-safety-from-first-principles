"""SafetyBench check with local data loading and deterministic stratification."""

from __future__ import annotations

import hashlib
import json
import random
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Iterable, Mapping

from ..mcq import safetybench_prompt


class SafetyBenchCheck:
    def __init__(self, manifest: dict[str, Any], dataset_path: Path, frozen_item_ids: list[str] | None = None) -> None:
        self.manifest = manifest
        self.dataset_path = dataset_path.resolve()
        self._items = self._load_items(self.dataset_path)
        self._frozen_item_ids = frozen_item_ids
        dataset = manifest["dataset"]
        scoring = manifest["scoring"]
        self._identity = {
            "check_id": manifest["check_id"],
            "dataset_revision": dataset["revision"],
            "prompt_version": scoring["prompt_renderer"],
            "scoring_version": scoring["scoring_version"],
            "dataset_path": str(self.dataset_path),
            "dataset_sha256": hashlib.sha256(self.dataset_path.read_bytes()).hexdigest(),
        }

    @property
    def identity(self) -> dict[str, Any]:
        return dict(self._identity)

    @staticmethod
    def _load_items(path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            raise FileNotFoundError(f"SafetyBench data was not found: {path}")
        if path.suffix == ".jsonl":
            raw_items = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
        elif path.suffix == ".json":
            payload = json.loads(path.read_text(encoding="utf-8"))
            raw_items = payload if isinstance(payload, list) else payload.get("items", payload.get("data"))
        else:
            raise ValueError("SafetyBench data must be a .json or .jsonl file")
        if not isinstance(raw_items, list):
            raise ValueError("SafetyBench file must contain a list or an object with an items/data list")
        return [SafetyBenchCheck._normalise(item) for item in raw_items]

    @staticmethod
    def _normalise(item: Mapping[str, Any]) -> dict[str, Any]:
        question = str(item.get("question", item.get("prompt", ""))).strip()
        options = item.get("options", item.get("choices"))
        if not question or not isinstance(options, list) or len(options) < 2:
            raise ValueError("Each SafetyBench item needs a question and at least two options")
        answer = item.get("answer", item.get("label"))
        if isinstance(answer, str) and len(answer.strip()) == 1 and answer.upper() in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
            answer_index = ord(answer.upper()) - ord("A")
        else:
            answer_index = int(answer)
        if answer_index < 0 or answer_index >= len(options):
            raise ValueError("SafetyBench answer index is outside the options")
        language = str(item.get("language", item.get("lang", "en"))).lower()
        language = "zh" if language.startswith("zh") or language.startswith("chinese") else "en"
        category = str(item.get("category", item.get("domain", "uncategorised")))
        stable_content = json.dumps([question, options, language, category], ensure_ascii=False, separators=(",", ":"))
        item_id = str(item.get("item_id", item.get("id", hashlib.sha256(stable_content.encode()).hexdigest()[:20])))
        return {"item_id": item_id, "question": question, "options": [str(option) for option in options], "answer_index": answer_index, "language": language, "category": category}

    def items(self, *, seed: int, max_items: int | None) -> Iterable[Mapping[str, Any]]:
        if self._frozen_item_ids is not None:
            by_id = {item["item_id"]: item for item in self._items}
            missing = [item_id for item_id in self._frozen_item_ids if item_id not in by_id]
            if missing:
                raise ValueError(f"Frozen sample contains {len(missing)} IDs absent from the dataset")
            if len(self._frozen_item_ids) != len(set(self._frozen_item_ids)):
                raise ValueError("Frozen sample contains duplicate item IDs")
            return [by_id[item_id] for item_id in self._frozen_item_ids]
        groups: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
        for item in self._items:
            groups[(item["language"], item["category"])].append(item)
        generator = random.Random(seed)
        strata = []
        for key in sorted(groups):
            values = list(groups[key])
            generator.shuffle(values)
            strata.append(deque(values))
        generator.shuffle(strata)
        selected = []
        while strata and (max_items is None or len(selected) < max_items):
            remaining = []
            for stratum in strata:
                if max_items is not None and len(selected) >= max_items:
                    break
                selected.append(stratum.popleft())
                if stratum:
                    remaining.append(stratum)
            strata = remaining
        return selected

    def render(self, item: Mapping[str, Any]) -> dict[str, Any]:
        prompt, labels = safetybench_prompt(str(item["question"]), item["options"], language=str(item["language"]))
        return {"prompt": prompt, "options": labels, "language": item["language"], "category": item["category"]}

    def score(self, item: Mapping[str, Any], model_output: Mapping[str, Any]) -> dict[str, Any]:
        prediction = int(model_output["predicted_index_total"])
        return {"predicted_index": prediction, "answer_index": int(item["answer_index"]), "correct": prediction == int(item["answer_index"]), "category": item["category"], "language": item["language"]}
